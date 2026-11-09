"""Run the agent on cases, grade, and summarize."""

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from repodoctor.config import Settings
from repodoctor.evals.cases import EvalCase, changed_files
from repodoctor.evals.grade import grade_tree
from repodoctor.runner import run_fix_issue

_PATCH_KINDS = (
    ("paths in need_more_files do not exist", "asked_for_unknown_file"),
    ("Moved to the top", "asked_for_files"),
    ("already shown to you", "asked_for_shown_file"),
    ("Not shown because of the context budget", "budget_cut"),
    ("No edits were proposed", "no_edits"),
    ("not valid structured output", "malformed_output"),
    ("protected test file", "edits_test_file"),
    ("'old' text not found", "old_not_found"),
    ("'old' text appears", "old_ambiguous"),
    ("does not exist", "bad_path"),
    ("outside the repository", "bad_path"),
    ("did not change any code", "no_change"),
)


def attempt_kind(feedback: str) -> str:
    """Name the way one fix attempt ended, from the feedback it produced."""
    first = feedback.split("\n", 1)[0]  # test output below it could contain anything
    if "pass and no tests regressed" in first:
        return "passed"
    if "still fail" in first:
        return "test_still_fails"
    if "timed out" in first:
        return "test_timeout"
    if "broke tests" in first:
        return "regression"
    if "full test run produced no report" in first:
        return "no_report"
    head = feedback[:400]
    for needle, kind in _PATCH_KINDS:
        if needle in head:
            return kind
    return "other"


@dataclass
class CaseResult:
    case_id: str
    trial: int
    outcome: str
    resolved: bool
    agent_success: bool
    attempts: int
    repro_attempts: int
    seconds: float
    files_changed: list[str] = field(default_factory=list)
    touched_gold_files: bool = False
    detail: str = ""
    diff: str = ""
    gold_pr: str = ""
    history: list[dict] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    hypothesis: str = ""
    llm_calls: int = 0
    slowest_calls: list[dict] = field(default_factory=list)
    failed_calls: int = 0
    calls_by_model: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    context_omitted: list[str] = field(default_factory=list)
    explored_files: list[str] = field(default_factory=list)
    edited_tests: list[str] = field(default_factory=list)
    explore_trace: list[str] = field(default_factory=list)
    f2p_failed: list[str] = field(default_factory=list)
    p2p_broken: list[str] = field(default_factory=list)
    p2p_unseen: list[str] = field(default_factory=list)


def is_infrastructure_failure(r: CaseResult) -> bool:
    """Transient: the models were unavailable, or setup failed before any model call
    (for example an install). A baseline test run that fails is deterministic for a
    case, so it is a separate outcome ('baseline_failed') and does not stop a pass."""
    return r.outcome == "model_error" or (r.outcome == "setup_error" and r.llm_calls == 0)


def _slim_history(history: list[dict]) -> list[dict]:
    """Per-attempt record, trimmed so results.jsonl stays readable."""
    return [
        {
            "attempt": h.get("attempt"),
            "reasoning": (h.get("reasoning") or "")[:500],
            "edits": [
                {"path": e["path"], "old": e["old"][:300], "new": e["new"][:300]}
                for e in h.get("edits", [])
            ],
            "patch_errors": [err[:400] for err in h.get("patch_errors", [])],
            "feedback": (h.get("feedback") or "")[:800],
            "context_rounds": h.get("context_rounds", 0),
        }
        for h in history
    ]


def classify(final: dict, grade: dict | None) -> str:
    err = final.get("error")
    if err:
        if err.startswith("The run exceeded"):
            return "run_timeout"
        if err.startswith("Could not write a test"):
            return "no_repro"
        if "model failed" in err:
            return "model_error"
        if "explorer did not" in err:
            return "explore_failed"
        if err.startswith("The baseline test run"):
            return "baseline_failed"
        return "setup_error"
    if not final.get("success"):
        return "fix_failed"
    if grade is None:
        return "grading_error"
    if grade["resolved"]:
        return "resolved"
    broken = grade.get("p2p_broken", [])
    only_unseen = (
        not grade.get("f2p_failed") and bool(broken)
        and set(broken) <= set(grade.get("p2p_unseen", []))
    )
    return "unseen_test" if only_unseen else "wrong_fix"


def _detail(outcome: str, final: dict, grade: dict | None) -> str:
    if final.get("error"):
        return " | ".join(final["error"].splitlines()[:3])[:300]
    if outcome == "wrong_fix" and grade:
        bad = (grade["f2p_failed"] + grade["p2p_broken"])[:3]
        return ("gold tests failing: " + ", ".join(bad)) if bad else grade.get("note", "")
    if outcome == "fix_failed":
        return ((final.get("feedback") or "").strip().splitlines() or [""])[0][:200]
    return ""


def run_case(
    case: EvalCase,
    trial: int,
    settings: Settings,
    *,
    label: str,
    progress: Callable[[str], None] | None = None,
) -> CaseResult:
    started = time.monotonic()

    def post(sandbox, info, workdir, final):
        if not final.get("success"):
            return None   # nothing to grade
        return asdict(grade_tree(sandbox, info, workdir, case, settings.sandbox_timeout_s))

    try:
        final, _ = run_fix_issue(
            case.ref, settings, replay=True, targets=[], fix_pr=case.fix_pr, post=post,
            extra_tags=["eval", f"eval:{label}", f"case:{case.id}", f"trial:{trial}"],
            progress=progress,
        )
    except Exception as e:  # one broken case must not stop the whole batch
        return CaseResult(
            case.id, trial, "crash", False, False, 0, 0, time.monotonic() - started,
            detail=f"{type(e).__name__}: {e}"[:300],
        )

    grade = final.get("post")
    outcome = classify(final, grade)
    files = changed_files(final.get("full_diff") or final.get("diff") or "")
    return CaseResult(
        case_id=case.id,
        trial=trial,
        outcome=outcome,
        resolved=outcome == "resolved",
        agent_success=bool(final.get("success")),
        attempts=final.get("attempts", 0),
        repro_attempts=final.get("repro_attempts", 0),
        seconds=time.monotonic() - started,
        files_changed=files,
        touched_gold_files=bool(set(files) & set(case.src_files)),
        detail=_detail(outcome, final, grade),
        diff=final.get("full_diff") or final.get("diff") or "",
        gold_pr=f"https://github.com/{case.repo}/pull/{case.fix_pr}",
        history=_slim_history(final.get("history", [])),
        relevant_files=final.get("relevant_files", []),
        hypothesis=(final.get("hypothesis") or "")[:600],
        llm_calls=final.get("llm_calls", 0),
        slowest_calls=sorted(final.get("llm_timeline", []), key=lambda c: -c["seconds"])[:3],
        failed_calls=sum(1 for c in final.get("llm_timeline", []) if not c["ok"]),
        calls_by_model=dict(Counter(c["model"] for c in final.get("llm_timeline", []))),
        timings=final.get("timings", {}),
        context_omitted=final.get("context_omitted", []),
        explored_files=final.get("explored_files", []),
        edited_tests=final.get("edited_tests", []),
        explore_trace=final.get("explore_trace", []),
        f2p_failed=(grade or {}).get("f2p_failed", [])[:20],
        p2p_broken=(grade or {}).get("p2p_broken", [])[:20],
        p2p_unseen=(grade or {}).get("p2p_unseen", [])[:20],
    )


def run_eval(
    cases: list[EvalCase],
    settings: Settings,
    *,
    trials: int,
    label: str,
    results_file: Path,
    done: set[tuple[str, int]] | None = None,
    stop_after_model_errors: int = 2,
    log: Callable[[str], None] = lambda _m: None,
) -> list[CaseResult]:
    """Run every (case, trial) not in `done`. Stops early if the models keep failing."""
    done = done or set()
    results: list[CaseResult] = []
    results_file.parent.mkdir(parents=True, exist_ok=True)
    consecutive_model_errors = 0
    for case in cases:
        for trial in range(1, trials + 1):
            if (case.id, trial) in done:
                continue
            log(f"Running {case.id} (trial {trial}/{trials}) ...")
            r = run_case(case, trial, settings, label=label, progress=log)
            results.append(r)
            with results_file.open("a", encoding="utf-8") as f:   # crash-safe
                f.write(json.dumps(asdict(r)) + "\n")
            slow = max((c["seconds"] for c in r.slowest_calls), default=0)
            log(
                f"  -> {r.outcome} in {r.seconds:.0f}s, {r.llm_calls} calls "
                f"(slowest {slow:.0f}s) {r.detail}".rstrip()
            )
            consecutive_model_errors = (
                consecutive_model_errors + 1 if is_infrastructure_failure(r) else 0
            )
            if consecutive_model_errors >= stop_after_model_errors:
                log(
                    "Stopping: the models keep failing (quota, rate limit or network?). "
                    "Resume later with --resume " + str(results_file)
                )
                return results
    return results


def load_results(path: Path) -> list[CaseResult]:
    """The latest record for each (case, trial), so resumed runs replace earlier errors.
    Keys this version does not know (from older or newer versions) are ignored."""
    known = {f.name for f in fields(CaseResult)}
    latest: dict[tuple[str, int], CaseResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = CaseResult(**{k: v for k, v in json.loads(line).items() if k in known})
            latest[(r.case_id, r.trial)] = r
    return list(latest.values())


def compare(
    a: list[CaseResult], b: list[CaseResult], names: tuple[str, str] = ("A", "B")
) -> str:
    """Per-case and total comparison of two sets of runs on the same cases."""
    cases = sorted({r.case_id for r in a} | {r.case_id for r in b})

    def tally(rs: list[CaseResult], cid: str) -> tuple[int, int]:
        mine = [r for r in rs if r.case_id == cid]
        return sum(r.resolved for r in mine), len(mine)

    def attempts(rs: list[CaseResult]) -> list[dict]:
        return [h for r in rs for h in r.history]

    def no_edit(rs: list[CaseResult]) -> tuple[int, int]:
        at = attempts(rs)
        return sum(1 for h in at if not h.get("edits")), len(at)

    def used_rounds(rs: list[CaseResult]) -> tuple[int, int]:
        at = attempts(rs)
        return sum(1 for h in at if h.get("context_rounds", 0) > 0), len(at)

    def reached(rs: list[CaseResult]) -> tuple[int, int]:
        agent = [r for r in rs if r.llm_calls > 0]
        return sum(r.resolved for r in agent), len(agent)

    lines = [f"{'case':<40} {names[0]:>7} {names[1]:>7}"]
    better = worse = same = 0
    for cid in cases:
        (ra, na), (rb, nb) = tally(a, cid), tally(b, cid)
        lines.append(f"{cid:<40} {ra:>3}/{na:<3} {rb:>3}/{nb:<3}")
        if na and nb:
            fa, fb = ra / na, rb / nb
            better += fb > fa
            worse += fb < fa
            same += fb == fa
    (ea, ta), (eb, tb) = no_edit(a), no_edit(b)
    (ua, _), (ub, _) = used_rounds(a), used_rounds(b)
    (xa, ya), (xb, yb) = reached(a), reached(b)
    lines += [
        "",
        f"Resolved runs:      {names[0]} {sum(r.resolved for r in a)}/{len(a)}   "
        f"{names[1]} {sum(r.resolved for r in b)}/{len(b)}",
        f"Runs that reached the agent: {names[0]} {xa}/{ya} resolved   "
        f"{names[1]} {xb}/{yb} resolved",
        f"Cases by resolve rate: {names[1]} better in {better}, worse in {worse}, same in {same}",
        f"Fix attempts with no edit: {names[0]} {ea}/{ta}   {names[1]} {eb}/{tb}",
        f"Fix attempts that used a free file-request round: "
        f"{names[0]} {ua}/{ta}   {names[1]} {ub}/{tb}",
    ]
    return "\n".join(lines)


def summarize(
    results: list[CaseResult], scope_ids: set[str] | frozenset[str] = frozenset()
) -> str:
    if not results:
        return "No results."
    n = len(results)
    resolved = sum(r.resolved for r in results)
    claimed = sum(r.agent_success for r in results)
    case_ids = sorted({r.case_id for r in results})

    lines = [
        f"{'case':<38} {'trial':>5}  {'outcome':<14} {'fix':>3} {'repro':>5} {'sec':>5}"
    ]
    for r in results:
        lines.append(
            f"{r.case_id:<38} {r.trial:>5}  {r.outcome:<14} "
            f"{r.attempts:>3} {r.repro_attempts:>5} {r.seconds:>5.0f}"
        )
    lines += ["", f"Resolved:           {resolved}/{n} ({100 * resolved / n:.0f}%)"]
    if claimed:
        lines.append(
            f"Agent said success: {claimed}/{n}  (precision {100 * resolved / claimed:.0f}%)"
        )
    else:
        lines.append(f"Agent said success: 0/{n}")
    hidden = sum(r.outcome == "unseen_test" for r in results)
    if hidden:
        lines.append(
            f"Fixed the issue and every test the agent could see: {resolved + hidden}/{n} "
            f"(resolved {resolved} + {hidden} that failed only tests whose ids are absent "
            "from the base repository)"
        )
    if len(case_ids) < n:
        once = sum(any(r.resolved for r in results if r.case_id == c) for c in case_ids)
        lines.append(f"Solved in at least one trial: {once}/{len(case_ids)} cases")
    counts = Counter(r.outcome for r in results)
    lines.append("Outcomes:           " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    calls = sum(r.llm_calls for r in results)
    lines.append(f"Model calls:        {calls} total, {calls / n:.0f} per run")
    node_time: Counter = Counter()
    for r in results:
        node_time.update(r.timings)
    if node_time:
        other = max(0.0, sum(r.seconds for r in results) - sum(node_time.values()))
        parts = [f"{k}={v / n:.0f}s" for k, v in node_time.most_common()]
        parts.append(f"other={other / n:.0f}s")
        lines.append("Mean time by step:  " + ", ".join(parts))
    scope = [r for r in results if r.case_id in scope_ids]
    if scope:
        core = [r for r in results if r.case_id not in scope_ids]
        lines.append(
            f"  without scope cases: {sum(r.resolved for r in core)}/{len(core)}   "
            f"scope cases only: {sum(r.resolved for r in scope)}/{len(scope)}"
        )
    worst = max(
        (c for r in results for c in r.slowest_calls), key=lambda c: c["seconds"], default=None
    )
    if worst:
        lines.append(f"Slowest model call: {worst['seconds']:.0f}s ({worst['model']})")
    failed_calls = sum(r.failed_calls for r in results)
    if failed_calls:
        lines.append(f"Failed model calls: {failed_calls} (retried or sent to a fallback)")
    edited = [r for r in results if r.edited_tests]
    if edited:
        lines.append(
            f"Runs that edited existing tests: {len(edited)} "
            f"(resolved: {sum(r.resolved for r in edited)})"
        )
    by_model: Counter = Counter()
    for r in results:
        by_model.update(r.calls_by_model)
    if by_model:
        total = sum(by_model.values())
        lines.append(
            "Calls by model:     "
            + ", ".join(f"{m}={n} ({100 * n / total:.0f}%)" for m, n in by_model.most_common())
        )
    secs = sorted(r.seconds for r in results)
    lines.append(f"Median time:        {secs[len(secs) // 2]:.0f}s per run")
    lines.append(f"Mean time:          {sum(r.seconds for r in results) / n:.0f}s per run")
    return "\n".join(lines)
