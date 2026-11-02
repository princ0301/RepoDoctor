"""Run the agent on cases, grade, and summarize."""

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from repodoctor.config import Settings
from repodoctor.evals.cases import EvalCase, changed_files
from repodoctor.evals.grade import grade_tree
from repodoctor.runner import run_fix_issue


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
        }
        for h in history
    ]


def classify(final: dict, grade: dict | None) -> str:
    err = final.get("error")
    if err:
        if err.startswith("Could not write a test"):
            return "no_repro"
        if "model failed" in err:
            return "model_error"
        if "explorer did not" in err:
            return "explore_failed"
        return "setup_error"
    if not final.get("success"):
        return "fix_failed"
    if grade is None:
        return "grading_error"
    return "resolved" if grade["resolved"] else "wrong_fix"


def _detail(outcome: str, final: dict, grade: dict | None) -> str:
    if final.get("error"):
        return final["error"].splitlines()[0][:200]
    if outcome == "wrong_fix" and grade:
        bad = (grade["f2p_failed"] + grade["p2p_broken"])[:3]
        return ("gold tests failing: " + ", ".join(bad)) if bad else grade.get("note", "")
    if outcome == "fix_failed":
        return ((final.get("feedback") or "").strip().splitlines() or [""])[0][:200]
    return ""


def run_case(
    case: EvalCase, trial: int, settings: Settings, *, label: str
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
            r = run_case(case, trial, settings, label=label)
            results.append(r)
            with results_file.open("a", encoding="utf-8") as f:   # crash-safe
                f.write(json.dumps(asdict(r)) + "\n")
            log(f"  -> {r.outcome} in {r.seconds:.0f}s, {r.llm_calls} calls {r.detail}".rstrip())
            consecutive_model_errors = (
                consecutive_model_errors + 1 if r.outcome == "model_error" else 0
            )
            if consecutive_model_errors >= stop_after_model_errors:
                log(
                    "Stopping: the models keep failing (quota or rate limit?). "
                    "Resume later with --resume " + str(results_file)
                )
                return results
    return results


def load_results(path: Path) -> list[CaseResult]:
    """The latest record for each (case, trial), so resumed runs replace earlier errors."""
    latest: dict[tuple[str, int], CaseResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = CaseResult(**json.loads(line))
            latest[(r.case_id, r.trial)] = r
    return list(latest.values())


def summarize(results: list[CaseResult]) -> str:
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
    if len(case_ids) < n:
        once = sum(any(r.resolved for r in results if r.case_id == c) for c in case_ids)
        lines.append(f"Solved in at least one trial: {once}/{len(case_ids)} cases")
    counts = Counter(r.outcome for r in results)
    lines.append("Outcomes:           " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    calls = sum(r.llm_calls for r in results)
    lines.append(f"Model calls:        {calls} total, {calls / n:.0f} per run")
    lines.append(f"Mean time:          {sum(r.seconds for r in results) / n:.0f}s per run")
    return "\n".join(lines)
