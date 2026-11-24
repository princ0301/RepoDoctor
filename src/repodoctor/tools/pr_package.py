"""Turn a successful run into a reviewable pull-request package on disk. Nothing is pushed."""

import difflib
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from repodoctor.config import Settings
from repodoctor.tools.conventions import (
    classify,
    convention_summary_lines,
    draft_changelog_entry,
    fragment_path,
    new_test_findings,
    render_checklist,
)
from repodoctor.tools.diffcheck import (
    check_diff,
    parse_diff,
    repo_line_length,
    syntax_findings,
    unused_import_findings,
)
from repodoctor.tools.findings import TODO_MARKER, Report
from repodoctor.tools.github import IssueData, RepoInfo
from repodoctor.tools.issue_ref import IssueRef
from repodoctor.tools.repo import commit_files, get_diff_since, remove_dir, run_git
from repodoctor.tools.review_hints import changelog_findings

REPODOCTOR_URL = "https://github.com/princ0301/RepoDoctor"
MIN_ROOT_CAUSE_WORDS = 8
_LINES = re.compile(
    r"\s*\(?\b(?:on |at |in )?lines? \d+(?:\s*[-–]\s*\d+)?"
    r"(?: of [\w./\\-]*\w)?\)?",
    re.I,
)
_SECTION = re.compile(r"^## Root cause[ \t]*\n(.*?)(?=^## |\Z)", re.S | re.M)
_FIX_SENTENCE = re.compile(
    r"^(?:the |a )?(?:proposed |suggested )?(?:fix|solution|remedy|workaround)\b"
    r"|^to (?:fix|resolve|solve)\b|^(?:we|you|it) (?:should|could|must|need)\b",
    re.I,
)

class PackageError(RuntimeError):
    """The package cannot be prepared or approved."""


def tidy_hypothesis(text: str, max_sentences: int = 3) -> str:
    """The agent's hypothesis as a root cause: no line numbers, no prescribed fix, a few sentences."""
    t = _LINES.sub("", " ".join((text or "").split()))
    t = re.sub(r"\s+([,.;:])", r"\1", t)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", t) if s]
    kept = [s for s in sentences if not _FIX_SENTENCE.match(s)] or sentences
    return " ".join(kept[:max_sentences]).strip()


def with_agent_cause(text: str, cause: str) -> str:
    """pr.md with the Root cause section replaced by `cause`."""
    return _SECTION.sub(lambda m: f"## Root cause\n\n{cause}\n\n", text, count=1)


def trailer_text(settings: Settings, model: str) -> str:
    return settings.commit_trailer.replace("{model}", model).strip()


def commit_message(title: str, issue: int, trailer: str = "") -> str:
    msg = f"{title}\n\nFixes #{issue}"
    return f"{msg}\n\n{trailer}" if trailer else msg


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_package(pkg: Path) -> dict:
    return json.loads((pkg / "package.json").read_text(encoding="utf-8"))


def save_package(pkg: Path, meta: dict) -> None:
    (pkg / "package.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def root_cause_section(pr_text: str) -> str:
    m = re.search(r"^## Root cause[ \t]*\n(.*?)(?=^## |\Z)", pr_text, re.S | re.M)
    return m.group(1).strip() if m else ""


def _issue_test_path(root: Path, old_rel: str, number: int) -> str:
    folder = old_rel.rsplit("/", 1)[0] + "/" if "/" in old_rel else ""
    candidate = f"{folder}test_issue_{number}.py"
    n = 2
    while (root / candidate).exists():
        candidate = f"{folder}test_issue_{number}_{n}.py"
        n += 1
    return candidate


def build_pr_text(
    *, ref: IssueRef, issue: IssueData, final: dict, user: str, test_id: str,
    template: str = "", facts: dict | None = None, trailer: str = "", agent_cause: str = "",
) -> tuple[str, str]:
    title = re.sub(r"^[#>*\-\s]+", "", " ".join(issue.title.split()))[:100] or f"Fix #{ref.number}"
    base = final.get("baseline") or {}
    preexisting = (base.get("failed") or 0) + (base.get("errors") or 0)
    suite = f"the unmodified code had {base.get('passed', 0)} passing tests"
    if preexisting:
        suite += f" and {preexisting} that already failed, which were ignored"
    body = [
        f"Fixes #{ref.number}.",
        "",
        "## Root cause",
        "",
        f"{TODO_MARKER}: one or two sentences, in your own words, on why the bug happens. "
        "Delete this line.",
        "",
        "## Verification",
        "",
        f"- The new test `{test_id}` fails on the unmodified code and passes with this change.",
        f"- The existing test suite shows no new failures ({suite}).",
        "",
    ]
    checklist = render_checklist(template, facts or {}) if template.strip() else ""
    if checklist:
        body += [checklist, ""]
    body += [
        "## AI assistance",
        "",
        f"This change was prepared with the help of an AI agent ([RepoDoctor]({REPODOCTOR_URL})) "
        f"and reviewed by @{user} before it was submitted. Please say if you would like "
        "it changed or closed.",
        "",
    ]
    if trailer:
        body += [trailer, ""]
    text = "\n".join(body)
    return title, (with_agent_cause(text, agent_cause) if agent_cause else text)


def prepare_package(
    *,
    ref: IssueRef,
    issue: IssueData,
    final: dict,
    workdir: Path,
    settings: Settings,
    triage: Report,
    repo_info: RepoInfo,
    policy_hits: list[str],
    ai_threads: list[str] | None = None,
    conventions: dict | None = None,
) -> Path:
    if not final.get("success"):
        raise PackageError("the run did not succeed, so there is nothing to submit")
    base_sha = final.get("base_sha") or ""
    if not base_sha:
        raise PackageError("the run has no base commit recorded")
    user = settings.github_username
    if not user:
        raise PackageError("GITHUB_USERNAME is not set")

    pkg = settings.prs_dir.resolve() / f"{ref.owner}__{ref.repo}-{ref.number}"
    pkg.mkdir(parents=True, exist_ok=True)
    clone = pkg / "clone"   # the package owns its clone: later runs cannot erase it
    remove_dir(clone)
    shutil.copytree(workdir, clone, symlinks=True)

    paths = parse_diff(get_diff_since(clone, base_sha)).files
    if not paths:
        raise PackageError("the run produced no changes")

    # One clean commit on a branch: undo the local test commit, keep every file.
    branch = f"repodoctor/issue-{ref.number}"
    run_git("checkout", "-q", "-B", branch, cwd=clone)
    run_git("reset", "-q", "--mixed", base_sha, cwd=clone)

    test_id = (final.get("target_tests") or [""])[0]
    repro = final.get("repro_file")
    if repro and repro in paths:
        new_rel = _issue_test_path(clone, repro, ref.number)
        (clone / repro).rename(clone / new_rel)
        paths = [new_rel if p == repro else p for p in paths]
        test_id = test_id.replace(repro, new_rel, 1)

    # Projects that keep one small file per change get an empty one, to be written by a human.
    if conventions:
        fp = fragment_path(conventions["stats"]["changelog"], ref.number)
        if fp and not any(classify(p) == "fragment" for p in paths) and not (clone / fp).exists():
            (clone / fp).parent.mkdir(parents=True, exist_ok=True)
            (clone / fp).write_bytes(
                f"{TODO_MARKER}: one line on what changed, for users of the project.\n".encode()
            )
            paths.append(fp)

    model = settings.llm_fix.partition(":")[2] or settings.llm_fix
    trailer = trailer_text(settings, model)
    agent_mode = settings.root_cause_mode.strip().lower() == "agent"
    cause = tidy_hypothesis(final.get("hypothesis") or "") if agent_mode else ""
    title, body = build_pr_text(
        ref=ref, issue=issue, final=final, user=user, test_id=test_id,
        template=(conventions or {}).get("pr_template", ""),
        trailer=trailer, agent_cause=cause,
    )
    author = (
        settings.git_author_name or user,
        settings.git_author_email or f"{user}@users.noreply.github.com",
    )
    head = commit_files(clone, paths, commit_message(title, ref.number, trailer), author)
    diff = get_diff_since(clone, base_sha)
    patch = run_git("format-patch", "-1", "--stdout", cwd=clone)

    style = check_diff(
        diff,
        line_length=repo_line_length(clone),
        max_lines=settings.pr_max_diff_lines,
        max_files=settings.pr_max_files,
        secrets=(settings.nvidia_api_key, settings.openrouter_api_key, settings.github_token),
    )
    style.extend(syntax_findings(clone, paths))
    style.extend(unused_import_findings(clone, diff))
    cl_kind = ((conventions or {}).get("stats") or {}).get("changelog", {}).get("kind")
    if cl_kind != "none":   # when past fixes never touch the changelog, a reminder is noise
        style.extend(changelog_findings(clone, paths))
    style.extend(new_test_findings(conventions, clone, base_sha, paths, ref.number))
    if cause and (policy_hits or ai_threads):
        style.add(
            "warn", "agent_written_description",
            "The root cause in pr.md was drafted by the agent, and this project's docs or "
            "threads mention AI: check that AI-written descriptions are allowed there.",
        )
    style.add(
        "warn", "not_reverified",
        "Lint, the final re-verification and the checklist have not run yet: run repackage.",
    )

    changelog = None
    cl = conventions["stats"]["changelog"] if conventions else None
    if cl and cl["kind"] == "file" and cl.get("path") and cl["path"] not in paths \
            and (clone / cl["path"]).is_file():
        st = conventions.get("changelog_style") or {}
        changelog = {
            "path": cl["path"],
            "excerpt": st.get("excerpt", ""),
            "suggestion": draft_changelog_entry(st, ref.number),
        }

    (pkg / "change.patch").write_bytes(patch.encode("utf-8"))
    (pkg / "pr.md").write_bytes(f"{title}\n\n{body}".encode())

    base_counts = final.get("baseline") or {}
    save_package(pkg, {
        "version": 3,
        "repo": ref.full_name,
        "issue": ref.number,
        "issue_url": issue.url,
        "branch": branch,
        "base_sha": base_sha,
        "head_sha": head,
        "default_branch": repo_info.default_branch,
        "workdir": str(clone),
        "files": paths,
        "author": {"name": author[0], "email": author[1]},
        "dry_run": settings.dry_run,
        "created_at": _now(),
        "triage": [vars(f) for f in triage.findings],
        "style": [vars(f) for f in style.findings],
        "policy_hits": policy_hits,
        "ai_threads": ai_threads or [],
        "hypothesis": final.get("hypothesis") or "",
        "root_cause_mode": "agent" if cause else "human",
        "baseline": final.get("baseline"),
        "conventions": conventions,
        "changelog": changelog,
        "evidence": {
            "test_id": test_id,
            "fix_attempts": final.get("attempts"),
            "repro_attempts": final.get("repro_attempts"),
            "baseline": {k: base_counts.get(k) for k in ("total", "passed", "failed", "errors")},
            "models": {"explore": settings.llm_explore, "fix": settings.llm_fix},
        },
        "approval": None,
    })
    return pkg


def blocking_findings(meta: dict) -> list[dict]:
    return [f for f in (*meta["triage"], *meta["style"]) if f["level"] == "skip"]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def copies_hypothesis(root_cause: str, hypothesis: str) -> bool:
    """True when the root cause mostly repeats the agent's hypothesis."""
    a, b = _words(root_cause), _words(hypothesis)
    if len(a) < 6 or len(b) < 6:
        return False
    mine = {tuple(a[i:i + 5]) for i in range(len(a) - 4)}
    theirs = {tuple(b[i:i + 5]) for i in range(len(b) - 4)}
    if mine and len(mine & theirs) / len(mine) >= 0.4:
        return True
    return difflib.SequenceMatcher(None, " ".join(a), " ".join(b)).ratio() >= 0.6


def approve(pkg: Path, *, policy_acknowledged: bool) -> dict:
    """Record approval together with hashes of the patch and the PR text."""
    meta = load_package(pkg)
    blocking = blocking_findings(meta)
    if blocking:
        raise PackageError("cannot approve: " + "; ".join(f["message"] for f in blocking))
    text = (pkg / "pr.md").read_text(encoding="utf-8")
    if TODO_MARKER in text:
        raise PackageError(
            f"pr.md still contains {TODO_MARKER}: write the cause in your own words "
            "and delete that line"
        )
    if TODO_MARKER in (pkg / "change.patch").read_text(encoding="utf-8"):
        raise PackageError(
            f"the patch still contains {TODO_MARKER} (a changelog entry?): write it in the "
            "work clone, then run repackage"
        )
    if meta.get("root_cause_mode") != "agent":
        words = len(root_cause_section(text).split())
        if words < MIN_ROOT_CAUSE_WORDS:
            raise PackageError(
                f"the 'Root cause' section of pr.md has {words} word(s): write at least "
                f"{MIN_ROOT_CAUSE_WORDS}, saying why the bug happens in your own words"
            )
        if copies_hypothesis(root_cause_section(text), meta.get("hypothesis") or ""):
            raise PackageError(
                "the 'Root cause' section is the agent's own hypothesis, barely reworded: "
                "write the cause in your own words (and without line numbers, which go stale)"
            )
    if (meta["policy_hits"] or meta.get("ai_threads")) and not policy_acknowledged:
        raise PackageError(
            "the contribution docs or recent threads mention AI: confirm you read them"
        )
    meta["approval"] = {
        "at": _now(),
        "policy_acknowledged": policy_acknowledged,
        "patch_sha256": _sha256(pkg / "change.patch"),
        "pr_sha256": _sha256(pkg / "pr.md"),
    }
    save_package(pkg, meta)
    return meta


def verify_approval(pkg: Path) -> list[str]:
    """Empty list = the package is approved and unchanged since."""
    meta = load_package(pkg)
    approval = meta.get("approval")
    if not approval:
        return ["the package is not approved"]
    problems: list[str] = []
    if _sha256(pkg / "change.patch") != approval["patch_sha256"]:
        problems.append("change.patch changed after approval")
    if _sha256(pkg / "pr.md") != approval["pr_sha256"]:
        problems.append("pr.md changed after approval")
    problems += [f"blocking finding: {f['message']}" for f in blocking_findings(meta)]
    return problems


def render_review(meta: dict, patch: str, pr_text: str) -> str:
    def section(title: str, lines: list[str]) -> list[str]:
        return ["", title, *(lines or ["  (nothing)"])]

    def fmt(findings: list[dict]) -> list[str]:
        return [f"  [{f['level']:<4}] {f['message']}" for f in findings]

    ev = meta["evidence"]
    base = ev["baseline"]
    out = [
        "=" * 72,
        f"REVIEW  {meta['repo']}  issue #{meta['issue']}  {meta['issue_url']}",
        "=" * 72,
        f"Branch {meta['branch']} (from {meta['base_sha'][:10]}); files: {', '.join(meta['files'])}",
        f"Work clone: {meta['workdir']}",
        "DRY_RUN is on: nothing will be pushed." if meta["dry_run"] else "DRY_RUN is OFF.",
    ]
    out += section("TRIAGE", fmt(meta["triage"]))
    out += section("POLICY (contribution docs that mention AI)",
                   [f"  {h}" for h in meta["policy_hits"]])
    out += section("RECENT THREADS ABOUT AI, AGENTS OR BOTS",
                   [f"  {t}" for t in meta.get("ai_threads", [])])
    out += section("PROJECT CONVENTIONS (learned from its merged fix pull requests)",
                   convention_summary_lines(meta.get("conventions")))
    cl = meta.get("changelog")
    if cl:
        lines = [f"  The project keeps its changelog in {cl['path']}; the top of the file:"]
        lines += [f"    | {ln}" for ln in cl["excerpt"].splitlines()]
        lines += ["  An entry in the same style (write it in the work clone, then run repackage):",
                  f"    {cl['suggestion']}"]
        out += section("CHANGELOG", lines)
    out += section("DIFF CHECKS", fmt(meta["style"]))
    v = meta.get("verification")
    if v:
        lines = [
            f"  The test fails without the source change: {v['test_fails_without_fix']}",
            f"  The test passes with it: {v['test_passes_with_fix']}",
            f"  Tests that newly fail: {len(v['new_failures'])}",
            f"  Linters run: {', '.join(v.get('lint_tools') or []) or 'none detected'}",
        ]
        if v.get("hooks_not_run"):
            lines.append(f"  Pre-commit hooks not run here: {', '.join(v['hooks_not_run'])}")
        if v.get("failure_without_fix"):
            lines.append("  Without the source change the test fails with:")
            lines += [f"    {ln}" for ln in v["failure_without_fix"]]
        out += section("RE-VERIFICATION (after your edits)", lines)
    out += section("EVIDENCE", [
        f"  New test: {ev['test_id']} "
        f"({'re-verified by repackage' if v else 'accepted by the agent because it failed on the unmodified code'})",
        f"  Fix attempts: {ev['fix_attempts']}; reproduction attempts: {ev['repro_attempts']}",
        f"  Baseline suite: {base['total']} tests, {base['passed']} passed, "
        f"{(base['failed'] or 0) + (base['errors'] or 0)} already failing (ignored)",
        f"  Models: explore {ev['models']['explore']}, fix {ev['models']['fix']}",
    ])
    out += section(
        "THE AGENT'S HYPOTHESIS (unverified: write the cause in pr.md in your own words)",
        [f"  {line}" for line in (meta.get("hypothesis") or "(none)").splitlines()],
    )
    out += section("PULL REQUEST TEXT (posted as written; edit pr.md before approving)",
                   [f"  {line}" for line in pr_text.splitlines()])
    out += ["", "PATCH", patch]
    return "\n".join(out)
