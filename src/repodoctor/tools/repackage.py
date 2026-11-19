"""Rebuild a package after the reviewer edited its work clone, and re-verify it."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repodoctor.config import Settings
from repodoctor.tools.conventions import classify, new_test_findings, update_checkboxes
from repodoctor.tools.diffcheck import (
    check_diff,
    parse_diff,
    repo_line_length,
    syntax_findings,
    unused_import_findings,
)
from repodoctor.tools.findings import TODO_MARKER, Finding
from repodoctor.tools.lint import detect_lint_tools, exists_at, install_command, run_lint
from repodoctor.tools.pr_package import (
    PackageError,
    commit_message,
    load_package,
    save_package,
    trailer_text,
)
from repodoctor.tools.repo import detect_project, get_diff_since, head_sha, run_git
from repodoctor.tools.repo_config import apply_overrides
from repodoctor.tools.review_hints import changelog_findings, unknown_test_id
from repodoctor.tools.sandbox import Sandbox
from repodoctor.tools.testreport import SuiteReport, new_failures

__all__ = ["repackage"]


def failure_summary(output: str, limit: int = 8) -> list[str]:
    """The lines pytest prints to say why a test failed (they start with 'E ')."""
    return [ln.rstrip() for ln in output.splitlines() if ln.startswith("E ")][:limit]


def repackage(
    pkg: Path,
    settings: Settings,
    *,
    test_id: str | None = None,
    sandbox_factory: Callable[[Path], Any] | None = None,
) -> dict:
    meta = load_package(pkg)
    if not meta.get("baseline"):
        raise PackageError(
            "this package was prepared by an older version: run fix-issue --prepare-pr again"
        )
    root = Path(meta["workdir"])
    if not root.is_dir():
        raise PackageError(f"the work clone is gone: {root}")
    if run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=root).strip() != meta["branch"]:
        raise PackageError(f"the work clone is not on branch {meta['branch']}")
    base = meta["base_sha"]
    name, email = meta["author"]["name"], meta["author"]["email"]

    tid = test_id or meta["evidence"]["test_id"]
    problem = unknown_test_id(root, tid)   # before anything is changed
    if problem:
        raise PackageError(f"{problem} Pass --test FILE::NAME for the test as it is now.")

    # Fold every edit in the clone into the single commit, titled like pr.md.
    title = (pkg / "pr.md").read_text(encoding="utf-8").split("\n", 1)[0].strip()
    fix_model = meta["evidence"]["models"]["fix"]
    model = fix_model.partition(":")[2] or fix_model
    trailer = trailer_text(settings, model)
    run_git("add", "-A", cwd=root)
    run_git(
        "-c", f"user.name={name}", "-c", f"user.email={email}",
        "commit", "--amend", "-q", "--no-verify", "--no-gpg-sign",
        "-m", commit_message(title, meta["issue"], trailer), cwd=root,
    )
    head = head_sha(root)
    diff = get_diff_since(root, base)
    files = parse_diff(diff).files
    if not files:
        raise PackageError("no changes are left in the work clone")

    code_files = [f for f in files if classify(f) in ("source", "other")]
    if not code_files:
        raise PackageError("the change touches no source file, only tests or docs")
    modified = [f for f in code_files if exists_at(root, base, f)]

    info = apply_overrides(detect_project(root), meta["repo"])
    if not info.supported:
        raise PackageError(f"unsupported project: {info.reason}")
    tools, hooks_not_run = detect_lint_tools(root)
    factory = sandbox_factory or (
        lambda r: Sandbox(r, memory=settings.sandbox_memory, cpus=settings.sandbox_cpus)
    )
    timeout = settings.sandbox_timeout_s
    findings: list[Finding] = []
    fails_before: bool | None = None
    without_fix: list[str] = []
    lint_findings: list[Finding] = []

    with factory(root) as sb:
        if not sb.install(info).ok:
            raise PackageError("dependency install failed while re-verifying")
        if tools:
            installed = sb.run(install_command(tools), settings.build_install_timeout_s)
            if not installed.ok:
                findings.append(Finding(
                    "info", "lint_unavailable",
                    "The project's linters could not be installed here; run them yourself.",
                ))
                tools = []
        sb.lock_network()

        if modified and len(modified) == len(code_files):
            run_git("checkout", base, "--", *modified, cwd=root)
            try:
                res, _ = sb.run_suite(info, args=[tid], timeout=timeout)
            finally:
                run_git("checkout", head, "--", *modified, cwd=root)
            without_fix = failure_summary(res.output)
            fails_before = res.exit_code == 1 and not res.timed_out
            if res.exit_code in (2, 3, 4, 5):
                fails_before = None
                findings.append(Finding(
                    "skip", "test_not_runnable",
                    f"pytest could not run {tid} without the source change "
                    f"(exit {res.exit_code}). Check the test id.",
                ))
            elif not fails_before:
                findings.append(Finding(
                    "skip", "test_does_not_fail_without_fix",
                    f"Without the source change the test did not fail (exit {res.exit_code}), "
                    "so it does not show the bug.",
                ))
        else:
            findings.append(Finding(
                "warn", "fails_before_unchecked",
                "The change adds source files, so 'the test fails without the fix' "
                "could not be re-checked.",
            ))

        res, _ = sb.run_suite(info, args=[tid], timeout=timeout)
        passes = res.ok
        if res.exit_code in (2, 3, 4, 5):
            findings.append(Finding(
                "skip", "test_not_runnable",
                f"pytest could not run {tid} (exit {res.exit_code}). Check the test id.",
            ))
        elif not passes:
            findings.append(Finding("skip", "test_fails_with_fix", f"{tid} does not pass."))

        _, report = sb.run_suite(info, timeout=timeout)
        regressions: dict[str, str] = {}
        if report is None:
            findings.append(Finding("skip", "no_report", "The full test run produced no report."))
        else:
            regressions = new_failures(SuiteReport(**meta["baseline"]), report)
            if regressions:
                shown = ", ".join(list(regressions)[:5])
                findings.append(Finding(
                    "skip", "regression",
                    f"{len(regressions)} test(s) that passed before now fail: {shown}",
                ))

        lint_findings = run_lint(sb, root, tools, files, diff, base, head)

    style = check_diff(
        diff,
        line_length=repo_line_length(root),
        max_lines=settings.pr_max_diff_lines,
        max_files=settings.pr_max_files,
        secrets=(settings.nvidia_api_key, settings.openrouter_api_key, settings.github_token),
    )
    style.extend(syntax_findings(root, files))
    style.extend(unused_import_findings(root, diff))
    style.extend(changelog_findings(root, files))
    style.extend(new_test_findings(meta.get("conventions"), root, base, files, meta["issue"]))
    style.extend(lint_findings)
    if hooks_not_run:
        style.add(
            "info", "precommit_not_run",
            f"pre-commit also runs {', '.join(hooks_not_run)} (not run here): "
            "run `pre-commit run --files ...` yourself.",
        )
    style.extend(findings)

    (pkg / "change.patch").write_bytes(
        run_git("format-patch", "-1", "--stdout", cwd=root).encode("utf-8")
    )

    # Tick the checklist boxes this run has verified; never untick, never touch the reviewer's.
    lint_problems = any(f.level == "warn" and f.code in ("lint", "format") for f in lint_findings)
    facts = {
        "tests_added": any(classify(f) == "test" for f in files),
        "test_fails_without_fix": fails_before,
        "changelog_touched": any(classify(f) in ("changelog", "fragment") for f in files)
        and TODO_MARKER not in diff,
        "lint_passed": True if (tools and not lint_problems and not hooks_not_run) else None,
        "suite_passed": bool(passes and report is not None and not regressions),
    }
    old_id = meta["evidence"]["test_id"]
    text = (pkg / "pr.md").read_text(encoding="utf-8")
    new_text = update_checkboxes(text.replace(old_id, tid) if tid != old_id else text, facts)
    if new_text != text:
        (pkg / "pr.md").write_bytes(new_text.encode("utf-8"))

    meta["evidence"]["test_id"] = tid
    meta.update(
        head_sha=head,
        files=files,
        style=[vars(f) for f in style.findings],
        approval=None,   # any edit invalidates an earlier approval
        verification={
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "test_fails_without_fix": fails_before,
            "test_passes_with_fix": passes,
            "failure_without_fix": without_fix,
            "new_failures": list(regressions)[:20],
            "lint_tools": [t.name for t in tools],
            "hooks_not_run": hooks_not_run,
        },
    )
    save_package(pkg, meta)
    return meta
