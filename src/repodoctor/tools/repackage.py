"""Rebuild a package after the reviewer edited its work clone, and re-verify it."""

import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repodoctor.config import Settings
from repodoctor.tools.diffcheck import (
    check_diff,
    parse_diff,
    repo_line_length,
    syntax_findings,
    unused_import_findings,
)
from repodoctor.tools.findings import Finding
from repodoctor.tools.paths import is_test_path
from repodoctor.tools.pr_package import PackageError, load_package, save_package
from repodoctor.tools.repo import (
    GitError,
    added_files,
    detect_project,
    get_diff_since,
    head_sha,
    run_git,
)
from repodoctor.tools.repo_config import ignored_paths, with_ignores
from repodoctor.tools.review_hints import (
    changelog_findings,
    hint_new_test_files,
    unknown_test_id,
)
from repodoctor.tools.sandbox import Sandbox
from repodoctor.tools.testreport import SuiteReport, new_failures

__all__ = ["repackage"]


def _exists_at(root: Path, rev: str, rel: str) -> bool:
    try:
        run_git("cat-file", "-e", f"{rev}:{rel}", cwd=root)
        return True
    except GitError:
        return False


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

    # Fold every edit in the clone into the single commit.
    run_git("add", "-A", cwd=root)
    run_git(
        "-c", f"user.name={name}", "-c", f"user.email={email}",
        "commit", "--amend", "--no-edit", "-q", "--no-verify", "--no-gpg-sign", cwd=root,
    )
    head = head_sha(root)
    diff = get_diff_since(root, base)
    files = parse_diff(diff).files
    if not files:
        raise PackageError("no changes are left in the work clone")

    sources = [f for f in files if not is_test_path(f)]
    if not sources:
        raise PackageError("the change touches no source file, only tests")
    modified = [f for f in sources if _exists_at(root, base, f)]

    info = with_ignores(detect_project(root), ignored_paths(meta["repo"]))
    if not info.supported:
        raise PackageError(f"unsupported project: {info.reason}")
    factory = sandbox_factory or (
        lambda r: Sandbox(r, memory=settings.sandbox_memory, cpus=settings.sandbox_cpus)
    )
    timeout = settings.sandbox_timeout_s
    findings: list[Finding] = []
    fails_before: bool | None = None

    with factory(root) as sb:
        if not sb.install(info).ok:
            raise PackageError("dependency install failed while re-verifying")
        sb.lock_network()

        if modified and len(modified) == len(sources):
            run_git("checkout", base, "--", *modified, cwd=root)
            try:
                res, _ = sb.run_suite(info, args=[tid], timeout=timeout)
            finally:
                run_git("checkout", head, "--", *modified, cwd=root)
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
    style.extend(hint_new_test_files(
        root, [p for p in added_files(root, base) if is_test_path(p)]
    ))
    style.extend(findings)

    (pkg / "change.patch").write_bytes(
        run_git("format-patch", "-1", "--stdout", cwd=root).encode("utf-8")
    )
    old_id = meta["evidence"]["test_id"]
    if tid != old_id:
        text = (pkg / "pr.md").read_text(encoding="utf-8").replace(old_id, tid)
        (pkg / "pr.md").write_bytes(text.encode("utf-8"))
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
            "new_failures": list(regressions)[:20],
        },
    )
    save_package(pkg, meta)
    return meta
