"""Turn a spec into a validated case, with fail-to-pass / pass-to-pass ground truth."""

from collections.abc import Callable

import httpx

from repodoctor.config import Settings
from repodoctor.evals.cases import CaseSpec, EvalCase, case_id, is_test_file, is_test_path
from repodoctor.tools.github import fetch_issue, fetch_pr, list_pr_files
from repodoctor.tools.issue_ref import IssueRef, check_allowed
from repodoctor.tools.repo import (
    checkout_commit,
    clone_issue_repo,
    detect_project,
    fetch_pull_head,
    remove_dir,
    run_git,
)
from repodoctor.tools.sandbox import Sandbox, ensure_image

MAX_SRC_FILES = 4


class CaseRejected(RuntimeError):
    """The spec is not usable as an eval case (the message says why)."""


def build_case(
    spec: CaseSpec,
    settings: Settings,
    client: httpx.Client,
    log: Callable[[str], None] = lambda _m: None,
) -> EvalCase:
    owner, _, repo = spec.repo.partition("/")
    ref = IssueRef(owner, repo, spec.issue)
    check_allowed(ref, settings.allowed_repo_list)

    issue = fetch_issue(client, ref)
    if issue.is_pull_request:
        raise CaseRejected("the issue number is a pull request")
    if issue.state != "closed":
        raise CaseRejected("the issue is not closed")
    pr = fetch_pr(client, ref, spec.fix_pr)

    files = list_pr_files(client, ref, spec.fix_pr)
    test_files = [f.path for f in files if f.status != "removed" and is_test_path(f.path)]
    run_targets = [p for p in test_files if is_test_file(p)]
    src_files = [f.path for f in files if not is_test_path(f.path)]
    if not run_targets:
        raise CaseRejected("the PR changes no test module")
    if not src_files:
        raise CaseRejected("the PR changes only tests")
    if len(src_files) > MAX_SRC_FILES:
        raise CaseRejected(f"the PR changes {len(src_files)} source files (max {MAX_SRC_FILES})")
    log(f"  PR #{spec.fix_pr}: {len(run_targets)} test module(s), {len(src_files)} source file(s)")

    dest = settings.work_dir.resolve() / f"{owner}__{repo}__build"
    remove_dir(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    clone_issue_repo(ref, dest)
    checkout_commit(dest, pr.base_sha)
    head = fetch_pull_head(dest, spec.fix_pr)

    info = detect_project(dest)
    if not info.supported:
        raise CaseRejected(f"unsupported project: {info.reason}")

    ensure_image()
    with Sandbox(dest, memory=settings.sandbox_memory, cpus=settings.sandbox_cpus) as sb:
        installed = sb.install(info)
        if not installed.ok:
            raise CaseRejected("dependency install failed at the base commit")
        sb.lock_network()

        # A: code before the fix + the maintainers' tests
        run_git("checkout", head, "--", *test_files, cwd=dest)
        _, before = sb.run_outcomes(info, run_targets, settings.sandbox_timeout_s)
        # B: the PR's own tree (fix + tests)
        run_git("reset", "--hard", "-q", "HEAD", cwd=dest)
        checkout_commit(dest, head)
        _, after = sb.run_outcomes(info, run_targets, settings.sandbox_timeout_s)

    if before is None or after is None:
        raise CaseRejected("the maintainers' tests produced no report")
    fail_to_pass = sorted(
        t for t, s in before.items() if s in ("failed", "error") and after.get(t) == "passed"
    )
    pass_to_pass = sorted(
        t for t, s in before.items() if s == "passed" and after.get(t) == "passed"
    )
    if not fail_to_pass:
        raise CaseRejected(
            "the PR's tests do not fail before the fix and pass after it "
            "(they may not be collectable on the old code)"
        )
    return EvalCase(
        id=case_id(spec.repo, spec.issue), repo=spec.repo, issue=spec.issue,
        fix_pr=spec.fix_pr, title=issue.title, base_sha=pr.base_sha, head_sha=head,
        test_files=test_files, run_targets=run_targets, src_files=src_files,
        fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass,
    )