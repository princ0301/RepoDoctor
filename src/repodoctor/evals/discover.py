"""Find (issue, fixing PR) pairs worth evaluating, from merged PRs of a repo."""

import httpx

from repodoctor.evals.cases import CaseSpec, is_test_file, is_test_path
from repodoctor.tools.github import (
    GitHubError,
    closed_issue_numbers,
    list_merged_prs,
    list_pr_files,
)
from repodoctor.tools.issue_ref import IssueRef


def discover(
    client: httpx.Client,
    repo: str,
    *,
    limit: int = 15,
    max_prs: int = 200,
    max_src_files: int = 4,
) -> list[CaseSpec]:
    """Merged PRs that say 'Fixes #N' for exactly one issue, change a test module and a
    small amount of Python source. Final validation happens in eval-build."""
    owner, _, name = repo.partition("/")
    ref = IssueRef(owner, name, 0)
    specs: list[CaseSpec] = []
    for pr in list_merged_prs(client, repo, max_prs):
        issues = closed_issue_numbers(pr.get("body") or "")
        if len(issues) != 1:
            continue
        try:
            files = list_pr_files(client, ref, pr["number"])
        except GitHubError:
            continue
        tests = [f.path for f in files if f.status != "removed" and is_test_file(f.path)]
        src = [f.path for f in files if not is_test_path(f.path)]
        if not tests or not (1 <= len(src) <= max_src_files):
            continue
        if not any(p.endswith(".py") for p in src):
            continue
        specs.append(CaseSpec(repo, issues[0], pr["number"]))
        if len(specs) >= limit:
            break
    return specs