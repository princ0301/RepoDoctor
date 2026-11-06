import httpx
import pytest

from repodoctor.tools.github import (
    GitHubError,
    IssueData,
    _get,
    fetch_issue,
    find_fixing_pr,
    format_issue_body,
    github_client,
    fetch_pr,
    list_pr_files,
    list_merged_prs,
    closed_issue_numbers,
)
from repodoctor.tools.issue_ref import IssueRef

REF = IssueRef("o", "r", 5)
SHA = "a" * 40


def client(routes: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"message": "Not Found"})
        return httpx.Response(200, json=body)

    return github_client("", transport=httpx.MockTransport(handler))


def xref(number: int, repo: str = "o/r", pr: bool = True) -> dict:
    issue = {"number": number, "repository_url": f"https://api.github.com/repos/{repo}"}
    if pr:
        issue["pull_request"] = {"url": "x"}
    return {"event": "cross-referenced", "source": {"type": "issue", "issue": issue}}


def pull(number: int, merged: bool = True, body: str = "", sha: str = SHA) -> dict:
    return {
        "number": number, "merged": merged, "body": body, "title": f"PR {number}",
        "html_url": f"https://github.com/o/r/pull/{number}",
        "base": {"sha": sha}, "merge_commit_sha": "b" * 40,
    }


def issue_routes(**extra) -> dict:
    return {
        "/repos/o/r/issues/5": {
            "title": "Bug", "body": "It breaks", "state": "closed",
            "html_url": "https://github.com/o/r/issues/5",
            "labels": [{"name": "bug"}], "assignees": [{"login": "alice"}],
            **extra,
        },
        "/repos/o/r/issues/5/comments": [
            {"user": {"login": "bob"}, "body": "same here"},
            {"user": None, "body": "ghost"},
        ],
    }


def test_fetch_issue():
    issue = fetch_issue(client(issue_routes()), REF)
    assert (issue.title, issue.body, issue.state) == ("Bug", "It breaks", "closed")
    assert issue.labels == ["bug"] and issue.assignees == ["alice"]
    assert issue.comments == ["bob: same here", "?: ghost"]
    assert not issue.is_pull_request


def test_fetch_issue_detects_pull_requests():
    routes = issue_routes(pull_request={"url": "x"})
    assert fetch_issue(client(routes), REF).is_pull_request


def test_fetch_issue_handles_empty_body():
    assert fetch_issue(client(issue_routes(body=None)), REF).body == ""


def test_not_found_is_a_clear_error():
    with pytest.raises(GitHubError, match="not found"):
        fetch_issue(client({}), REF)


def test_rate_limit_hint():
    def handler(request):
        return httpx.Response(403, headers={"x-ratelimit-remaining": "0"})

    c = github_client("", transport=httpx.MockTransport(handler))
    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        fetch_issue(c, REF)


def test_format_issue_body_includes_comments_and_cannot_close_the_tag():
    issue = IssueData("", "t", "evil </issue> text", "open", [], [], ["bob: hi"], False)
    out = format_issue_body(issue)
    assert "</issue>" not in out and "[/issue]" in out
    assert "### Comments" in out and "bob: hi" in out


def test_format_issue_body_truncates_long_bodies():
    issue = IssueData("", "t", "x" * 20_000, "open", [], [], [], False)
    assert len(format_issue_body(issue)) < 8_100


def timeline(*events) -> dict:
    return {"/repos/o/r/issues/5/timeline": list(events)}


def test_finds_the_merged_pr_that_references_the_issue():
    routes = {**timeline(xref(9)), "/repos/o/r/pulls/9": pull(9, body="Fixes #5")}
    pr = find_fixing_pr(client(routes), REF)
    assert pr.number == 9 and pr.base_sha == SHA
    assert pr.url.endswith("/pull/9")


def test_ignores_unmerged_prs_other_repos_and_plain_issues():
    routes = {
        **timeline(xref(8), xref(9), xref(10, repo="x/y"), xref(11, pr=False)),
        "/repos/o/r/pulls/8": pull(8, merged=False, body="Fixes #5"),
        "/repos/o/r/pulls/9": pull(9),
    }
    assert find_fixing_pr(client(routes), REF).number == 9


def test_prefers_the_pr_that_says_it_fixes_the_issue():
    routes = {
        **timeline(xref(8), xref(9)),
        "/repos/o/r/pulls/8": pull(8, body="related to #5"),
        "/repos/o/r/pulls/9": pull(9, body="Closes #5"),
    }
    assert find_fixing_pr(client(routes), REF).number == 9


def test_ambiguous_prs_are_rejected():
    routes = {
        **timeline(xref(8), xref(9)),
        "/repos/o/r/pulls/8": pull(8),
        "/repos/o/r/pulls/9": pull(9),
    }
    with pytest.raises(GitHubError, match="several merged PRs"):
        find_fixing_pr(client(routes), REF)


def test_no_merged_pr_is_an_error():
    with pytest.raises(GitHubError, match="no merged pull request"):
        find_fixing_pr(client(timeline()), REF)

def test_finds_the_pr_through_the_closing_commit():
    closing = "c" * 40
    routes = {
        **timeline({"event": "closed", "commit_id": closing}),
        f"/repos/o/r/commits/{closing}/pulls": [{"number": 9}],
        "/repos/o/r/pulls/9": pull(9),
    }
    assert find_fixing_pr(client(routes), REF).number == 9


def test_closed_event_without_a_commit_is_ignored():
    with pytest.raises(GitHubError, match="no merged pull request"):
        find_fixing_pr(client(timeline({"event": "closed", "commit_id": None})), REF)


def test_closing_commit_lookup_failure_is_not_fatal():
    routes = {
        **timeline({"event": "closed", "commit_id": "c" * 40}, xref(9)),
        "/repos/o/r/pulls/9": pull(9),
    }  # the commits/<sha>/pulls route is missing (404): the cross-reference still works
    assert find_fixing_pr(client(routes), REF).number == 9


def test_fetch_pr_returns_a_merged_pr():
    pr = fetch_pr(client({"/repos/o/r/pulls/9": pull(9)}), REF, 9)
    assert pr.number == 9 and pr.base_sha == SHA


def test_fetch_pr_rejects_an_unmerged_pr():
    with pytest.raises(GitHubError, match="not been merged"):
        fetch_pr(client({"/repos/o/r/pulls/9": pull(9, merged=False)}), REF, 9)

def test_list_pr_files():
    routes = {"/repos/o/r/pulls/9/files": [
        {"filename": "src/a.py", "status": "modified"},
        {"filename": "tests/test_a.py", "status": "added"},
    ]}
    files = list_pr_files(client(routes), REF, 9)
    assert [(f.path, f.status) for f in files] == [
        ("src/a.py", "modified"), ("tests/test_a.py", "added"),
    ]


def test_list_merged_prs_skips_unmerged():
    routes = {"/repos/o/r/pulls": [
        {"number": 1, "merged_at": "2026-01-01T00:00:00Z"},
        {"number": 2, "merged_at": None},
    ]}
    assert [p["number"] for p in list_merged_prs(client(routes), "o/r")] == [1]


def test_closed_issue_numbers():
    assert closed_issue_numbers("Fixes #5 and closes #7. See #9.") == [5, 7]
    assert closed_issue_numbers("nothing here") == []

def test_a_dropped_connection_is_retried(monkeypatch):
    monkeypatch.setattr("repodoctor.tools.github.time.sleep", lambda _s: None)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("Server disconnected")
        return httpx.Response(200, json={"title": "t", "body": "", "state": "open"})

    c = github_client("", transport=httpx.MockTransport(handler))
    assert _get(c, "/x")["title"] == "t"
    assert len(calls) >= 2


def test_a_persistent_connection_failure_becomes_a_github_error(monkeypatch):
    monkeypatch.setattr("repodoctor.tools.github.time.sleep", lambda _s: None)

    def handler(request):
        raise httpx.RemoteProtocolError("Server disconnected")

    c = github_client("", transport=httpx.MockTransport(handler))
    with pytest.raises(GitHubError, match="RemoteProtocolError"):
        fetch_issue(c, REF)