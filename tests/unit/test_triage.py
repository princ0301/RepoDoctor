from datetime import UTC, datetime

import httpx

from repodoctor.tools.github import (
    IssueData,
    LinkedPR,
    RepoInfo,
    github_client,
    linked_pull_requests,
)
from repodoctor.tools.issue_ref import IssueRef
from repodoctor.tools.triage import (
    policy_findings,
    run_triage,
    triage_issue,
    triage_links,
    triage_repo,
    scan_repo
)

NOW = datetime(2026, 11, 10, tzinfo=UTC)
REF = IssueRef("o", "r", 5)
GOOD_BODY = "Repro:\n```python\nf()\n```\nIt crashes, and this description is long enough."


def issue(**over) -> IssueData:
    base = dict(
        url="https://github.com/o/r/issues/5", title="Bug", body=GOOD_BODY, state="open",
        labels=["bug"], assignees=[], comments=[], is_pull_request=False,
        created_at="2026-10-01T00:00:00Z", updated_at="2026-11-01T00:00:00Z",
    )
    return IssueData(**{**base, **over})


def codes(report) -> set[str]:
    return {f.code for f in report.findings}


def test_a_clean_bug_report_passes():
    r = triage_issue(issue(), [], me="me", now=NOW)
    assert r.verdict == "ok" and {"has_code_sample", "helpful_label"} <= codes(r)


def test_closed_issues_and_pull_requests_are_skipped():
    assert "not_open" in codes(triage_issue(issue(state="closed"), [], me="me", now=NOW))
    assert "is_pull_request" in codes(
        triage_issue(issue(is_pull_request=True), [], me="me", now=NOW)
    )


def test_assignment_to_someone_else_is_a_skip_but_to_me_is_not():
    assert triage_issue(issue(assignees=["bob"]), [], me="me", now=NOW).verdict == "skip"
    assert triage_issue(issue(assignees=["Me"]), [], me="me", now=NOW).verdict == "ok"


def test_labels():
    assert triage_issue(issue(labels=["Status: Wontfix"]), [], me="me", now=NOW).verdict == "skip"
    assert "feature_label" in codes(triage_issue(issue(labels=["enhancement"]), [], me="me", now=NOW))


def test_missing_code_sample_and_short_text_warn():
    r = triage_issue(issue(body="it breaks"), [], me="me", now=NOW)
    assert {"no_code_sample", "short_description"} <= codes(r) and r.verdict == "warn"


def test_stale_issues_warn():
    r = triage_issue(issue(updated_at="2024-01-01T00:00:00Z"), [], me="me", now=NOW)
    assert "stale" in codes(r)


def test_comments_claiming_the_issue_warn_but_my_own_do_not():
    claim = ("bob", "I'll work on this one")
    assert "claimed" in codes(triage_issue(issue(), [claim], me="me", now=NOW))
    assert "claimed" not in codes(triage_issue(issue(), [("me", claim[1])], me="me", now=NOW))


def test_discouraging_comments_warn():
    r = triage_issue(issue(), [("maint", "We won't fix this.")], me="me", now=NOW)
    assert "discouraged" in codes(r)


def test_linked_pull_requests():
    def pr(state, merged=False, closes=False):
        return LinkedPR(9, state, merged, closes, "u")

    assert triage_links([pr("open", closes=True)]).verdict == "skip"
    assert triage_links([pr("open")]).verdict == "warn"
    assert triage_links([pr("closed", merged=True)]).verdict == "warn"
    assert triage_links([pr("closed")]).verdict == "ok"


def test_unusable_repositories_are_skipped():
    assert triage_repo(RepoInfo("main", True, False, False)).verdict == "skip"
    assert triage_repo(RepoInfo("main", False, False, False)).verdict == "ok"


def test_policy_findings_distinguish_a_ban_from_a_mention():
    ban = ["CONTRIBUTING.md:2: AI-generated contributions are not accepted."]
    mention = ["CONTRIBUTING.md:2: You may use AI tools if you disclose it."]
    assert policy_findings(ban)[0].level == "skip"
    assert policy_findings(mention)[0].level == "warn"
    assert policy_findings([]) == []


def mock_client(routes: dict, texts: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in routes:
            return httpx.Response(200, json=routes[path])
        if texts and path in texts:
            return httpx.Response(200, text=texts[path])
        return httpx.Response(404, json={})

    return github_client("", transport=httpx.MockTransport(handler))


def issue_routes(**over) -> dict:
    body = {
        "title": "Bug", "body": GOOD_BODY + " " * 100, "state": "open", "html_url": "u",
        "labels": [{"name": "bug"}], "assignees": [], "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-11-01T00:00:00Z", **over,
    }
    return {
        "/repos/o/r/issues/5": body,
        "/repos/o/r/issues/5/comments": [],
        "/repos/o/r/issues/5/timeline": [],
        "/repos/o/r": {"default_branch": "main", "archived": False, "disabled": False,
                       "private": False},
    }


def test_linked_pull_requests_reads_state_and_closing_keywords():
    routes = {
        "/repos/o/r/issues/5/timeline": [{
            "event": "cross-referenced",
            "source": {"issue": {
                "number": 9, "repository_url": "https://api.github.com/repos/o/r",
                "pull_request": {"url": "x"},
            }},
        }],
        "/repos/o/r/pulls/9": {"state": "open", "merged": False, "body": "Fixes #5",
                               "html_url": "u9"},
    }
    assert linked_pull_requests(mock_client(routes), REF) == [LinkedPR(9, "open", False, True, "u9")]


def test_run_triage_stops_early_for_closed_issues():
    result = run_triage(mock_client(issue_routes(state="closed")), REF, me="me", now=NOW)
    assert result.report.verdict == "skip" and result.policy_hits == []


def test_run_triage_blocks_a_repo_whose_docs_ban_ai():
    texts = {"/repos/o/r/contents/CONTRIBUTING.md": "Be kind.\nAI-generated PRs are not accepted.\n"}
    result = run_triage(mock_client(issue_routes(), texts), REF, me="me", now=NOW)
    assert result.report.verdict == "skip"
    assert "policy_blocks_ai" in codes(result.report)
    assert result.policy_hits == ["CONTRIBUTING.md:2: AI-generated PRs are not accepted."]


def test_run_triage_passes_a_clean_issue():
    result = run_triage(mock_client(issue_routes()), REF, me="me", now=NOW)
    assert result.report.verdict == "ok" and result.repo.default_branch == "main"

def test_the_claim_finding_quotes_the_comment():
    r = triage_issue(issue(), [("bob", "Sure, I'll work on this one soon")], me="me", now=NOW)
    (f,) = [f for f in r.findings if f.code == "claimed"]
    assert "bob" in f.message and "work on this" in f.message


def list_item(number: int, **over) -> dict:
    d = {
        "number": number, "title": f"Issue {number}", "body": GOOD_BODY, "state": "open",
        "html_url": f"u{number}", "labels": [{"name": "bug"}], "assignees": [],
        "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-11-01T00:00:00Z",
    }
    return {**d, **over}


def scan_routes(items: list[dict], **repo) -> dict:
    routes = {
        "/repos/o/r": {"default_branch": "main", "archived": False, "disabled": False,
                       "private": False, **repo},
        "/repos/o/r/issues": items,
    }
    for item in items:
        routes[f"/repos/o/r/issues/{item['number']}/comments"] = []
        routes[f"/repos/o/r/issues/{item['number']}/timeline"] = []
    return routes


def test_scan_ranks_clean_issues_first_and_leaves_out_pull_requests():
    items = [
        list_item(1, body="short"), list_item(2), list_item(3, pull_request={"url": "x"}),
    ]
    result = scan_repo(mock_client(scan_routes(items)), "o/r", me="me", now=NOW)
    assert [(r.number, r.verdict) for r in result.rows] == [(2, "ok"), (1, "warn")]


def test_scan_skips_issues_that_already_have_an_open_fixing_pr():
    routes = scan_routes([list_item(2)])
    routes["/repos/o/r/issues/2/timeline"] = [{
        "event": "cross-referenced",
        "source": {"issue": {"number": 9, "pull_request": {"url": "x"},
                             "repository_url": "https://api.github.com/repos/o/r"}},
    }]
    routes["/repos/o/r/pulls/9"] = {"state": "open", "merged": False, "body": "Fixes #2",
                                    "html_url": "u9"}
    (row,) = scan_repo(mock_client(routes), "o/r", me="me", now=NOW).rows
    assert row.verdict == "skip" and "Open PR #9" in row.notes[0]


def test_scan_stops_for_an_archived_repository():
    result = scan_repo(mock_client(scan_routes([list_item(1)], archived=True)), "o/r", now=NOW)
    assert result.repo_report.verdict == "skip" and result.rows == []


def test_scan_stops_when_the_docs_ban_ai():
    texts = {"/repos/o/r/contents/CONTRIBUTING.md": "AI-generated PRs are not accepted.\n"}
    result = scan_repo(mock_client(scan_routes([list_item(1)]), texts), "o/r", now=NOW)
    assert result.repo_report.verdict == "skip" and result.rows == []