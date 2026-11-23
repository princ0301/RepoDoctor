import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from repodoctor.config import Settings
from repodoctor.tools.findings import Report
from repodoctor.tools.github import IssueData, RepoInfo, github_client
from repodoctor.tools.issue_ref import IssueRef
from repodoctor.tools.pr_package import (
    PackageError,
    approve,
    load_package,
    prepare_package,
    save_package,
)
from repodoctor.tools.repo import GitError, commit_files, get_diff, head_sha, run_git
from repodoctor.tools.submit import (
    load_ledger,
    preflight,
    push_branch,
    recent_submissions,
    scrub,
    split_pr_text,
    submit_package,
)

REF = IssueRef("o", "r", 5)
ISSUE = IssueData(
    url="https://github.com/o/r/issues/5", title="x is wrong", body="body", state="open",
    labels=[], assignees=[], comments=[], is_pull_request=False,
)
CAUSE = "The comparison used the wrong operator, so text of exactly the maximum length was cut."
ROUTES = {
    ("GET", "/user"): (200, {"login": "me"}),
    ("GET", "/repos/o/r/issues/5/comments"): (200, [{
        "user": {"login": "me"}, "body": "I found the cause",
        "created_at": "2026-01-01T00:00:00Z", "author_association": "NONE",
    }]),
    ("POST", "/repos/o/r/forks"): (202, {"full_name": "me/r"}),
    ("GET", "/repos/me/r"): (200, {}),
    ("POST", "/repos/me/r/merge-upstream"): (200, {}),
    ("GET", "/repos/o/r/pulls"): (200, []),
    ("POST", "/repos/o/r/pulls"): (201, {"html_url": "https://github.com/o/r/pull/9"}),
}


def cfg(tmp_path, **over) -> Settings:
    base = dict(
        github_username="me", github_push_token="tok", allowed_repos="o/r",
        prs_dir=tmp_path / "prs", dry_run=False,
    )
    return Settings(_env_file=None, **{**base, **over})


def api(routes=None, calls=None):
    table = ROUTES if routes is None else routes

    def handler(request):
        body = json.loads(request.content) if request.content else None
        if calls is not None:
            calls.append((request.method, request.url.path, body))
        hit = table.get((request.method, request.url.path))
        if hit is None:
            return httpx.Response(404, json={"message": "Not Found"})
        return httpx.Response(hit[0], json=hit[1])

    return github_client("tok", transport=httpx.MockTransport(handler))


def triage_ok(client, ref, me=""):
    return SimpleNamespace(report=Report())


@pytest.fixture
def approved(make_git_repo, tmp_path):
    root = make_git_repo({
        "app.py": "x = 1\n", "requirements.txt": "\n",
        "tests/test_app.py": "def test_ok():\n    assert True\n",
    })
    base = head_sha(root)
    (root / "tests" / "test_repodoctor_repro.py").write_bytes(
        b"from app import x\n\n\ndef test_x():\n    assert x == 2\n"
    )
    commit_files(root, ["tests/test_repodoctor_repro.py"], "test: reproduce")
    (root / "app.py").write_bytes(b"x = 2\n")
    get_diff(root, ["app.py"])
    final = {
        "success": True, "base_sha": base, "attempts": 1, "repro_attempts": 1,
        "repro_file": "tests/test_repodoctor_repro.py",
        "target_tests": ["tests/test_repodoctor_repro.py::test_x"],
        "hypothesis": "h", "baseline": {"total": 3, "passed": 3, "failed": 0, "errors": 0},
    }
    settings = cfg(tmp_path)
    pkg = prepare_package(
        ref=REF, issue=ISSUE, final=final, workdir=root, settings=settings, triage=Report(),
        repo_info=RepoInfo("main", False, False, False), policy_hits=[],
    )
    meta = load_package(pkg)
    meta["verification"] = {
        "test_fails_without_fix": True, "test_passes_with_fix": True, "new_failures": [],
        "lint_tools": [], "hooks_not_run": [], "failure_without_fix": [],
    }
    save_package(pkg, meta)
    p = pkg / "pr.md"
    lines = [CAUSE if "TODO(reviewer)" in ln else ln
             for ln in p.read_text(encoding="utf-8").splitlines()]
    p.write_bytes(("\n".join(lines) + "\n").encode())
    approve(pkg, policy_acknowledged=True)
    return pkg, settings


def check(approved, **over):
    pkg, settings = approved
    settings = settings.model_copy(update=over)
    return preflight(
        pkg, settings, read_client=api(), push_client=api(), triage_fn=triage_ok,
    ), pkg, settings


def test_preflight_passes_for_an_approved_unchanged_package(approved):
    plan, _, _ = check(approved)
    assert plan.ok, plan.problems
    assert not plan.own_repo and plan.title == "x is wrong"


def test_preflight_blocks_a_package_that_was_never_reverified(approved):
    pkg, _ = approved
    meta = load_package(pkg)
    meta["verification"] = None
    save_package(pkg, meta)
    plan, _, _ = check(approved)
    assert any("never re-verified" in p for p in plan.problems)


def test_third_party_submissions_need_an_older_comment_by_you(approved):
    pkg, settings = approved
    none = {**ROUTES, ("GET", "/repos/o/r/issues/5/comments"): (200, [])}
    plan = preflight(pkg, settings, read_client=api(none), push_client=api(), triage_fn=triage_ok)
    assert any("have not commented" in p for p in plan.problems)

    fresh = {**ROUTES, ("GET", "/repos/o/r/issues/5/comments"): (200, [{
        "user": {"login": "me"}, "body": "x", "author_association": "NONE",
        "created_at": datetime.now(UTC).isoformat(),
    }])}
    plan = preflight(
        pkg, settings, read_client=api(fresh), push_client=api(), triage_fn=triage_ok
    )
    assert any("less than 24 hours old" in p for p in plan.problems)

    off = settings.model_copy(update={"pr_min_comment_hours": 0})
    assert preflight(
        pkg, off, read_client=api(none), push_client=api(), triage_fn=triage_ok
    ).ok


def test_preflight_blocks_edits_after_approval(approved):
    pkg, _ = approved
    (pkg / "pr.md").write_bytes(b"changed\n")
    plan, _, _ = check(approved)
    assert any("pr.md changed" in p for p in plan.problems)


def test_preflight_blocks_a_modified_clone(approved):
    pkg, _ = approved
    (Path(load_package(pkg)["workdir"]) / "app.py").write_bytes(b"x = 3\n")
    plan, _, _ = check(approved)
    assert any("uncommitted" in p for p in plan.problems)


def test_preflight_blocks_other_repos_accounts_and_missing_tokens(approved):
    assert any("ALLOWED_REPOS" in p for p in check(approved, allowed_repos="a/b")[0].problems)
    assert any("GITHUB_PUSH_TOKEN" in p for p in check(approved, github_push_token="")[0].problems)
    pkg, settings = approved
    plan = preflight(
        pkg, settings, read_client=api(),
        push_client=api({("GET", "/user"): (200, {"login": "someone-else"})}),
        triage_fn=triage_ok,
    )
    assert any("belongs to 'someone-else'" in p for p in plan.problems)


def test_a_missing_push_token_is_only_a_note_in_a_dry_run(approved):
    plan, _, _ = check(approved, github_push_token="", dry_run=True)
    assert plan.ok and any("GITHUB_PUSH_TOKEN" in n for n in plan.notes)


def test_preflight_blocks_when_triage_says_skip(approved):
    from repodoctor.tools.findings import Finding

    def skip(client, ref, me=""):
        r = Report()
        r.findings.append(Finding("skip", "assigned", "Already assigned to bob."))
        return SimpleNamespace(report=r)

    pkg, settings = approved
    plan = preflight(pkg, settings, read_client=api(), push_client=api(), triage_fn=skip)
    assert "triage: Already assigned to bob." in plan.problems


def test_preflight_enforces_the_daily_limit(approved):
    pkg, settings = approved
    settings.prs_dir.mkdir(parents=True, exist_ok=True)
    (settings.prs_dir / "ledger.json").write_text(json.dumps([{
        "repo": "a/b", "issue": 1, "url": "https://github.com/a/b/pull/1",
        "at": datetime.now(UTC).isoformat(),
    }]))
    plan, _, _ = check(approved)
    assert any("already submitted in the last 24 hours" in p for p in plan.problems)


def test_submit_forks_pushes_and_opens_the_pull_request(approved):
    pkg, settings = approved
    calls: list = []
    pushes: list = []
    meta = submit_package(
        pkg, settings, client=api(calls=calls),
        pusher=lambda clone, url, branch, token: pushes.append((Path(clone), url, branch, token)),
        sleep=lambda _s: None,
    )
    clone = Path(load_package(pkg)["workdir"])
    assert pushes == [(clone, "https://github.com/me/r.git", "repodoctor/issue-5", "tok")]
    (pr_call,) = [c for c in calls if c[:2] == ("POST", "/repos/o/r/pulls")]
    payload = pr_call[2]
    assert payload["head"] == "me:repodoctor/issue-5" and payload["base"] == "main"
    assert payload["title"] == "x is wrong" and payload["body"].startswith("Fixes #5.")
    assert meta["submission"]["stage"] == "opened"
    assert meta["submission"]["url"] == "https://github.com/o/r/pull/9"
    assert [e["url"] for e in load_ledger(settings)] == ["https://github.com/o/r/pull/9"]


def test_submitting_to_your_own_repository_skips_the_fork(approved):
    pkg, settings = approved
    own = settings.model_copy(update={"github_username": "o"})
    calls: list = []
    pushes: list = []
    submit_package(
        pkg, own,
        client=api({**ROUTES, ("GET", "/repos/o/r/pulls"): (200, [])}, calls),
        pusher=lambda clone, url, branch, token: pushes.append(url), sleep=lambda _s: None,
    )
    assert pushes == ["https://github.com/o/r.git"]
    assert not any(c[1].endswith("/forks") for c in calls)
    (pr_call,) = [c for c in calls if c[:2] == ("POST", "/repos/o/r/pulls")]
    assert pr_call[2]["head"] == "o:repodoctor/issue-5"


def test_an_existing_pull_request_is_reused_and_a_second_submission_is_refused(approved):
    pkg, settings = approved
    routes = {**ROUTES, ("GET", "/repos/o/r/pulls"): (200, [{"html_url": "https://x/pull/3"}])}
    calls: list = []
    meta = submit_package(
        pkg, settings, client=api(routes, calls), pusher=lambda *a: None, sleep=lambda _s: None
    )
    assert meta["submission"]["url"] == "https://x/pull/3"
    assert not any(c[:2] == ("POST", "/repos/o/r/pulls") for c in calls)
    with pytest.raises(PackageError, match="already opened"):
        submit_package(pkg, settings, client=api(), pusher=lambda *a: None)


def test_a_resumed_submission_does_not_push_twice(approved):
    pkg, settings = approved
    pushes: list = []
    pusher = lambda *a: pushes.append(a)  # noqa: E731
    with pytest.raises(GitHubError_for_test()):
        submit_package(  # the PR call fails after the push succeeded
            pkg, settings, client=api({**{k: v for k, v in ROUTES.items()
                                         if k != ("POST", "/repos/o/r/pulls")}}),
            pusher=pusher, sleep=lambda _s: None,
        )
    assert load_package(pkg)["submission"]["stage"] == "pushed" and len(pushes) == 1
    submit_package(pkg, settings, client=api(), pusher=pusher, sleep=lambda _s: None)
    assert len(pushes) == 1 and load_package(pkg)["submission"]["stage"] == "opened"


def GitHubError_for_test():
    from repodoctor.tools.github import GitHubError

    return GitHubError


def test_push_branch_to_a_local_remote_and_failures_hide_the_token(make_git_repo, tmp_path):
    src = make_git_repo({"a.txt": "x\n"})
    remote = tmp_path / "remote.git"
    run_git("init", "-q", "--bare", str(remote))
    run_git("checkout", "-q", "-b", "feature/x", cwd=src)
    push_branch(src, remote.as_uri(), "feature/x", "secret-token")
    assert "feature/x" in run_git("branch", "--list", cwd=remote)
    with pytest.raises(GitError) as e:
        push_branch(src, (tmp_path / "missing.git").as_uri(), "feature/x", "secret-token")
    assert "secret-token" not in str(e.value)


def test_scrub_removes_the_token_and_its_header():
    import base64

    header = base64.b64encode(b"x-access-token:tok123").decode()
    assert scrub(f"bad tok123 and {header}", "tok123") == "bad *** and ***"


def test_split_pr_text_and_recent_submissions():
    assert split_pr_text("Title\n\nBody line\n") == ("Title", "Body line\n")
    now = datetime(2026, 11, 20, tzinfo=UTC)
    ledger = [
        {"at": (now - timedelta(hours=2)).isoformat()},
        {"at": (now - timedelta(hours=30)).isoformat()},
    ]
    assert len(recent_submissions(ledger, now)) == 1