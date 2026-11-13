import subprocess
import sys

import pytest

from repodoctor.config import Settings
from repodoctor.tools.findings import Report
from repodoctor.tools.github import IssueData, RepoInfo
from repodoctor.tools.issue_ref import IssueRef
from repodoctor.tools.pr_package import (
    PackageError,
    approve,
    blocking_findings,
    load_package,
    prepare_package,
    render_review,
    save_package,
    verify_approval,
)
from repodoctor.tools.repo import commit_files, get_diff, get_diff_since, head_sha, run_git

REF = IssueRef("o", "r", 5)
ISSUE = IssueData(
    url="https://github.com/o/r/issues/5", title="x is wrong", body="body", state="open",
    labels=[], assignees=[], comments=[], is_pull_request=False,
)
REPO = RepoInfo("main", False, False, False)


@pytest.fixture
def run(make_git_repo):
    """A work clone in the state the agent leaves it: test committed, fix staged."""
    root = make_git_repo({"app.py": "x = 1\n", "tests/test_app.py": "def test_ok():\n    assert True\n"})
    base = head_sha(root)
    (root / "tests" / "test_repodoctor_repro.py").write_text(
        "from app import x\n\n\ndef test_x():\n    assert x == 2\n"
    )
    commit_files(root, ["tests/test_repodoctor_repro.py"], "test: reproduce")
    (root / "app.py").write_text("x = 2\n")
    get_diff(root, ["app.py"])
    final = {
        "success": True, "base_sha": base, "attempts": 1, "repro_attempts": 1,
        "repro_file": "tests/test_repodoctor_repro.py",
        "target_tests": ["tests/test_repodoctor_repro.py::test_x"],
        "hypothesis": "x has the wrong value.",
        "baseline": {"total": 3, "passed": 3, "failed": 0, "errors": 0, "skipped": 0},
    }
    return root, final, base


def settings(tmp_path, **over) -> Settings:
    return Settings(_env_file=None, github_username="me", prs_dir=tmp_path / "prs", **over)


def prepare(run, tmp_path, **over):
    root, final, _ = run
    return prepare_package(
        ref=REF, issue=ISSUE, final={**final, **over}, workdir=root,
        settings=settings(tmp_path), triage=Report(), repo_info=REPO, policy_hits=[],
    )


def test_prepare_makes_one_clean_commit_on_a_branch(run, tmp_path):
    root, _, base = run
    pkg = prepare(run, tmp_path)
    meta = load_package(pkg)
    assert pkg.name == "o__r-5" and meta["branch"] == "repodoctor/issue-5"
    assert run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=root).strip() == "repodoctor/issue-5"
    assert run_git("rev-parse", "HEAD~1", cwd=root).strip() == base
    assert run_git("log", "-1", "--format=%an <%ae>", cwd=root).strip() == (
        "me <me@users.noreply.github.com>"
    )
    assert sorted(meta["files"]) == ["app.py", "tests/test_issue_5.py"]
    assert meta["evidence"]["test_id"] == "tests/test_issue_5.py::test_x"


def test_the_generated_test_is_renamed_and_still_passes(run, tmp_path):
    root, _, _ = run
    pkg = prepare(run, tmp_path)
    assert "test_repodoctor_repro" not in (pkg / "change.patch").read_text(encoding="utf-8")
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_issue_5.py"],
        cwd=root, capture_output=True, text=True,
    )
    assert p.returncode == 0, p.stdout


def test_the_pr_text_discloses_the_agent_and_links_the_issue(run, tmp_path):
    text = (prepare(run, tmp_path) / "pr.md").read_text(encoding="utf-8")
    assert text.startswith("x is wrong\n") and "Fixes #5." in text
    assert "AI agent" in text and "@me" in text and "tests/test_issue_5.py::test_x" in text


def test_prepare_refuses_without_a_success_or_a_username(run, tmp_path):
    root, final, _ = run
    with pytest.raises(PackageError, match="did not succeed"):
        prepare(run, tmp_path, success=False)
    with pytest.raises(PackageError, match="GITHUB_USERNAME"):
        prepare_package(
            ref=REF, issue=ISSUE, final=final, workdir=root,
            settings=Settings(_env_file=None, prs_dir=tmp_path / "prs"),
            triage=Report(), repo_info=REPO, policy_hits=[],
        )


def test_approval_records_hashes_and_detects_later_edits(run, tmp_path):
    pkg = prepare(run, tmp_path)
    assert verify_approval(pkg) == ["the package is not approved"]
    approve(pkg, policy_acknowledged=True)
    assert verify_approval(pkg) == []
    (pkg / "pr.md").write_text("changed\n", encoding="utf-8")
    assert any("pr.md" in problem for problem in verify_approval(pkg))


def test_a_skip_finding_blocks_approval(run, tmp_path):
    pkg = prepare(run, tmp_path)
    meta = load_package(pkg)
    meta["style"].append({"level": "skip", "code": "secret_in_diff", "message": "a secret"})
    save_package(pkg, meta)
    assert blocking_findings(meta)
    with pytest.raises(PackageError, match="a secret"):
        approve(pkg, policy_acknowledged=True)


def test_policy_mentions_need_an_acknowledgement(run, tmp_path):
    pkg = prepare(run, tmp_path)
    meta = load_package(pkg)
    meta["policy_hits"] = ["CONTRIBUTING.md:1: AI tools are fine if disclosed"]
    save_package(pkg, meta)
    with pytest.raises(PackageError, match="mention AI"):
        approve(pkg, policy_acknowledged=False)
    approve(pkg, policy_acknowledged=True)


def test_the_review_screen_shows_every_section(run, tmp_path):
    pkg = prepare(run, tmp_path)
    shown = render_review(
        load_package(pkg),
        (pkg / "change.patch").read_text(encoding="utf-8"),
        (pkg / "pr.md").read_text(encoding="utf-8"),
    )
    for part in ("REVIEW  o/r", "TRIAGE", "POLICY", "DIFF CHECKS", "EVIDENCE",
                 "PULL REQUEST TEXT", "PATCH", "DRY_RUN is on", "x is wrong"):
        assert part in shown