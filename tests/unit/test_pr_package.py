import subprocess
import sys
from pathlib import Path

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
from repodoctor.tools.repackage import repackage
from repodoctor.tools.repo import commit_files, get_diff, head_sha, remove_dir, run_git
from repodoctor.tools.sandbox import ExecResult
from repodoctor.tools.testreport import SuiteReport

REF = IssueRef("o", "r", 5)
ISSUE = IssueData(
    url="https://github.com/o/r/issues/5", title="x is wrong", body="body", state="open",
    labels=[], assignees=[], comments=[], is_pull_request=False,
)
REPO = RepoInfo("main", False, False, False)


@pytest.fixture
def run(make_git_repo):
    """A work clone in the state the agent leaves it: test committed, fix staged."""
    root = make_git_repo({
        "app.py": "x = 1\n",
        "requirements.txt": "\n",
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


def fill_todo(pkg: Path) -> None:
    p = pkg / "pr.md"
    lines = [
        "The cause, in my own words." if "TODO(reviewer)" in ln else ln
        for ln in p.read_text(encoding="utf-8").splitlines()
    ]
    p.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


class FakeSandbox:
    """Scripted test runs: `target` exit codes for single-test runs, `suite` for the full run."""

    def __init__(self, target, suite):
        self.target, self.suite = list(target), suite

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def install(self, info):
        return ExecResult(0, "", False, 0.0)

    def lock_network(self):
        pass

    def run_suite(self, info, args=None, timeout=300):
        if args:
            return ExecResult(self.target.pop(0), "", False, 0.0), SuiteReport()
        return ExecResult(0, "", False, 0.0), self.suite


def test_prepare_makes_one_clean_commit_on_a_branch_in_its_own_clone(run, tmp_path):
    root, _, base = run
    pkg = prepare(run, tmp_path)
    meta = load_package(pkg)
    clone = Path(meta["workdir"])
    assert pkg.name == "o__r-5" and meta["branch"] == "repodoctor/issue-5"
    assert clone == pkg.resolve() / "clone" and clone != root
    assert run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=clone).strip() == "repodoctor/issue-5"
    assert run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=root).strip() == "main"  # untouched
    assert run_git("rev-parse", "HEAD~1", cwd=clone).strip() == base
    assert run_git("log", "-1", "--format=%an <%ae>", cwd=clone).strip() == (
        "me <me@users.noreply.github.com>"
    )
    assert sorted(meta["files"]) == ["app.py", "tests/test_issue_5.py"]
    assert meta["evidence"]["test_id"] == "tests/test_issue_5.py::test_x"


def test_the_package_survives_removal_of_the_work_directory(run, tmp_path):
    root, _, _ = run
    pkg = prepare(run, tmp_path)
    remove_dir(root)
    clone = Path(load_package(pkg)["workdir"])
    assert clone.is_dir() and run_git("log", "-1", "--format=%s", cwd=clone).strip()


def test_the_generated_test_is_renamed_and_still_passes(run, tmp_path):
    pkg = prepare(run, tmp_path)
    assert "test_repodoctor_repro" not in (pkg / "change.patch").read_text(encoding="utf-8")
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_issue_5.py"],
        cwd=load_package(pkg)["workdir"], capture_output=True, text=True,
    )
    assert p.returncode == 0, p.stdout


def test_the_pr_text_discloses_the_agent_and_leaves_the_cause_to_the_reviewer(run, tmp_path):
    text = (prepare(run, tmp_path) / "pr.md").read_text(encoding="utf-8")
    assert text.startswith("x is wrong\n") and "Fixes #5." in text
    assert "AI agent" in text and "@me" in text and "tests/test_issue_5.py::test_x" in text
    assert "TODO(reviewer)" in text and "x has the wrong value" not in text


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


def test_approval_waits_for_the_reviewer_to_write_the_cause(run, tmp_path):
    pkg = prepare(run, tmp_path)
    with pytest.raises(PackageError, match="TODO"):
        approve(pkg, policy_acknowledged=True)
    fill_todo(pkg)
    approve(pkg, policy_acknowledged=True)


def test_approval_records_hashes_and_detects_later_edits(run, tmp_path):
    pkg = prepare(run, tmp_path)
    assert verify_approval(pkg) == ["the package is not approved"]
    fill_todo(pkg)
    approve(pkg, policy_acknowledged=True)
    assert verify_approval(pkg) == []
    (pkg / "pr.md").write_bytes(b"changed\n")
    assert any("pr.md" in problem for problem in verify_approval(pkg))


def test_a_skip_finding_blocks_approval(run, tmp_path):
    pkg = prepare(run, tmp_path)
    meta = load_package(pkg)
    meta["style"].append({"level": "skip", "code": "secret_in_diff", "message": "a secret"})
    save_package(pkg, meta)
    assert blocking_findings(meta)
    with pytest.raises(PackageError, match="a secret"):
        approve(pkg, policy_acknowledged=True)


@pytest.mark.parametrize("field, value", [
    ("policy_hits", ["CONTRIBUTING.md:1: AI tools are fine if disclosed"]),
    ("ai_threads", ["#512 Low quality duplicate PRs caused by agent projects"]),
])
def test_policy_mentions_and_ai_threads_need_an_acknowledgement(run, tmp_path, field, value):
    pkg = prepare(run, tmp_path)
    fill_todo(pkg)
    meta = load_package(pkg)
    meta[field] = value
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
    for part in ("REVIEW  o/r", "Work clone:", "TRIAGE", "POLICY", "RECENT THREADS",
                 "DIFF CHECKS", "EVIDENCE", "THE AGENT'S HYPOTHESIS", "x has the wrong value",
                 "PULL REQUEST TEXT", "PATCH", "DRY_RUN is on", "x is wrong"):
        assert part in shown


# ---------- repackage ----------

def test_repackage_reverifies_and_clears_approval(run, tmp_path):
    pkg = prepare(run, tmp_path)
    fill_todo(pkg)
    approve(pkg, policy_acknowledged=True)
    clone = Path(load_package(pkg)["workdir"])
    (clone / "app.py").write_bytes(b"x = 2  # the answer\n")

    sb = FakeSandbox(target=[1, 0], suite=SuiteReport(total=3, passed=3))
    meta = repackage(pkg, settings(tmp_path), sandbox_factory=lambda _root: sb)

    assert meta["approval"] is None
    v = meta["verification"]
    assert (v["test_fails_without_fix"], v["test_passes_with_fix"], v["new_failures"]) == (
        True, True, [],
    )
    assert "the answer" in (pkg / "change.patch").read_text(encoding="utf-8")
    assert run_git("rev-list", "--count", f"{meta['base_sha']}..HEAD", cwd=clone).strip() == "1"
    assert (clone / "app.py").read_bytes() == b"x = 2  # the answer\n"   # restored after the check
    assert not blocking_findings(meta)


def test_repackage_blocks_a_test_that_passes_without_the_fix(run, tmp_path):
    pkg = prepare(run, tmp_path)
    sb = FakeSandbox(target=[0, 0], suite=SuiteReport(total=3, passed=3))
    meta = repackage(pkg, settings(tmp_path), sandbox_factory=lambda _root: sb)
    assert meta["verification"]["test_fails_without_fix"] is False
    assert any(f["code"] == "test_does_not_fail_without_fix" for f in blocking_findings(meta))


def test_repackage_blocks_regressions(run, tmp_path):
    pkg = prepare(run, tmp_path)
    suite = SuiteReport(total=3, passed=2, failed=1, failing={"tests.test_app::test_ok": "boom"})
    meta = repackage(
        pkg, settings(tmp_path), sandbox_factory=lambda _root: FakeSandbox([1, 0], suite)
    )
    assert meta["verification"]["new_failures"] == ["tests.test_app::test_ok"]
    assert any(f["code"] == "regression" for f in blocking_findings(meta))


def test_repackage_refuses_when_the_clone_is_on_another_branch(run, tmp_path):
    pkg = prepare(run, tmp_path)
    run_git("checkout", "-q", "-b", "other", cwd=load_package(pkg)["workdir"])
    with pytest.raises(PackageError, match="not on branch"):
        repackage(pkg, settings(tmp_path), sandbox_factory=lambda _root: FakeSandbox([], None))


def test_repackage_rejects_an_unknown_test_id_without_touching_the_package(run, tmp_path):
    pkg = prepare(run, tmp_path)
    before = ((pkg / "pr.md").read_bytes(), load_package(pkg))
    with pytest.raises(PackageError, match="not a test in"):
        repackage(
            pkg, settings(tmp_path), test_id="tests/test_issue_5.py::test_nope",
            sandbox_factory=lambda _root: FakeSandbox([], None),
        )
    assert ((pkg / "pr.md").read_bytes(), load_package(pkg)) == before


def test_repackage_says_when_pytest_cannot_run_the_test(run, tmp_path):
    pkg = prepare(run, tmp_path)
    sb = FakeSandbox(target=[4, 4], suite=SuiteReport(total=3, passed=3))
    meta = repackage(pkg, settings(tmp_path), sandbox_factory=lambda _root: sb)
    codes = {f["code"] for f in blocking_findings(meta)}
    assert "test_not_runnable" in codes and "test_does_not_fail_without_fix" not in codes
