import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from repodoctor.config import Settings
from repodoctor.tools.conventions import analyze
from repodoctor.tools.findings import Report
from repodoctor.tools.github import IssueData, RepoInfo
from repodoctor.tools.issue_ref import IssueRef
from repodoctor.tools.pr_package import (
    PackageError,
    approve,
    blocking_findings,
    copies_hypothesis,
    load_package,
    prepare_package,
    render_review,
    save_package,
    tidy_hypothesis,
    verify_approval,
)
from repodoctor.tools.repackage import failure_summary, repackage
from repodoctor.tools.repo import commit_files, get_diff, head_sha, remove_dir, run_git
from repodoctor.tools.sandbox import ExecResult
from repodoctor.tools.testreport import SuiteReport

REF = IssueRef("o", "r", 5)
ISSUE = IssueData(
    url="https://github.com/o/r/issues/5", title="x is wrong", body="body", state="open",
    labels=[], assignees=[], comments=[], is_pull_request=False,
)
REPO = RepoInfo("main", False, False, False)
HYPOTHESIS = (
    "The bug is caused by the `or None` pattern in the constructor, which treats 0 as falsy "
    "and converts it to None, so the padding falls back to the default width."
)


def build_run(make_git_repo, extra=None):
    """A work clone in the state the agent leaves it: test committed, fix staged."""
    root = make_git_repo({
        "app.py": "x = 1\n",
        "requirements.txt": "\n",
        "tests/test_app.py": "def test_ok():\n    assert True\n",
        **(extra or {}),
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


@pytest.fixture
def run(make_git_repo):
    return build_run(make_git_repo)

CONV_TEMPLATE = (
    "<!-- c -->\n\nChecklist:\n\n"
    "- [ ] Add tests that demonstrate the behavior. Tests should fail without the change.\n"
    "- [ ] Run `pre-commit` hooks and fix any issues.\n"
    "- [ ] Run `pytest` and `tox`.\n"
)
PRECOMMIT = (
    "repos:\n  - repo: https://github.com/astral-sh/ruff-pre-commit\n"
    "    rev: v0.4.4\n    hooks:\n      - id: ruff\n"
)


def make_conv():
    records = [{
        "number": n, "title": "t", "issues": [n], "merged_at": "x",
        "files": [
            {"path": "app.py", "status": "modified", "additions": 1, "deletions": 1, "kind": "source"},
            {"path": "tests/test_app.py", "status": "modified", "additions": 3, "deletions": 0,
             "kind": "test"},
            {"path": f"changelog/{n}.bugfix.rst", "status": "added", "additions": 1, "deletions": 0,
             "kind": "fragment"},
        ],
        "new_tests": [f"test_issue{n}_x"],
    } for n in (1, 2, 3)]
    return {"prs": records, "stats": analyze(records), "pr_template": CONV_TEMPLATE,
            "changelog_head": "", "changelog_style": None}


def prepare_conv(root, final, tmp_path):
    return prepare_package(
        ref=REF, issue=ISSUE, final=final, workdir=root, settings=settings(tmp_path),
        triage=Report(), repo_info=REPO, policy_hits=[], conventions=make_conv(),
    )


def test_prepare_uses_the_projects_conventions(run, tmp_path):
    root, final, _ = run
    pkg = prepare_conv(root, final, tmp_path)
    meta = load_package(pkg)
    clone = Path(meta["workdir"])
    assert "changelog/5.bugfix.rst" in meta["files"]
    assert "TODO(reviewer)" in (clone / "changelog" / "5.bugfix.rst").read_text(encoding="utf-8")
    text = (pkg / "pr.md").read_text(encoding="utf-8")
    assert "Checklist:" in text and "- [ ] Add tests that demonstrate" in text and "<!--" not in text
    assert "Assisted-by: RepoDoctor (" in text
    style = {f["code"]: f["message"] for f in meta["style"]}
    assert {"new_test_file", "test_name_convention", "not_reverified"} <= style.keys()
    assert "tests/test_app.py" in style["new_test_file"]


def test_no_changelog_reminder_when_past_fixes_never_touch_it(run, tmp_path):
    root, final, _ = run
    (root / "CHANGELOG.md").write_text("# Changes\n", encoding="utf-8")
    records = [{
        "number": n, "title": "t", "issues": [n], "merged_at": "x", "new_tests": [],
        "files": [{"path": "app.py", "status": "modified", "additions": 1, "deletions": 1,
                   "kind": "source"}],
    } for n in (1, 2, 3)]
    conv = {"prs": records, "stats": analyze(records), "pr_template": "",
            "changelog_head": "", "changelog_style": None}
    pkg = prepare_package(
        ref=REF, issue=ISSUE, final=final, workdir=root, settings=settings(tmp_path),
        triage=Report(), repo_info=REPO, policy_hits=[], conventions=conv,
    )
    assert "changelog_not_updated" not in {f["code"] for f in load_package(pkg)["style"]}
    plain = prepare(run, tmp_path)   # no conventions learned: the reminder stays
    assert "changelog_not_updated" in {f["code"] for f in load_package(plain)["style"]}


def test_a_todo_left_in_the_patch_blocks_approval(run, tmp_path):
    root, final, _ = run
    pkg = prepare_conv(root, final, tmp_path)
    fill_todo(pkg)
    with pytest.raises(PackageError, match="patch still contains"):
        approve(pkg, policy_acknowledged=True)


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


class LintSandbox(FakeSandbox):
    def __init__(self, target, suite, replies):
        super().__init__(target, suite)
        self.replies = replies
        self.commands: list[str] = []

    def run(self, cmd, timeout):
        self.commands.append(cmd)
        for key, queue in self.replies.items():
            if key in cmd:
                code, out = queue[0]
                return ExecResult(code, out, False, 0.0)
        return ExecResult(0, "", False, 0.0)


@pytest.mark.parametrize("ruff_reply, lint_box", [
    ((0, ""), "x"), ((1, "app.py:1:1: E501 too long\n"), " "),
])
def test_repackage_runs_the_projects_linters_and_ticks_what_it_verified(
    make_git_repo, tmp_path, ruff_reply, lint_box
):
    root, final, _ = build_run(make_git_repo, {".pre-commit-config.yaml": PRECOMMIT})
    pkg = prepare_conv(root, final, tmp_path)
    clone = Path(load_package(pkg)["workdir"])
    (clone / "changelog" / "5.bugfix.rst").write_text("Fix the value of x.\n", encoding="utf-8")
    sb = LintSandbox([1, 0], SuiteReport(total=3, passed=3),
                     {"pip install": [(0, "")], "ruff check": [ruff_reply]})
    meta = repackage(pkg, settings(tmp_path), sandbox_factory=lambda _r: sb)

    assert "ruff==0.4.4" in sb.commands[0]
    assert meta["verification"]["lint_tools"] == ["ruff-check"]
    text = (pkg / "pr.md").read_text(encoding="utf-8")
    assert "- [x] Add tests that demonstrate" in text
    assert "- [ ] Run `pytest` and `tox`." in text
    clean = ruff_reply[0] == 0
    assert ("- [x] Run `pre-commit` hooks" in text) is clean
    assert any(f["code"] == "lint" for f in meta["style"]) is (not clean)
    assert not any(f["code"] == "not_reverified" for f in meta["style"])

def settings(tmp_path, **over) -> Settings:
    over.setdefault("root_cause_mode", "human")
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
        "The cause, in my own words: the value was set wrong, so the check failed."
        if "TODO(reviewer)" in ln else ln
        for ln in p.read_text(encoding="utf-8").splitlines()
    ]
    p.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


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
    assert "## AI assistance" in text
    assert "TODO(reviewer)" in text and "x has the wrong value" not in text


def test_the_commit_credits_the_agent_with_a_trailer(run, tmp_path):
    pkg = prepare(run, tmp_path)
    body = run_git("log", "-1", "--format=%B", cwd=Path(load_package(pkg)["workdir"]))
    assert "Fixes #5" in body and "Assisted-by: RepoDoctor (" in body


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


def test_copies_hypothesis_detects_copies_and_light_rewording_but_not_a_fresh_explanation():
    assert copies_hypothesis(HYPOTHESIS, HYPOTHESIS)
    assert copies_hypothesis(HYPOTHESIS.replace("constructor", "init method"), HYPOTHESIS)
    fresh = "Zero is falsy in Python, so `x or None` quietly discards an explicit 0 argument."
    assert not copies_hypothesis(fresh, HYPOTHESIS)
    assert not copies_hypothesis("short", HYPOTHESIS)


def test_a_root_cause_copied_from_the_agents_hypothesis_is_refused(run, tmp_path):
    pkg = prepare(run, tmp_path)
    meta = load_package(pkg)
    meta["hypothesis"] = HYPOTHESIS
    save_package(pkg, meta)
    original = (pkg / "pr.md").read_text(encoding="utf-8")

    def write_cause(cause: str) -> None:
        text = "\n".join(
            cause if "TODO(reviewer)" in ln else ln for ln in original.splitlines()
        )
        (pkg / "pr.md").write_bytes((text + "\n").encode("utf-8"))

    write_cause(HYPOTHESIS)
    with pytest.raises(PackageError, match="own hypothesis"):
        approve(pkg, policy_acknowledged=True)
    write_cause("Zero is falsy in Python, so `x or None` quietly discards an explicit 0 argument.")
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

def test_failure_summary_keeps_the_error_lines():
    out = "x\nE   AssertionError: boom\nE   assert 1 == 2\ny\n" + "E  more\n" * 10
    assert failure_summary(out, 3) == ["E   AssertionError: boom", "E   assert 1 == 2", "E  more"]
    assert failure_summary("no errors here") == []


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
    assert meta["verification"]["failure_without_fix"] == []
    assert "the answer" in (pkg / "change.patch").read_text(encoding="utf-8")
    assert run_git("rev-list", "--count", f"{meta['base_sha']}..HEAD", cwd=clone).strip() == "1"
    assert (clone / "app.py").read_bytes() == b"x = 2  # the answer\n"   # restored after the check
    assert not blocking_findings(meta)


def test_repackage_omits_changelog_reminder_when_conventions_say_none(
    make_git_repo, tmp_path
):
    root, final, _ = build_run(make_git_repo, {"CHANGELOG.md": "Past changes\n"})
    conventions = make_conv()
    conventions["stats"]["changelog"]["kind"] = "none"
    pkg = prepare_package(
        ref=REF, issue=ISSUE, final=final, workdir=root, settings=settings(tmp_path),
        triage=Report(), repo_info=REPO, policy_hits=[], conventions=conventions,
    )
    meta = repackage(
        pkg, settings(tmp_path),
        sandbox_factory=lambda _root: FakeSandbox([1, 0], SuiteReport(total=3, passed=3)),
    )
    assert not any(f["code"] == "changelog_not_updated" for f in meta["style"])


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


def test_a_one_word_root_cause_is_not_enough(run, tmp_path):
    pkg = prepare(run, tmp_path)
    lines = ["fixed" if "TODO(reviewer)" in ln else ln
             for ln in (pkg / "pr.md").read_text(encoding="utf-8").splitlines()]
    (pkg / "pr.md").write_bytes(("\n".join(lines) + "\n").encode())
    with pytest.raises(PackageError, match="Root cause"):
        approve(pkg, policy_acknowledged=True)


def test_markdown_heading_marks_are_stripped_from_the_title(run, tmp_path):
    root, final, _ = run
    pkg = prepare_package(
        ref=REF, issue=replace(ISSUE, title="# truncate() shortens text"), final=final,
        workdir=root, settings=settings(tmp_path), triage=Report(), repo_info=REPO,
        policy_hits=[],
    )
    assert (pkg / "pr.md").read_text(encoding="utf-8").startswith("truncate() shortens text\n")


def test_the_disclosure_trailer_can_be_switched_off(run, tmp_path):
    root, final, _ = run
    pkg = prepare_package(
        ref=REF, issue=ISSUE, final=final, workdir=root,
        settings=settings(tmp_path, commit_trailer=""), triage=Report(),
        repo_info=REPO, policy_hits=[],
    )
    body = run_git("log", "-1", "--format=%B", cwd=Path(load_package(pkg)["workdir"]))
    assert "Assisted-by" not in body
    assert "Assisted-by" not in (pkg / "pr.md").read_text(encoding="utf-8")
    assert "AI agent" in (pkg / "pr.md").read_text(encoding="utf-8")


def test_repackage_uses_the_title_from_pr_md_as_the_commit_message(run, tmp_path):
    pkg = prepare(run, tmp_path)
    rest = (pkg / "pr.md").read_text(encoding="utf-8").split("\n", 1)[1]
    (pkg / "pr.md").write_bytes(("Fix the wrong value of x\n" + rest).encode())
    sb = FakeSandbox(target=[1, 0], suite=SuiteReport(total=3, passed=3))
    meta = repackage(pkg, settings(tmp_path), sandbox_factory=lambda _root: sb)
    assert run_git("log", "-1", "--format=%s", cwd=Path(meta["workdir"])).strip() == (
        "Fix the wrong value of x"
    )
    assert "Subject: [PATCH] Fix the wrong value of x" in (pkg / "change.patch").read_text(
        encoding="utf-8"
    )
    assert "Assisted-by: RepoDoctor (" in run_git(
        "log", "-1", "--format=%B", cwd=Path(meta["workdir"])
    )

def test_tidy_hypothesis_removes_line_numbers_and_keeps_a_few_sentences():
    raw = (
        "The bug is caused by the `or None` pattern on lines 474-475 of src/x/y.py, which "
        "treats `0` as falsy and converts it to `None`. When `left=0` is passed, the "
        "`_get` method (lines 2147-2156) falls back to the default. Third. Fourth."
    )
    out = tidy_hypothesis(raw)
    assert "474" not in out and "2147" not in out and "lines" not in out
    assert out.startswith("The bug is caused by the `or None` pattern, which treats `0` as falsy")
    assert "method falls back to the default." in out and out.endswith("Third.")


def agent_package(run, tmp_path, hypothesis, **extra):
    root, final, _ = run
    return prepare_package(
        ref=REF, issue=ISSUE, final={**final, "hypothesis": hypothesis}, workdir=root,
        settings=settings(tmp_path, root_cause_mode="agent"), triage=Report(),
        repo_info=REPO, **{"policy_hits": [], **extra},
    )


def test_agent_mode_fills_the_root_cause(run, tmp_path):
    pkg = agent_package(
        run, tmp_path,
        "The value is wrong on line 10 of app.py, so the check fails. It never recovers.",
    )
    text = (pkg / "pr.md").read_text(encoding="utf-8")
    assert "TODO(reviewer)" not in text and "line 10" not in text
    assert "The value is wrong, so the check fails." in text
    assert "drafted by the agent" not in text
    assert "prepared with the help of an AI agent" in text
    assert load_package(pkg)["root_cause_mode"] == "agent"
    approve(pkg, policy_acknowledged=True)     # no rewriting needed


def test_agent_mode_warns_when_the_project_mentions_ai(run, tmp_path):
    pkg = agent_package(run, tmp_path, "The value is wrong. It fails.",
                        policy_hits=["CONTRIBUTING.md:1: AI tools welcome"])
    assert "agent_written_description" in {f["code"] for f in load_package(pkg)["style"]}


def test_human_mode_is_unchanged_by_the_new_setting(run, tmp_path):
    pkg = prepare(run, tmp_path)
    assert load_package(pkg)["root_cause_mode"] == "human"
    with pytest.raises(PackageError, match="TODO"):
        approve(pkg, policy_acknowledged=True)


def test_repackage_switches_to_the_agents_root_cause_only_when_asked(run, tmp_path):
    root, final, _ = run
    pkg = prepare_package(
        ref=REF, issue=ISSUE,
        final={**final, "hypothesis": "The value is wrong on line 3 of app.py. It fails."},
        workdir=root, settings=settings(tmp_path), triage=Report(), repo_info=REPO, policy_hits=[],
    )
    custom = "My own words about why the comparison used the wrong operator here."
    lines = [custom if "TODO(reviewer)" in ln else ln
             for ln in (pkg / "pr.md").read_text(encoding="utf-8").splitlines()]
    (pkg / "pr.md").write_bytes(("\n".join(lines) + "\n").encode())

    def sandbox():
        return lambda _r: FakeSandbox(target=[1, 0], suite=SuiteReport(total=3, passed=3))

    meta = repackage(pkg, settings(tmp_path, root_cause_mode="agent"), sandbox_factory=sandbox())
    assert meta.get("root_cause_mode") != "agent"
    assert custom in (pkg / "pr.md").read_text(encoding="utf-8")        # your text is left alone
    meta = repackage(pkg, settings(tmp_path), sandbox_factory=sandbox(), agent_cause=True)
    text = (pkg / "pr.md").read_text(encoding="utf-8")
    assert meta["root_cause_mode"] == "agent" and "The value is wrong. It fails." in text
    assert custom not in text