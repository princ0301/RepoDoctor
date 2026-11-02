import json
from dataclasses import asdict

import httpx
import pytest

from repodoctor.config import Settings
from repodoctor.evals.build import pair_problem
from repodoctor.evals.cases import (
    CaseSpec,
    EvalCase,
    case_id,
    changed_files,
    is_test_file,
    is_test_path,
    load_cases,
    load_specs,
    save_cases,
    save_specs,
)
from repodoctor.evals.discover import discover
from repodoctor.evals.grade import grade_tree
from repodoctor.evals.run import CaseResult, classify, load_results, run_eval, summarize
from repodoctor.tools.github import github_client
from repodoctor.tools.repo import ProjectInfo, clone_repo, head_sha, run_git
from repodoctor.tools.sandbox import ExecResult


def case(**over) -> EvalCase:
    base = dict(
        id="o__r-5", repo="o/r", issue=5, fix_pr=7, title="t", base_sha="a" * 40,
        head_sha="b" * 40, test_files=["tests/test_x.py"], run_targets=["tests/test_x.py"],
        src_files=["src/x.py"], fail_to_pass=["tests.test_x::test_a"], pass_to_pass=[],
    )
    return EvalCase(**{**base, **over})


# ---------- paths and diffs ----------

@pytest.mark.parametrize(
    "path, is_test",
    [
        ("tests/test_time.py", True), ("tests/data/x.json", True), ("test/helper.py", True),
        ("src/pkg/test_utils.py", True), ("pkg/x_test.py", True), ("tests/conftest.py", True),
        ("conftest.py", True), ("src/humanize/time.py", False), ("README.md", False),
        ("docs/tests.md", False),
    ],
)
def test_is_test_path(path, is_test):
    assert is_test_path(path) is is_test


def test_is_test_file_only_matches_collectable_modules():
    assert is_test_file("tests/test_time.py") and is_test_file("a/b_test.py")
    assert not is_test_file("tests/conftest.py") and not is_test_file("tests/data.json")


def test_changed_files_from_a_diff():
    diff = (
        "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n"
        "diff --git a/tests/new.py b/tests/new.py\nnew file mode 100644\n"
        "diff --git a/src/a.py b/src/a.py\n"
    )
    assert changed_files(diff) == ["src/a.py", "tests/new.py"]


def test_case_id_and_ref():
    assert case_id("python-humanize/humanize", 154) == "python-humanize__humanize-154"
    ref = case(repo="o/r", issue=5).ref
    assert (ref.owner, ref.repo, ref.number) == ("o", "r", 5)


def test_cases_and_specs_round_trip(tmp_path):
    save_cases(tmp_path / "c.json", [case()])
    assert load_cases(tmp_path / "c.json") == [case()]
    save_specs(tmp_path / "s.json", [CaseSpec("o/r", 5, 7)])
    assert load_specs(tmp_path / "s.json") == [CaseSpec("o/r", 5, 7)]


# ---------- classification and summary ----------

@pytest.mark.parametrize(
    "final, grade, expected",
    [
        ({"error": "Could not write a test that reproduces the issue after 3"}, None, "no_repro"),
        ({"error": "The fix model failed on attempt 1: boom"}, None, "model_error"),
        ({"error": "The explorer did not identify any existing source file."}, None,
         "explore_failed"),
        ({"error": "Dependency install failed (exit 1)"}, None, "setup_error"),
        ({"success": False}, None, "fix_failed"),
        ({"success": True}, None, "grading_error"),
        ({"success": True}, {"resolved": False}, "wrong_fix"),
        ({"success": True}, {"resolved": True}, "resolved"),
    ],
)
def test_classify(final, grade, expected):
    assert classify(final, grade) == expected


def result(case_id_="o__r-5", trial=1, outcome="resolved", seconds=60.0, success=True):
    return CaseResult(case_id_, trial, outcome, outcome == "resolved", success, 1, 1, seconds)


def test_summarize_reports_rates_and_precision():
    out = summarize([result(), result("o__r-6", outcome="wrong_fix")])
    assert "1/2 (50%)" in out
    assert "precision 50%" in out
    assert "resolved=1" in out and "wrong_fix=1" in out


def test_summarize_pass_at_k_with_multiple_trials():
    out = summarize([result(trial=1, outcome="fix_failed", success=False), result(trial=2)])
    assert "Solved in at least one trial: 1/1" in out


def test_summarize_empty():
    assert summarize([]) == "No results."


def test_pair_problem():
    assert pair_problem("2026-07-02T00:00:00Z", "2026-07-01T00:00:00Z") is None
    assert "already closed" in pair_problem("2024-01-01T00:00:00Z", "2026-07-01T00:00:00Z")
    assert pair_problem("", "2026-07-01T00:00:00Z") is None


def test_run_eval_stops_after_consecutive_model_errors(tmp_path, monkeypatch):
    ran = []

    def fake(c, trial, settings, *, label):
        ran.append(c.id)
        return result(c.id, trial, "model_error", success=False)

    monkeypatch.setattr("repodoctor.evals.run.run_case", fake)
    cases = [case(id=f"c{i}") for i in range(5)]
    run_eval(cases, Settings(_env_file=None), trials=1, label="t",
             results_file=tmp_path / "r.jsonl")
    assert ran == ["c0", "c1"]


def test_run_eval_skips_finished_runs(tmp_path, monkeypatch):
    ran = []

    def fake(c, trial, settings, *, label):
        ran.append((c.id, trial))
        return result(c.id, trial)

    monkeypatch.setattr("repodoctor.evals.run.run_case", fake)
    run_eval([case(id="a"), case(id="b")], Settings(_env_file=None), trials=1, label="t",
             results_file=tmp_path / "r.jsonl", done={("a", 1)})
    assert ran == [("b", 1)]


def test_load_results_keeps_the_latest_record_per_run(tmp_path):
    path = tmp_path / "r.jsonl"
    for outcome in ("model_error", "resolved"):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result("a", 1, outcome))) + "\n")
    (only,) = load_results(path)
    assert only.outcome == "resolved"


# ---------- discovery (mocked GitHub) ----------

def mock_client(routes: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        return httpx.Response(404) if body is None else httpx.Response(200, json=body)

    return github_client("", transport=httpx.MockTransport(handler))


def pr(number: int, body: str) -> dict:
    return {"number": number, "body": body, "merged_at": "2026-01-01T00:00:00Z"}


def files(*paths: str) -> list[dict]:
    return [{"filename": p, "status": "modified"} for p in paths]


def test_discover_keeps_only_small_single_issue_fixes_with_tests():
    routes = {
        "/repos/o/r/pulls": [
            pr(10, "Fixes #1"),                 # good
            pr(11, "Fixes #2"),                 # no test change
            pr(12, "Fixes #3 and fixes #4"),    # two issues
            pr(13, "no issue mentioned"),
            pr(14, "Fixes #6"),                 # docs-only source change
        ],
        "/repos/o/r/pulls/10/files": files("src/a.py", "tests/test_a.py"),
        "/repos/o/r/pulls/11/files": files("src/a.py"),
        "/repos/o/r/pulls/14/files": files("docs/a.md", "tests/test_a.py"),
    }
    assert discover(mock_client(routes), "o/r") == [CaseSpec("o/r", 1, 10)]


# ---------- grading (real git, fake sandbox) ----------

class FakeOutcomes:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.seen_test_file = False

    def run_outcomes(self, info, args=None, timeout=300):
        self.args = args
        return ExecResult(0, "", False, 0.0), self.outcomes


@pytest.fixture
def gold(make_git_repo, tmp_path):
    """A repo whose PR (refs/pull/7/head) adds tests/test_gold.py; clone at the base."""
    src = make_git_repo({"app.py": "x = 1\n"}, name="src")
    run_git("checkout", "-q", "-b", "pr", cwd=src)
    (src / "tests").mkdir()
    (src / "tests" / "test_gold.py").write_text("def test_a(): ...\n")
    run_git("add", "-A", cwd=src)
    run_git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "pr", cwd=src)
    sha = head_sha(src)
    run_git("update-ref", "refs/pull/7/head", sha, cwd=src)
    run_git("checkout", "-q", "main", cwd=src)
    work = clone_repo(src.as_uri(), tmp_path / "work")
    return work, case(head_sha=sha, test_files=["tests/test_gold.py"],
                      run_targets=["tests/test_gold.py"],
                      fail_to_pass=["tests.test_gold::test_a"],
                      pass_to_pass=["tests.test_gold::test_b"])


INFO = ProjectInfo(True, test_command=["python", "-m", "pytest"])


def test_grade_resolved_when_gold_tests_pass(gold):
    work, c = gold
    sandbox = FakeOutcomes({"tests.test_gold::test_a": "passed", "tests.test_gold::test_b": "passed"})
    g = grade_tree(sandbox, INFO, work, c, 60)
    assert g.resolved and sandbox.args == ["tests/test_gold.py"]
    assert (work / "tests" / "test_gold.py").exists()   # the gold test was copied in


def test_grade_unresolved_names_the_failing_tests(gold):
    work, c = gold
    sandbox = FakeOutcomes({"tests.test_gold::test_a": "failed", "tests.test_gold::test_b": "passed"})
    g = grade_tree(sandbox, INFO, work, c, 60)
    assert not g.resolved and g.f2p_failed == ["tests.test_gold::test_a"] and not g.p2p_broken


def test_grade_counts_missing_tests_as_failures(gold):
    work, c = gold
    g = grade_tree(FakeOutcomes({"tests.test_gold::test_a": "passed"}), INFO, work, c, 60)
    assert not g.resolved and g.p2p_broken == ["tests.test_gold::test_b"]


def test_grade_without_a_report_is_unresolved(gold):
    work, c = gold
    g = grade_tree(FakeOutcomes(None), INFO, work, c, 60)
    assert not g.resolved and "no report" in g.note
