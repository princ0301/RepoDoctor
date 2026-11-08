import json
from dataclasses import asdict

import httpx
import pytest

from repodoctor.config import Settings
from repodoctor.evals.build import apply_pr_changes, feature_signals, pair_problem
from repodoctor.evals.cases import (
    CaseSpec,
    EvalCase,
    case_id,
    changed_files,
    is_test_file,
    is_test_path,
    load_cases,
    load_specs,
    parse_case_flags,
    removed_line_count,
    save_cases,
    save_specs,
)
from repodoctor.evals.discover import discover
from repodoctor.evals.gold import gold_test_source
from repodoctor.evals.grade import grade_tree
from repodoctor.evals.run import (
    CaseResult,
    attempt_kind,
    classify,
    compare,
    is_infrastructure_failure,
    load_results,
    run_eval,
    summarize,
)
from repodoctor.tools.github import PRFile, fetch_file_text, github_client
from repodoctor.tools.issue_ref import IssueRef
from repodoctor.tools.repo import ProjectInfo, clone_repo, fetch_pull_head, head_sha, run_git
from repodoctor.tools.sandbox import ExecResult


def case(**over) -> EvalCase:
    base = dict(
        id="o__r-5", repo="o/r", issue=5, fix_pr=7, title="t", base_sha="a" * 40,
        head_sha="b" * 40, test_files=["tests/test_x.py"], run_targets=["tests/test_x.py"],
        src_files=["src/x.py"], fail_to_pass=["tests.test_x::test_a"], pass_to_pass=[],
    )
    return EvalCase(**{**base, **over})


def test_feature_signals():
    assert feature_signals(["bug"], "Fix crash on empty input") == []
    assert feature_signals(["Type: Enhancement"], "Fix x") == ["label 'Type: Enhancement'"]
    assert "PR title" in feature_signals([], "Add option to truncate")[0]


def test_parse_case_flags():
    text = "# header [spec]\na__b-1  [spec]  # why\nc__d-2 [scope]\ne__f-3  # untagged\n\n"
    assert parse_case_flags(text) == {"a__b-1": "spec", "c__d-2": "scope", "e__f-3": "spec"}


def test_removed_line_count():
    patch = "@@ -1,3 +1,3 @@\n a\n-b\n+c\n-\n-  \n--- weird\n\\ No newline at end of file"
    assert removed_line_count(patch) == 2


def test_summarize_splits_scope_cases():
    out = summarize([result("a"), result("b", outcome="wrong_fix")], scope_ids={"b"})
    assert "without scope cases: 1/1" in out and "scope cases only: 0/1" in out


GOLD_SRC = '''
import pytest


def test_top():
    assert 1


class TestA:
    @pytest.mark.parametrize("x", [1, 2])
    def test_method(self, x):
        assert x
'''


def test_gold_test_source_finds_functions_and_methods():
    texts = {"tests/test_a.py": GOLD_SRC}
    path, src = gold_test_source("tests.test_a::test_top", texts)
    assert path == "tests/test_a.py" and "def test_top" in src
    _, src = gold_test_source("tests.test_a.TestA::test_method[1]", texts)
    assert "@pytest.mark.parametrize" in src and "def test_method" in src
    assert gold_test_source("tests.test_a::missing", texts) is None
    assert gold_test_source("tests.other::test_top", texts) is None


def test_fetch_file_text_returns_raw_text():
    def handler(request):
        assert request.url.params["ref"] == "abc"
        return httpx.Response(200, text="print('hi')\n")

    c = github_client("", transport=httpx.MockTransport(handler))
    assert fetch_file_text(c, IssueRef("o", "r", 1), "tests/t.py", "abc") == "print('hi')\n"


@pytest.mark.parametrize(
    "feedback, kind",
    [
        ("The target test(s) pass and no tests regressed.", "passed"),
        (
            "Your edits could not be applied:\nNo edits were proposed. You must return",
            "no_edits",
        ),
        (
            "Your edits could not be applied:\nNo edits were proposed. "
            "Moved to the top of your context: a.py.",
            "asked_for_files",
        ),
        (
            "Your edits could not be applied:\nNo edits were proposed. "
            "These files are already shown to you in full, use them: a.py.",
            "asked_for_shown_file",
        ),
        (
            "Your edits could not be applied:\nNo edits were proposed. "
            "Not shown because of the context budget: b.py.",
            "budget_cut",
        ),
        (
            "Your edits could not be applied:\nNo edits were proposed. "
            "These paths in need_more_files do not exist: x.py.",
            "asked_for_unknown_file",
        ),
        (
            "Your edits could not be applied:\nedit 1 (a.py): 'old' text not found. It",
            "old_not_found",
        ),
        (
            "Your edits could not be applied:\nedit 1 (a.py): 'old' text appears 3 times.",
            "old_ambiguous",
        ),
        ("Your edits could not be applied:\nedit 1 (x.py): file does not exist", "bad_path"),
        (
            "Your edits could not be applied:\nt.py is a protected test file and must",
            "edits_test_file",
        ),
        ("Your edits could not be applied:\nThe edits did not change any code.", "no_change"),
        (
            "Your edits could not be applied:\nYour last reply was not valid structured output",
            "malformed_output",
        ),
        (
            "The target test(s) still fail after your edits. Output:\nE timed out waiting",
            "test_still_fails",
        ),
        ("The target test(s) timed out after your edits. Output:", "test_timeout"),
        ("Your edits broke tests that passed before:\n- a::b: boom", "regression"),
        ("The full test run produced no report:\nlog", "no_report"),
        ("something unexpected", "other"),
    ],
)
def test_attempt_kind(feedback, kind):
    assert attempt_kind(feedback) == kind


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


def test_apply_pr_changes_touches_only_the_files_the_pr_changed(make_git_repo, tmp_path):
    src = make_git_repo({"a.py": "v1\n", "c.py": "c\n", "other.py": "base\n"}, name="src")
    run_git("checkout", "-q", "-b", "pr", cwd=src)
    (src / "a.py").write_bytes(b"v2\n")
    (src / "c.py").unlink()
    (src / "t.py").write_bytes(b"t\n")
    (src / "other.py").write_bytes(b"unrelated change merged from main\n")
    run_git("add", "-A", cwd=src)
    run_git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "pr", cwd=src)
    run_git("update-ref", "refs/pull/7/head", head_sha(src), cwd=src)
    run_git("checkout", "-q", "main", cwd=src)
    work = clone_repo(src.as_uri(), tmp_path / "work")
    head = fetch_pull_head(work, 7)

    files = [PRFile("a.py", "modified"), PRFile("c.py", "removed"), PRFile("t.py", "added")]
    apply_pr_changes(work, head, files)

    assert (work / "a.py").read_text() == "v2\n"
    assert not (work / "c.py").exists()
    assert (work / "t.py").exists()
    assert (work / "other.py").read_text() == "base\n"   # not dragged in


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
        (
            {
                "error": "The run exceeded the time limit of 900s "
                "and was stopped after the fix step."
            },
            None,
            "run_timeout",
        ),
        (
            {"error": "The explorer did not identify any existing source file (it named: a.py)."},
            None,
            "explore_failed",
        ),
        ({"error": "Dependency install failed (exit 1)"}, None, "setup_error"),
        ({"success": False}, None, "fix_failed"),
        ({"success": True}, None, "grading_error"),
        ({"success": True}, {"resolved": False}, "wrong_fix"),
        (
            {"success": True},
            {
                "resolved": False,
                "f2p_failed": [],
                "p2p_broken": ["test_a"],
                "p2p_unseen": ["test_a"],
            },
            "unseen_test",
        ),
        (
            {"success": True},
            {
                "resolved": False,
                "f2p_failed": [],
                "p2p_broken": ["t::b"],
                "p2p_unseen": ["t::b"],
            },
            "unseen_test",
        ),
        (
            {"success": True},
            {
                "resolved": False,
                "f2p_failed": [],
                "p2p_broken": ["t::b"],
                "p2p_unseen": [],
            },
            "wrong_fix",
        ),
        (
            {"success": True},
            {
                "resolved": False,
                "f2p_failed": ["t::a"],
                "p2p_broken": [],
                "p2p_unseen": [],
            },
            "wrong_fix",
        ),
        ({"success": True}, {"resolved": True}, "resolved"),
    ],
)
def test_classify(final, grade, expected):
    assert classify(final, grade) == expected


def result(case_id_="o__r-5", trial=1, outcome="resolved", seconds=60.0, success=True):
    return CaseResult(case_id_, trial, outcome, outcome == "resolved", success, 1, 1, seconds)


def test_setup_errors_without_model_calls_are_infrastructure_failures():
    r = result(outcome="setup_error")
    assert is_infrastructure_failure(r)          # llm_calls defaults to 0
    r.llm_calls = 5
    assert not is_infrastructure_failure(r)
    assert not is_infrastructure_failure(result(outcome="fix_failed"))


def test_summarize_reports_rates_and_precision():
    out = summarize([result(), result("o__r-6", outcome="wrong_fix")])
    assert "1/2 (50%)" in out
    assert "precision 50%" in out
    assert "resolved=1" in out and "wrong_fix=1" in out


def test_compare_reports_per_case_and_totals():
    a = [result("x", 1, "wrong_fix"), result("y", 1)]
    b = [result("x", 1), result("y", 1)]
    a[0].history = [{"edits": []}, {"edits": [{"path": "p"}]}]
    out = compare(a, b)
    assert "A 1/2   B 2/2" in out
    assert "B better in 1, worse in 0, same in 1" in out
    assert "A 1/2   B 0/0" in out


def test_summarize_reports_tests_the_agent_could_not_see():
    out = summarize([result(), result("o__r-6", outcome="unseen_test")])
    assert "Fixed the issue and every test the agent could see: 2/2" in out


def test_summarize_counts_unseen_test_failures_separately():
    out = summarize([result("a"), result("b", outcome="unseen_test")])
    assert "1/2 (50%)" in out
    assert "2/2" in out and "absent from the base repository" in out


def test_summarize_reports_time_by_step():
    r = result()
    r.timings = {"setup": 40.0, "fix": 20.0}
    assert "setup=40s" in summarize([r])


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

    def fake(c, trial, settings, *, label, progress=None):
        ran.append(c.id)
        return result(c.id, trial, "model_error", success=False)

    monkeypatch.setattr("repodoctor.evals.run.run_case", fake)
    cases = [case(id=f"c{i}") for i in range(5)]
    run_eval(cases, Settings(_env_file=None), trials=1, label="t",
             results_file=tmp_path / "r.jsonl")
    assert ran == ["c0", "c1"]


def test_run_eval_skips_finished_runs(tmp_path, monkeypatch):
    ran = []

    def fake(c, trial, settings, *, label, progress=None):
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


def test_summarize_reports_calls_by_model():
    r = result()
    r.calls_by_model = {"big": 9, "free": 1}
    assert "big=9 (90%)" in summarize([r])


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
    sandbox = FakeOutcomes(
        {"tests.test_gold::test_a": "passed", "tests.test_gold::test_b": "passed"}
    )
    g = grade_tree(sandbox, INFO, work, c, 60)
    assert g.resolved and sandbox.args == ["tests/test_gold.py"]
    assert (work / "tests" / "test_gold.py").exists()   # the gold test was copied in


def test_grade_unresolved_names_the_failing_tests(gold):
    work, c = gold
    sandbox = FakeOutcomes(
        {"tests.test_gold::test_a": "failed", "tests.test_gold::test_b": "passed"}
    )
    g = grade_tree(sandbox, INFO, work, c, 60)
    assert not g.resolved and g.f2p_failed == ["tests.test_gold::test_a"] and not g.p2p_broken


def test_grade_counts_missing_tests_as_failures(gold):
    work, c = gold
    g = grade_tree(FakeOutcomes({"tests.test_gold::test_a": "passed"}), INFO, work, c, 60)
    assert not g.resolved and g.p2p_broken == ["tests.test_gold::test_b"]


def test_grade_separates_broken_tests_the_agent_could_not_see(gold):
    work, c = gold
    c.new_pass_to_pass = ["tests.test_gold::test_b"]
    outcomes = {"tests.test_gold::test_a": "passed", "tests.test_gold::test_b": "failed"}
    g = grade_tree(FakeOutcomes(outcomes), INFO, work, c, 60)
    assert not g.resolved
    assert g.p2p_broken == g.p2p_unseen == ["tests.test_gold::test_b"]


def test_grade_without_a_report_is_unresolved(gold):
    work, c = gold
    g = grade_tree(FakeOutcomes(None), INFO, work, c, 60)
    assert not g.resolved and "no report" in g.note

def test_load_results_ignores_unknown_fields(tmp_path):
    path = tmp_path / "r.jsonl"
    record = asdict(result("a", 1))
    record["field_from_another_version"] = ["x"]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    (only,) = load_results(path)
    assert only.case_id == "a"

def test_discover_skips_candidates_already_known():
    routes = {
        "/repos/o/r/pulls": [pr(10, "Fixes #1")],
        "/repos/o/r/pulls/10/files": files("src/a.py", "tests/test_a.py"),
    }
    assert discover(mock_client(routes), "o/r", exclude={("o/r", 1)}) == []
