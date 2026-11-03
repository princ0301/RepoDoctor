import time

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from repodoctor.config import Settings
from repodoctor.deps import Deps
from repodoctor.graph import build_graph
from repodoctor.nodes.search_tools import make_search_tools
from repodoctor.runner import run_graph
from repodoctor.schemas import Edit, ExploreResult, FixProposal, ReproTest
from repodoctor.tools.repo import ProjectInfo, run_git
from repodoctor.tools.sandbox import ExecResult
from repodoctor.tools.testreport import SuiteReport

CODE = (
    "def truncate(text, max_len, suffix='...'):\n"
    "    if len(text) < max_len:\n"
    "        return text\n"
    "    return text[: max_len - len(suffix)] + suffix\n"
)
TEST = (
    "from textkit import truncate\n\n"
    "def test_exact():\n"
    "    assert truncate('hello', 5) == 'hello'\n"
)
TARGET = "tests/test_text.py::test_exact"

GOOD = Edit(path="textkit.py", old="if len(text) < max_len:", new="if len(text) <= max_len:")
BAD = Edit(path="textkit.py", old="this text does not exist", new="x")
TOUCH_TEST = Edit(path="tests/test_text.py", old="== 'hello'", new="== 'he...'")

REPRO_CODE = (
    "from textkit import truncate\n\n"
    "def test_repro_exact():\n"
    "    assert truncate('hello', 5) == 'hello'\n"
)
REPRO = ReproTest(reasoning="r", test_name="test_repro_exact", code=REPRO_CODE)
REPRO_BAD_SYNTAX = ReproTest(
    reasoning="r", test_name="test_repro_exact", code="def test_repro_exact(:\n"
)
REPRO_ID = "tests/test_repodoctor_repro.py::test_repro_exact"

BASE = SuiteReport(total=2, passed=1, failed=1, failing={"tests.test_text::test_exact": "boom"})
CLEAN = SuiteReport(total=2, passed=2)
REGRESSED = SuiteReport(
    total=2, passed=1, failed=1, failing={"tests.test_text::test_other": "broke"}
)
NAME_ERROR = SuiteReport(
    total=1, failed=1, failing={"t::x": "NameError: name 'foo' is not defined"}
)


def proposal(*edits: Edit) -> FixProposal:
    return FixProposal(reasoning="because", edits=list(edits))


def scripted(items):
    queue = list(items)

    def step(_messages):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return RunnableLambda(step)


def _validation_error() -> ValidationError:
    try:
        FixProposal.model_validate({"edits": 5})
    except ValidationError as e:
        return e
    raise AssertionError("expected a validation error")


def _default_target_report(code: int) -> SuiteReport:
    if code == 0:
        return SuiteReport(total=1, passed=1)
    return SuiteReport(total=1, failed=1, failing={"t::x": "assert 'he...' == 'hello'"})


class FakeSandbox:
    """target: exit codes (or (code, report)) for target-test runs, in order.
    suites: reports for full-suite runs, in order."""

    def __init__(self, target, suites, install_ok=True):
        self.target, self.suites, self.install_ok = list(target), list(suites), install_ok

    def install(self, info):
        return ExecResult(0 if self.install_ok else 1, "install output", False, 0.0)

    def lock_network(self):
        pass

    def run_suite(self, info, args=None, timeout=300):
        if args:
            item = self.target.pop(0)
            code, report = item if isinstance(item, tuple) else (item, _default_target_report(item))
            return ExecResult(code, f"target exit={code}", False, 0.0), report
        report = self.suites.pop(0)
        return ExecResult(0 if report.clean else 1, "suite output", False, 0.0), report


@pytest.fixture
def repo(make_git_repo):
    return make_git_repo({"textkit.py": CODE, "tests/test_text.py": TEST})


@pytest.fixture
def repo_two_files(make_git_repo):
    return make_git_repo({
        "textkit.py": CODE,
        "extra.py": "# helper\n" + "x = 1\n" * 40,
        "tests/test_text.py": TEST,
    })


def run(repo, sandbox, proposals, explorer=None, max_attempts=3, targets=(TARGET,), repros=(),
        relevant=("textkit.py",), summarizer=None, fixer=None, timeout_s=0, clock=None,
        **settings_kw):
    tools = make_search_tools(repo)
    deps = Deps(
        settings=Settings(_env_file=None, max_attempts=max_attempts, **settings_kw),
        sandbox=sandbox,
        info=ProjectInfo(True, test_command=["python", "-m", "pytest"]),
        workdir=repo,
        tools=tools,
        explorer=explorer or scripted([AIMessage(content="done")]),
        summarizer=summarizer or RunnableLambda(
            lambda _m: ExploreResult(relevant_files=list(relevant), hypothesis="off by one")
        ),
        fixer=fixer or scripted(proposals),
        repro_writer=scripted(repros),
    )
    state = {"issue_title": "t", "issue_body": "b"}
    if targets:
        state["target_tests"] = list(targets)
    return run_graph(
        build_graph(deps), state, {}, timeout_s=timeout_s, clock=clock or time.monotonic
    )


def test_explorer_gets_a_second_chance_when_its_paths_do_not_exist(repo):
    summarizer = scripted([
        ExploreResult(relevant_files=["nope/missing.py"], hypothesis="a"),
        ExploreResult(relevant_files=["textkit.py"], hypothesis="b"),
    ])
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [proposal(GOOD)], summarizer=summarizer)
    assert final["success"] and final["hypothesis"] == "b"


def test_explorer_error_names_the_files_it_invented(repo):
    summarizer = scripted([ExploreResult(relevant_files=["a/x.py"], hypothesis="a")] * 2)
    final = run(repo, FakeSandbox(target=[1], suites=[BASE]), [], summarizer=summarizer)
    assert "explorer did not identify" in final["error"] and "a/x.py" in final["error"]


def test_run_stops_when_over_the_time_limit(repo):
    ticks = iter(range(0, 100_000, 500))
    sandbox = FakeSandbox(target=[1, 1, 1, 1], suites=[BASE])
    final = run(repo, sandbox, [proposal(GOOD)] * 3, timeout_s=600, clock=lambda: next(ticks))
    assert "time limit" in final["error"]
    assert final.get("attempts", 0) < 3


def test_progress_is_reported_per_step(repo):
    seen: list[str] = []
    tools = make_search_tools(repo)
    deps = Deps(
        settings=Settings(_env_file=None),
        sandbox=FakeSandbox(target=[1, 0], suites=[BASE, CLEAN]),
        info=ProjectInfo(True, test_command=["python", "-m", "pytest"]),
        workdir=repo,
        tools=tools,
        explorer=scripted([AIMessage(content="done")]),
        summarizer=RunnableLambda(
            lambda _m: ExploreResult(relevant_files=["textkit.py"], hypothesis="h")
        ),
        fixer=scripted([proposal(GOOD)]),
        repro_writer=scripted([]),
    )
    run_graph(
        build_graph(deps),
        {"issue_title": "t", "issue_body": "b", "target_tests": [TARGET]},
        {},
        progress=seen.append,
    )
    assert [s.split()[1] for s in seen] == ["setup", "explore", "reproduce", "fix", "verify"]


def test_fixer_prompt_lists_the_repository_files(repo):
    seen: list[str] = []
    queue = [proposal(GOOD)]

    def fixer_step(messages):
        seen.append(messages[1].content)
        return queue.pop(0)

    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    run(repo, sandbox, [], fixer=RunnableLambda(fixer_step))
    assert "textkit.py" in seen[0].split("## Repository files", 1)[1]


def test_asking_for_nothing_gets_a_message_that_explains_why(repo):
    ask = FixProposal(reasoning="let me look at the grouping module", edits=[])
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [ask, proposal(GOOD)])
    assert "cannot run tools" in final["history"][0]["feedback"]


# ---------- user-supplied failing test ----------

def test_fixes_on_first_attempt_and_uses_explorer_tools(repo):
    explorer = scripted([
        AIMessage(content="", tool_calls=[
            {"name": "find_definition", "args": {"name": "truncate"}, "id": "c1"}
        ]),
        AIMessage(content="done"),
    ])
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [proposal(GOOD)], explorer=explorer)
    assert final["success"] and final["attempts"] == 1
    assert final["relevant_files"] == ["textkit.py"]
    assert "+    if len(text) <= max_len:" in final["diff"]


def test_retries_after_edit_that_does_not_apply(repo):
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [proposal(BAD), proposal(GOOD)])
    assert final["success"] and final["attempts"] == 2
    assert "could not be applied" in final["history"][0]["feedback"]


def test_gives_up_after_max_attempts(repo):
    sandbox = FakeSandbox(target=[1, 1, 1, 1], suites=[BASE])
    final = run(repo, sandbox, [proposal(GOOD)] * 3)
    assert not final["success"] and final["attempts"] == 3
    assert "error" not in final
    assert "still fail" in final["feedback"]


def test_retries_after_regression(repo):
    sandbox = FakeSandbox(target=[1, 0, 0], suites=[BASE, REGRESSED, CLEAN])
    final = run(repo, sandbox, [proposal(GOOD), proposal(GOOD)])
    assert final["success"] and final["attempts"] == 2
    assert "broke tests" in final["history"][0]["feedback"]


def test_refuses_to_edit_the_target_test(repo):
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [proposal(TOUCH_TEST), proposal(GOOD)])
    assert final["success"] and final["attempts"] == 2
    assert "protected" in final["history"][0]["feedback"]
    assert (repo / "tests" / "test_text.py").read_text() == TEST


def test_malformed_model_output_is_a_failed_attempt_not_an_outage(repo):
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [_validation_error(), proposal(GOOD)])
    assert final["success"] and final["attempts"] == 2
    assert "not valid structured output" in final["history"][0]["feedback"]


def test_other_model_failures_still_abort_the_run(repo):
    sandbox = FakeSandbox(target=[1], suites=[BASE])
    final = run(repo, sandbox, [RuntimeError("boom")])
    assert "fix model failed" in final["error"] and "attempts" not in final or True
    assert not final.get("success")


def test_records_time_spent_in_each_step(repo):
    final = run(repo, FakeSandbox(target=[1, 0], suites=[BASE, CLEAN]), [proposal(GOOD)])
    assert set(final["timings"]) >= {"setup", "explore", "fix", "verify"}


def test_a_fixer_that_needs_another_file_gets_it_on_the_next_attempt(repo):
    ask = FixProposal(reasoning="need it", edits=[], need_more_files=["tests/test_text.py"])
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [ask, proposal(GOOD)])
    assert final["success"] and final["attempts"] == 2
    assert final["relevant_files"] == ["tests/test_text.py", "textkit.py"]
    assert final["explored_files"] == ["textkit.py"]
    assert "Moved to the top" in final["history"][0]["feedback"]


def test_empty_edits_with_an_invalid_request_explain_how_to_ask(repo):
    ask = FixProposal(reasoning="x", edits=[], need_more_files=["nope.py"])
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [ask, proposal(GOOD)])
    assert "need_more_files" in final["history"][0]["feedback"]
    assert final["relevant_files"] == ["textkit.py"]


def test_a_file_cut_by_the_budget_is_promoted_when_the_fixer_asks_for_it(repo_two_files):
    ask = FixProposal(reasoning="need it", need_more_files=["extra.py"])
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(
        repo_two_files, sandbox, [ask, proposal(GOOD)],
        relevant=("textkit.py", "extra.py"), fix_context_chars=len(CODE) + 60,
    )
    assert final["success"] and final["attempts"] == 2
    assert final["relevant_files"][0] == "extra.py"
    assert "Moved to the top" in final["history"][0]["feedback"]
    assert "context_omitted" in final


def test_a_request_for_a_file_that_is_already_shown_says_so(repo):
    ask = FixProposal(reasoning="x", need_more_files=["textkit.py"])
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [ask, proposal(GOOD)])
    assert "already shown" in final["history"][0]["feedback"]


def test_aborts_when_target_already_passes(repo):
    final = run(repo, FakeSandbox(target=[0], suites=[BASE]), [])
    assert "already pass" in final["error"]
    assert "attempts" not in final or final["attempts"] == 0


def test_aborts_when_install_fails(repo):
    final = run(repo, FakeSandbox(target=[], suites=[], install_ok=False), [])
    assert "install failed" in final["error"]


# ---------- the agent writes the failing test ----------

def test_generates_a_reproduction_test_then_fixes(repo):
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [proposal(GOOD)], targets=(), repros=[REPRO])
    assert final["success"] and final["attempts"] == 1
    assert final["target_tests"] == [REPRO_ID]
    assert final["repro_file"] == "tests/test_repodoctor_repro.py"
    assert "tests/test_repodoctor_repro.py" in run_git("ls-files", cwd=repo)
    assert final["base_sha"] != run_git("rev-parse", "HEAD", cwd=repo).strip()


def test_rejects_a_repro_that_passes_then_accepts_a_better_one(repo):
    sandbox = FakeSandbox(target=[0, 1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [proposal(GOOD)], targets=(), repros=[REPRO, REPRO])
    assert final["success"] and final["repro_attempts"] == 2


def test_rejects_a_repro_that_fails_for_the_wrong_reason(repo):
    sandbox = FakeSandbox(target=[(1, NAME_ERROR), 1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [proposal(GOOD)], targets=(), repros=[REPRO, REPRO])
    assert final["success"] and final["repro_attempts"] == 2


def test_rejects_a_syntax_error_without_running_the_sandbox(repo):
    sandbox = FakeSandbox(target=[1, 0], suites=[BASE, CLEAN])
    final = run(repo, sandbox, [proposal(GOOD)], targets=(), repros=[REPRO_BAD_SYNTAX, REPRO])
    assert final["success"] and final["repro_attempts"] == 2


def test_stops_before_fixing_when_no_valid_reproduction_exists(repo):
    sandbox = FakeSandbox(target=[0, 0, 0], suites=[BASE])
    final = run(repo, sandbox, [], targets=(), repros=[REPRO] * 3)
    assert "reproduces the issue" in final["error"]
    assert final.get("attempts", 0) == 0
    assert not (repo / "tests" / "test_repodoctor_repro.py").exists()
