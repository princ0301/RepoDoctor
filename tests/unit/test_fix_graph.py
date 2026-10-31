import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from repodoctor.config import Settings
from repodoctor.deps import Deps
from repodoctor.graph import build_graph
from repodoctor.nodes.search_tools import make_search_tools
from repodoctor.schemas import Edit, ExploreResult, FixProposal
from repodoctor.tools.repo import ProjectInfo
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

BASE = SuiteReport(total=2, passed=1, failed=1, failing={"tests.test_text::test_exact": "boom"})
CLEAN = SuiteReport(total=2, passed=2)
REGRESSED = SuiteReport(
    total=2, passed=1, failed=1, failing={"tests.test_text::test_other": "broke"}
)


def proposal(*edits: Edit) -> FixProposal:
    return FixProposal(reasoning="because", edits=list(edits))


def scripted(items):
    queue = list(items)
    return RunnableLambda(lambda _messages: queue.pop(0))


class FakeSandbox:
    """target: exit codes for target-test runs; suites: reports for full runs."""

    def __init__(self, target, suites, install_ok=True):
        self.target, self.suites, self.install_ok = list(target), list(suites), install_ok

    def install(self, info):
        return ExecResult(0 if self.install_ok else 1, "install output", False, 0.0)

    def lock_network(self):
        pass

    def run_suite(self, info, args=None, timeout=300):
        if args:
            code = self.target.pop(0)
            return ExecResult(code, f"target exit={code}", False, 0.0), SuiteReport()
        report = self.suites.pop(0)
        return ExecResult(0 if report.clean else 1, "suite output", False, 0.0), report


@pytest.fixture
def repo(make_git_repo):
    return make_git_repo({"textkit.py": CODE, "tests/test_text.py": TEST})


def run(repo, sandbox, proposals, explorer=None, max_attempts=3):
    tools = make_search_tools(repo)
    deps = Deps(
        settings=Settings(_env_file=None, max_attempts=max_attempts),
        sandbox=sandbox,
        info=ProjectInfo(True, test_command=["python", "-m", "pytest"]),
        workdir=repo,
        tools=tools,
        explorer=explorer or scripted([AIMessage(content="done")]),
        summarizer=RunnableLambda(
            lambda _m: ExploreResult(relevant_files=["textkit.py"], hypothesis="off by one")
        ),
        fixer=scripted(proposals),
    )
    return build_graph(deps).invoke(
        {"issue_title": "t", "issue_body": "b", "target_tests": [TARGET]}
    )


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


def test_aborts_when_target_already_passes(repo):
    final = run(repo, FakeSandbox(target=[0], suites=[BASE]), [])
    assert "already pass" in final["error"]
    assert "attempts" not in final or final["attempts"] == 0


def test_aborts_when_install_fails(repo):
    final = run(repo, FakeSandbox(target=[], suites=[], install_ok=False), [])
    assert "install failed" in final["error"]