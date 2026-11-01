import pytest

from repodoctor.nodes.reproduce import check_repro, static_problem
from repodoctor.schemas import ReproTest
from repodoctor.tools.sandbox import ExecResult
from repodoctor.tools.testreport import SuiteReport


def res(code: int, timed_out: bool = False) -> ExecResult:
    return ExecResult(code, "output", timed_out, 0.1)


def failing(msg: str) -> SuiteReport:
    return SuiteReport(total=1, failed=1, failing={"t::x": msg})


def test_static_accepts_a_valid_test():
    t = ReproTest(reasoning="r", test_name="test_a", code="def test_a():\n    assert 1\n")
    assert static_problem(t) is None


@pytest.mark.parametrize(
    "name, code",
    [
        ("test_a", "def test_a(:\n"),                  # syntax error
        ("test_a", "def test_other():\n    pass\n"),   # named function missing
        ("check_a", "def check_a():\n    pass\n"),     # name must start with 'test'
    ],
)
def test_static_rejects(name, code):
    assert static_problem(ReproTest(reasoning="r", test_name=name, code=code))


def test_accepts_an_assertion_failure():
    assert check_repro(res(1), failing("assert 'a' == 'b'")) is None


def test_accepts_a_bug_triggered_exception():
    assert check_repro(res(1), failing("KeyError: 'x'")) is None


def test_rejects_a_passing_test():
    assert "PASSES" in check_repro(res(0), SuiteReport(total=1, passed=1))


def test_rejects_a_timeout():
    assert "timed out" in check_repro(res(-1, timed_out=True), None)


def test_rejects_when_pytest_cannot_run_it():
    assert "could not run" in check_repro(res(4), None)


def test_rejects_collection_or_setup_errors():
    report = SuiteReport(total=1, errors=1, failing={"t::x": "failed on setup with boom"})
    assert "errored" in check_repro(res(1), report)


@pytest.mark.parametrize(
    "msg",
    [
        "ImportError: cannot import name 'x'",
        "ModuleNotFoundError: No module named 'x'",
        "NameError: name 'y' is not defined",
        "SyntaxError: invalid syntax",
    ],
)
def test_rejects_wrong_reason_failures(msg):
    assert "wrong reason" in check_repro(res(1), failing(msg))