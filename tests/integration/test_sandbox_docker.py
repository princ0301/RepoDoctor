from pathlib import Path

import pytest

from repodoctor.tools.repo import ProjectInfo
from repodoctor.tools.sandbox import Sandbox, docker_available, ensure_image

pytestmark = pytest.mark.docker

TEST_CMD = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]


@pytest.fixture(scope="session", autouse=True)
def _docker_ready():
    if not docker_available():
        pytest.skip("Docker is not running")
    ensure_image()


def make_project(root: Path, failing: bool) -> ProjectInfo:
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "requirements.txt").write_text("")
    (root / "tests").mkdir()
    expected = 4 if failing else 3
    (root / "tests" / "test_calc.py").write_text(
        f"from calc import add\n\ndef test_add():\n    assert add(1, 2) == {expected}\n"
    )
    return ProjectInfo(True, test_command=TEST_CMD, requirement_files=["requirements.txt"])


def test_run_returns_output_and_exit_code(tmp_path):
    with Sandbox(tmp_path) as sb:
        r = sb.run("echo hi", 20)
        assert r.ok and "hi" in r.output
        assert sb.run("exit 3", 20).exit_code == 3


def test_files_are_shared_with_host(tmp_path):
    (tmp_path / "from_host.txt").write_text("hello")
    with Sandbox(tmp_path) as sb:
        assert "hello" in sb.run("cat /work/from_host.txt", 20).output
        sb.run("echo data > /work/from_container.txt", 20)
    assert (tmp_path / "from_container.txt").read_text().strip() == "data"


def test_timeout_kills_the_command(tmp_path):
    with Sandbox(tmp_path) as sb:
        r = sb.run("sleep 60", timeout=2)
    assert r.timed_out and r.duration_s < 20


def test_network_is_blocked_after_lock(tmp_path):
    probe = "python -c \"import urllib.request as u; u.urlopen('https://pypi.org', timeout=5)\""
    with Sandbox(tmp_path) as sb:
        sb.lock_network()
        assert sb.run(probe, 30).exit_code != 0


def test_container_is_removed_on_exit(tmp_path):
    from repodoctor.tools.sandbox import _docker

    with Sandbox(tmp_path) as sb:
        name = sb.name
    assert name not in _docker("ps", "-a", "--format", "{{.Names}}").stdout


@pytest.mark.slow
def test_install_and_run_passing_tests(tmp_path):
    info = make_project(tmp_path, failing=False)
    with Sandbox(tmp_path) as sb:
        assert sb.install(info).ok          # needs internet (pip install pytest)
        sb.lock_network()
        r = sb.run_tests(info)
    assert r.ok and "1 passed" in r.output


@pytest.mark.slow
def test_failing_tests_are_reported(tmp_path):
    info = make_project(tmp_path, failing=True)
    with Sandbox(tmp_path) as sb:
        assert sb.install(info).ok
        sb.lock_network()
        r = sb.run_tests(info)
    assert r.exit_code == 1 and "1 failed" in r.output

def test_dead_container_raises_clear_error(tmp_path):
    from repodoctor.tools.sandbox import SandboxError, _docker

    with Sandbox(tmp_path) as sb:
        _docker("kill", sb.name)
        with pytest.raises(SandboxError, match="not running"):
            sb.run("echo hi", 10)


@pytest.mark.slow
def test_run_suite_reports_failing_test_ids(tmp_path):
    info = make_project(tmp_path, failing=True)
    with Sandbox(tmp_path) as sb:
        assert sb.install(info).ok
        sb.lock_network()
        result, report = sb.run_suite(info)
    assert report is not None
    assert report.failed == 1
    assert any(k.endswith("::test_add") for k in report.failing)