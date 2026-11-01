import subprocess
import sys

from repodoctor.demo import DEMO_TEST_ID, make_demo


def _pytest(cwd):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=cwd, capture_output=True, text=True,
    )


def test_demo_repo_has_exactly_one_failing_test(tmp_path):
    p = _pytest(make_demo(tmp_path / "demo").repo_dir)
    assert p.returncode == 1
    assert "1 failed, 4 passed" in p.stdout
    assert DEMO_TEST_ID.split("::")[1] in p.stdout


def test_blind_demo_repo_passes_despite_the_bug(tmp_path):
    p = _pytest(make_demo(tmp_path / "demo").blind_repo_dir)
    assert p.returncode == 0 and "4 passed" in p.stdout


def test_make_demo_can_run_twice(tmp_path):
    make_demo(tmp_path / "demo")
    make_demo(tmp_path / "demo")