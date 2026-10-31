import subprocess
import sys

from repodoctor.demo import DEMO_TEST_ID, make_demo


def test_demo_has_exactly_one_failing_test(tmp_path):
    demo = make_demo(tmp_path / "demo")
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=demo.repo_dir, capture_output=True, text=True,
    )
    assert p.returncode == 1
    assert "1 failed, 4 passed" in p.stdout
    assert DEMO_TEST_ID.split("::")[1] in p.stdout