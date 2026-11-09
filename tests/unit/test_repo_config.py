import subprocess
import sys

from repodoctor.tools.repo import ProjectInfo
from repodoctor.tools.repo_config import ignored_paths, with_ignores


def test_ignored_paths_are_per_repo_and_case_insensitive(tmp_path):
    cfg = tmp_path / "o.json"
    cfg.write_text('{"Pallets/Werkzeug": {"ignore": ["tests/test_serving.py"]}}')
    assert ignored_paths("pallets/werkzeug", cfg) == ["tests/test_serving.py"]
    assert ignored_paths("other/repo", cfg) == []
    assert ignored_paths("pallets/werkzeug", tmp_path / "missing.json") == []


def test_with_ignores_appends_flags_without_touching_the_original():
    info = ProjectInfo(True, test_command=["python", "-m", "pytest"])
    out = with_ignores(info, ["a.py", "b/"])
    assert out.test_command == ["python", "-m", "pytest", "--ignore=a.py", "--ignore=b/"]
    assert info.test_command == ["python", "-m", "pytest"]
    assert with_ignores(info, []) is info


def test_pytest_really_skips_an_ignored_file_in_a_full_run(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_a():\n    assert True\n")
    (tmp_path / "test_hang.py").write_text("def test_b():\n    assert False\n")
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--ignore=test_hang.py"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert p.returncode == 0 and "1 passed" in p.stdout