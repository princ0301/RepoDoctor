from pathlib import Path
import os
import stat
import pytest

from repodoctor.tools.repo import (
    GitError,
    clone_repo,
    detect_project,
    get_diff,
    reset_changes,
    run_git,
    remove_dir,
)


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def make_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    run_git("init", "-q", "-b", "main", cwd=path)
    for rel, text in files.items():
        write(path, rel, text)
    run_git("add", "-A", cwd=path)
    run_git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=path)
    return path


@pytest.fixture
def cloned(tmp_path) -> Path:
    src = make_repo(tmp_path / "src", {"app.py": "x = 1\n", "pkg/mod.py": "y = 2\n"})
    return clone_repo(src.as_uri(), tmp_path / "clone")


def test_clone_has_files_and_lf_config(cloned):
    assert (cloned / "app.py").read_text() == "x = 1\n"
    assert run_git("config", "core.autocrlf", cwd=cloned).strip() == "false"


def test_clone_failure_raises_git_error(tmp_path):
    with pytest.raises(GitError):
        clone_repo((tmp_path / "does-not-exist").as_uri(), tmp_path / "clone")


def test_diff_is_empty_for_clean_clone(cloned):
    assert get_diff(cloned) == ""


def test_diff_includes_edits_and_new_files(cloned):
    write(cloned, "app.py", "x = 2\n")
    write(cloned, "tests/test_new.py", "def test_x(): ...\n")
    diff = get_diff(cloned)
    assert "-x = 1" in diff and "+x = 2" in diff
    assert "tests/test_new.py" in diff


def test_diff_ignores_cache_files(cloned):
    write(cloned, "pkg/__pycache__/mod.cpython-312.pyc", "junk")
    write(cloned, ".pytest_cache/README.md", "junk")
    write(cloned, "new.py", "z = 3\n")
    diff = get_diff(cloned)
    assert "new.py" in diff
    assert "__pycache__" not in diff and ".pytest_cache" not in diff


def test_reset_restores_clean_state(cloned):
    write(cloned, "app.py", "x = 99\n")
    write(cloned, "extra.py", "q = 1\n")
    get_diff(cloned)  # stages changes, like the real flow
    reset_changes(cloned)
    assert (cloned / "app.py").read_text() == "x = 1\n"
    assert not (cloned / "extra.py").exists()
    assert get_diff(cloned) == ""


def test_detect_pyproject_with_pytest_extras_and_groups(tmp_path):
    write(
        tmp_path,
        "pyproject.toml",
        '[project]\nname = "p"\nrequires-python = ">=3.9"\n'
        '[project.optional-dependencies]\ntest = ["pytest"]\ndocs = ["sphinx"]\n'
        '[dependency-groups]\ndev = ["pytest"]\n'
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
    )
    info = detect_project(tmp_path)
    assert info.supported
    assert info.extras == ["test"]  # 'docs' is ignored
    assert info.dependency_groups == ["dev"]
    assert info.requires_python == ">=3.9"
    assert info.test_command[:3] == ["python", "-m", "pytest"]


def test_detect_requirements_only_project(tmp_path):
    write(tmp_path, "requirements.txt", "requests\n")
    write(tmp_path, "requirements-dev.txt", "pytest\n")
    write(tmp_path, "tests/test_a.py", "def test_a(): ...\n")
    info = detect_project(tmp_path)
    assert info.supported
    assert info.requirement_files == ["requirements.txt", "requirements-dev.txt"]


def test_detect_unsupported_when_nothing_python(tmp_path):
    write(tmp_path, "README.md", "hi")
    info = detect_project(tmp_path)
    assert not info.supported and info.reason


def test_detect_unsupported_without_pytest(tmp_path):
    write(tmp_path, "pyproject.toml", '[project]\nname = "p"\n')
    info = detect_project(tmp_path)
    assert not info.supported and "pytest" in info.reason

def test_remove_dir_missing_path_is_noop(tmp_path):
    remove_dir(tmp_path / "does-not-exist")  # must not raise


def test_remove_dir_deletes_readonly_files(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    f = d / "x.txt"
    f.write_text("x")
    os.chmod(f, stat.S_IREAD)
    remove_dir(d)
    assert not d.exists()