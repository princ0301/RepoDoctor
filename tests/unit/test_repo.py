from pathlib import Path
import os
import re
import stat
import pytest

from repodoctor.tools.repo import (
    GitError,
    clone_repo,
    detect_project,
    get_diff,
    # reset_changes,
    run_git,
    remove_dir,
    reset_to_clean,
    list_untracked,
    head_sha,
    commit_files,
    get_diff_since,
    checkout_commit,
    ensure_commit,
    fetch_pull_head,
    pretend_version,
    scm_version_env,
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


# def test_reset_restores_clean_state(cloned):
#     write(cloned, "app.py", "x = 99\n")
#     write(cloned, "extra.py", "q = 1\n")
#     get_diff(cloned)  # stages changes, like the real flow
#     reset_changes(cloned)
#     assert (cloned / "app.py").read_text() == "x = 1\n"
#     assert not (cloned / "extra.py").exists()
#     assert get_diff(cloned) == ""

def test_diff_can_be_limited_to_paths(cloned):
    write(cloned, "app.py", "x = 2\n")
    write(cloned, "generated.txt", "junk\n")
    diff = get_diff(cloned, ["app.py"])
    assert "app.py" in diff and "generated.txt" not in diff


def test_reset_to_clean_removes_only_new_files(cloned):
    write(cloned, "build_artifact.txt", "keep me\n")
    keep = list_untracked(cloned)
    write(cloned, "app.py", "x = 99\n")
    write(cloned, "created_by_agent.py", "q = 1\n")
    get_diff(cloned, ["app.py", "created_by_agent.py"])
    reset_to_clean(cloned, keep)
    assert (cloned / "app.py").read_text() == "x = 1\n"
    assert not (cloned / "created_by_agent.py").exists()
    assert (cloned / "build_artifact.txt").exists()

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


@pytest.mark.parametrize("described, expected", [
    ("9.0.1-0-gabc1234", "9.0.1"),
    ("v9.0.1-12-gabc1234", "9.0.2.dev12+gabc1234"),
    ("release-2.0-3-gdeadbee", "2.1.dev3+gdeadbee"),
    ("nodigits-1-gabc1234", None),
    ("not a describe", None),
])
def test_pretend_version(described, expected):
    assert pretend_version(described) == expected


PYPROJECT = '[build-system]\nrequires = ["setuptools_scm"]\n\n[project]\nname = "my-lib"\n'


def test_scm_version_env_uses_the_nearest_tag(make_git_repo):
    root = make_git_repo({"pyproject.toml": PYPROJECT})
    run_git("tag", "v1.2.3", cwd=root)
    (root / "a.txt").write_text("x\n")
    commit_files(root, ["a.txt"], "more")
    ((key, value),) = scm_version_env(root).items()
    assert key == "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MY_LIB"
    assert re.fullmatch(r"1\.2\.4\.dev1\+g[0-9a-f]+", value)


def test_projects_without_tag_based_versions_or_without_tags_get_nothing(make_git_repo):
    assert scm_version_env(make_git_repo({"pyproject.toml": "[project]\nname = 'x'\n"})) == {}
    assert scm_version_env(make_git_repo({"pyproject.toml": PYPROJECT}, name="scm-no-tags")) == {}

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

def test_commit_files_survives_reset_and_leaves_other_files_alone(cloned):
    write(cloned, "tests/test_new.py", "def test_x(): ...\n")
    write(cloned, "other.txt", "x\n")
    before = head_sha(cloned)
    assert commit_files(cloned, ["tests/test_new.py"], "msg") != before
    reset_to_clean(cloned, list_untracked(cloned))
    assert (cloned / "tests" / "test_new.py").exists()
    assert (cloned / "other.txt").exists()


def test_get_diff_since_includes_commits_and_staged_changes(cloned):
    base = head_sha(cloned)
    write(cloned, "tests/test_new.py", "def test_x(): ...\n")
    commit_files(cloned, ["tests/test_new.py"], "msg")
    write(cloned, "app.py", "x = 2\n")
    get_diff(cloned, ["app.py"])  # stages the change
    diff = get_diff_since(cloned, base)
    assert "tests/test_new.py" in diff and "+x = 2" in diff

def test_checkout_commit_detaches_at_an_existing_commit(tmp_path):
    src = make_repo(tmp_path / "src", {"app.py": "v1\n"})
    first = head_sha(src)
    write(src, "app.py", "v2\n")
    run_git("add", "-A", cwd=src)
    run_git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "two", cwd=src)
    clone = clone_repo(src.as_uri(), tmp_path / "clone", depth=10)
    checkout_commit(clone, first)
    assert head_sha(clone) == first
    assert (clone / "app.py").read_text() == "v1\n"


def test_checkout_commit_rejects_anything_but_a_full_sha(cloned):
    with pytest.raises(GitError):
        checkout_commit(cloned, "--upload-pack=evil")

def test_fetch_pull_head_returns_the_pr_commit(tmp_path):
    src = make_repo(tmp_path / "src", {"app.py": "v1\n"})
    write(src, "app.py", "v2\n")
    run_git("add", "-A", cwd=src)
    run_git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "pr", cwd=src)
    pr_sha = head_sha(src)
    run_git("update-ref", "refs/pull/7/head", pr_sha, cwd=src)
    run_git("reset", "--hard", "-q", "HEAD~1", cwd=src)
    clone = clone_repo(src.as_uri(), tmp_path / "clone")
    assert fetch_pull_head(clone, 7) == pr_sha
    ensure_commit(clone, pr_sha)   # present now: must not raise


def test_ensure_commit_rejects_non_sha(cloned):
    with pytest.raises(GitError):
        ensure_commit(cloned, "main")
