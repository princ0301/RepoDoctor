"""Git operations and project detection for a cloned repo."""

import os
import shutil
import stat
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from repodoctor.tools.issue_ref import IssueRef

_EXCLUDES = [
    "__pycache__/", "*.pyc", ".pytest_cache/", ".mypy_cache/",
    ".ruff_cache/", ".venv/", ".tox/", "*.egg-info/", ".coverage",
]
_DEV_NAMES = ("test", "tests", "testing", "dev")
_REQ_CANDIDATES = (
    "requirements-dev.txt", "requirements-test.txt", "requirements-tests.txt",
    "dev-requirements.txt", "test-requirements.txt",
    "requirements/dev.txt", "requirements/test.txt", "requirements/tests.txt",
)

class GitError(RuntimeError):
    """A git command failed."""

def run_git(*args: str, cwd: Path | None = None, timeout: int = 120) -> str:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired as e:
        raise GitError(f"git {args[0]} timed out after {timeout}s") from e
    if p.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout

def _on_rm_error(func, path, _exc) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)

def remove_dir(path: Path) -> None:
    if not path.exists:
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_on_rm_error)
    else:
        shutil.rmtree(path, onerror=_on_rm_error)

def clone_repo(url: str, dest: Path, depth: int = 1) -> Path:
    run_git(
        "clone", "-c", "core.autocrlf=false", "-c", "core.longpaths=true",
        "--depth", str(depth), url, str(dest),
        timeout=600,
    )
    exclude = dest / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(_EXCLUDES) + "\n")
    return dest

def clone_issue_repo(ref: IssueRef, dest: Path) -> Path:
    return clone_repo(f"https://github.com/{ref.full_name}.git", dest)

def get_diff(root: Path) -> str:
    """Unified diff of all changes vs HEAD, including new files."""
    run_git("add", "-A", cwd=root)
    return run_git("diff", "--cached", "--no-color", cwd=root)

def reset_changes(root: Path) -> None:
    """Throw away every change and untracked file (used between retries)."""
    run_git("reset", "--hard", "HEAD", cwd=root)
    run_git("clean", "-fd", cwd=root)

@dataclass
class ProjectInfo:
    supported: bool
    reason: str = ""
    test_command: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)
    dependency_groups: list[str] = field(default_factory=list)
    requirement_files: list[str] = field(default_factory=list)
    requires_python: str | None = None

def detect_project(root: Path) -> ProjectInfo:
    pyproject: dict = {}
    pp = root / "pyproject.toml"
    if pp.is_file():
        try:
            pyproject = tomllib.loads(pp.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            pyproject = {}

    has_build = pp.is_file() or (root / "setup.py").is_file() or (root / "setup.cfg").is_file()
    req_files = [r for r in _REQ_CANDIDATES if (root / r).is_file()]
    if not has_build and (root / "requirements.txt").is_file():
        req_files.insert(0, "requirements.txt")
    if not has_build and not req_files:
        return ProjectInfo(False, "no pyproject.toml, setup.py, setup.cfg or requirements file")

    def has_text(name: str, needle: str) -> bool:
        f = root / name
        return f.is_file() and needle in f.read_text(encoding="utf-8", errors="ignore")

    uses_pytest = (
        "pytest" in pyproject.get("tool", {})
        or (root / "pytest.ini").is_file()
        or has_text("tox.ini", "[pytest]")
        or has_text("setup.cfg", "[tool:pytest]")
        or (root / "conftest.py").is_file()
        or (root / "tests").is_dir()
        or (root / "test").is_dir()
    )
    if not uses_pytest:
        return ProjectInfo(False, "no pytest setup found (other test runners not supported yet)")

    project = pyproject.get("project", {})
    return ProjectInfo(
        supported=True,
        test_command=["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        extras=[n for n in project.get("optional-dependencies", {}) if n in _DEV_NAMES],
        dependency_groups=[n for n in pyproject.get("dependency-groups", {}) if n in _DEV_NAMES],
        requirement_files=req_files,
        requires_python=project.get("requires-python"),
    )