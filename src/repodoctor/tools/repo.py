"""Git operations and project detection for a cloned repo."""

import re
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
_SHA = re.compile(r"[0-9a-f]{40}")

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
    try:
        os.chmod(path, stat.S_IWRITE)  # git marks object files read-only on Windows
        func(path)
    except FileNotFoundError:
        pass  # already gone, nothing to do

def remove_dir(path: Path) -> None:
    if not path.exists():
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

def get_diff(root: Path, paths: list[str] | None = None) -> str:
    """Unified diff vs HEAD of `paths` (default: everything), including new files."""
    run_git("add", "-A", "--", *(paths or ["."]), cwd=root)
    spec = ["--", *paths] if paths else []
    return run_git("diff", "--cached", "--no-color", *spec, cwd=root)

# def reset_changes(root: Path) -> None:
#     """Throw away every change and untracked file (used between retries)."""
#     run_git("reset", "--hard", "HEAD", cwd=root)
#     run_git("clean", "-fd", cwd=root)

def list_untracked(root: Path) -> set[str]:
    out = run_git("ls-files", "--others", "--exclude-standard", "-z", cwd=root)
    return {p for p in out.split("\0") if p}


def reset_to_clean(root: Path, keep_untracked: set[str]) -> None:
    """Undo all edits. Untracked files that existed in `keep_untracked` are left alone,
    so build artifacts from the install step survive between retries."""
    run_git("reset", "--hard", "HEAD", cwd=root)
    for rel in list_untracked(root) - set(keep_untracked):
        (root / rel).unlink(missing_ok=True)

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

def head_sha(root: Path) -> str:
    return run_git("rev-parse", "HEAD", cwd=root).strip()


def commit_files(root: Path, paths: list[str], message: str) -> str:
    """Commit exactly `paths` in the LOCAL work clone (nothing is pushed). Returns new HEAD.

    The generated reproduction test is committed so `git reset --hard` between fix
    attempts keeps it, while the agent's edits are still thrown away.
    """
    run_git("add", "-A", "--", *paths, cwd=root)
    run_git(
        "-c", "user.name=RepoDoctor",
        "-c", "user.email=repodoctor@users.noreply.github.com",
        "commit", "-q", "--no-verify", "--no-gpg-sign", "-m", message,
        cwd=root,
    )
    return head_sha(root)


def get_diff_since(root: Path, base: str) -> str:
    """Everything that differs from `base`: committed test + staged fix."""
    return run_git("diff", "--no-color", base, cwd=root)

def checkout_commit(root: Path, sha: str) -> None:
    """Detach HEAD at `sha`, fetching it (shallow) if the clone does not have it."""
    if not _SHA.fullmatch(sha):
        raise GitError(f"not a full commit sha: {sha!r}")
    try:
        run_git("cat-file", "-e", f"{sha}^{{commit}}", cwd=root)
    except GitError:
        run_git("fetch", "--depth", "1", "origin", sha, cwd=root, timeout=300)
    run_git("checkout", "--detach", "-q", sha, cwd=root)