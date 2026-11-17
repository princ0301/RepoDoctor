"""Per-repository test-run overrides: tests that cannot run in the sandbox, flags that distort a
project's own tests, extra arguments such as xdist workers."""

import json
from dataclasses import replace
from pathlib import Path

from repodoctor.tools.repo import ProjectInfo

DEFAULT_PATH = Path("evals/repo_overrides.json")


def _entry(repo: str, path: Path) -> dict:
    if not path.exists():
        return {}
    data = {k.lower(): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    return data.get(repo.lower(), {})


def ignored_paths(repo: str, path: Path = DEFAULT_PATH) -> list[str]:
    """Test paths to skip in full-suite runs of `repo` ('owner/name', case-insensitive)."""
    return list(_entry(repo, path).get("ignore", []))


def with_ignores(info: ProjectInfo, paths: list[str]) -> ProjectInfo:
    if not paths:
        return info
    return replace(info, test_command=[*info.test_command, *(f"--ignore={p}" for p in paths)])


def _drop(cmd: list[str], drop: list[str]) -> list[str]:
    """Remove every occurrence of the consecutive arguments `drop` (for example -p no:x)."""
    if not drop:
        return list(cmd)
    n, out, i = len(drop), [], 0
    while i < len(cmd):
        if cmd[i:i + n] == drop:
            i += n
        else:
            out.append(cmd[i])
            i += 1
    return out


def apply_overrides(info: ProjectInfo, repo: str, path: Path = DEFAULT_PATH) -> ProjectInfo:
    """The project's test command with this repository's overrides applied."""
    if not info.supported:
        return info
    entry = _entry(repo, path)
    cmd = _drop(list(info.test_command), [str(a) for a in entry.get("drop_args", [])])
    cmd += [f"--ignore={p}" for p in entry.get("ignore", [])]
    cmd += [str(a) for a in entry.get("extra_args", [])]
    return replace(info, test_command=cmd)