"""Per-repository test-run overrides: tests that cannot run in the sandbox."""

import json
from dataclasses import replace
from pathlib import Path

from repodoctor.tools.repo import ProjectInfo

DEFAULT_PATH = Path("evals/repo_overrides.json")


def ignored_paths(repo: str, path: Path = DEFAULT_PATH) -> list[str]:
    """Test paths to skip in full-suite runs of `repo` ('owner/name', case-insensitive)."""
    if not path.exists():
        return []
    data = {k.lower(): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    return list(data.get(repo.lower(), {}).get("ignore", []))


def with_ignores(info: ProjectInfo, paths: list[str]) -> ProjectInfo:
    if not paths:
        return info
    return replace(info, test_command=[*info.test_command, *(f"--ignore={p}" for p in paths)])