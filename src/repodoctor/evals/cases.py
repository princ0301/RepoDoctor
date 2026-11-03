"""Eval data model: specs (what to build), cases (built, with ground truth), file I/O."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from repodoctor.tools.issue_ref import IssueRef


@dataclass
class CaseSpec:
    repo: str        # "owner/name"
    issue: int
    fix_pr: int      # the merged PR that fixed it


@dataclass
class EvalCase:
    id: str
    repo: str
    issue: int
    fix_pr: int
    title: str
    base_sha: str
    head_sha: str                  # tip of the PR branch (the "gold" tree)
    test_files: list[str]          # every test-side file the PR changed (copied in to grade)
    run_targets: list[str]         # the test modules among them (what pytest runs)
    src_files: list[str]           # files the maintainers changed (for analysis only)
    fail_to_pass: list[str]        # fail before the fix, pass after it
    pass_to_pass: list[str]        # pass both before and after

    @property
    def ref(self) -> IssueRef:
        owner, _, name = self.repo.partition("/")
        return IssueRef(owner, name, self.issue)


def case_id(repo: str, issue: int) -> str:
    return f"{repo.replace('/', '__')}-{issue}"


def is_test_path(path: str) -> bool:
    """A file on the test side of a PR (test modules, files under tests/, conftest)."""
    parts = path.split("/")
    name = parts[-1]
    return (
        "tests" in parts[:-1]
        or "test" in parts[:-1]
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def is_test_file(path: str) -> bool:
    """A test module pytest would collect on its own."""
    name = path.rsplit("/", 1)[-1]
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


_DIFF_HEADER = re.compile(r"^diff --git a/.+? b/(.+)$", re.M)


def changed_files(diff: str) -> list[str]:
    seen: list[str] = []
    for path in _DIFF_HEADER.findall(diff):
        if path not in seen:
            seen.append(path)
    return seen


def removed_line_count(patch: str) -> int:
    """Lines a diff hunk removes (blank ones ignored): existing code that was changed or deleted."""
    return sum(1 for ln in patch.splitlines() if ln.startswith("-") and ln[1:].strip())


def save_cases(path: Path, cases: list[EvalCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in cases], indent=2) + "\n", encoding="utf-8")


def load_cases(path: Path) -> list[EvalCase]:
    return [EvalCase(**d) for d in json.loads(path.read_text(encoding="utf-8"))]


def save_specs(path: Path, specs: list[CaseSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(s) for s in specs], indent=2) + "\n", encoding="utf-8")


def load_specs(path: Path) -> list[CaseSpec]:
    return [CaseSpec(**d) for d in json.loads(path.read_text(encoding="utf-8"))]

def parse_case_flags(text: str) -> dict[str, str]:
    """Lines like '<case id>  [spec]  # reason' -> {id: 'spec'}. Untagged means spec."""
    flags: dict[str, str] = {}
    for line in text.splitlines():
        body = line.split("#", 1)[0].split()
        if body:
            flags[body[0]] = body[1].strip("[]").lower() if len(body) > 1 else "spec"
    return flags
