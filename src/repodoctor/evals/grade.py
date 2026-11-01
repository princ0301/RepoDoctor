"""Grade an agent's final tree against the maintainers' tests."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from repodoctor.evals.cases import EvalCase
from repodoctor.tools.repo import ProjectInfo, fetch_pull_head, run_git
from repodoctor.tools.sandbox import ExecResult


class OutcomeRunner(Protocol):
    def run_outcomes(
        self, info: ProjectInfo, args: list[str] | None = None, timeout: int = 300
    ) -> tuple[ExecResult, dict[str, str] | None]: ...


@dataclass
class Grade:
    resolved: bool
    f2p_failed: list[str] = field(default_factory=list)
    p2p_broken: list[str] = field(default_factory=list)
    note: str = ""


def grade_tree(
    sandbox: OutcomeRunner, info: ProjectInfo, workdir: Path, case: EvalCase, timeout: int
) -> Grade:
    """Copy the maintainers' test files over the agent's tree, run them, check the sets."""
    fetch_pull_head(workdir, case.fix_pr)   # makes the gold commit available locally
    run_git("checkout", case.head_sha, "--", *case.test_files, cwd=workdir)
    result, outcomes = sandbox.run_outcomes(info, case.run_targets, timeout)
    if outcomes is None:
        return Grade(False, note=f"the gold tests produced no report (exit {result.exit_code})")
    f2p = [t for t in case.fail_to_pass if outcomes.get(t) != "passed"]
    p2p = [t for t in case.pass_to_pass if outcomes.get(t) != "passed"]
    return Grade(not f2p and not p2p, f2p, p2p)