from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from repodoctor.config import Settings
from repodoctor.tools.repo import ProjectInfo
from repodoctor.tools.sandbox import ExecResult
from repodoctor.tools.testreport import SuiteReport


class SandboxLike(Protocol):
    """The slice of Sandbox the nodes use (lets tests substitute a fake)."""

    def install(self, info: ProjectInfo) -> ExecResult: ...

    def lock_network(self) -> None: ...

    def run_suite(
        self, info: ProjectInfo, args: list[str] | None = None, timeout: int = 300
    ) -> tuple[ExecResult, SuiteReport | None]: ...


@dataclass
class Deps:
    """Everything the nodes need from the outside world, injected once."""

    settings: Settings
    sandbox: SandboxLike
    info: ProjectInfo
    workdir: Path
    tools: list[BaseTool]
    explorer: Runnable      # chat model with search tools bound
    summarizer: Runnable    # -> ExploreResult
    fixer: Runnable         # -> FixProposal
    repro_writer: Runnable  # -> ReproTest