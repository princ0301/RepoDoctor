"""Wires everything together: workspace -> sandbox -> models -> graph -> final state."""

from dataclasses import dataclass, field
from pathlib import Path

from repodoctor.config import Settings
from repodoctor.deps import Deps
from repodoctor.graph import build_graph
from repodoctor.models import build_chain
from repodoctor.nodes.search_tools import make_search_tools
from repodoctor.schemas import ExploreResult, FixProposal, ReproTest
from repodoctor.tools.repo import clone_repo, detect_project, get_diff_since, remove_dir
from repodoctor.tools.sandbox import Sandbox, ensure_image
from repodoctor.tracing import run_config


class RunnerError(RuntimeError):
    """The request cannot be run (not a git repo, unsupported project, ...)."""


@dataclass
class FixRequest:
    repo_dir: Path             # local git repo containing the bug (committed)
    issue_title: str
    issue_body: str
    # pytest node ids that must pass after the fix; empty = the agent writes the test
    target_tests: list[str] = field(default_factory=list)


def parse_issue_file(text: str) -> tuple[str, str]:
    """First line is the title (leading '#' allowed); the rest is the body."""
    lines = text.strip().splitlines()
    if not lines:
        raise ValueError("the issue file is empty")
    return lines[0].lstrip("#").strip(), "\n".join(lines[1:]).strip()


def prepare_workdir(src: Path, settings: Settings) -> Path:
    """Copy the repo into the work dir. The agent never touches your original."""
    src = src.resolve()
    if not (src / ".git").exists():
        raise RunnerError(f"{src} is not a git repository (commit your changes first)")
    dest = settings.work_dir.resolve() / f"local__{src.name}"
    remove_dir(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    clone_repo(src.as_uri(), dest)
    return dest


def run_fix(req: FixRequest, settings: Settings) -> dict:
    workdir = prepare_workdir(req.repo_dir, settings)
    info = detect_project(workdir)
    if not info.supported:
        raise RunnerError(f"unsupported project: {info.reason}")

    # Build models first: config and credential problems should fail before Docker starts.
    tools = make_search_tools(workdir)
    explorer = build_chain("explore", settings, tools=tools)
    summarizer = build_chain("explore", settings, schema=ExploreResult)
    fixer = build_chain("fix", settings, schema=FixProposal)
    repro_writer = build_chain("fix", settings, schema=ReproTest)

    ensure_image()
    with Sandbox(workdir, memory=settings.sandbox_memory, cpus=settings.sandbox_cpus) as sb:
        deps = Deps(
            settings=settings, sandbox=sb, info=info, workdir=workdir, tools=tools,
            explorer=explorer, summarizer=summarizer, fixer=fixer, repro_writer=repro_writer,
        )
        config = run_config(
            f"fix:{req.repo_dir.name}",
            tags=["fix-loop", "own-test" if req.target_tests else "generated-test"],
            metadata={
                "repo": req.repo_dir.name,
                "llm_explore": settings.llm_explore,
                "llm_fix": settings.llm_fix,
                "max_attempts": settings.max_attempts,
            },
        )
        initial: dict = {"issue_title": req.issue_title, "issue_body": req.issue_body}
        if req.target_tests:
            initial["target_tests"] = list(req.target_tests)

        final = build_graph(deps).invoke(initial, config=config)
        if final.get("success") and final.get("base_sha"):
            final["full_diff"] = get_diff_since(workdir, final["base_sha"])
        return final