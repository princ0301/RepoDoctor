"""Wires everything together: workspace -> sandbox -> models -> graph -> final state."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from repodoctor.config import Settings
from repodoctor.deps import Deps
from repodoctor.graph import build_graph
from repodoctor.models import build_chain
from repodoctor.nodes.search_tools import make_search_tools
from repodoctor.schemas import ExploreResult, FixProposal, ReproTest
from repodoctor.tools.github import (
    fetch_issue,
    fetch_pr,
    find_fixing_pr,
    format_issue_body,
    github_client,
)
from repodoctor.tools.issue_ref import IssueRef
from repodoctor.tools.policy import policy_mentions
from repodoctor.tools.repo import (
    checkout_commit,
    clone_issue_repo,
    clone_repo,
    detect_project,
    get_diff_since,
    ProjectInfo,
    remove_dir,
)
from repodoctor.tools.sandbox import Sandbox, ensure_image
from repodoctor.tracing import CallCounter, run_config


PostHook = Callable[[Sandbox, ProjectInfo, Path, dict], dict | None]


class RunnerError(RuntimeError):
    """The request cannot be run (not a git repo, unsupported project, ...)."""


@dataclass
class FixRequest:
    repo_dir: Path             # local git repo containing the bug (committed)
    issue_title: str
    issue_body: str
    # pytest node ids that must pass after the fix; empty = the agent writes the test
    target_tests: list[str] = field(default_factory=list)


@dataclass
class IntakeInfo:
    issue_state: str
    fixing_pr_url: str | None  # set in --replay mode
    policy_hits: list[str]


def parse_issue_file(text: str) -> tuple[str, str]:
    """First line is the title (leading '#' allowed); the rest is the body."""
    lines = text.strip().splitlines()
    if not lines:
        raise ValueError("the issue file is empty")
    return lines[0].lstrip("#").strip(), "\n".join(lines[1:]).strip()


def prepare_workdir(src: Path, settings: Settings) -> Path:
    """Copy a local repo into the work dir. The agent never touches your original."""
    src = src.resolve()
    if not (src / ".git").exists():
        raise RunnerError(f"{src} is not a git repository (commit your changes first)")
    dest = settings.work_dir.resolve() / f"local__{src.name}"
    remove_dir(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    clone_repo(src.as_uri(), dest)
    return dest


def execute(
    workdir: Path,
    title: str,
    body: str,
    targets: list[str],
    settings: Settings,
    *,
    run_name: str,
    tags: list[str],
    metadata: dict,
    post: PostHook | None = None,
) -> dict:
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
            run_name,
            tags=["fix-loop", "own-test" if targets else "generated-test", *tags],
            metadata={
                "llm_explore": settings.llm_explore,
                "llm_fix": settings.llm_fix,
                "max_attempts": settings.max_attempts,
                **metadata,
            },
        )
        counter = CallCounter()
        config["callbacks"] = [counter]
        initial: dict = {"issue_title": title, "issue_body": body}
        if targets:
            initial["target_tests"] = list(targets)

        final = build_graph(deps).invoke(initial, config=config)
        final["llm_calls"] = counter.calls
        if final.get("success") and final.get("base_sha"):
            final["full_diff"] = get_diff_since(workdir, final["base_sha"])
        if post is not None:  # runs while the sandbox is still up (used by the evals)
            final["post"] = post(sb, info, workdir, final)
        return final


def run_fix(req: FixRequest, settings: Settings) -> dict:
    """Local mode: a repo directory plus an issue file."""
    workdir = prepare_workdir(req.repo_dir, settings)
    return execute(
        workdir, req.issue_title, req.issue_body, req.target_tests, settings,
        run_name=f"fix:{req.repo_dir.name}",
        tags=["local"],
        metadata={"repo": req.repo_dir.name},
    )


def run_fix_issue(
    ref: IssueRef,
    settings: Settings,
    *,
    replay: bool,
    targets: list[str],
    fix_pr: int | None = None,
    post: PostHook | None = None,
    extra_tags: list[str] | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> tuple[dict, IntakeInfo]:
    """GitHub mode: fetch the issue, clone the repo, run the loop. Nothing is pushed."""
    with github_client(settings.github_token) as client:
        issue = fetch_issue(client, ref)
        if issue.is_pull_request:
            raise RunnerError(f"{ref.full_name}#{ref.number} is a pull request, not an issue")
        log(f"Issue:    {issue.title} [{issue.state}] labels={issue.labels or '-'}")
        if issue.assignees:
            log(f"WARN      already assigned to {', '.join(issue.assignees)}")

        pr = None
        if replay:
            if issue.state != "closed":
                raise RunnerError("--replay needs a closed issue with a merged fix")
            pr = fetch_pr(client, ref, fix_pr) if fix_pr else find_fixing_pr(client, ref)
        elif issue.state == "closed":
            raise RunnerError(
                "this issue is closed, so the fix is probably already in the default branch. "
                "Use --replay to run the agent on the code as it was before the fix."
            )

    dest = settings.work_dir.resolve() / f"{ref.owner}__{ref.repo}"
    remove_dir(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Cloning {ref.full_name} ...")
    clone_issue_repo(ref, dest)
    if pr is not None:
        checkout_commit(dest, pr.base_sha)
        log(f"Replay:   code as of {pr.base_sha[:10]}, before the fix in PR #{pr.number}")

    hits = policy_mentions(dest)
    if hits:
        log("WARN      contribution docs mention AI. Read them before opening any PR:")
        for h in hits:
            log(f"            {h}")

    final = execute(
        dest, issue.title, format_issue_body(issue), targets, settings,
        run_name=f"fix:{ref.full_name}#{ref.number}",
        tags=["github-replay" if pr else "github", *(extra_tags or [])],
        metadata={"repo": ref.full_name, "issue": ref.number, "replay": replay},
        post=post,
    )
    return final, IntakeInfo(issue.state, pr.url if pr else None, hits)
