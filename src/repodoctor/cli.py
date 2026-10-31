import shutil
from pathlib import Path

import typer
from dotenv import load_dotenv

from repodoctor.config import get_settings
from repodoctor.graph import build_hello_graph
from repodoctor.tools.issue_ref import (
    InvalidIssueUrl,
    IssueRef,
    RepoNotAllowed,
    check_allowed,
    parse_issue_url,
)
from repodoctor.tools.repo import (
    GitError,
    ProjectInfo,
    clone_issue_repo,
    detect_project,
    remove_dir,
)
from repodoctor.tools.sandbox import Sandbox, SandboxError, ensure_image, image_exists

# Loads .env into os.environ so LangSmith picks up LANGSMITH_* variables
load_dotenv()

app = typer.Typer(no_args_is_help=True, help="RepoDoctor: autonomous issue fixer")


def _resolve_issue(url: str) -> IssueRef:
    try:
        ref = parse_issue_url(url)
        check_allowed(ref, get_settings().allowed_repo_list)
    except (InvalidIssueUrl, RepoNotAllowed) as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    return ref


def _clone_and_detect(ref: IssueRef) -> tuple[Path, ProjectInfo]:
    dest = get_settings().work_dir.resolve() / f"{ref.owner}__{ref.repo}"
    remove_dir(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Cloning {ref.full_name} ...")
    try:
        clone_issue_repo(ref, dest)
    except GitError as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e
    return dest, detect_project(dest)


@app.command()
def hello(name: str = "world") -> None:
    """Run the placeholder graph (check that it shows up in LangSmith)."""
    result = build_hello_graph().invoke({"name": name}, config={"run_name": "hello"})
    typer.echo(result["greeting"])


@app.command()
def doctor() -> None:
    """Check that your environment is ready."""
    for tool in ("uv", "git", "rg", "docker"):
        typer.echo(f"{'OK     ' if shutil.which(tool) else 'MISSING'} {tool}")

    s = get_settings()
    required = {
        "GOOGLE_API_KEY": s.google_api_key,
        "OPENROUTER_API_KEY": s.openrouter_api_key,
        "GITHUB_TOKEN": s.github_token,
        "GITHUB_USERNAME": s.github_username,
        "ALLOWED_REPOS": s.allowed_repos,
    }
    for label, value in required.items():
        typer.echo(f"{'OK     ' if value else 'MISSING'} {label}")

    typer.echo(f"INFO    DRY_RUN={s.dry_run}")
    if not s.dry_run:
        typer.echo("WARN    DRY_RUN is off: the agent can push branches and open PRs")


@app.command("check-issue")
def check_issue(url: str) -> None:
    """Validate an issue URL against the allowlist."""
    ref = _resolve_issue(url)
    typer.echo(f"OK      {ref.full_name} #{ref.number}")


@app.command("inspect")
def inspect_repo(url: str) -> None:
    """Clone the repo behind an issue and detect how to test it."""
    ref = _resolve_issue(url)
    dest, info = _clone_and_detect(ref)
    typer.echo(f"Cloned to      {dest}")
    if not info.supported:
        typer.echo(f"UNSUPPORTED    {info.reason}")
        raise typer.Exit(2)
    typer.echo(f"Test command   {' '.join(info.test_command)}")
    typer.echo(f"Extras         {info.extras or '-'}")
    typer.echo(f"Dep groups     {info.dependency_groups or '-'}")
    typer.echo(f"Req files      {info.requirement_files or '-'}")
    typer.echo(f"Python         {info.requires_python or 'not specified'}")


@app.command()
def smoke(url: str) -> None:
    """Run the repo's own test suite in the sandbox and record the baseline."""
    ref = _resolve_issue(url)
    dest, info = _clone_and_detect(ref)
    if not info.supported:
        typer.echo(f"UNSUPPORTED    {info.reason}")
        raise typer.Exit(2)

    s = get_settings()
    try:
        if not image_exists():
            typer.echo("Building sandbox image (first run only, takes a few minutes) ...")
            ensure_image()
        with Sandbox(dest, memory=s.sandbox_memory, cpus=s.sandbox_cpus) as sb:
            typer.echo("Installing dependencies (network on) ...")
            r = sb.install(info)
            if not r.ok:
                typer.echo(r.output)
                typer.echo(f"\nERROR   install failed (exit {r.exit_code}, "
                           f"{'timeout' if r.timed_out else 'error'})")
                raise typer.Exit(3)
            sb.lock_network()
            typer.echo("Running tests (network off) ...")
            r, report = sb.run_suite(info, timeout=s.sandbox_timeout_s)
    except SandboxError as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1) from e

    if report is None:
        typer.echo(r.output)
        why = "timeout" if r.timed_out else f"exit {r.exit_code}"
        typer.echo(f"\nERROR   no test report produced ({why}); "
                   "pytest may have crashed during collection")
        raise typer.Exit(1)

    typer.echo(
        f"Baseline: {report.total} tests | {report.passed} passed, {report.failed} failed, "
        f"{report.errors} errors, {report.skipped} skipped | {r.duration_s:.0f}s"
    )
    if report.clean:
        typer.echo("Result: CLEAN baseline")
        return
    typer.echo(f"Result: {len(report.failing)} pre-existing failures "
               "(the agent will ignore these)")
    for test_id, msg in list(report.failing.items())[:10]:
        typer.echo(f"  - {test_id}: {msg}")
    if len(report.failing) > 10:
        typer.echo(f"  ... and {len(report.failing) - 10} more")


if __name__ == "__main__":
    app()