import shutil
import typer
from dotenv import load_dotenv

from repodoctor.config import get_settings
from repodoctor.graph import build_hello_graph
from repodoctor.tools.issue_ref import InvalidIssueUrl, RepoNotAllowed, check_allowed, parse_issue_url

load_dotenv()

app = typer.Typer(no_args_is_help=True, help="RepoDoctor: autonomous issue fixer")

@app.command()
def hello(name: str = "world") -> None:
    result = build_hello_graph().invoke({"name": name}, config={"run_name": "hello"})
    typer.echo(result["greeting"])

@app.command("check-issue")
def check_issue(url: str) -> None:
    """Validate an issue URL against the allowlist."""
    try:
        ref = parse_issue_url(url)
        check_allowed(ref, get_settings().allowed_repo_list)
    except (InvalidIssueUrl, RepoNotAllowed) as e:
        typer.echo(f"ERROR   {e}")
        raise typer.Exit(1)
    typer.echo(f"OK      {ref.full_name} #{ref.number}")

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