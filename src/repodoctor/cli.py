import shutil
import typer
from dotenv import load_dotenv

from repodoctor.config import get_settings
from repodoctor.graph import build_hello_graph

load_dotenv()

app = typer.Typer(no_args_is_help=True, help="RepoDoctor: autonomous issue fixer")

@app.command()
def hello(name: str = "world") -> None:
    result = build_hello_graph().invoke({"name": name}, config={"run_name": "hello"})
    typer.echo(result["greeting"])

@app.command()
def doctor() -> None:
    for tool in ("uv", "git", "rg", "docker"):
        typer.echo(f"{'OK' if shutil.which(tool) else 'MISSING'} {tool}")

    s = get_settings()
    for label, value in {
        "GOOGLE_API_KEY": s.google_api_key,
        "OPENROUTER_API_KEY": s.openrouter_api_key,
        "GITHUB_TOKEN": s.github_token,
        "GITHUB_TEST_REPO": s.github_test_repo,
    }.items():
        typer.echo(f"{'OK' if value else 'MISSING'} {label}")

if __name__ == "__main__":
    app()