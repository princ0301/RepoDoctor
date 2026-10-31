from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env also holds LANGSMITH_* vars that we don't model here
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM providers ---
    google_api_key: str = ""
    openrouter_api_key: str = ""

    # --- GitHub (classic token, public_repo scope) ---
    github_token: str = ""
    github_username: str = ""
    allowed_repos: str = ""        # comma-separated "owner/repo,owner2/repo2"
    dry_run: bool = True           # True = never push or open a PR

    # --- Limits ---
    max_attempts: int = 3          # fix/verify retries
    sandbox_timeout_s: int = 300

    # --- Paths ---
    work_dir: Path = Path(".repodoctor/work")   # where repos get cloned (gitignored)

    sandbox_memory: str = "2g"
    sandbox_cpus: float = 2.0

    @property
    def allowed_repo_list(self) -> list[str]:
        return [r.strip() for r in self.allowed_repos.split(",") if r.strip()]


def get_settings() -> Settings:
    return Settings()