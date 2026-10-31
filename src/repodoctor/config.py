from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = ""
    openrouter_api_key: str = ""

    github_token: str = ""
    github_username: str = ""
    allowed_repos: str = ""      # "owner/repo,owner2/repo2"
    dry_run: bool = True         # True = never push or open a PR

    max_attempts: int = 3
    sandbox_timeout_s: int = 300

    @property
    def allowed_repo_list(self) -> list[str]:
        return [r.strip() for r in self.allowed_repos.split(",") if r.strip()]

def get_settings() -> Settings:
    return Settings()

