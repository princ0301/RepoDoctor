from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env also holds LANGSMITH_* vars that we don't model here
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- API keys ---
    nvidia_api_key: str = ""
    openrouter_api_key: str = ""

    # --- GitHub (classic token, public_repo scope) ---
    github_token: str = ""
    github_push_token: str = ""     # used only by `submit`; keep it separate and short-lived
    github_username: str = ""
    allowed_repos: str = ""        # comma-separated "owner/repo,owner2/repo2"
    dry_run: bool = True           # True = never push or open a PR

    # --- Models: "provider:model", provider is "google" or "openrouter" ---
    llm_explore: str = "openrouter:nvidia/nemotron-3.5-super-vl:free"
    llm_fix: str = "openrouter:nvidia/nemotron-3.5-super-vl:free"
    llm_fallbacks: str = ""
    llm_timeout_s: int = 120
    llm_call_deadline_s: int = 240
    llm_max_retries: int = 2

    # --- Agent limits ---
    max_attempts: int = 3          # fix/verify retries
    explore_max_steps: int = 10    # tool-call rounds in the explorer
    repro_max_attempts: int = 3    # tries to write a valid reproduction test
    fix_context_chars: int = 240_000
    fix_context_rounds: int = 2
    run_timeout_s: int = 900       # set RUN_TIMEOUT_S=0 to disable; checked between graph steps
    build_install_timeout_s: int = 300   # eval-build: one install attempt, no retry
    build_step_timeout_s: int = 240      # eval-build: each test run

    # --- Sandbox ---
    sandbox_timeout_s: int = 300
    sandbox_memory: str = "2g"
    sandbox_cpus: float = 2.0

    allow_test_updates: bool = False

    # --- Paths ---
    work_dir: Path = Path(".repodoctor/work")   # where repos get cloned (gitignored)
    conventions_dir: Path = Path(".repodoctor/conventions")

    # --- PR preparation ---
    prs_dir: Path = Path(".repodoctor/prs")
    pr_max_diff_lines: int = 300
    pr_max_files: int = 8
    # Third-party repos: your first comment on the issue must be this old (0 = off).
    pr_min_comment_hours: int = 24
    max_prs_per_day: int = 1
    git_author_name: str = ""     # default: your GitHub username
    git_author_email: str = ""    # default: <username>@users.noreply.github.com
    commit_trailer: str = "Assisted-by: RepoDoctor ({model})"   # empty = no trailer

    root_cause_mode: str = "agent"   # "agent" uses and discloses the agent's explanation

    @property
    def allowed_repo_list(self) -> list[str]:
        return [r.strip() for r in self.allowed_repos.split(",") if r.strip()]

    @property
    def fallback_specs(self) -> list[str]:
        return [s.strip() for s in self.llm_fallbacks.split(",") if s.strip()]


def get_settings() -> Settings:
    return Settings()
