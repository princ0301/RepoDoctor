from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = ""
    openrouter_api_key: str = ""
    github_token: str = ""
    github_test_repo: str = ""

    max_attempts: int = 3
    sandbox_timeout_s: int = 300

def get_settings() -> Settings:
    return Settings()

