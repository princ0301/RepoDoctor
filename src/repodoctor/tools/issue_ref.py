import re
from dataclasses import dataclass

_ISSUE_URL = re.compile(
    r"^https?://(?:www\.)?github\.com/"
    r"(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/issues/(?P<number>\d+)"
    r"/?(?:[?#].*)?$"
)

class InvalidIssueUrl(ValueError):
    """The string is not a GitHub issue URL."""


class RepoNotAllowed(PermissionError):
    """The repo is not in the ALLOWED_REPOS allowlist."""

@dataclass(frozen=True)
class IssueRef:
    owner: str
    repo: str
    number: int

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

def parse_issue_url(url: str) -> IssueRef:
    m = _ISSUE_URL.match(url.strip())
    if not m:
        raise InvalidIssueUrl(
            f"Not a GitHub issue URL: {url!r} "
            "(expected https://github.com/<owner>/<repo>/issues/<number>)"
        )
    return IssueRef(m["owner"], m["repo"], int(m["number"]))

def check_allowed(ref: IssueRef, allowed: list[str]) -> None:
    """Raise RepoNotAllowed unless ref's repo is in the allowlist (case-insensitive)."""
    if ref.full_name.lower() not in {r.lower() for r in allowed}:
        raise RepoNotAllowed(
            f"{ref.full_name} is not in ALLOWED_REPOS. Add it to .env to allow it."
        )