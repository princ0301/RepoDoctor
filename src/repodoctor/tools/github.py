"""Read-only GitHub REST access (issues, linked pull requests). No LLM."""

import re
from dataclasses import dataclass

import httpx

from repodoctor.tools.issue_ref import IssueRef

API = "https://api.github.com"
MAX_BODY_CHARS = 8_000
MAX_COMMENTS = 5
MAX_COMMENT_CHARS = 1_500

_CLOSING = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\b[:\s]+(?:[\w.-]+/[\w.-]+)?#(\d+)", re.I
)


class GitHubError(RuntimeError):
    """A GitHub request failed or the data is not usable."""


@dataclass
class IssueData:
    url: str
    title: str
    body: str
    state: str                 # "open" | "closed"
    labels: list[str]
    assignees: list[str]
    comments: list[str]        # "user: text"
    is_pull_request: bool


@dataclass
class FixingPR:
    number: int
    url: str
    title: str
    base_sha: str              # the commit the PR was opened against
    merge_sha: str | None


def github_client(
    token: str = "", transport: httpx.BaseTransport | None = None
) -> httpx.Client:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=API, headers=headers, timeout=30, transport=transport)


def _get(client: httpx.Client, path: str, **params):
    try:
        r = client.get(path, params=params or None)
    except httpx.HTTPError as e:
        raise GitHubError(f"GitHub request failed: {type(e).__name__}: {e}") from e
    if r.status_code == 404:
        raise GitHubError(f"not found: {path}")
    if r.status_code in (403, 429):
        hint = " (rate limited: set GITHUB_TOKEN)" if (
            r.headers.get("x-ratelimit-remaining") == "0"
        ) else ""
        raise GitHubError(f"GitHub refused the request ({r.status_code}){hint}")
    if r.status_code >= 400:
        raise GitHubError(f"GitHub error {r.status_code} for {path}")
    return r.json()


def fetch_issue(client: httpx.Client, ref: IssueRef) -> IssueData:
    base = f"/repos/{ref.full_name}/issues/{ref.number}"
    d = _get(client, base)
    comments = _get(client, f"{base}/comments", per_page=MAX_COMMENTS)
    return IssueData(
        url=d.get("html_url", ""),
        title=d.get("title", ""),
        body=d.get("body") or "",
        state=d.get("state", ""),
        labels=[x["name"] for x in d.get("labels", []) if isinstance(x, dict)],
        assignees=[a["login"] for a in d.get("assignees", [])],
        comments=[
            f"{(c.get('user') or {}).get('login', '?')}: {c.get('body') or ''}"
            for c in comments[:MAX_COMMENTS]
        ],
        is_pull_request="pull_request" in d,
    )


def _head(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[... truncated ...]"


def format_issue_body(issue: IssueData) -> str:
    """Body plus comments as one block of untrusted text for the prompts."""
    text = _head(issue.body, MAX_BODY_CHARS)
    if issue.comments:
        text += "\n\n### Comments\n" + "\n\n".join(
            _head(c, MAX_COMMENT_CHARS) for c in issue.comments
        )
    # the prompts wrap this text in <issue> tags: make sure it cannot close them
    return text.replace("</issue>", "[/issue]")


def _closed_numbers(pr_body: str) -> set[int]:
    return {int(n) for n in _CLOSING.findall(pr_body)}


def find_fixing_pr(client: httpx.Client, ref: IssueRef) -> FixingPR:
    """The merged PR in the same repo that fixed this issue (via the issue timeline)."""
    events = _get(
        client, f"/repos/{ref.full_name}/issues/{ref.number}/timeline", per_page=100
    )
    suffix = f"/repos/{ref.full_name}".lower()
    numbers: list[int] = []
    for ev in events:
        if ev.get("event") != "cross-referenced":
            continue
        src = (ev.get("source") or {}).get("issue") or {}
        if "pull_request" not in src:
            continue
        if not str(src.get("repository_url", "")).lower().endswith(suffix):
            continue
        n = src.get("number")
        if isinstance(n, int) and n not in numbers:
            numbers.append(n)

    merged = []
    for n in numbers:
        pr = _get(client, f"/repos/{ref.full_name}/pulls/{n}")
        if pr.get("merged"):
            merged.append(pr)
    if not merged:
        raise GitHubError("no merged pull request is linked to this issue")

    closers = [p for p in merged if ref.number in _closed_numbers(p.get("body") or "")]
    chosen = closers or merged
    if len(chosen) > 1:
        nums = ", ".join(f"#{p['number']}" for p in chosen)
        raise GitHubError(f"several merged PRs reference this issue ({nums}); pick another issue")
    pr = chosen[0]
    return FixingPR(
        number=pr["number"],
        url=pr.get("html_url", ""),
        title=pr.get("title", ""),
        base_sha=pr["base"]["sha"],
        merge_sha=pr.get("merge_commit_sha"),
    )