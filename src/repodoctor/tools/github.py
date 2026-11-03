"""Read-only GitHub REST access (issues, linked pull requests). No LLM."""

import re
from dataclasses import dataclass

import httpx

from repodoctor.tools.issue_ref import IssueRef

API = "https://api.github.com"
MAX_BODY_CHARS = 8_000
MAX_COMMENTS = 5
MAX_COMMENT_CHARS = 1_500
TIMELINE_PAGE_SIZE = 100
TIMELINE_MAX_PAGES = 3

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
    closed_at: str = ""


@dataclass
class FixingPR:
    number: int
    url: str
    title: str
    base_sha: str              # the commit the PR was opened against
    merge_sha: str | None
    head_sha: str = ""
    created_at: str = ""


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
        closed_at=d.get("closed_at") or "",
    )


def fetch_pr_text(client: httpx.Client, ref: IssueRef, number: int) -> tuple[str, str]:
    pr = _get(client, f"/repos/{ref.full_name}/pulls/{number}")
    return pr.get("title", ""), pr.get("body") or ""


def fetch_file_text(client: httpx.Client, ref: IssueRef, path: str, sha: str) -> str:
    try:
        r = client.get(
            f"/repos/{ref.full_name}/contents/{path}",
            params={"ref": sha},
            headers={"Accept": "application/vnd.github.raw+json"},
        )
    except httpx.HTTPError as e:
        raise GitHubError(f"GitHub request failed: {type(e).__name__}: {e}") from e
    if r.status_code >= 400:
        raise GitHubError(f"could not fetch {path} at {sha[:10]} (HTTP {r.status_code})")
    return r.text


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


def _to_fixing_pr(pr: dict) -> FixingPR:
    return FixingPR(
        number=pr["number"],
        url=pr.get("html_url", ""),
        title=pr.get("title", ""),
        base_sha=pr["base"]["sha"],
        merge_sha=pr.get("merge_commit_sha"),
        head_sha=(pr.get("head") or {}).get("sha", ""),
        created_at=pr.get("created_at") or "",
    )


def _timeline(client: httpx.Client, ref: IssueRef) -> list[dict]:
    events: list[dict] = []
    for page in range(1, TIMELINE_MAX_PAGES + 1):
        batch = _get(
            client,
            f"/repos/{ref.full_name}/issues/{ref.number}/timeline",
            per_page=TIMELINE_PAGE_SIZE,
            page=page,
        )
        events += batch
        if len(batch) < TIMELINE_PAGE_SIZE:
            break
    return events


def _candidate_numbers(client: httpx.Client, ref: IssueRef, events: list[dict]) -> list[int]:
    """PR numbers that might have fixed the issue, from two kinds of timeline evidence:
    pull requests that mention it, and the pull request(s) behind the closing commit."""
    repo_suffix = f"/repos/{ref.full_name}".lower()
    numbers: list[int] = []

    def add(n) -> None:
        if isinstance(n, int) and n not in numbers:
            numbers.append(n)

    for ev in events:
        kind = ev.get("event")
        if kind == "cross-referenced":
            src = (ev.get("source") or {}).get("issue") or {}
            if "pull_request" in src and str(src.get("repository_url", "")).lower().endswith(
                repo_suffix
            ):
                add(src.get("number"))
        elif kind == "closed" and ev.get("commit_id"):
            try:
                prs = _get(client, f"/repos/{ref.full_name}/commits/{ev['commit_id']}/pulls")
            except GitHubError:
                continue
            for pr in prs:
                add(pr.get("number"))
    return numbers


def find_fixing_pr(client: httpx.Client, ref: IssueRef) -> FixingPR:
    """The merged PR in the same repo that fixed this issue."""
    numbers = _candidate_numbers(client, ref, _timeline(client, ref))

    merged = []
    for n in numbers:
        pr = _get(client, f"/repos/{ref.full_name}/pulls/{n}")
        if pr.get("merged"):
            merged.append(pr)
    if not merged:
        raise GitHubError(
            "no merged pull request found for this issue (checked references and the "
            "closing commit). Pass --fix-pr NUMBER to choose the pull request yourself."
        )

    closers = [p for p in merged if ref.number in _closed_numbers(p.get("body") or "")]
    chosen = closers or merged
    if len(chosen) > 1:
        nums = ", ".join(f"#{p['number']}" for p in chosen)
        raise GitHubError(
            f"several merged PRs reference this issue ({nums}). "
            "Pass --fix-pr NUMBER to choose one."
        )
    return _to_fixing_pr(chosen[0])


def fetch_pr(client: httpx.Client, ref: IssueRef, number: int) -> FixingPR:
    """A specific merged PR of the same repo (manual override for find_fixing_pr)."""
    pr = _get(client, f"/repos/{ref.full_name}/pulls/{number}")
    if not pr.get("merged"):
        raise GitHubError(f"PR #{number} has not been merged")
    return _to_fixing_pr(pr)


@dataclass
class PRFile:
    path: str
    status: str   # added | modified | removed | renamed | ...
    patch: str = ""


def list_pr_files(client: httpx.Client, ref: IssueRef, number: int) -> list[PRFile]:
    files: list[PRFile] = []
    for page in range(1, 4):
        batch = _get(
            client, f"/repos/{ref.full_name}/pulls/{number}/files", per_page=100, page=page
        )
        files += [PRFile(f["filename"], f.get("status", ""), f.get("patch") or "") for f in batch]
        if len(batch) < 100:
            break
    return files


def list_merged_prs(client: httpx.Client, full_name: str, max_prs: int = 200) -> list[dict]:
    """Recently updated merged PRs of a repo (newest first)."""
    merged: list[dict] = []
    for page in range(1, 11):
        batch = _get(
            client, f"/repos/{full_name}/pulls",
            state="closed", sort="updated", direction="desc", per_page=100, page=page,
        )
        merged += [p for p in batch if p.get("merged_at")]
        if len(batch) < 100 or len(merged) >= max_prs:
            break
    return merged[:max_prs]


def closed_issue_numbers(text: str) -> list[int]:
    """Issue numbers a PR description closes ('Fixes #12', 'Closes #3')."""
    return sorted(_closed_numbers(text))
