"""Deterministic triage: is this issue a reasonable target for an automated change?

Uses only GitHub's API. No model calls, no cloning.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from repodoctor.tools.findings import Finding, Report
from repodoctor.tools.github import (
    GitHubError,
    IssueData,
    LinkedPR,
    RepoInfo,
    fetch_comments,
    fetch_file_text,
    fetch_issue,
    fetch_repo_info,
    linked_pull_requests,
    list_open_issues,
    list_recent_threads,
)
from repodoctor.tools.issue_ref import IssueRef
from repodoctor.tools.policy import POLICY_FILES, scan_text

STALE_DAYS = 730
MAX_POLICY_HITS = 8
MIN_DESCRIPTION_LENGTH = 60

_BLOCKING_LABEL = re.compile(
    r"\b(wontfix|won't fix|invalid|duplicate|question|discussion|needs[- ]info"
    r"|needs[- ]repro\w*|blocked|on hold|stale)\b"
)
_FEATURE_LABEL = re.compile(r"\b(feature|enhancement|proposal|rfc|idea)\b")
_HELPFUL_LABEL = re.compile(r"\b(good first issue|help wanted|bug)\b")
_CLAIM = re.compile(
    r"\bi(?:'ll| will| am|'m)\s+(?:going to\s+)?(?:work|take|fix|try|submit|open|send)"
    r"|\bcan i (?:take|work on|try|have|pick)"
    r"|\blet me (?:take|work on|try|fix)"
    r"|\b(?:working|work) on (?:this|it|a fix)\b"
    r"|\bi(?:'ve| have) (?:opened|sent|submitted|started)"
    r"|\bassign (?:this|it) to me\b",
    re.I,
)
_DISCOURAGE = re.compile(
    r"\bwe (?:don't|do not|won't|will not) (?:want|accept|plan)"
    r"|\bnot (?:planning|accepting)\b|\bwon't (?:fix|do|implement)\b"
    r"|\bout of scope\b|\bworking as intended\b|\bnot a bug\b",
    re.I,
)
_POLICY_BLOCK = re.compile(
    r"\bnot (?:accept|allow|permit)\w*|\b(?:banned?|prohibit\w*|forbid\w*|disallow\w*)\b"
    r"|\bwill be (?:closed|rejected)\b"
    r"|\b(?:do not|don't|must not|never) (?:submit|use|send|contribute|open|post)\b"
    r"|\bno (?:ai|llm)\b",
    re.I,
)
_AI_TITLE = re.compile(
    r"\b(?:ai|llms?|gpt|chatgpt|copilot|claude|slop|spam)\b"
    r"|(?<!user )(?<!user-)\bagents?\b|\bbots?\b",
    re.I,
)


def ai_thread_findings(threads: list[tuple[int, str, bool]]) -> tuple[list[str], list[Finding]]:
    hits = [
        f"#{n} {'(pull request) ' if is_pr else ''}{title}"
        for n, title, is_pr in threads if _AI_TITLE.search(title)
    ]
    if not hits:
        return [], []
    more = f" (+{len(hits) - 1} more)" if len(hits) > 1 else ""
    return hits[:5], [Finding(
        "warn", "ai_threads",
        f"Recent issues or pull requests discuss AI, agents or bots; read them before "
        f"contributing: {hits[0]}{more}",
    )]


def _ai_threads(client: httpx.Client, full_name: str) -> tuple[list[str], list[Finding]]:
    try:
        return ai_thread_findings(list_recent_threads(client, full_name))
    except GitHubError:
        return [], []


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def triage_issue(
    issue: IssueData, comments: list[tuple[str, str]], *, me: str, now: datetime
) -> Report:
    r = Report()
    if issue.is_pull_request:
        r.add("skip", "is_pull_request", "This is a pull request, not an issue.")
        return r
    if issue.state != "open":
        r.add("skip", "not_open", f"The issue is {issue.state or 'not open'}.")
        return r

    others = [a for a in issue.assignees if a.lower() != me.lower()]
    if others:
        r.add("skip", "assigned", f"Already assigned to {', '.join(others)}.")

    for label in issue.labels:
        norm = label.lower().replace("_", " ")
        if _BLOCKING_LABEL.search(norm):
            r.add("skip", "blocking_label", f"Label '{label}' says this is not ready for a fix.")
        elif _FEATURE_LABEL.search(norm):
            r.add(
                "warn", "feature_label",
                f"Label '{label}' suggests a feature request; in our evaluation these often "
                "need design choices the issue does not state.",
            )
        elif _HELPFUL_LABEL.search(norm):
            r.add("info", "helpful_label", f"Label '{label}'.")

    body = issue.body or ""
    if "```" in body or "Traceback (most recent call last)" in body or ">>> " in body:
        r.add("info", "has_code_sample", "The issue includes a code sample or traceback.")
    else:
        r.add(
            "warn", "no_code_sample",
            "No code sample or traceback: the agent has little to reproduce the bug from.",
        )
    if len(body.strip()) < MIN_DESCRIPTION_LENGTH:
        r.add("warn", "short_description", "The description is very short.")

    updated = _parse_time(issue.updated_at)
    if updated and (now - updated).days > STALE_DAYS:
        r.add("warn", "stale", f"No activity for {(now - updated).days} days.")

    claims: dict[str, str] = {}
    for who, text in comments:
        if who.lower() == me.lower() or who in claims:
            continue
        m = _CLAIM.search(text)
        if m:
            claims[who] = text[max(0, m.start() - 20): m.end() + 40].replace("\n", " ")
    if claims:
        shown = "; ".join(f'{who}: "...{snip}..."' for who, snip in list(claims.items())[:2])
        r.add("warn", "claimed", f"Comments suggest someone may be working on it ({shown}).")
    if any(_DISCOURAGE.search(text) for _, text in comments):
        r.add("warn", "discouraged", "A comment suggests the maintainers may not want this.")
    return r


def triage_links(linked: list[LinkedPR]) -> Report:
    r = Report()
    for pr in linked:
        if pr.state == "open" and pr.closes_issue:
            r.add("skip", "open_pr", f"Open PR #{pr.number} already says it fixes this ({pr.url}).")
        elif pr.state == "open":
            r.add("warn", "open_pr_mentions", f"Open PR #{pr.number} mentions this issue.")
        elif pr.merged:
            r.add(
                "warn", "merged_pr_mentions",
                f"Merged PR #{pr.number} mentions this issue; it may already be fixed.",
            )
    return r


def triage_repo(repo: RepoInfo) -> Report:
    r = Report()
    if repo.archived:
        r.add("skip", "archived", "The repository is archived.")
    if repo.disabled:
        r.add("skip", "disabled", "The repository is disabled.")
    if repo.private:
        r.add("skip", "private", "The repository is private.")
    return r


def policy_findings(hits: list[str]) -> list[Finding]:
    if not hits:
        return []
    blocking = [h for h in hits if _POLICY_BLOCK.search(h.split(": ", 1)[-1])]
    if blocking:
        return [Finding(
            "skip", "policy_blocks_ai",
            f"The contribution docs seem to restrict AI-generated contributions: {blocking[0]}",
        )]
    return [Finding(
        "warn", "policy_mentions_ai",
        f"The contribution docs mention AI ({len(hits)} line(s)); read them first. "
        f"First: {hits[0]}",
    )]


def remote_policy_hits(client: httpx.Client, ref: IssueRef, branch: str) -> list[str]:
    hits: list[str] = []
    for rel in POLICY_FILES:
        try:
            text = fetch_file_text(client, ref, rel, branch)
        except GitHubError:
            continue   # the file does not exist (or could not be read)
        hits += scan_text(rel, text, MAX_POLICY_HITS - len(hits))
        if len(hits) >= MAX_POLICY_HITS:
            break
    return hits


@dataclass
class TriageResult:
    report: Report
    issue: IssueData
    repo: RepoInfo
    policy_hits: list[str]
    ai_threads: list[str]


def run_triage(
    client: httpx.Client, ref: IssueRef, *, me: str = "", now: datetime | None = None
) -> TriageResult:
    now = now or datetime.now(UTC)
    issue = fetch_issue(client, ref)
    repo = fetch_repo_info(client, ref.full_name)
    report = Report()
    report.extend(triage_repo(repo).findings)

    open_issue = issue.state == "open" and not issue.is_pull_request
    comments = fetch_comments(client, ref) if open_issue else []
    report.extend(triage_issue(issue, comments, me=me, now=now).findings)
    if report.verdict == "skip":
        return TriageResult(report, issue, repo, [], [])

    report.extend(triage_links(linked_pull_requests(client, ref)).findings)
    hits = remote_policy_hits(client, ref, repo.default_branch)
    report.extend(policy_findings(hits))
    threads, thread_findings = _ai_threads(client, ref.full_name)
    report.extend(thread_findings)
    return TriageResult(report, issue, repo, hits, threads)


@dataclass
class ScanRow:
    number: int
    title: str
    verdict: str
    labels: list[str]
    notes: list[str]      # the warn and skip messages


@dataclass
class ScanResult:
    repo: RepoInfo
    repo_report: Report
    policy_hits: list[str]
    rows: list[ScanRow]   # ok first, then warn, then skip; recent first within each
    ai_threads: list[str]


_ORDER = {"ok": 0, "warn": 1, "skip": 2}


def scan_repo(
    client: httpx.Client,
    full_name: str,
    *,
    me: str = "",
    scan: int = 30,
    now: datetime | None = None,
) -> ScanResult:
    """Triage the most recent open, unassigned issues of a repository.

    Costs roughly 1 request per issue for comments, plus 2 to 4 for linked pull requests
    on those that pass.
    """
    now = now or datetime.now(UTC)
    owner, _, name = full_name.partition("/")
    repo = fetch_repo_info(client, full_name)
    repo_report = triage_repo(repo)
    hits: list[str] = []
    threads: list[str] = []
    if repo_report.verdict != "skip":
        hits = remote_policy_hits(client, IssueRef(owner, name, 0), repo.default_branch)
        repo_report.extend(policy_findings(hits))
        threads, thread_findings = _ai_threads(client, full_name)
        repo_report.extend(thread_findings)
    if repo_report.verdict == "skip":
        return ScanResult(repo, repo_report, hits, [], threads)

    rows: list[ScanRow] = []
    for number, issue in list_open_issues(client, full_name, scan):
        ref = IssueRef(owner, name, number)
        report = triage_issue(issue, fetch_comments(client, ref), me=me, now=now)
        if report.verdict != "skip":
            try:
                report.extend(triage_links(linked_pull_requests(client, ref)).findings)
            except GitHubError as e:
                report.add("warn", "links_unchecked", f"Could not check linked pull requests ({e}).")
        rows.append(ScanRow(
            number, issue.title, report.verdict, issue.labels,
            [f.message for f in report.findings if f.level in ("warn", "skip")],
        ))
    rows.sort(key=lambda r: _ORDER[r.verdict])   # stable: keeps most-recent-first
    return ScanResult(repo, repo_report, hits, rows, threads)
