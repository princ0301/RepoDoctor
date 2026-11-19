"""Submit an approved package: fork, push, open the pull request.

This is the only code that writes to GitHub, and it refuses unless every guard passes."""

import base64
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from repodoctor.config import Settings
from repodoctor.tools.github import (
    GitHubError,
    create_fork,
    create_pull_request,
    current_user,
    find_pull_request,
    sync_fork,
    wait_for_repo,
)
from repodoctor.tools.issue_ref import IssueRef, RepoNotAllowed, check_allowed
from repodoctor.tools.pr_package import PackageError, load_package, save_package, verify_approval
from repodoctor.tools.repo import GitError, head_sha, run_git
from repodoctor.tools.triage import run_triage

Pusher = Callable[[Path, str, str, str], None]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def split_pr_text(text: str) -> tuple[str, str]:
    """pr.md is the title on the first line, then the body."""
    first, _, rest = text.partition("\n")
    return first.strip(), rest.strip() + "\n"


def scrub(text: str, token: str) -> str:
    out = text
    for secret in (token, base64.b64encode(f"x-access-token:{token}".encode()).decode()):
        if secret:
            out = out.replace(secret, "***")
    return out


def push_branch(clone: Path, url: str, branch: str, token: str) -> None:
    """git push with the token in a transient HTTP header (never stored in the repository)."""
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth}",
    }
    refspec = f"refs/heads/{branch}:refs/heads/{branch}"
    for attempt in (1, 2):
        p = subprocess.run(
            ["git", "push", url, refspec], cwd=clone, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env, timeout=600,
        )
        if p.returncode == 0:
            return
        err = scrub(p.stderr.strip(), token)
        if attempt == 1 and "shallow" in err.lower():
            run_git("fetch", "--unshallow", "origin", cwd=clone, timeout=900)
            continue
        raise GitError(f"git push failed: {err}")


# ---------- ledger ----------

def _ledger_path(settings: Settings) -> Path:
    return settings.prs_dir.resolve() / "ledger.json"


def load_ledger(settings: Settings) -> list[dict]:
    p = _ledger_path(settings)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def record_submission(settings: Settings, entry: dict) -> None:
    p = _ledger_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([*load_ledger(settings), entry], indent=2) + "\n", encoding="utf-8")


def recent_submissions(ledger: list[dict], now: datetime, hours: int = 24) -> list[dict]:
    cutoff = now - timedelta(hours=hours)
    return [e for e in ledger if datetime.fromisoformat(e["at"]) >= cutoff]


# ---------- preflight ----------

@dataclass
class Plan:
    repo: str
    issue: int
    branch: str
    base: str
    own_repo: bool
    title: str
    files: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def clone_problems(meta: dict, pkg: Path) -> list[str]:
    clone = Path(meta["workdir"])
    if not clone.is_dir():
        return [f"the work clone is gone: {clone}"]
    try:
        out: list[str] = []
        if run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=clone).strip() != meta["branch"]:
            out.append(f"the work clone is not on branch {meta['branch']}")
        if run_git("status", "--porcelain", cwd=clone).strip():
            out.append("the work clone has uncommitted changes: run repackage")
        if head_sha(clone) != meta["head_sha"]:
            out.append("the work clone's commit differs from the package: run repackage")
        elif run_git("format-patch", "-1", "--stdout", cwd=clone) != (
            pkg / "change.patch"
        ).read_text(encoding="utf-8"):
            out.append("change.patch does not match the commit: run repackage")
        return out
    except GitError as e:
        return [str(e)]


def preflight(
    pkg: Path,
    settings: Settings,
    *,
    read_client: httpx.Client,
    push_client: httpx.Client,
    triage_fn: Callable[..., Any] = run_triage,
    now: datetime | None = None,
) -> Plan:
    now = now or datetime.now(UTC)
    meta = load_package(pkg)
    repo = meta["repo"]
    owner, _, name = repo.partition("/")
    ref = IssueRef(owner, name, meta["issue"])
    title, _ = split_pr_text((pkg / "pr.md").read_text(encoding="utf-8"))
    plan = Plan(
        repo=repo, issue=meta["issue"], branch=meta["branch"], base=meta["default_branch"],
        own_repo=owner.lower() == settings.github_username.lower(), title=title,
        files=meta["files"],
    )
    try:
        check_allowed(ref, settings.allowed_repo_list)
    except RepoNotAllowed as e:
        plan.problems.append(str(e))
    plan.problems += verify_approval(pkg)
    plan.problems += clone_problems(meta, pkg)

    stage = (meta.get("submission") or {}).get("stage")
    if stage == "opened":
        plan.problems.append(f"already opened: {meta['submission'].get('url')}")
    elif stage in ("forked", "pushed"):
        plan.notes.append(f"an earlier attempt got as far as '{stage}'; submitting resumes from there")

    if not settings.github_push_token:
        (plan.notes if settings.dry_run else plan.problems).append(
            "GITHUB_PUSH_TOKEN is not set (it is needed to submit)"
        )
    else:
        try:
            who = current_user(push_client)
            if who.lower() != settings.github_username.lower():
                plan.problems.append(
                    f"the push token belongs to '{who}', not to GITHUB_USERNAME "
                    f"'{settings.github_username}'"
                )
        except GitHubError as e:
            plan.problems.append(f"the push token does not work: {e}")

    recent = recent_submissions(load_ledger(settings), now)
    if len(recent) >= settings.max_prs_per_day:
        plan.problems.append(
            f"{len(recent)} pull request(s) already submitted in the last 24 hours "
            f"(limit {settings.max_prs_per_day}): {recent[-1]['url']}"
        )

    try:
        triage = triage_fn(read_client, ref, me=settings.github_username)
    except GitHubError as e:
        plan.problems.append(f"triage could not run: {e}")
    else:
        for f in triage.report.findings:
            if f.level == "skip":
                plan.problems.append(f"triage: {f.message}")
            elif f.level == "warn":
                plan.notes.append(f"triage: {f.message}")
    return plan


def render_plan(plan: Plan) -> str:
    route = (
        "push the branch to this repository" if plan.own_repo
        else "fork the repository, push the branch to your fork, open the PR from the fork"
    )
    lines = [
        "=" * 72,
        f"SUBMIT PLAN  {plan.repo}  issue #{plan.issue}",
        "=" * 72,
        f"PR title: {plan.title}",
        f"Branch {plan.branch} into {plan.repo}:{plan.base}",
        f"Files: {', '.join(plan.files)}",
        f"Route: {route}",
    ]
    lines += [f"  [block] {p}" for p in plan.problems]
    lines += [f"  [note ] {n}" for n in plan.notes]
    lines.append("All guards passed." if plan.ok else "")
    return "\n".join(lines)


# ---------- submit ----------

def _record(pkg: Path, meta: dict, **fields: Any) -> None:
    meta["submission"] = {**(meta.get("submission") or {}), **fields, "updated_at": _now()}
    save_package(pkg, meta)


def submit_package(
    pkg: Path,
    settings: Settings,
    *,
    client: httpx.Client,
    pusher: Pusher = push_branch,
    draft: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    problems = verify_approval(pkg)
    if problems:
        raise PackageError("; ".join(problems))
    meta = load_package(pkg)
    stage = (meta.get("submission") or {}).get("stage")
    if stage == "opened":
        raise PackageError(f"already opened: {meta['submission'].get('url')}")

    title, body = split_pr_text((pkg / "pr.md").read_text(encoding="utf-8"))
    repo, branch = meta["repo"], meta["branch"]
    own = repo.partition("/")[0].lower() == settings.github_username.lower()
    clone = Path(meta["workdir"])

    if own:
        head_repo, head = repo, f"{repo.partition('/')[0]}:{branch}"
    else:
        head_repo = (meta.get("submission") or {}).get("fork") or create_fork(client, repo)
        wait_for_repo(client, head_repo, sleep=sleep)
        sync_fork(client, head_repo, meta["default_branch"])
        head = f"{head_repo.partition('/')[0]}:{branch}"
        if stage is None:
            _record(pkg, meta, stage="forked", fork=head_repo)

    if stage != "pushed":
        pusher(clone, f"https://github.com/{head_repo}.git", branch, settings.github_push_token)
        _record(pkg, meta, stage="pushed", head=head)

    url = find_pull_request(client, repo, head)   # makes a repeated run harmless
    if url is None:
        url = create_pull_request(
            client, repo, title=title, body=body, head=head, base=meta["default_branch"],
            draft=draft,
        )["html_url"]
    _record(pkg, meta, stage="opened", url=url)
    record_submission(settings, {"repo": repo, "issue": meta["issue"], "url": url, "at": _now()})
    return meta