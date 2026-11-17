"""What a project's own merged bug-fix pull requests look like.

Where the tests go, how the changelog is kept, what the PR template asks, how big changes are.
Deterministic: GitHub API only, no model calls."""

import json
import re
import statistics
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import httpx

from repodoctor.tools.findings import TODO_MARKER, Finding
from repodoctor.tools.github import (
    GitHubError,
    _closed_numbers,
    _get,
    fetch_file_text,
    fetch_repo_info,
)
from repodoctor.tools.issue_ref import IssueRef
from repodoctor.tools.paths import is_test_path
from repodoctor.tools.repo import added_files
from repodoctor.tools.review_hints import hint_new_test_files, list_test_ids

CACHE_DAYS = 7
TEMPLATE_PATHS = (
    ".github/pull_request_template.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "pull_request_template.md",
    "PULL_REQUEST_TEMPLATE.md",
    "docs/pull_request_template.md",
)
_BOT = re.compile(r"\[bot\]$|^(dependabot|renovate|pre-commit-ci|github-actions)", re.I)
_FIXISH = re.compile(r"\b(fix(es|ed)?|bug|regression|crash|error)\b", re.I)
_CHANGELOG_NAME = re.compile(r"^(changelog|changes|history|news|release[-_ ]?notes)(\.\w+)?$", re.I)
_FRAGMENT_DIRS = {"changelog", "changes", "news", "newsfragments", "changelog.d", "changes.d"}
_FRAGMENT_NAME = re.compile(
    r"^(?P<id>\d+)\.(?P<type>[a-z]+)(?:\.\d+)?(?:\.(?P<ext>rst|md|txt))?$"
)
_BUG_TYPES = ("bugfix", "fix", "bug", "fixed", "fixes")
_REF_TOKEN = re.compile(r":issue:`\d+`|:pr:`\d+`|\(#\d+\)|#\d+")
_BULLET = re.compile(r"^(\s*[-*+]\s+)(\S.*)$")
_TEST_DEF = re.compile(r"^\+\s*(?:async\s+)?def\s+(test\w*)\s*\(", re.M)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_CHECK = re.compile(r"^(\s*[-*]\s+)\[( |x|X)\]\s+(.*)$")


def classify(path: str) -> str:
    """test | fragment | changelog | docs | source | other"""
    p = PurePosixPath(path)
    if is_test_path(path):
        return "test"
    if p.parent.name.lower() in _FRAGMENT_DIRS and _FRAGMENT_NAME.match(p.name):
        return "fragment"
    if len(p.parts) <= 2 and _CHANGELOG_NAME.match(p.name):
        return "changelog"
    if p.parts[0].lower() in ("docs", "doc") or p.suffix.lower() in (".rst", ".md", ".txt"):
        return "docs"
    return "source" if p.suffix == ".py" else "other"


# ---------- analysis of mined pull requests ----------

def analyze(records: list[dict]) -> dict:
    n = len(records)
    touched = added_file = modified = with_new_tests = issue_named = 0
    homes: Counter = Counter()
    file_prs = frag_prs = 0
    cl_files: Counter = Counter()
    frag_dirs: Counter = Counter()
    frag_types: Counter = Counter()
    frag_exts: Counter = Counter()
    sizes: list[int] = []
    counts: list[int] = []
    for r in records:
        files = r["files"]
        tests = [f for f in files if f["kind"] == "test"]
        edited = [f for f in tests if f["status"] in ("modified", "renamed")]
        touched += bool(tests)
        added_file += any(f["status"] == "added" for f in tests)
        modified += bool(edited)
        for f in edited:
            homes[f["path"]] += 1
        if r.get("new_tests"):
            with_new_tests += 1
            if any(
                re.search(rf"(?<!\d){i}(?!\d)", t) for t in r["new_tests"] for i in r["issues"]
            ):
                issue_named += 1
        changed = [f for f in files if f["kind"] == "changelog"]
        frags = [f for f in files if f["kind"] == "fragment"]
        file_prs += bool(changed)
        frag_prs += bool(frags)
        for f in changed:
            cl_files[f["path"]] += 1
        for f in frags:
            p = PurePosixPath(f["path"])
            m = _FRAGMENT_NAME.match(p.name)
            frag_dirs[str(p.parent)] += 1
            if m:
                frag_types[m.group("type")] += 1
                if m.group("ext"):
                    frag_exts[m.group("ext")] += 1
        sizes.append(sum(f.get("additions", 0) + f.get("deletions", 0) for f in files))
        counts.append(len(files))

    kind = "fragment" if frag_prs >= max(2, file_prs) else "file" if file_prs >= 2 else "none"
    bug = [t for t, _ in frag_types.most_common() if t in _BUG_TYPES]
    return {
        "n": n,
        "tests": {
            "touched": touched,
            "added_file": added_file,
            "modified_existing": modified,
            "top_files": homes.most_common(5),
            "with_new_tests": with_new_tests,
            "issue_named": issue_named,
        },
        "changelog": {
            "kind": kind,
            "touched": frag_prs if kind == "fragment" else file_prs,
            "path": cl_files.most_common(1)[0][0] if cl_files else None,
            "fragment_dir": frag_dirs.most_common(1)[0][0] if frag_dirs else None,
            "fragment_type": (bug or [t for t, _ in frag_types.most_common(1)] or [None])[0],
            "fragment_ext": frag_exts.most_common(1)[0][0] if frag_exts else None,
        },
        "median_changed_lines": int(statistics.median(sizes)) if sizes else 0,
        "median_files": int(statistics.median(counts)) if counts else 0,
    }


def changelog_style(head: str) -> dict:
    """How entries in a single-file changelog look, from the top of the file."""
    lines = head.splitlines()
    bullets = [m for m in (_BULLET.match(line) for line in lines[:150]) if m][:15]
    refs: Counter = Counter()
    for m in bullets:
        for tok in _REF_TOKEN.findall(m.group(2)):
            refs[re.sub(r"\d+", "{n}", tok)] += 1
    return {
        "prefix": bullets[0].group(1) if bullets else "- ",
        "ref": refs.most_common(1)[0][0] if refs else "#{n}",
        "excerpt": "\n".join(lines[:14]),
    }


def draft_changelog_entry(style: dict | None, issue: int) -> str:
    s = style or {}
    ref = (s.get("ref") or "#{n}").replace("{n}", str(issue))
    return f"{s.get('prefix') or '- '}{TODO_MARKER}: one line on what changed, for users. {ref}"


def fragment_path(cl: dict, issue: int) -> str | None:
    """Where a changelog fragment for this issue goes, for projects that use fragment files."""
    if cl.get("kind") != "fragment" or not cl.get("fragment_dir") or not cl.get("fragment_type"):
        return None
    d = cl["fragment_dir"]
    if d.startswith("/") or ".." in d.split("/"):
        return None
    ext = f".{cl['fragment_ext']}" if cl.get("fragment_ext") else ""
    return f"{d}/{issue}.{cl['fragment_type']}{ext}"


def homes_for_sources(records: list[dict], sources: list[str], top: int = 3) -> list[tuple[str, int]]:
    """Existing test files that past fixes to the same code (or folder) edited, with counts."""
    src = set(sources)
    dirs = {str(PurePosixPath(s).parent) for s in sources}
    score: Counter = Counter()
    prs: Counter = Counter()
    for r in records:
        r_src = {f["path"] for f in r["files"] if f["kind"] == "source"}
        if r_src & src:
            weight = 3
        elif {str(PurePosixPath(p).parent) for p in r_src} & dirs:
            weight = 1
        else:
            continue
        for f in r["files"]:
            if f["kind"] == "test" and f["status"] in ("modified", "renamed"):
                score[f["path"]] += weight
                prs[f["path"]] += 1
    return [(p, prs[p]) for p, _ in score.most_common(top)]

def exact_homes(records: list[dict], sources: list[str], top: int = 3) -> list[tuple[str, int]]:
    """Existing test files that past fixes to the very same source files edited."""
    src = set(sources)
    prs: Counter = Counter()
    for r in records:
        if not ({f["path"] for f in r["files"] if f["kind"] == "source"} & src):
            continue
        for f in r["files"]:
            if f["kind"] == "test" and f["status"] in ("modified", "renamed"):
                prs[f["path"]] += 1
    return prs.most_common(top)


def convention_findings(
    conv: dict | None, root: Path, new_tests: list[str], names: list[str],
    sources: list[str], issue: int,
) -> list[Finding]:
    if not conv or not conv.get("prs"):
        return []
    t = conv["stats"]["tests"]
    out: list[Finding] = []

    def listed(items: list[tuple[str, int]]) -> str:
        return "; ".join(f"{p} ({c} past fix{'es' if c > 1 else ''})" for p, c in items)

    exact = [(p, c) for p, c in exact_homes(conv["prs"], sources) if (root / p).is_file()]
    folder = [(p, c) for p, c in homes_for_sources(conv["prs"], sources) if (root / p).is_file()]
    for test_file in new_tests:
        if exact:
            out.append(Finding(
                "warn", "new_test_file",
                f"The test is in a new file ({test_file}). Past fixes to this code in this "
                f"project put their tests in: {listed(exact)}.",
            ))
        elif folder:
            out.append(Finding(
                "warn", "new_test_file_folder",
                f"The test is in a new file ({test_file}). Past fixes in the same folder put "
                f"their tests in: {listed(folder)}; none touched this code itself, so check "
                "which test module covers it.",
            ))
        elif t["modified_existing"] > t["added_file"] and t["top_files"]:
            common = ", ".join(p for p, _ in t["top_files"][:3])
            out.append(Finding(
                "warn", "new_test_file_generic",
                f"The test is in a new file ({test_file}); most past fixes here add tests to "
                f"existing files, most often {common}.",
            ))
    if names and t["with_new_tests"] >= 3 and t["issue_named"] / t["with_new_tests"] >= 0.4:
        if not any(re.search(rf"(?<!\d){issue}(?!\d)", n) for n in names):
            out.append(Finding(
                "warn", "test_name_convention",
                f"{t['issue_named']} of {t['with_new_tests']} recent fix PRs here put the issue "
                f"number in the test name (for example test_issue{issue}_...); yours is {names[0]}.",
            ))
    return out


def new_test_findings(
    conv: dict | None, root: Path, base: str, files: list[str], issue: int
) -> list[Finding]:
    new_tests = [p for p in added_files(root, base) if is_test_path(p)]
    if not new_tests:
        return []
    sources = [f for f in files if classify(f) == "source"]
    names = [i.partition("::")[2] for t in new_tests for i in list_test_ids(root, t)]
    found = convention_findings(conv, root, new_tests, names, sources, issue)
    if not any(f.code == "new_test_file" for f in found):
        # no past fix touched this code: also point at tests that call the same functions
        found += hint_new_test_files(root, new_tests)
    return found


# ---------- the project's pull request template ----------

def checklist_tick(text: str, facts: dict) -> bool:
    """Tick a checklist item only when we verified what it says."""
    t = text.lower()
    if "tox" in t:
        return False
    if re.search(
        r"pytest|test suite|run (?:the )?tests|ran the tests|tests? (?:pass|run)"
        r"|no tests? (?:fail|broke)|all tests", t,
    ):
        return facts.get("suite_passed") is True
    if re.search(r"\btests?\b", t) and re.search(
        r"fail|demonstrat|cover|regression|add|include|write|new", t
    ):
        return bool(facts.get("tests_added") and facts.get("test_fails_without_fix") is True)
    if re.search(r"changelog|changes\.rst|release notes|\bnews\b", t):
        return bool(facts.get("changelog_touched"))
    if re.search(r"pre-commit|lint|flake8|ruff|black|format|style", t):
        return facts.get("lint_passed") is True
    return False


def render_checklist(template: str, facts: dict) -> str:
    """The template without its instruction comments; verified boxes ticked, pre-ticked kept."""
    lines = []
    for line in _COMMENT.sub("", template).splitlines():
        m = _CHECK.match(line)
        if m:
            on = m.group(2) in "xX" or checklist_tick(m.group(3), facts)
            lines.append(f"{m.group(1)}[{'x' if on else ' '}] {m.group(3)}")
        else:
            lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def update_checkboxes(pr_text: str, facts: dict) -> str:
    """Tick newly verified boxes. Never unticks and never touches the reviewer's own edits."""
    out = []
    for line in pr_text.split("\n"):
        m = _CHECK.match(line)
        if m and m.group(2) == " " and checklist_tick(m.group(3), facts):
            line = f"{m.group(1)}[x] {m.group(3)}"
        out.append(line)
    return "\n".join(out)


def convention_summary_lines(conv: dict | None) -> list[str]:
    if not conv:
        return ["  (not learned: run `repodoctor conventions OWNER/REPO`)"]
    if not conv.get("prs"):
        return ["  (no merged bug-fix pull requests found to learn from)"]
    s = conv["stats"]
    n, t, c = s["n"], s["tests"], s["changelog"]
    lines = [
        f"  From {n} recent merged pull requests that look like bug fixes:",
        f"  - tests: {t['touched']} of {n} touch tests; {t['modified_existing']} edit existing "
        f"test files, {t['added_file']} add a new test file",
    ]
    if t["top_files"]:
        lines.append("    most often edited: " + ", ".join(f"{p} ({k})" for p, k in t["top_files"][:3]))
    if t["with_new_tests"]:
        lines.append(
            f"  - test names: {t['issue_named']} of {t['with_new_tests']} include the issue number"
        )
    if c["kind"] == "fragment":
        ext = f".{c['fragment_ext']}" if c.get("fragment_ext") else ""
        lines.append(
            f"  - changelog: a new file in {c['fragment_dir']}/ named "
            f"<issue>.{c['fragment_type']}{ext} ({c['touched']} of {n})"
        )
    elif c["kind"] == "file":
        lines.append(f"  - changelog: {c['path']} is edited in {c['touched']} of {n}")
    else:
        lines.append("  - changelog: no pattern found")
    lines.append(
        f"  - size: typically {s['median_changed_lines']} changed lines in "
        f"{s['median_files']} file(s)"
    )
    lines.append("  - PR template: " + ("found" if conv.get("pr_template") else "none found"))
    return lines


# ---------- mining ----------

def _try_text(client: httpx.Client, ref: IssueRef, rel: str, branch: str) -> str:
    try:
        return fetch_file_text(client, ref, rel, branch) or ""
    except GitHubError:
        return ""


def mine(
    client: httpx.Client, full_name: str, *, limit: int = 20, now: datetime | None = None
) -> dict:
    now = now or datetime.now(UTC)
    owner, _, name = full_name.partition("/")
    ref = IssueRef(owner, name, 0)
    repo = fetch_repo_info(client, full_name)
    listing = _get(
        client, f"/repos/{full_name}/pulls",
        state="closed", sort="updated", direction="desc", per_page=100,
    )
    records: list[dict] = []
    for pr in listing:
        if len(records) >= limit:
            break
        if not pr.get("merged_at"):
            continue
        user = pr.get("user") or {}
        if user.get("type") == "Bot" or _BOT.search(user.get("login", "")):
            continue
        base_ref = (pr.get("base") or {}).get("ref")
        if base_ref and base_ref != repo.default_branch:
            continue   # backports repeat the original fix
        issues = sorted(set(_closed_numbers(pr.get("body") or "")))
        if not issues and not _FIXISH.search(pr.get("title") or ""):
            continue
        rows: list[dict] = []
        new_tests: list[str] = []
        for f in _get(client, f"/repos/{full_name}/pulls/{pr['number']}/files", per_page=100):
            path = f.get("filename", "")
            kind = classify(path)
            rows.append({
                "path": path, "status": f.get("status", ""), "kind": kind,
                "additions": f.get("additions", 0), "deletions": f.get("deletions", 0),
            })
            if kind == "test" and f.get("status") in ("added", "modified", "renamed"):
                new_tests += _TEST_DEF.findall(f.get("patch") or "")
        records.append({
            "number": pr["number"], "title": pr.get("title", ""), "issues": issues,
            "merged_at": pr["merged_at"], "files": rows, "new_tests": new_tests,
        })

    stats = analyze(records)
    template = ""
    for rel in TEMPLATE_PATHS:
        template = _try_text(client, ref, rel, repo.default_branch)
        if template.strip():
            break
    head = ""
    cl = stats["changelog"]
    if cl["kind"] == "file" and cl["path"]:
        head = _try_text(client, ref, cl["path"], repo.default_branch)[:4000]
    return {
        "repo": full_name,
        "fetched_at": now.isoformat(timespec="seconds"),
        "default_branch": repo.default_branch,
        "prs": records,
        "stats": stats,
        "pr_template": template[:6000],
        "changelog_head": head,
        "changelog_style": changelog_style(head) if head else None,
    }


def load_or_mine(
    client: httpx.Client, full_name: str, cache_dir: Path, *,
    limit: int = 20, refresh: bool = False, now: datetime | None = None,
) -> dict:
    now = now or datetime.now(UTC)
    path = Path(cache_dir) / f"{full_name.replace('/', '__')}.json"
    if path.is_file() and not refresh:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if datetime.fromisoformat(data["fetched_at"]) > now - timedelta(days=CACHE_DAYS):
                return data
        except (ValueError, KeyError):
            pass
    data = mine(client, full_name, limit=limit, now=now)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data