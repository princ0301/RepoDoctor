from pathlib import Path

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from repodoctor.tools.paths import safe_resolve
from repodoctor.tools.repo import GitError, run_git
from repodoctor.tools.search import SearchError, read_raw


def clip(text: str, limit: int, *, tail: bool = False) -> str:
    """Shorten text. tail=True keeps the end (where test failures are summarized)."""
    if len(text) <= limit:
        return text
    if tail:
        return f"[... truncated ...]\n{text[-limit:]}"
    return f"{text[:limit]}\n[... truncated ...]"


def is_malformed_output(e: Exception) -> bool:
    """The model answered but not in the requested structure: its mistake, not an outage."""
    return isinstance(e, ValidationError | OutputParserException)


def _tracked_files(root: Path) -> list[str]:
    try:
        return run_git("ls-files", cwd=root).splitlines()
    except GitError:
        return []


def repo_file_list(root: Path, max_files: int = 200, max_chars: int = 6000) -> str:
    """Tracked Python files (source first, tests last), so a model can name real paths."""

    def is_test(f: str) -> bool:
        parts = f.split("/")
        return "tests" in parts[:-1] or "test" in parts[:-1] or parts[-1].startswith("test_")

    files = sorted(
        (f for f in _tracked_files(root) if f.endswith(".py")), key=lambda f: (is_test(f), f)
    )
    shown: list[str] = []
    used = 0
    for f in files[:max_files]:
        if used + len(f) + 1 > max_chars:
            break
        shown.append(f)
        used += len(f) + 1
    text = "\n".join(shown)
    if len(shown) < len(files):
        text += f"\n[... {len(files) - len(shown)} more Python files not listed ...]"
    return text


def existing_files(root: Path, paths: list[str], limit: int = 5) -> list[str]:
    """Paths that exist inside the repo, normalized and de-duplicated.

    A path that does not exist is still accepted when exactly one tracked file ends with
    it (models often drop a leading 'src/').
    """
    tracked: list[str] | None = None
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        wanted = raw.strip().replace("\\", "/").removeprefix("./").lstrip("/")
        if not wanted:
            continue
        p = safe_resolve(root, wanted)
        if p is not None and p.is_file():
            rel = p.relative_to(root.resolve()).as_posix()
        else:
            if tracked is None:
                tracked = _tracked_files(root)
            hits = [t for t in tracked if t == wanted or t.endswith("/" + wanted)]
            if len(hits) != 1:
                continue
            rel = hits[0]
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out[:limit]


def render_files(root: Path, paths: list[str], budget: int) -> tuple[str, list[str]]:
    """Files as fenced blocks within a character budget; also returns the omitted paths."""
    chunks: list[str] = []
    omitted: list[str] = []
    used = 0
    for p in paths:
        try:
            text = read_raw(root, p)
        except SearchError as e:
            chunks.append(f"### {p}\n(unavailable: {e})")
            continue
        block = f"### {p}\n```\n{text}\n```"
        if chunks and used + len(block) > budget:
            omitted.append(p)
            chunks.append(
                f"### {p}\n(NOT SHOWN: the context budget is exhausted. If you cannot "
                "proceed without it, list it in need_more_files.)"
            )
            continue
        chunks.append(block)
        used += len(block)
    return "\n\n".join(chunks), omitted
