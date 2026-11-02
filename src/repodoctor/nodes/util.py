from pathlib import Path

from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from repodoctor.tools.paths import safe_resolve
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


def existing_files(root: Path, paths: list[str], limit: int = 5) -> list[str]:
    """Paths that exist inside the repo, normalized and de-duplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in paths:
        p = safe_resolve(root, raw.strip())
        if p is None or not p.is_file():
            continue
        rel = p.relative_to(root.resolve()).as_posix()
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
