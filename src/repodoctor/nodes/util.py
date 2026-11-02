from pathlib import Path

from repodoctor.tools.paths import safe_resolve
from repodoctor.tools.search import SearchError, read_raw


def clip(text: str, limit: int, *, tail: bool = False) -> str:
    """Shorten text. tail=True keeps the end (where test failures are summarized)."""
    if len(text) <= limit:
        return text
    if tail:
        return f"[... truncated ...]\n{text[-limit:]}"
    return f"{text[:limit]}\n[... truncated ...]"


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


def render_files(root: Path, paths: list[str], budget: int) -> str:
    """Files as fenced blocks (raw text, no line numbers) within a character budget."""
    chunks: list[str] = []
    used = 0
    for p in paths:
        try:
            text = read_raw(root, p)
        except SearchError as e:
            chunks.append(f"### {p}\n(unavailable: {e})")
            continue
        block = f"### {p}\n```\n{text}\n```"
        if chunks and used + len(block) > budget:
            chunks.append(f"### {p}\n(omitted: context budget exhausted)")
            continue
        chunks.append(block)
        used += len(block)
    return "\n\n".join(chunks)
