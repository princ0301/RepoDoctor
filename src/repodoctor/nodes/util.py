from pathlib import Path

from repodoctor.tools.search import SearchError, read_raw


def clip(text: str, limit: int, *, tail: bool = False) -> str:
    """Shorten text. tail=True keeps the end (where test failures are summarized)."""
    if len(text) <= limit:
        return text
    if tail:
        return f"[... truncated ...]\n{text[-limit:]}"
    return f"{text[:limit]}\n[... truncated ...]"


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