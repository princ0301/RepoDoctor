"""Scan contribution docs for mentions of AI, so a human reads them before any PR."""

import re
from pathlib import Path

POLICY_FILES = (
    "CONTRIBUTING.md",
    "CONTRIBUTING.rst",
    ".github/CONTRIBUTING.md",
    "docs/CONTRIBUTING.md",
    "docs/contributing.md",
    "AI_POLICY.md",
    "AI-POLICY.md",
    ".github/AI_POLICY.md",
)
_AI = re.compile(
    r"\bAI\b|\bLLMs?\b|large language model|copilot|chatgpt|generative"
    r"|machine[- ]generated",
    re.I,
)
MAX_FILE_BYTES = 200_000


def policy_mentions(root: Path, max_hits: int = 8) -> list[str]:
    """Lines like 'CONTRIBUTING.md:12: text' that mention AI/LLMs."""
    hits: list[str] = []
    for rel in POLICY_FILES:
        p = root / rel
        if not p.is_file() or p.stat().st_size > MAX_FILE_BYTES:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if _AI.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:200]}")
                if len(hits) >= max_hits:
                    return hits
    return hits