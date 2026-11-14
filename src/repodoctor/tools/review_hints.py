"""Hints for the human reviewer about project conventions the agent does not know."""

import ast
import math
import re
from pathlib import Path

from repodoctor.tools.findings import Finding

_CHANGELOG = re.compile(r"^(changelog|changes|history|news)(\.\w+)?$", re.I)
MIN_NAME_LEN = 4


def find_changelog(root: Path) -> str | None:
    for folder in (root, root / "docs"):
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.is_file() and _CHANGELOG.match(p.name):
                    return p.relative_to(root).as_posix()
    return None


def changelog_findings(root: Path, changed: list[str]) -> list[Finding]:
    name = find_changelog(root)
    if name is None or name in changed:
        return []
    return [Finding(
        "warn", "changelog_not_updated",
        f"The project keeps a changelog ({name}) and this change does not touch it. "
        "Check whether its contributing guide expects an entry.",
    )]


def _called_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                names.add(f.attr)
            elif isinstance(f, ast.Name):
                names.add(f.id)
    return {n for n in names if len(n) >= MIN_NAME_LEN and not n.startswith("test_")}


def suggest_test_homes(
    root: Path, new_test: str, limit: int = 3
) -> list[tuple[str, list[str]]]:
    """Existing test files that call the same functions as the new test (rarer names count more)."""
    path = root / new_test
    if not path.is_file():
        return []
    wanted = _called_names(path.read_text(encoding="utf-8", errors="ignore"))
    if not wanted:
        return []
    texts = {
        p: p.read_text(encoding="utf-8", errors="ignore")
        for p in sorted(root.rglob("test_*.py"))
        if p != path and ".git" not in p.parts
    }
    n = len(texts) or 1
    df = {w: sum(1 for t in texts.values() if f"{w}(" in t) for w in wanted}
    scored: list[tuple[float, str, list[str]]] = []
    for p, text in texts.items():
        shared = {w: min(text.count(f"{w}("), 10) for w in wanted if f"{w}(" in text}
        if shared:
            score = sum(c * math.log(1 + n / df[w]) for w, c in shared.items())
            top = sorted(shared, key=lambda w: -shared[w])[:3]
            scored.append((score, p.relative_to(root).as_posix(), top))
    scored.sort(key=lambda s: -s[0])
    return [(rel, names) for _, rel, names in scored[:limit]]


def hint_new_test_files(root: Path, new_tests: list[str]) -> list[Finding]:
    out: list[Finding] = []
    for t in new_tests:
        homes = suggest_test_homes(root, t)
        if homes:
            shown = "; ".join(f"{rel} (calls: {', '.join(names)})" for rel, names in homes)
            out.append(Finding(
                "warn", "new_test_file",
                f"The test is in a new file ({t}); the project already tests the same "
                f"functions in: {shown}. Tests are usually kept with the existing ones.",
            ))
    return out