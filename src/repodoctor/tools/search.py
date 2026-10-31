import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from repodoctor.tools.paths import safe_resolve

DEFAULT_MAX_RESULTS = 50
DEFAULT_MAX_LINES = 400
MAX_HIT_CHARS = 200
MAX_READ_LINE_CHARS = 1000
MAX_FILE_BYTES = 1_000_000
GIT_TIMEOUT_S = 30

_SKIP_NAMES = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".venv", ".tox", "node_modules",
}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HIT = re.compile(r"^(.+?):(\d+):(.*)$")
_OUTSIDE = "path is outside the repository or inside .git"

class SearchError(ValueError):
    """Bad input or failed search. The message is meant to be shown to an LLM."""

@dataclass
class GrepHit:
    path: str
    line: int
    text: str

@dataclass
class GrepResult:
    hits: list[GrepHit] = field(default_factory=list)
    truncated: bool = False

def _resolve(root: Path, path: str) -> Path:
    p = safe_resolve(root, path)
    if p is None:
        raise SearchError(f"{path!r}: {_OUTSIDE}")
    return p

def _run_grep(
    root: Path,
    pattern: str,
    pathspecs: list[str],
    *,
    fixed: bool,
    ignore_case: bool,
    max_results: int,
) -> GrepResult:
    if not pattern:
        raise SearchError("pattern must not be empty")
    args = [
        "git", "-c", "core.quotepath=false", "grep",
        "-n", "-I", "--no-color", "--untracked", "-F" if fixed else "-E",
    ]
    if ignore_case:
        args.append("-i")
    args += ["-e", pattern, "--", *pathspecs]
    try:
        p = subprocess.run(
            args, cwd=root, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=GIT_TIMEOUT_S,
        )
    except FileNotFoundError as e:
        raise SearchError("git was not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise SearchError(
            f"search timed out after {GIT_TIMEOUT_S}s; use a more specific pattern or path"
        ) from e

    if p.returncode == 1:
        return GrepResult()
    if p.returncode != 0:
        kind = "string" if fixed else "extended regex"
        raise SearchError(f"search failed (is the pattern a valid {kind}?): {p.stderr.strip()}")

    result = GrepResult()
    for raw in p.stdout.split("\n"):
        m = _HIT.match(raw)
        if not m:
            continue
        if len(result.hits) >= max_results:
            result.truncated = True
            break
        text = m[3].rstrip()
        if len(text) > MAX_HIT_CHARS:
            text = text[:MAX_HIT_CHARS] + "…"
        result.hits.append(GrepHit(m[1], int(m[2]), text))
    return result

def grep(
    root: Path,
    pattern: str,
    *,
    path: str = ".",
    fixed: bool = False,
    ignore_case: bool = False,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> GrepResult:
    """Search tracked and untracked (non-ignored) text files. Extended regex by default."""
    root = root.resolve()
    target = _resolve(root, path)
    if not target.exists():
        raise SearchError(f"path not found: {path}")
    scope = target.relative_to(root).as_posix()
    return _run_grep(
        root, pattern, [scope], fixed=fixed, ignore_case=ignore_case, max_results=max_results
    )

def find_symbol(root: Path, name: str, *, max_results: int = 20) -> GrepResult:
    """Find Python definitions (def / async def / class) of an exact name."""
    if not _IDENT.match(name):
        raise SearchError(f"{name!r} is not a plain Python identifier")
    pattern = (
        r"^[[:space:]]*(async[[:space:]]+def|def|class)[[:space:]]+"
        + name
        + r"([^[:alnum:]_]|$)"
    )
    return _run_grep(
        root.resolve(), pattern, ["*.py"],
        fixed=False, ignore_case=False, max_results=max_results,
    )

def format_grep(result: GrepResult) -> str:
    if not result.hits:
        return "No matches."
    lines = [f"{h.path}:{h.line}: {h.text}" for h in result.hits]
    if result.truncated:
        lines.append("[... more matches not shown; narrow with `path` or a more specific pattern ...]")
    return "\n".join(lines)

def read_file(
    root: Path,
    path: str,
    *,
    start: int = 1,
    end: int | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
) -> str:
    """Return numbered lines ('  12| code'). The prefix is NOT part of the file content."""
    root = root.resolve()
    p = _resolve(root, path)
    if not p.is_file():
        raise SearchError(f"file not found: {path}")
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise SearchError(f"{path}: file too large ({size} bytes); use grep to locate code instead")
    data = p.read_bytes()
    if b"\x00" in data[:8192]:
        raise SearchError(f"{path}: binary file")

    lines = data.decode("utf-8", errors="replace").replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    total = len(lines)
    rel = p.relative_to(root).as_posix()
    if total == 0:
        return f"# {rel} (empty file)"

    start = max(start, 1)
    if start > total:
        raise SearchError(f"{rel}: start line {start} is past the end of the file ({total} lines)")
    stop = total if end is None else min(end, total)
    if stop < start:
        raise SearchError(f"{rel}: end line {end} is before start line {start}")
    shown_stop = min(stop, start + max_lines - 1)

    width = len(str(shown_stop))
    body = []
    for i in range(start, shown_stop + 1):
        text = lines[i - 1]
        if len(text) > MAX_READ_LINE_CHARS:
            text = text[:MAX_READ_LINE_CHARS] + " [...line truncated]"
        body.append(f"{i:>{width}}| {text}")

    out = f"# {rel} (lines {start}-{shown_stop} of {total})\n" + "\n".join(body)
    if shown_stop < stop:
        out += f"\n[... more lines; call again with start={shown_stop + 1} ...]"
    return out

def list_dir(root: Path, path: str = ".", *, max_entries: int = 200) -> str:
    """Directories first (with trailing '/'), then files. Skips caches and VCS folders."""
    root = root.resolve()
    p = _resolve(root, path)
    if not p.is_dir():
        raise SearchError(f"not a directory: {path}")
    entries = sorted(
        (e for e in p.iterdir() if e.name not in _SKIP_NAMES and not e.name.endswith(".egg-info")),
        key=lambda e: (not e.is_dir(), e.name.lower()),
    )
    if not entries:
        return "(empty directory)"
    names = [e.name + "/" if e.is_dir() else e.name for e in entries]
    out = "\n".join(names[:max_entries])
    if len(names) > max_entries:
        out += f"\n[... {len(names) - max_entries} more entries not shown ...]"
    return out