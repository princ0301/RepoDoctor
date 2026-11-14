"""Hygiene checks on the final diff: a deterministic gate before a human reviews it."""

import ast
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from repodoctor.tools.findings import Finding, Report

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$")
_SECRET_PATTERNS = [
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"), "GitHub token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{30,}"), "GitHub token"),
    (re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}"), "NVIDIA API key"),
    (re.compile(r"lsv2_[A-Za-z0-9_]{20,}"), "LangSmith API key"),
    (re.compile(r"sk-[A-Za-z0-9]{32,}"), "API key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "private key"),
]
_PACKAGING = {
    "pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "Pipfile", "Pipfile.lock",
    "uv.lock", "poetry.lock",
}
_CI_PREFIXES = (".github/workflows/", ".circleci/")
_CI_FILES = {".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml"}


@dataclass
class AddedLine:
    path: str
    line: int
    text: str


@dataclass
class ParsedDiff:
    files: list[str]
    deleted: list[str]
    added: list[AddedLine]
    no_eol: list[AddedLine]
    n_removed: int

    @property
    def n_added(self) -> int:
        return len(self.added)


def parse_diff(diff: str) -> ParsedDiff:
    files: list[str] = []
    deleted: list[str] = []
    added: list[AddedLine] = []
    no_eol: list[AddedLine] = []
    removed = 0
    path: str | None = None
    in_hunk = False
    new_line = 0
    last_added: AddedLine | None = None

    for raw in diff.split("\n"):    # not splitlines(): a trailing \r must survive
        header = _DIFF_HEADER.match(raw.rstrip("\r"))
        if header:
            path = header.group(2)
            in_hunk = False
            last_added = None
            if path not in files:
                files.append(path)
            continue
        hunk = _HUNK.match(raw)
        if hunk:
            in_hunk = True
            new_line = int(hunk.group(1))
            last_added = None
            continue
        if not in_hunk:
            if raw.startswith("deleted file mode") and path:
                deleted.append(path)
            continue
        if raw.startswith("+"):
            last_added = AddedLine(path or "", new_line, raw[1:])
            added.append(last_added)
            new_line += 1
        elif raw.startswith("-"):
            removed += 1
            last_added = None
        elif raw.startswith("\\"):
            if last_added is not None:
                no_eol.append(last_added)
        else:
            new_line += 1
            last_added = None
    return ParsedDiff(files, deleted, added, no_eol, removed)


def repo_line_length(root: Path, default: int = 88) -> int:
    """The line length the project itself configures (ruff, black, flake8)."""
    pp = root / "pyproject.toml"
    if pp.is_file():
        try:
            data = tomllib.loads(pp.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            data = {}
        for name in ("ruff", "black"):
            value = data.get("tool", {}).get(name, {}).get("line-length")
            if isinstance(value, int):
                return value
    for name in ("setup.cfg", "tox.ini", ".flake8"):
        f = root / name
        if f.is_file():
            m = re.search(
                r"max[-_]line[-_]length\s*=\s*(\d+)", f.read_text(encoding="utf-8", errors="ignore")
            )
            if m:
                return int(m.group(1))
    return default


def syntax_findings(root: Path, paths: list[str]) -> list[Finding]:
    out: list[Finding] = []
    for rel in paths:
        p = root / rel
        if not rel.endswith(".py") or not p.is_file():
            continue
        try:
            ast.parse(p.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as e:
            out.append(Finding("warn", "syntax_error", f"{rel}:{e.lineno}: {e.msg}"))
        except UnicodeDecodeError:
            out.append(Finding("warn", "not_utf8", f"{rel} is not valid UTF-8."))
    return out


def _where(items: list[AddedLine]) -> str:
    return f"{items[0].path}:{items[0].line}"


def check_diff(
    diff: str,
    *,
    line_length: int = 88,
    max_lines: int = 300,
    max_files: int = 8,
    secrets: tuple[str, ...] = (),
) -> Report:
    r = Report()
    p = parse_diff(diff)
    if not p.files:
        r.add("skip", "empty_diff", "The change is empty.")
        return r

    for a in p.added:
        for pattern, label in _SECRET_PATTERNS:
            if pattern.search(a.text):
                r.add("skip", "secret_in_diff", f"{a.path}:{a.line} looks like a {label}.")
        if any(s and len(s) >= 8 and s in a.text for s in secrets):
            r.add("skip", "secret_in_diff", f"{a.path}:{a.line} contains one of your secrets.")

    for f in p.files:
        if f.startswith(_CI_PREFIXES) or f in _CI_FILES:
            r.add("skip", "touches_ci", f"The change edits CI configuration ({f}).")
        elif f.rsplit("/", 1)[-1] in _PACKAGING or re.match(r"requirements.*\.txt$", f):
            r.add("warn", "touches_packaging", f"The change edits packaging or dependencies ({f}).")
        if f.endswith("test_repodoctor_repro.py"):
            r.add("warn", "generated_test_name", f"{f} still has the generated test name.")
    if p.deleted:
        r.add("warn", "deletes_files", f"The change deletes: {', '.join(p.deleted)}.")
    if len(p.files) > max_files:
        r.add("warn", "many_files", f"{len(p.files)} files changed (limit {max_files}).")
    if p.n_added + p.n_removed > max_lines:
        r.add(
            "warn", "large_change",
            f"{p.n_added + p.n_removed} changed lines (limit {max_lines}); "
            "large for an automated change.",
        )

    py = [a for a in p.added if a.path.endswith(".py")]
    body = {id(a): a.text.rstrip("\r") for a in p.added}
    trailing = [a for a in py if body[id(a)].strip() and body[id(a)] != body[id(a)].rstrip()]
    tabs = [a for a in py if "\t" in a.text]
    long_lines = [a for a in py if len(body[id(a)]) > line_length]
    crlf = [a for a in p.added if a.text.endswith("\r")]
    if trailing:
        r.add(
            "warn", "trailing_whitespace",
            f"{len(trailing)} added line(s) have trailing whitespace (first: {_where(trailing)}).",
        )
    if tabs:
        r.add(
            "warn", "tab_character",
            f"{len(tabs)} added line(s) contain a tab (first: {_where(tabs)}).",
        )
    if long_lines:
        r.add(
            "warn", "long_lines",
            f"{len(long_lines)} added line(s) exceed {line_length} characters "
            f"(first: {_where(long_lines)}).",
        )
    if crlf:
        r.add("warn", "crlf", f"{len(crlf)} added line(s) use CRLF line endings.")
    if p.no_eol:
        r.add(
            "warn", "no_newline_at_eof",
            f"{len(p.no_eol)} file(s) end without a newline (first: {p.no_eol[0].path}).",
        )
    return r

def unused_import_findings(root: Path, diff: str) -> list[Finding]:
    """Imports added by this change that nothing uses (the project's linter will say so)."""
    added: dict[str, set[int]] = {}
    for a in parse_diff(diff).added:
        added.setdefault(a.path, set()).add(a.line)
    out: list[Finding] = []
    for rel, lines in added.items():
        p = root / rel
        if not rel.endswith(".py") or rel.endswith("__init__.py") or not p.is_file():
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [(a.asname or a.name).split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                names = [a.asname or a.name for a in node.names if a.name != "*"]
            else:
                continue
            if node.lineno not in lines:
                continue
            out += [
                Finding("warn", "unused_import", f"{rel}:{node.lineno}: '{n}' is imported but not used.")
                for n in names if n not in used
            ]
    return out