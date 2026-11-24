"""Run the linters a project itself uses (the ones its pre-commit config pins) on a change."""

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from repodoctor.tools.diffcheck import parse_diff
from repodoctor.tools.findings import Finding
from repodoctor.tools.repo import GitError, run_git


@dataclass(frozen=True)
class LintTool:
    name: str            # ruff-check | ruff-format | black | flake8 | isort
    package: str
    version: str | None
    kind: str            # "lint": problems on lines; "format": says a file would change


_HOOKS: dict[str, tuple[str, dict[str, tuple[str, str]]]] = {
    "ruff-pre-commit": ("ruff", {
        "ruff": ("ruff-check", "lint"), "ruff-check": ("ruff-check", "lint"),
        "ruff-format": ("ruff-format", "format"),
    }),
    "black": ("black", {"black": ("black", "format")}),
    "black-pre-commit-mirror": ("black", {"black": ("black", "format")}),
    "flake8": ("flake8", {"flake8": ("flake8", "lint")}),
    "isort": ("isort", {"isort": ("isort", "format")}),
}
_LINT_LINE = re.compile(r"^(?P<path>[^\s:][^:]*):(?P<line>\d+):(?:\d+:)?\s*(?P<msg>\S.*)$")
_MODE_RULES = ("EXE",)   # depend on file modes, which a Windows bind mount reports wrongly


def exists_at(root: Path, rev: str, rel: str) -> bool:
    try:
        run_git("cat-file", "-e", f"{rev}:{rel}", cwd=root)
        return True
    except GitError:
        return False


def _version(rev: str | None) -> str | None:
    if not rev:
        return None
    v = rev.lstrip("v")
    return v if re.fullmatch(r"\d+(\.\d+)*", v) else None


def parse_precommit(text: str) -> list[dict]:
    """repos with their url, rev and hook ids (a small reader: no YAML dependency)."""
    repos: list[dict] = []
    cur: dict | None = None
    for raw in text.splitlines():
        s = raw.strip()
        if m := re.match(r"-?\s*repo:\s*(\S+)", s):
            cur = {"url": m.group(1), "rev": None, "hooks": []}
            repos.append(cur)
        elif cur is None:
            continue
        elif m := re.match(r"rev:\s*['\"]?([^'\"\s#]+)", s):
            cur["rev"] = m.group(1)
        elif m := re.match(r"-?\s*id:\s*(\S+)", s):
            cur["hooks"].append(m.group(1))
    return repos


def _from_tool_config(root: Path) -> dict[str, LintTool]:
    def read(name: str) -> str:
        p = root / name
        return p.read_text(encoding="utf-8", errors="ignore") if p.is_file() else ""

    pyproject = read("pyproject.toml")
    out: dict[str, LintTool] = {}
    if (
        re.search(r"^\[tool\.ruff", pyproject, re.M)
        or (root / "ruff.toml").is_file() or (root / ".ruff.toml").is_file()
    ):
        out["ruff-check"] = LintTool("ruff-check", "ruff", None, "lint")
    if re.search(r"^\[tool\.black\]", pyproject, re.M):
        out["black"] = LintTool("black", "black", None, "format")
    if re.search(r"^\[tool\.isort\]", pyproject, re.M):
        out["isort"] = LintTool("isort", "isort", None, "format")
    if (root / ".flake8").is_file() or re.search(
        r"^\[flake8\]", read("setup.cfg") + read("tox.ini"), re.M
    ):
        out["flake8"] = LintTool("flake8", "flake8", None, "lint")
    return out


def detect_lint_tools(root: Path) -> tuple[list[LintTool], list[str]]:
    """(tools we can run, other pre-commit hooks we cannot run here)."""
    tools: dict[str, LintTool] = {}
    not_run: list[str] = []
    cfg = root / ".pre-commit-config.yaml"
    if cfg.is_file():
        for repo in parse_precommit(cfg.read_text(encoding="utf-8", errors="ignore")):
            url = repo["url"]
            if "pre-commit-hooks" in url:
                continue   # whitespace and end-of-file fixers: the diff checks cover them
            key = url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
            package, hooks = _HOOKS.get(key, ("", {}))
            for hook in repo["hooks"]:
                if hook in hooks:
                    name, kind = hooks[hook]
                    tools[name] = LintTool(name, package, _version(repo["rev"]), kind)
                else:
                    not_run.append(hook)
    if not tools:
        tools.update(_from_tool_config(root))
    return list(tools.values()), not_run


def install_command(tools: list[LintTool]) -> str:
    specs = sorted({f"{t.package}=={t.version}" if t.version else t.package for t in tools})
    return "python -m pip install -q " + " ".join(shlex.quote(s) for s in specs)


def check_command(tool: LintTool, files: list[str]) -> str:
    quoted = " ".join(shlex.quote(f) for f in files)
    commands = {
        "ruff-check": f"RUFF_OUTPUT_FORMAT=concise python -m ruff check --no-fix {quoted}",
        "ruff-format": f"python -m ruff format --check {quoted}",
        "black": f"python -m black --check -q {quoted}",
        "flake8": f"python -m flake8 {quoted}",
        "isort": f"python -m isort --check-only -q {quoted}",
    }
    return commands[tool.name]


def parse_lint_output(output: str) -> list[tuple[str, int, str]]:
    rows = []
    for raw in output.splitlines():
        m = _LINT_LINE.match(raw.strip())
        if m:
            rows.append((m.group("path").removeprefix("./"), int(m.group("line")), m.group("msg").strip()))
    return rows


def _passes_at_base(sb, root: Path, tool: LintTool, rel: str, base: str, head: str, timeout: int):
    """Did this file pass the formatter at the base commit? None if it did not exist there."""
    if not exists_at(root, base, rel):
        return None
    run_git("checkout", base, "--", rel, cwd=root)
    try:
        res = sb.run(check_command(tool, [rel]), timeout)
    finally:
        run_git("checkout", head, "--", rel, cwd=root)
    return res.exit_code == 0


def run_lint(
    sb, root: Path, tools: list[LintTool], files: list[str], diff: str,
    base: str, head: str, timeout: int = 180,
) -> list[Finding]:
    """Problems this change introduced: linter output on added lines, formatter failures on
    files that passed before."""
    py = [f for f in files if f.endswith(".py") and (root / f).is_file()]
    if not tools or not py:
        return []
    added = {(a.path, a.line) for a in parse_diff(diff).added}
    out: list[Finding] = []
    for tool in tools:
        if tool.kind == "lint":
            res = sb.run(check_command(tool, py), timeout)
            if res.timed_out or res.exit_code not in (0, 1):
                out.append(Finding(
                    "info", "lint_unavailable",
                    f"{tool.name} could not run (exit {res.exit_code}); run it yourself.",
                ))
                continue
            hits = [
                (p, n, m) for p, n, m in parse_lint_output(res.output)
                if (p, n) in added and not m.startswith(_MODE_RULES)
            ]
            out += [Finding("warn", "lint", f"{p}:{n}: {m} [{tool.name}]") for p, n, m in hits[:10]]
            continue
        for f in py:
            res = sb.run(check_command(tool, [f]), timeout)
            if res.timed_out or res.exit_code not in (0, 1):
                out.append(Finding(
                    "info", "lint_unavailable",
                    f"{tool.name} could not run on {f} (exit {res.exit_code}).",
                ))
                continue
            if res.exit_code == 0:
                continue
            before = _passes_at_base(sb, root, tool, f, base, head, timeout)
            if before is False:
                continue   # it already failed before this change
            note = "it passes at the base commit" if before else "a new file"
            out.append(Finding(
                "warn", "format", f"{f}: {tool.name} would reformat this file ({note})."
            ))
    return out