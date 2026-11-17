from repodoctor.tools.diffcheck import parse_diff  # noqa: F401  (kept for clarity of the diff helper)
from repodoctor.tools.lint import (
    LintTool,
    check_command,
    detect_lint_tools,
    install_command,
    parse_lint_output,
    parse_precommit,
    run_lint,
)
from repodoctor.tools.repo import commit_files, get_diff_since, head_sha
from repodoctor.tools.sandbox import ExecResult

PRECOMMIT = """repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
"""


def test_parse_precommit_reads_repos_revs_and_hooks():
    repos = parse_precommit(PRECOMMIT)
    assert [(r["url"].rsplit("/", 1)[-1], r["rev"], r["hooks"]) for r in repos] == [
        ("ruff-pre-commit", "v0.4.4", ["ruff", "ruff-format"]),
        ("pre-commit-hooks", "v4.6.0", ["trailing-whitespace"]),
        ("mirrors-mypy", "v1.10.0", ["mypy"]),
    ]


def test_tools_come_from_the_pinned_precommit_config(tmp_path):
    (tmp_path / ".pre-commit-config.yaml").write_text(PRECOMMIT)
    tools, not_run = detect_lint_tools(tmp_path)
    assert [(t.name, t.version, t.kind) for t in tools] == [
        ("ruff-check", "0.4.4", "lint"), ("ruff-format", "0.4.4", "format"),
    ]
    assert not_run == ["mypy"]
    assert install_command(tools) == "python -m pip install -q ruff==0.4.4"


def test_without_precommit_the_tool_config_is_used(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n\n[tool.black]\n")
    tools, not_run = detect_lint_tools(tmp_path)
    assert sorted(t.name for t in tools) == ["black", "ruff-check"] and not_run == []
    assert all(t.version is None for t in tools)
    assert detect_lint_tools(tmp_path / "missing")[0] == []


def test_commands_and_output_parsing():
    assert "ruff check --no-fix a.py" in check_command(LintTool("ruff-check", "ruff", None, "lint"), ["a.py"])
    assert check_command(LintTool("black", "black", None, "format"), ["a b.py"]).endswith("'a b.py'")
    rows = parse_lint_output(
        "tests/test_x.py:3:1: F401 `os` imported but unused\n./pkg/a.py:10: E501 long\nFound 2 errors.\n"
    )
    assert rows == [("tests/test_x.py", 3, "F401 `os` imported but unused"), ("pkg/a.py", 10, "E501 long")]


class LintBox:
    """A sandbox whose commands answer from scripted (exit code, output) queues."""

    def __init__(self, replies):
        self.replies = replies
        self.commands: list[str] = []

    def run(self, cmd, timeout):
        self.commands.append(cmd)
        for key, queue in self.replies.items():
            if key in cmd:
                code, out = queue.pop(0) if len(queue) > 1 else queue[0]
                return ExecResult(code, out, False, 0.0)
        return ExecResult(0, "", False, 0.0)


def test_run_lint_keeps_only_what_this_change_introduced(make_git_repo):
    root = make_git_repo({"a.py": "x = 1\n", "b.py": "y = 1\n"})
    base = head_sha(root)
    (root / "a.py").write_bytes(b"import os\nx = 1\n")
    (root / "b.py").write_bytes(b"y = 2\n")
    (root / "c.py").write_bytes(b"z = 1\n")
    head = commit_files(root, ["a.py", "b.py", "c.py"], "change")
    diff = get_diff_since(root, base)
    tools = [LintTool("ruff-check", "ruff", None, "lint"), LintTool("black", "black", None, "format")]
    box = LintBox({
        "ruff check": [(1, "a.py:1:1: F401 `os` imported but unused\na.py:2:1: E501 old\n")],
        # a.py fails now but passed at base; b.py failed at base too; c.py is new
        "black": [(1, ""), (0, ""), (1, ""), (1, ""), (1, "")],
    })
    found = run_lint(box, root, tools, ["a.py", "b.py", "c.py"], diff, base, head)
    assert [(f.code, f.message) for f in found] == [
        ("lint", "a.py:1: F401 `os` imported but unused [ruff-check]"),
        ("format", "a.py: black would reformat this file (it passes at the base commit)."),
        ("format", "c.py: black would reformat this file (a new file)."),
    ]
    assert (root / "a.py").read_bytes() == b"import os\nx = 1\n"   # restored after the base check


def test_a_tool_that_cannot_run_is_reported_not_fatal(make_git_repo):
    root = make_git_repo({"a.py": "x = 1\n"})
    base = head_sha(root)
    (root / "a.py").write_bytes(b"x = 2\n")
    head = commit_files(root, ["a.py"], "change")
    box = LintBox({"flake8": [(2, "boom")]})
    found = run_lint(box, root, [LintTool("flake8", "flake8", None, "lint")], ["a.py"],
                     get_diff_since(root, base), base, head)
    assert [f.code for f in found] == ["lint_unavailable"] and found[0].level == "info"