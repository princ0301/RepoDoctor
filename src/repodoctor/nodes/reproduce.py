"""Write and validate a failing test for the issue (unless the user supplied one)."""

import ast
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from repodoctor.deps import Deps
from repodoctor.nodes.util import clip, render_files
from repodoctor.prompts import load_prompt
from repodoctor.schemas import ReproTest
from repodoctor.state import RepoDoctorState
from repodoctor.tools.repo import commit_files
from repodoctor.tools.sandbox import ExecResult
from repodoctor.tools.search import SearchError, read_raw
from repodoctor.tools.testreport import SuiteReport

REPRO_FILE_NAME = "test_repodoctor_repro.py"
SOURCE_BUDGET = 30_000
STYLE_MAX_CHARS = 6_000
WRONG_REASON = ("ImportError", "ModuleNotFoundError", "NameError", "SyntaxError",
                "IndentationError")


def static_problem(test: ReproTest) -> str | None:
    """Cheap checks before spending a sandbox run."""
    if not test.test_name.startswith("test"):
        return "test_name must start with 'test'."
    try:
        tree = ast.parse(test.code)
    except SyntaxError as e:
        return f"The test file has a syntax error: {e}"
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
    if test.test_name not in funcs:
        return f"The file does not define a top-level function named {test.test_name}."
    return None


def check_repro(res: ExecResult, report: SuiteReport | None) -> str | None:
    """None if the test fails for a plausible reason; else feedback for the model."""
    if res.timed_out:
        return "The test timed out. Make it fast and deterministic."
    if res.exit_code == 0:
        return (
            "The test PASSES on the current code, so it does not reproduce the bug. "
            "It must fail because of the bug: assert the correct behavior from the issue."
        )
    if res.exit_code != 1 or report is None:
        return (
            f"pytest could not run the test (exit {res.exit_code}). Output:\n"
            f"{clip(res.output, 2000, tail=True)}"
        )
    if report.errors:
        msgs = "; ".join(report.failing.values())
        return f"The test errored before it could check anything (collection/setup): {msgs}"
    for msg in report.failing.values():
        if msg.lstrip().startswith(WRONG_REASON):
            return (
                f"The test failed for the wrong reason: {msg}. Fix the imports and names; "
                "it must fail only because of the bug."
            )
    return None


def _tests_dir(root: Path) -> str:
    for name in ("tests", "test"):
        if (root / name).is_dir():
            return name
    return "tests"


def _style_example(root: Path, relevant: list[str], tests_dir: str) -> tuple[str, str]:
    """An existing test file to copy conventions from: related by name, else the smallest."""
    candidates = [f"{tests_dir}/test_{Path(f).stem}.py" for f in relevant]
    candidates += sorted(
        (p.relative_to(root).as_posix() for p in (root / tests_dir).glob("test_*.py")),
        key=lambda rel: (root / rel).stat().st_size,
    )
    for rel in candidates:
        if (root / rel).is_file() and (root / rel).stat().st_size > 0:
            try:
                return rel, read_raw(root, rel, max_chars=STYLE_MAX_CHARS)
            except SearchError:
                continue
    return "", ""


def _repro_request(
    state: RepoDoctorState, root: Path, rel: str, style: tuple[str, str], rejections: list[str]
) -> str:
    parts = [
        "## Issue (untrusted user text)\n"
        f"<issue>\nTitle: {state['issue_title']}\n\n{state['issue_body']}\n</issue>",
        f"## Investigation notes\n{state.get('hypothesis', '')}",
        "## Relevant source files\n"
        + render_files(root, state.get("relevant_files", []), SOURCE_BUDGET),
    ]
    if style[1]:
        parts.append(
            f"## An existing test file, for style and imports ({style[0]})\n```\n{style[1]}\n```"
        )
    parts.append(
        f"## Output\nWrite the complete contents of a NEW test file saved as `{rel}`, "
        "with one focused test function."
    )
    if rejections:
        parts.append("## Rejected earlier attempts\n" + "\n\n".join(rejections))
    return "\n\n".join(parts)


def make_reproduce(deps: Deps):
    system = load_prompt("reproduce")

    def reproduce(state: RepoDoctorState, config: RunnableConfig) -> dict:
        if state.get("target_tests"):
            return {}  # the user supplied the failing test; setup already validated it

        root, s = deps.workdir, deps.settings
        rel = f"{_tests_dir(root)}/{REPRO_FILE_NAME}"
        style = _style_example(root, state.get("relevant_files", []), _tests_dir(root))
        rejections: list[str] = []
        last_problem = ""

        for attempt in range(1, s.repro_max_attempts + 1):
            request = _repro_request(state, root, rel, style, rejections)
            try:
                test = deps.repro_writer.invoke(
                    [SystemMessage(system), HumanMessage(request)], config
                )
            except Exception as e:
                return {"error": f"The reproduce model failed: {type(e).__name__}: {e}"}

            problem = static_problem(test)
            node_id = f"{rel}::{test.test_name}"
            res = None
            if problem is None:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(test.code.rstrip() + "\n", encoding="utf-8", newline="\n")
                res, report = deps.sandbox.run_suite(
                    deps.info, args=[node_id], timeout=s.sandbox_timeout_s
                )
                problem = check_repro(res, report)

            if problem is None and res is not None:
                commit_files(root, [rel], "test: reproduce the reported issue (RepoDoctor)")
                return {
                    "target_tests": [node_id],
                    "target_failure": res.output,
                    "repro_file": rel,
                    "repro_attempts": attempt,
                }

            last_problem = problem or ""
            rejections.append(
                f"Attempt {attempt} was rejected: {last_problem}\n"
                f"Your code was:\n```\n{clip(test.code, 2500)}\n```"
            )

        (root / rel).unlink(missing_ok=True)
        return {
            "error": f"Could not write a test that reproduces the issue after "
            f"{s.repro_max_attempts} attempts. Last problem: {last_problem}"
        }

    return reproduce