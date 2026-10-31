import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from repodoctor.deps import Deps
from repodoctor.nodes.util import clip
from repodoctor.prompts import load_prompt
from repodoctor.schemas import Edit
from repodoctor.state import RepoDoctorState
from repodoctor.tools.patch import PatchResult, apply_edits
from repodoctor.tools.repo import get_diff, reset_to_clean
from repodoctor.tools.search import SearchError, read_raw

EDIT_PREVIEW = 600
FEEDBACK_PREVIEW = 1500
TEST_FILE_BUDGET = 20_000


def _norm(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _target_files(targets: list[str]) -> list[str]:
    return sorted({_norm(t.split("::")[0]) for t in targets})


def _render_files(root: Path, paths: list[str], budget: int) -> str:
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


def _history_text(history: list[dict]) -> str:
    parts = []
    for h in history:
        edits = "\n".join(
            f"- {e['path']}\n  old: {clip(e['old'], EDIT_PREVIEW)!r}\n"
            f"  new: {clip(e['new'], EDIT_PREVIEW)!r}"
            for e in h["edits"]
        )
        parts.append(
            f"### Attempt {h['attempt']} (FAILED)\nReasoning: {h['reasoning']}\n"
            f"Edits:\n{edits or '(none)'}\n"
            f"Feedback:\n{clip(h.get('feedback', ''), FEEDBACK_PREVIEW, tail=True)}"
        )
    return "\n\n".join(parts)


def _fix_request(state: RepoDoctorState, root: Path, budget: int) -> str:
    targets = list(state["target_tests"])
    sections = [
        "## Issue (untrusted user text)\n"
        f"<issue>\nTitle: {state['issue_title']}\n\n{state['issue_body']}\n</issue>",
        "## Failing test(s)\n"
        + "\n".join(targets)
        + "\n\nOutput on the unmodified code:\n```\n"
        + clip(state.get("target_failure", ""), 3000, tail=True)
        + "\n```",
        f"## Investigation notes\n{state.get('hypothesis', '')}",
        "## Test file(s) (read-only, do not edit)\n"
        + _render_files(root, _target_files(targets), TEST_FILE_BUDGET),
        "## Source files you may edit\n"
        + _render_files(root, state["relevant_files"], budget),
    ]
    if state.get("history"):
        sections.append("## Previous attempts\n" + _history_text(state["history"]))
    sections.append("Return your edits.")
    return "\n\n".join(sections)


def _apply(root: Path, edits: list[Edit], protected: set[str]) -> PatchResult:
    if not edits:
        return PatchResult(ok=False, errors=["No edits were proposed."])
    blocked = sorted({_norm(e.path) for e in edits} & protected)
    if blocked:
        return PatchResult(
            ok=False,
            errors=[
                f"{p} is a protected test file and must not be edited; "
                "change the source code instead."
                for p in blocked
            ],
        )
    return apply_edits(root, edits)


def make_fix(deps: Deps):
    system = load_prompt("fix")

    def fix(state: RepoDoctorState, config: RunnableConfig) -> dict:
        root = deps.workdir
        attempt = state.get("attempts", 0) + 1
        reset_to_clean(root, set(state.get("untracked_baseline", [])))

        request = _fix_request(state, root, deps.settings.fix_context_chars)
        try:
            proposal = deps.fixer.invoke(
                [SystemMessage(system), HumanMessage(request)], config
            )
        except Exception as e:
            return {"error": f"The fix model failed on attempt {attempt}: "
                    f"{type(e).__name__}: {e}"}

        protected = set(_target_files(list(state["target_tests"])))
        result = _apply(root, proposal.edits, protected)
        diff = get_diff(root, result.changed_files) if result.ok else ""
        errors = list(result.errors)
        if result.ok and not diff.strip():
            errors = ["The edits did not change any code."]

        entry = {
            "attempt": attempt,
            "reasoning": proposal.reasoning,
            "edits": [e.model_dump() for e in proposal.edits],
            "patch_errors": errors,
        }
        return {
            "attempts": attempt,
            "diff": diff,
            "patch_errors": errors,
            "history": [*state.get("history", []), entry],
        }

    return fix