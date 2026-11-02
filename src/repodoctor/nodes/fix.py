import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from repodoctor.deps import Deps
from repodoctor.nodes.util import clip, existing_files, render_files
from repodoctor.prompts import load_prompt
from repodoctor.schemas import Edit
from repodoctor.state import RepoDoctorState
from repodoctor.tools.patch import PatchResult, apply_edits
from repodoctor.tools.repo import get_diff, reset_to_clean

EDIT_PREVIEW = 600
FEEDBACK_PREVIEW = 1500
TEST_FILE_BUDGET = 20_000
MAX_CONTEXT_FILES = 8


def _norm(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _target_files(targets: list[str]) -> list[str]:
    return sorted({_norm(t.split("::")[0]) for t in targets})


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
        + render_files(root, _target_files(targets), TEST_FILE_BUDGET),
        "## Source files you may edit\n"
        + render_files(root, state["relevant_files"], budget),
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


def _no_edits_message(added: list[str]) -> str:
    if added:
        return (
            "No edits were proposed. The files you asked for were added to your context "
            f"for the next attempt: {', '.join(added)}."
        )
    return (
        "No edits were proposed. You must return edits. If you need to see another file, "
        "list its path in need_more_files."
    )


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

        relevant = list(state["relevant_files"])
        protected = set(_target_files(list(state["target_tests"])))
        result = _apply(root, proposal.edits, protected)
        errors = list(result.errors)
        if not proposal.edits:
            wanted = existing_files(root, proposal.need_more_files)
            added = [f for f in wanted if f not in relevant]
            relevant = [*relevant, *added][:MAX_CONTEXT_FILES]
            errors = [_no_edits_message(added)]

        diff = get_diff(root, result.changed_files) if result.ok else ""
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
            "relevant_files": relevant,
            "history": [*state.get("history", []), entry],
        }

    return fix
