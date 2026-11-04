import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from repodoctor.deps import Deps
from repodoctor.evals.cases import is_test_path
from repodoctor.nodes.util import (
    clip,
    existing_files,
    is_malformed_output,
    render_files,
    repo_file_list,
    test_file_of,
)
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


def _fix_request(state: RepoDoctorState, root: Path, budget: int) -> tuple[str, list[str]]:
    """The prompt, plus the relevant files that did not fit into the context budget."""
    targets = list(state["target_tests"])
    tests_text, _ = render_files(root, _target_files(targets), TEST_FILE_BUDGET)
    source_text, omitted = render_files(root, state["relevant_files"], budget)
    sections = [
        "## Issue (untrusted user text)\n"
        f"<issue>\nTitle: {state['issue_title']}\n\n{state['issue_body']}\n</issue>",
        "## Failing test(s)\n"
        + "\n".join(targets)
        + "\n\nOutput on the unmodified code:\n```\n"
        + clip(state.get("target_failure", ""), 3000, tail=True)
        + "\n```",
        f"## Investigation notes\n{state.get('hypothesis', '')}",
    ]
    sections += [
        "## Test file(s) (read-only, do not edit)\n" + tests_text,
        "## Source files you may edit\n" + source_text,
        "## Repository files (use exact paths in need_more_files)\n" + repo_file_list(root),
    ]
    if state.get("history"):
        sections.append("## Previous attempts\n" + _history_text(state["history"]))
    sections.append("Return your edits.")
    return "\n\n".join(sections), omitted

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


def _no_edits_message(
    wanted: list[str], shown: list[str], omitted: list[str], unknown: list[str]
) -> str:
    parts = ["No edits were proposed."]
    if shown:
        parts.append(
            f"These files are already shown to you in full, use them: {', '.join(shown)}."
        )
    promoted = [f for f in wanted if f not in shown]
    if promoted:
        parts.append(
            f"Moved to the top of your context for the next attempt: {', '.join(promoted)}."
        )
    if unknown:
        parts.append(
            f"These paths in need_more_files do not exist: {', '.join(unknown)}. "
            "Use paths exactly as listed under 'Repository files'."
        )
    if omitted and not wanted:
        parts.append(
            f"Not shown because of the context budget: {', '.join(omitted)}. List what you "
            "need in need_more_files, or write edits with what you have."
        )
    if len(parts) == 1:
        parts.append(
            "Saying that you want to look at something does nothing: you cannot run tools "
            "or open files. Either list exact paths from 'Repository files' in "
            "need_more_files, or write your best edits now from the files shown."
        )
    return " ".join(parts)


def _failed_attempt(state: RepoDoctorState, attempt: int, message: str) -> dict:
    entry = {"attempt": attempt, "reasoning": "", "edits": [], "patch_errors": [message]}
    return {
        "attempts": attempt, "diff": "", "patch_errors": [message],
        "history": [*state.get("history", []), entry],
    }


def make_fix(deps: Deps):
    system = load_prompt("fix")
    if deps.settings.allow_test_updates:
        system += "\n\n" + load_prompt("fix_test_updates")

    def fix(state: RepoDoctorState, config: RunnableConfig) -> dict:
        root = deps.workdir
        attempt = state.get("attempts", 0) + 1
        reset_to_clean(root, set(state.get("untracked_baseline", [])))

        relevant = list(state["relevant_files"])
        if deps.settings.allow_test_updates:
            # show the tests the last attempt broke, so they can be updated if the issue demands it
            promoted = list(dict.fromkeys(
                f for f in (test_file_of(root, t) for t in state.get("regressed_tests", [])) if f
            ))
            relevant = [*promoted, *(f for f in relevant if f not in promoted)]
            relevant = relevant[:MAX_CONTEXT_FILES]

        request, omitted = _fix_request(
            {**state, "relevant_files": relevant}, root, deps.settings.fix_context_chars
        )
        try:
            proposal = deps.fixer.invoke(
                [SystemMessage(system), HumanMessage(request)], config
            )
        except Exception as e:
            if not is_malformed_output(e):
                return {"error": f"The fix model failed on attempt {attempt}: "
                        f"{type(e).__name__}: {e}"}
            return _failed_attempt(
                state, attempt,
                "Your last reply was not valid structured output "
                f"({type(e).__name__}: {clip(str(e), 400)}). Reply again with valid `edits`.",
            )
        if proposal is None:
            return _failed_attempt(
                state, attempt,
                "Your last reply was not valid structured output (it contained none). "
                "Reply again with valid `edits`.",
            )

        protected = set(_target_files(list(state["target_tests"])))
        result = _apply(root, proposal.edits, protected)
        errors = list(result.errors)
        if not proposal.edits:
            wanted = existing_files(root, proposal.need_more_files)
            shown = [f for f in wanted if f in relevant and f not in omitted]
            unknown = [
                r for r in proposal.need_more_files
                if r.strip() and not existing_files(root, [r])
            ]
            relevant = [*wanted, *(f for f in relevant if f not in wanted)][:MAX_CONTEXT_FILES]
            errors = [_no_edits_message(wanted, shown, omitted, unknown)]

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
            "context_omitted": omitted,
            "edited_tests": [p for p in result.changed_files if is_test_path(p)],
            "history": [*state.get("history", []), entry],
        }

    return fix
