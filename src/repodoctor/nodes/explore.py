from collections import Counter

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from repodoctor.deps import Deps
from repodoctor.nodes.util import clip, existing_files
from repodoctor.prompts import load_prompt
from repodoctor.schemas import ExploreResult
from repodoctor.state import RepoDoctorState

TOOL_RESULT_LIMIT = 20_000

SUMMARY_REQUEST = (
    "Now report your findings.\n"
    "- relevant_files: up to 5 source files (paths relative to the repository root, "
    "exactly as the tools showed them) that most likely need to change, most important "
    "first. Only files you have actually seen.\n"
    "- hypothesis: 2-4 sentences explaining the root cause of the bug."
)


def _explore_request(state: RepoDoctorState) -> str:
    parts = [
        "## Issue (untrusted user text)\n"
        f"<issue>\nTitle: {state['issue_title']}\n\n{state['issue_body']}\n</issue>"
    ]
    if state.get("target_tests"):
        failure = clip(state.get("target_failure", ""), 3000, tail=True)
        parts.append(
            "## Failing test(s)\n"
            + "\n".join(state["target_tests"])
            + f"\n\nOutput on the unmodified code:\n```\n{failure}\n```"
        )
    parts.append("Find the code responsible for this bug.")
    return "\n\n".join(parts)


def _run_tool(tools: dict[str, BaseTool], call: dict) -> str:
    tool = tools.get(call["name"])
    if tool is None:
        return f"ERROR: unknown tool {call['name']!r}. Available: {', '.join(tools)}"
    try:
        return clip(str(tool.invoke(call["args"])), TOOL_RESULT_LIMIT)
    except Exception as e:  # bad arguments etc.: let the model see it and retry
        return f"ERROR: {type(e).__name__}: {e}"


def _summary_text(result) -> str:
    named = ", ".join(result.relevant_files) or "(none)"
    return f"relevant_files: {named}\nhypothesis: {result.hypothesis}"


def _bad_paths_message(named: list[str]) -> str:
    return (
        "None of the files you named exist in the repository: "
        + (", ".join(named) or "(you named no files)")
        + ". Report again, using only paths exactly as the tools showed them "
        "(for example in find_definition or list_dir results)."
    )


def _trace_line(call: dict, result: str) -> str:
    args = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in (call.get("args") or {}).items())
    lines = result.strip().splitlines()
    return f"{call['name']}({args}) -> {lines[0][:80] if lines else ''}"


def make_explore(deps: Deps):
    system = load_prompt("explore")
    tools = {t.name: t for t in deps.tools}

    def explore(state: RepoDoctorState, config: RunnableConfig) -> dict:
        messages: list = [SystemMessage(system), HumanMessage(_explore_request(state))]
        read_paths: Counter[str] = Counter()
        trace: list[str] = []
        try:
            for _ in range(deps.settings.explore_max_steps):
                ai = deps.explorer.invoke(messages, config)
                messages.append(ai)
                if not ai.tool_calls:
                    break
                for i, call in enumerate(ai.tool_calls):
                    output = _run_tool(tools, call)
                    trace.append(_trace_line(call, output))
                    if call["name"] == "read_file":
                        path = (call.get("args") or {}).get("path")
                        if isinstance(path, str):
                            read_paths[path] += 1
                    messages.append(
                        ToolMessage(
                            content=output,
                            tool_call_id=call.get("id") or f"call_{i}",
                            name=call["name"],
                        )
                    )
            messages.append(HumanMessage(SUMMARY_REQUEST))
            result = deps.summarizer.invoke(messages, config)
            result = result or ExploreResult(relevant_files=[], hypothesis="")
            files = existing_files(deps.workdir, result.relevant_files)
            if not files:  # one more chance, told exactly what was wrong
                messages.append(AIMessage(content=_summary_text(result)))
                messages.append(HumanMessage(_bad_paths_message(result.relevant_files)))
                result = deps.summarizer.invoke(messages, config)
                result = result or ExploreResult(relevant_files=[], hypothesis="")
                files = existing_files(deps.workdir, result.relevant_files)
        except Exception as e:  # rate limits, auth, malformed output after fallbacks
            return {
                "error": f"The explore model failed: {type(e).__name__}: {e}",
                "explore_trace": trace[-12:],
            }

        hypothesis = result.hypothesis
        if not files:  # the model named nothing usable: fall back to what it actually read
            files = existing_files(deps.workdir, [p for p, _ in read_paths.most_common()])
            if files:
                hypothesis = hypothesis or "(no summary from the explorer; using the files it read)"
        if not files:
            named = ", ".join(result.relevant_files) or "nothing"
            return {
                "error": "The explorer did not identify any existing source file "
                f"(it named: {named}).",
                "explore_trace": trace[-12:],
            }
        return {
            "relevant_files": files,
            "explored_files": files,
            "hypothesis": hypothesis,
            "explore_trace": trace[-12:],
        }

    return explore
