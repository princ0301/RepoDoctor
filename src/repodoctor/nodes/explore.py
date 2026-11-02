from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from repodoctor.deps import Deps
from repodoctor.nodes.util import clip, existing_files
from repodoctor.prompts import load_prompt
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


def make_explore(deps: Deps):
    system = load_prompt("explore")
    tools = {t.name: t for t in deps.tools}

    def explore(state: RepoDoctorState, config: RunnableConfig) -> dict:
        messages: list = [SystemMessage(system), HumanMessage(_explore_request(state))]
        try:
            for _ in range(deps.settings.explore_max_steps):
                ai = deps.explorer.invoke(messages, config)
                messages.append(ai)
                if not ai.tool_calls:
                    break
                for i, call in enumerate(ai.tool_calls):
                    messages.append(
                        ToolMessage(
                            content=_run_tool(tools, call),
                            tool_call_id=call.get("id") or f"call_{i}",
                            name=call["name"],
                        )
                    )
            messages.append(HumanMessage(SUMMARY_REQUEST))
            result = deps.summarizer.invoke(messages, config)
        except Exception as e:  # rate limits, auth, malformed output after fallbacks
            return {"error": f"The explore model failed: {type(e).__name__}: {e}"}

        files = existing_files(deps.workdir, result.relevant_files)
        if not files:
            return {"error": "The explorer did not identify any existing source file."}
        return {
            "relevant_files": files,
            "explored_files": files,
            "hypothesis": result.hypothesis,
        }

    return explore
