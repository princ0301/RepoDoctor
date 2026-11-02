from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class CallCounter(BaseCallbackHandler):
    """Counts model requests (fallback attempts included; HTTP-level retries are not)."""

    def __init__(self) -> None:
        self.calls = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.calls += 1


def run_config(
    run_name: str,
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    recursion_limit: int = 40,
) -> dict[str, Any]:
    """The `config` passed to graph.invoke: names the LangSmith trace and sets limits."""
    return {
        "run_name": run_name,
        "tags": list(tags or []),
        "metadata": dict(metadata or {}),
        "recursion_limit": recursion_limit,
    }
