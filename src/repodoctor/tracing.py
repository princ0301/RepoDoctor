from typing import Any
import time
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler


class CallCounter(BaseCallbackHandler):
    """Counts model requests and times each one (fallback attempts included;
    HTTP-level retries inside a client are not visible)."""

    def __init__(self) -> None:
        self.calls = 0
        self.timeline: list[dict[str, Any]] = []   # {"model", "seconds", "ok"}
        self._open: dict[UUID, tuple[float, str]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, **kwargs) -> None:
        self.calls += 1
        params = kwargs.get("invocation_params") or {}
        name = str(params.get("model") or params.get("model_name") or "?")
        self._open[run_id] = (time.monotonic(), name)

    def on_llm_end(self, response, *, run_id: UUID, **kwargs) -> None:
        self._close(run_id, ok=True)

    def on_llm_error(self, error, *, run_id: UUID, **kwargs) -> None:
        self._close(run_id, ok=False)

    def _close(self, run_id: UUID, *, ok: bool) -> None:
        started = self._open.pop(run_id, None)
        if started:
            self.timeline.append(
                {"model": started[1], "seconds": round(time.monotonic() - started[0], 1), "ok": ok}
            )


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
