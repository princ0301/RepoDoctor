from typing import Any

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