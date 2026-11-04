"""Model factory: 'provider:model' specs -> LangChain chat models with fallbacks."""

import concurrent.futures
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from repodoctor.config import Settings

NVIDIA_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# provider -> (base url, Settings field holding the key, env var name for messages)
ENDPOINTS = {
    "nvidia": (NVIDIA_URL, "nvidia_api_key", "NVIDIA_API_KEY"),
    "openrouter": (OPENROUTER_URL, "openrouter_api_key", "OPENROUTER_API_KEY"),
}
ROLES = ("explore", "fix")


class ModelConfigError(ValueError):
    """Bad model spec or missing credentials."""


def parse_spec(spec: str) -> tuple[str, str]:
    provider, sep, model = spec.partition(":")
    provider, model = provider.strip().lower(), model.strip()
    if not sep or not model or provider not in ENDPOINTS:
        raise ModelConfigError(
            f"bad model spec {spec!r}: expected 'provider:model' "
            f"with provider one of {', '.join(ENDPOINTS)}"
        )
    return provider, model


def role_specs(role: str, settings: Settings) -> list[str]:
    if role not in ROLES:
        raise ModelConfigError(f"unknown model role {role!r}")
    primary = settings.llm_explore if role == "explore" else settings.llm_fix
    return [primary, *settings.fallback_specs]


def make_chat_model(spec: str, settings: Settings) -> BaseChatModel:
    provider, model = parse_spec(spec)
    base_url, key_field, env_name = ENDPOINTS[provider]
    key = getattr(settings, key_field)
    if not key:
        raise ModelConfigError(f"{env_name} is not set")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=key,
        temperature=0,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_s,
    )


def adapt(
    model: BaseChatModel,
    *,
    tools: Sequence[BaseTool] | None = None,
    schema: type[BaseModel] | None = None,
) -> Runnable:
    """Bind tools or a structured-output schema to ONE model.

    Done per model, before combining with fallbacks, so every fallback is bound too.
    """
    if tools and schema:
        raise ValueError("pass tools or schema, not both")
    if tools:
        return model.bind_tools(list(tools))
    if schema:
        # hosted open models rarely support native JSON-schema mode; tool calling works
        return model.with_structured_output(schema, method="function_calling")
    return model


def with_deadline(chain: Runnable, seconds: float) -> Runnable:
    """Give up on a model call after `seconds` of wall-clock time. The client timeout only
    limits the gap between bytes, so a slow trickle can otherwise run for many minutes."""
    if seconds <= 0:
        return chain

    def call(value, config: RunnableConfig):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(chain.invoke, value, config)
        try:
            return future.result(timeout=seconds)
        except concurrent.futures.TimeoutError as e:
            raise TimeoutError(f"model call exceeded {seconds}s") from e
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    return RunnableLambda(call)


def build_chain(
    role: str,
    settings: Settings,
    *,
    tools: Sequence[BaseTool] | None = None,
    schema: type[BaseModel] | None = None,
) -> Runnable:
    chains: list[Runnable] = []
    for i, spec in enumerate(role_specs(role, settings)):
        try:
            model = make_chat_model(spec, settings)
        except ModelConfigError:
            if i == 0:
                raise
            continue  # a fallback without credentials is simply not used
        chains.append(with_deadline(
            adapt(model, tools=tools, schema=schema), settings.llm_call_deadline_s
        ))
    primary, *fallbacks = chains
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


# ---------- diagnostics (used by `repodoctor check-llm`) ----------

@tool
def ping(message: str) -> str:
    """Reply to a ping."""
    return message


class _Probe(BaseModel):
    answer: str


def probe(model: BaseChatModel) -> dict[str, str | None]:
    """Check tool calling and structured output. Value None means the check passed."""
    results: dict[str, str | None] = {}
    try:
        reply = adapt(model, tools=[ping]).invoke("Call the ping tool with the message 'hi'.")
        results["tool calling"] = None if reply.tool_calls else "model did not call the tool"
    except Exception as e:  # report any failure, whatever its type
        results["tool calling"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        out = adapt(model, schema=_Probe).invoke("Set answer to the word OK.")
        results["structured output"] = None if isinstance(out, _Probe) else "bad output type"
    except Exception as e:
        results["structured output"] = f"{type(e).__name__}: {str(e)[:200]}"
    return results
