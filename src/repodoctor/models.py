from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from repodoctor.config import Settings

OPENROUTER_URL = "https://openrouter.ai/api/v1"
PROVIDERS = ("google", "openrouter")
ROLES = ("explore", "fix")

class ModelConfigError(ValueError):
    """Bad model spec or missing credentials."""

def parse_spec(spec: str) -> tuple[str, str]:
    provider, sep, model = spec.partition(":")
    provider, model = provider.strip().lower(), model.strip()
    if not sep or not model or provider not in PROVIDERS:
        raise ModelConfigError(
            f"bad model spec {spec!r}: expected 'provider:model' "
            f"with provider one of {', '.join(PROVIDERS)}"
        )
    return provider, model

def role_specs(role: str, settings: Settings) -> list[str]:
    if role not in ROLES:
        raise ModelConfigError(f"unknown model role {role!r}")
    primary = settings.llm_explore if role == "explore" else settings.llm_fix
    return [primary, *settings.fallback_specs]

def make_chat_model(spec: str, settings: Settings) -> BaseChatModel:
    provider, model = parse_spec(spec)
    if provider == "google":
        if not settings.google_api_key:
            raise ModelConfigError("GOOGLE_API_KEY is not set")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            max_retries=2,
            timeout=settings.llm_timeout_s,
        )
    if not settings.openrouter_api_key:
        raise ModelConfigError("OPENROUTER_API_KEY is not set")
    return ChatOpenAI(
        model=model,
        base_url=OPENROUTER_URL,
        api_key=settings.openrouter_api_key,
        temperature=0,
        max_retries=2,
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
        if isinstance(model, ChatOpenAI):
            return model.with_structured_output(schema, method="function_calling")
        return model.with_structured_output(schema)
    return model

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
        chains.append(adapt(model, tools=tools, schema=schema))
    primary, *fallbacks = chains
    return primary.with_fallbacks(fallbacks) if fallbacks else primary

@tool
def ping(message: str) -> str:
    """Reply to ping."""
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