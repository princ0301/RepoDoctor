import time

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.fallbacks import RunnableWithFallbacks

from repodoctor.config import Settings
from repodoctor.models import ModelConfigError, build_chain, parse_spec, role_specs, with_deadline


def _boom(_x):
    raise ValueError("boom")


def test_parse_spec_splits_on_first_colon_only():
    assert parse_spec("openrouter:nvidia/x:free") == ("openrouter", "nvidia/x:free")
    assert parse_spec("nvidia:nvidia/nemotron-x") == ("nvidia", "nvidia/nemotron-x")


@pytest.mark.parametrize(
    "bad", ["", "nemotron", "nvidia:", ":model", "google:gemini-3.8-flash"]
)
def test_parse_spec_rejects_bad_specs(bad):
    with pytest.raises(ModelConfigError):
        parse_spec(bad)


def test_role_specs_lists_primary_then_fallbacks():
    s = Settings(
        _env_file=None, llm_explore="nvidia:a", llm_fix="nvidia:b",
        llm_fallbacks="openrouter:c, openrouter:d",
    )
    assert role_specs("explore", s) == ["nvidia:a", "openrouter:c", "openrouter:d"]
    assert role_specs("fix", s)[0] == "nvidia:b"


def test_unknown_role_is_rejected():
    with pytest.raises(ModelConfigError):
        role_specs("nope", Settings(_env_file=None))


def test_missing_primary_key_raises_clearly():
    s = Settings(_env_file=None, nvidia_api_key="", llm_fix="nvidia:nvidia/x")
    with pytest.raises(ModelConfigError, match="NVIDIA_API_KEY"):
        build_chain("fix", s)


def test_fallback_without_key_is_skipped():
    s = Settings(
        _env_file=None, openrouter_api_key="dummy", nvidia_api_key="",
        llm_fix="openrouter:m", llm_fallbacks="nvidia:n",
    )
    assert not isinstance(build_chain("fix", s), RunnableWithFallbacks)


def test_fallback_with_key_is_chained():
    s = Settings(
        _env_file=None, openrouter_api_key="dummy", nvidia_api_key="dummy",
        llm_fix="nvidia:n", llm_fallbacks="openrouter:m",
    )
    assert isinstance(build_chain("fix", s), RunnableWithFallbacks)


def test_with_deadline_gives_up_on_a_slow_call_and_falls_back():
    slow = RunnableLambda(lambda _x: time.sleep(3) or "late")
    chain = with_deadline(slow, 0.3).with_fallbacks([RunnableLambda(lambda _x: "fallback")])
    started = time.monotonic()
    assert chain.invoke("q") == "fallback"
    assert time.monotonic() - started < 2


def test_with_deadline_passes_results_and_errors_through():
    assert with_deadline(RunnableLambda(lambda x: x + 1), 5).invoke(1) == 2
    with pytest.raises(ValueError):
        with_deadline(RunnableLambda(_boom), 5).invoke(1)
