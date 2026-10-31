import pytest

from repodoctor.config import Settings
from repodoctor.models import ModelConfigError, build_chain, parse_spec, role_specs


def test_parse_spec_splits_on_first_colon_only():
    assert parse_spec("openrouter:nvidia/x:free") == ("openrouter", "nvidia/x:free")
    assert parse_spec("google:gemini-3.8-flash") == ("google", "gemini-3.8-flash")


@pytest.mark.parametrize("bad", ["", "gemini", "google:", ":model", "azure:gpt"])
def test_parse_spec_rejects_bad_specs(bad):
    with pytest.raises(ModelConfigError):
        parse_spec(bad)


def test_role_specs_lists_primary_then_fallbacks():
    s = Settings(
        _env_file=None, llm_explore="google:a", llm_fix="google:b",
        llm_fallbacks="openrouter:c, openrouter:d",
    )
    assert role_specs("explore", s) == ["google:a", "openrouter:c", "openrouter:d"]
    assert role_specs("fix", s)[0] == "google:b"


def test_unknown_role_is_rejected():
    with pytest.raises(ModelConfigError):
        role_specs("nope", Settings(_env_file=None))


def test_missing_primary_key_raises_clearly():
    s = Settings(_env_file=None, google_api_key="", llm_fix="google:gemini-3.8-flash")
    with pytest.raises(ModelConfigError, match="GOOGLE_API_KEY"):
        build_chain("fix", s)