from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import dotenv_values

from voltproof.config import (
    DEFAULTS,
    PROJECT_ROOT,
    ConfigError,
    Secret,
    get_settings,
    load_settings,
)

SENTINEL = "sk-test-SENTINEL-0123456789abcdef"


def test_env_example_matches_defaults_and_parses() -> None:
    example = dotenv_values(PROJECT_ROOT / ".env.example")
    assert example.pop("BASETEN_API_KEY") == ""
    assert example == dict(DEFAULTS)

    s = load_settings(example)
    assert s.llm_base_url == "https://inference.baseten.co/v1"
    assert s.llm_model == "meta-models/muse-spark-open-1026"
    assert s.llm_reasoning_effort == "medium"
    assert s.llm_max_output_tokens == 8192 and isinstance(s.llm_max_output_tokens, int)
    assert s.llm_timeout_s == 120.0
    assert (s.price_input_usd_per_m, s.price_output_usd_per_m, s.price_cached_input_usd_per_m) == (1.25, 4.25, 0.15)
    assert (s.budget_usd_cap, s.demo_budget_usd_cap) == (85.0, 10.0)
    assert s.live_mode is False


def test_missing_key_loads_without_error() -> None:
    s = load_settings({})
    assert not s.baseten_api_key
    assert s.baseten_api_key.reveal() == ""


@pytest.mark.parametrize("value", ["low", "medium", "high", "xhigh", "XHIGH", " High "])
def test_reasoning_effort_accepts_supported_values(value: str) -> None:
    assert load_settings({"LLM_REASONING_EFFORT": value}).llm_reasoning_effort == value.strip().lower()


def test_empty_reasoning_effort_means_not_sent() -> None:
    assert load_settings({"LLM_REASONING_EFFORT": ""}).llm_reasoning_effort is None


@pytest.mark.parametrize(("text", "expected"), [("true", True), ("1", True), ("YES", True), ("off", False)])
def test_live_mode_parses_booleans(text: str, expected: bool) -> None:
    assert load_settings({"LIVE_MODE": text}).live_mode is expected


def test_all_invalid_values_reported_at_once() -> None:
    bad = {
        "LLM_BASE_URL": "ftp://example.org",
        "LLM_MODEL": "",
        "LLM_REASONING_EFFORT": "turbo",
        "LLM_MAX_OUTPUT_TOKENS": "lots",
        "LLM_TIMEOUT_S": "0",
        "PRICE_INPUT_USD_PER_M": "-1",
        "PRICE_OUTPUT_USD_PER_M": "nan",
        "BUDGET_USD_CAP": "0",
        "LIVE_MODE": "maybe",
    }
    with pytest.raises(ConfigError) as info:
        load_settings(bad)
    message = str(info.value)
    for name in bad:
        assert name in message, name


def test_demo_cap_must_not_exceed_budget_cap() -> None:
    with pytest.raises(ConfigError, match="DEMO_BUDGET_USD_CAP"):
        load_settings({"BUDGET_USD_CAP": "10", "DEMO_BUDGET_USD_CAP": "20"})


def test_max_tokens_range() -> None:
    with pytest.raises(ConfigError, match="LLM_MAX_OUTPUT_TOKENS"):
        load_settings({"LLM_MAX_OUTPUT_TOKENS": "0"})


def test_key_hidden_in_repr_str_and_errors() -> None:
    s = load_settings({"BASETEN_API_KEY": f"  {SENTINEL}  "})
    assert s.baseten_api_key.reveal() == SENTINEL
    for text in (repr(s), str(s), repr(s.baseten_api_key), str(s.baseten_api_key), f"{s.baseten_api_key}"):
        assert SENTINEL not in text
    with pytest.raises(ConfigError) as info:
        load_settings({"BASETEN_API_KEY": SENTINEL, "LLM_TIMEOUT_S": "x"})
    assert SENTINEL not in str(info.value)


def test_secret_is_not_hashable_and_compares_by_value() -> None:
    assert Secret("a") == Secret("a") and Secret("a") != Secret("b")
    with pytest.raises(TypeError):
        hash(Secret("a"))


def test_get_settings_reads_dotenv_file_and_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dotenv = tmp_path / "test.env"
    dotenv.write_text("LLM_MODEL=model-from-file\nLLM_TIMEOUT_S=30\nLIVE_MODE=true\n", encoding="utf-8")
    monkeypatch.setenv("LLM_TIMEOUT_S", "45")

    s = get_settings(dotenv)

    assert s.llm_model == "model-from-file"
    assert s.llm_timeout_s == 45.0
    assert s.live_mode is True


def test_get_settings_without_dotenv_file_uses_defaults() -> None:
    # conftest points DOTENV_PATH at a file that does not exist.
    s = get_settings()
    assert s.llm_model == DEFAULTS["LLM_MODEL"]
    assert not s.baseten_api_key


def test_get_settings_does_not_modify_os_environ(tmp_path: Path) -> None:
    import os

    dotenv = tmp_path / "test.env"
    dotenv.write_text("LLM_MODEL=model-from-file\n", encoding="utf-8")
    get_settings(dotenv)
    assert "LLM_MODEL" not in os.environ


def test_trusted_proxy_hops_default_range_and_type() -> None:
    assert load_settings({}).trusted_proxy_hops == 0
    assert load_settings({"TRUSTED_PROXY_HOPS": "0"}).trusted_proxy_hops == 0
    assert load_settings({"TRUSTED_PROXY_HOPS": "5"}).trusted_proxy_hops == 5
    for bad in ("-1", "6", "two"):
        with pytest.raises(ConfigError, match="TRUSTED_PROXY_HOPS"):
            load_settings({"TRUSTED_PROXY_HOPS": bad})
