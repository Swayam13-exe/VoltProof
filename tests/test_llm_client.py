from __future__ import annotations

import dataclasses
import json
import logging
import traceback
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.fakes import (
    MESSAGES,
    FakeSDK,
    connection_error,
    make_client,
    make_response,
    status_error,
    timeout_error,
)
from voltproof.config import Secret, Settings
from voltproof.llm.client import (
    MAX_ATTEMPTS,
    BudgetExceeded,
    LLMClient,
    LLMError,
    LLMRequestRejected,
    LLMUnavailable,
    MissingAPIKey,
    Usage,
    cache_key,
    estimate_cost_usd,
    extract_reasoning,
    normalize_usage,
    read_ledger_totals,
    redact,
    worst_case_cost_usd,
)

SENTINEL = "sk-test-SENTINEL-0123456789abcdef"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_import_price",
            "description": "Price for one hour.",
            "parameters": {"type": "object", "properties": {"hour": {"type": "integer"}}, "required": ["hour"]},
        },
    }
]


def ledger_lines(settings: Settings) -> list[dict[str, Any]]:
    return [json.loads(line) for line in settings.ledger_path.read_text(encoding="utf-8").splitlines()]


def seed_ledger(settings: Settings, cost_usd: float, tags: dict[str, Any] | None = None) -> None:
    settings.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps({"cost_usd": cost_usd, "tags": tags or {}}) + "\n")


# --- cache key -------------------------------------------------------------------------

KEY_ARGS: dict[str, Any] = dict(
    model="m",
    messages=MESSAGES,
    tools=TOOLS,
    tool_choice="auto",
    response_format=None,
    reasoning_effort="medium",
    max_tokens=8192,
)


def test_cache_key_is_deterministic_and_ignores_dict_order() -> None:
    reordered = [dict(reversed(list(m.items()))) for m in MESSAGES]
    assert cache_key(**KEY_ARGS) == cache_key(**KEY_ARGS)
    assert cache_key(**KEY_ARGS) == cache_key(**{**KEY_ARGS, "messages": reordered})
    assert len(cache_key(**KEY_ARGS)) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other-model"),
        ("messages", [{"role": "user", "content": "different"}]),
        ("tools", None),
        ("tool_choice", "required"),
        ("response_format", {"type": "json_object"}),
        ("reasoning_effort", "xhigh"),
        ("max_tokens", 4096),
    ],
)
def test_cache_key_changes_with_every_field(field: str, value: Any) -> None:
    assert cache_key(**KEY_ARGS) != cache_key(**{**KEY_ARGS, field: value})


def test_cache_key_ignores_purpose_and_tags(settings: Settings) -> None:
    sdk = FakeSDK(make_response())
    client = make_client(settings, sdk)
    first = client.chat(MESSAGES, purpose="a", tags={"arm": "A2"})
    second = client.chat(MESSAGES, purpose="b", tags={"arm": "A4"})
    assert second.cache_hit and second.cache_key == first.cache_key
    assert len(sdk.calls) == 1


def test_non_serializable_messages_rejected_before_any_call(settings: Settings) -> None:
    sdk = FakeSDK()
    with pytest.raises(TypeError):
        make_client(settings, sdk).chat([{"role": "user", "content": object()}], purpose="t")
    assert sdk.calls == []


# --- cache hit -------------------------------------------------------------------------


def test_second_identical_call_is_a_free_cache_hit(settings: Settings) -> None:
    sdk = FakeSDK(make_response("hello", reasoning_content="thinking"))
    client = make_client(settings, sdk)

    first = client.chat(MESSAGES, purpose="t")
    second = client.chat(MESSAGES, purpose="t")

    assert len(sdk.calls) == 1
    assert not first.cache_hit and first.cost_usd > 0
    assert second.cache_hit and second.cost_usd == 0.0
    assert (second.content, second.reasoning, second.usage) == ("hello", "thinking", first.usage)
    lines = ledger_lines(settings)
    assert [line["cache_hit"] for line in lines] == [False, True]
    assert lines[1]["cost_usd"] == 0.0 and lines[1]["attempts"] == 0


def test_cache_hit_needs_no_key_and_no_budget(settings: Settings) -> None:
    make_client(settings, FakeSDK(make_response())).chat(MESSAGES, purpose="t")
    seed_ledger(settings, 1000.0)
    keyless = LLMClient(settings)  # no SDK, no key: a miss would raise MissingAPIKey
    assert not settings.baseten_api_key
    assert keyless.chat(MESSAGES, purpose="t").cache_hit


def test_cache_entry_written_atomically_to_sharded_path(settings: Settings) -> None:
    response = make_client(settings, FakeSDK(make_response())).chat(MESSAGES, purpose="t")
    path = settings.cache_dir / response.cache_key[:2] / f"{response.cache_key}.json"
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["response"]["content"] == "ok" and "raw" in entry
    assert not list(path.parent.glob(".tmp-*"))


def test_corrupt_cache_entry_is_a_miss(settings: Settings) -> None:
    sdk = FakeSDK(make_response("first"), make_response("second"))
    client = make_client(settings, sdk)
    first = client.chat(MESSAGES, purpose="t")
    path = settings.cache_dir / first.cache_key[:2] / f"{first.cache_key}.json"
    path.write_text("{not json", encoding="utf-8")
    again = client.chat(MESSAGES, purpose="t")
    assert not again.cache_hit and again.content == "second"


def test_errors_are_not_cached(settings: Settings) -> None:
    sdk = FakeSDK(status_error(400), make_response())
    client = make_client(settings, sdk)
    with pytest.raises(LLMRequestRejected):
        client.chat(MESSAGES, purpose="t")
    assert not client.chat(MESSAGES, purpose="t").cache_hit
    assert len(sdk.calls) == 2


# --- ledger ----------------------------------------------------------------------------


def test_ledger_line_has_documented_fields_and_no_content(settings: Settings) -> None:
    usage = {
        "prompt_tokens": 812,
        "completion_tokens": 240,
        "prompt_tokens_details": {"cached_tokens": 12},
        "completion_tokens_details": {"reasoning_tokens": 180},
    }
    sdk = FakeSDK(make_response("SECRET-CONTENT", finish_reason="tool_calls", usage=usage, reasoning="RSN"))
    make_client(settings, sdk).chat(MESSAGES, tools=TOOLS, purpose="smoke", tags={"arm": "A4", "episode": "e1"})

    (line,) = ledger_lines(settings)
    assert set(line) == {
        "ts", "purpose", "tags", "model", "cache_key", "cache_hit", "status", "error", "attempts",
        "finish_reason", "reasoning_effort", "prompt_tokens", "completion_tokens", "cached_tokens",
        "reasoning_tokens", "cost_usd", "possible_unbilled_usd", "usage_missing", "latency_s",
    }  # fmt: skip
    assert line["purpose"] == "smoke" and line["tags"] == {"arm": "A4", "episode": "e1"}
    assert (line["prompt_tokens"], line["completion_tokens"], line["cached_tokens"], line["reasoning_tokens"]) == (
        812,
        240,
        12,
        180,
    )
    assert line["status"] == "ok" and line["attempts"] == 1 and line["possible_unbilled_usd"] == 0.0
    assert line["ts"].endswith("Z")
    text = settings.ledger_path.read_text(encoding="utf-8")
    for leaked in ("SECRET-CONTENT", "RSN", MESSAGES[1]["content"]):
        assert leaked not in text


def test_ledger_totals_split_demo_and_skip_malformed(settings: Settings) -> None:
    seed_ledger(settings, 1.5)
    seed_ledger(settings, 2.0, {"demo": True})
    seed_ledger(settings, 3.0, {"demo": "yes"})  # only a real `true` counts as demo
    with settings.ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write("{broken\n[]\n" + json.dumps({"cost_usd": "nan"}) + "\n\n")
    totals = read_ledger_totals(settings.ledger_path)
    assert totals.total_usd == pytest.approx(6.5)
    assert totals.demo_usd == pytest.approx(2.0)
    assert (totals.calls, totals.malformed_lines) == (3, 3)


def test_missing_ledger_counts_as_zero(tmp_path: Path) -> None:
    assert read_ledger_totals(tmp_path / "nope.jsonl").total_usd == 0.0


# --- budget guard ----------------------------------------------------------------------


def test_budget_guard_refuses_before_calling(settings: Settings) -> None:
    seed_ledger(settings, settings.budget_usd_cap - 0.01)
    sdk = FakeSDK(make_response())
    with pytest.raises(BudgetExceeded, match="Budget cap"):
        make_client(settings, sdk).chat(MESSAGES, purpose="t")
    assert sdk.calls == []
    assert ledger_lines(settings)[-1]["status"] == "refused"


def test_budget_guard_allows_call_that_fits(settings: Settings) -> None:
    worst = worst_case_cost_usd(MESSAGES, None, None, settings)
    seed_ledger(settings, settings.budget_usd_cap - worst - 1e-6)
    assert make_client(settings, FakeSDK(make_response())).chat(MESSAGES, purpose="t").content == "ok"


def test_demo_cap_applies_only_to_demo_calls(settings: Settings) -> None:
    seed_ledger(settings, settings.demo_budget_usd_cap - 0.01, {"demo": True})
    sdk = FakeSDK(make_response())
    client = make_client(settings, sdk)
    with pytest.raises(BudgetExceeded, match="Demo budget cap"):
        client.chat(MESSAGES, purpose="demo", tags={"demo": True})
    assert sdk.calls == []
    assert client.chat(MESSAGES, purpose="bench", tags={"arm": "A4"}).content == "ok"


def test_in_flight_reservation_blocks_concurrent_call(settings: Settings) -> None:
    worst = worst_case_cost_usd(MESSAGES, None, None, settings)
    tight = dataclasses.replace(settings, budget_usd_cap=worst * 1.5, demo_budget_usd_cap=worst * 1.5)
    other = [{"role": "user", "content": "another request"}]
    refused: list[BaseException] = []

    def first_call(_kwargs: dict[str, Any]) -> dict[str, Any]:
        # While this call is in flight, a second call must not fit under the cap.
        try:
            client.chat(other, purpose="t")
        except BudgetExceeded as exc:
            refused.append(exc)
        return make_response()

    client = make_client(tight, FakeSDK(first_call))
    client.chat(MESSAGES, purpose="t")
    assert len(refused) == 1 and "in flight" in str(refused[0])
    # After the first call finishes, its reservation is released and only real cost remains.
    assert read_ledger_totals(tight.ledger_path).total_usd < worst


def test_worst_case_counts_full_max_tokens_output(settings: Settings) -> None:
    worst = worst_case_cost_usd(MESSAGES, TOOLS, None, settings)
    output_part = settings.llm_max_output_tokens * settings.price_output_usd_per_m / 1e6
    assert output_part < worst < output_part + 0.01


# --- retries ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [status_error(429), status_error(500), status_error(503), timeout_error(), connection_error()],
    ids=["429", "500", "503", "timeout", "connection"],
)
def test_retryable_errors_are_retried(settings: Settings, error: Exception) -> None:
    sleeps: list[float] = []
    sdk = FakeSDK(error, make_response("recovered"))
    response = make_client(settings, sdk, sleeps).chat(MESSAGES, purpose="t")
    assert response.content == "recovered"
    assert len(sdk.calls) == 2 and len(sleeps) == 1
    assert 0 <= sleeps[0] <= 1.0
    assert ledger_lines(settings)[-1]["attempts"] == 2


def test_gives_up_after_max_attempts_with_growing_backoff_bounds(settings: Settings) -> None:
    sleeps: list[float] = []
    sdk = FakeSDK(*[status_error(503) for _ in range(MAX_ATTEMPTS)])
    with pytest.raises(LLMUnavailable) as info:
        make_client(settings, sdk, sleeps).chat(MESSAGES, purpose="t")
    assert info.value.attempts == MAX_ATTEMPTS == len(sdk.calls) == 4
    assert info.value.status_code == 503
    assert len(sleeps) == MAX_ATTEMPTS - 1
    assert all(0 <= delay <= 2**i for i, delay in enumerate(sleeps))


def test_retry_after_header_is_honored(settings: Settings) -> None:
    sleeps: list[float] = []
    sdk = FakeSDK(status_error(429, headers={"retry-after": "7"}), make_response())
    make_client(settings, sdk, sleeps).chat(MESSAGES, purpose="t")
    assert sleeps == [7.0]


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_other_4xx_are_not_retried(settings: Settings, status: int) -> None:
    sleeps: list[float] = []
    sdk = FakeSDK(status_error(status), make_response())
    with pytest.raises(LLMRequestRejected) as info:
        make_client(settings, sdk, sleeps).chat(MESSAGES, purpose="t")
    assert len(sdk.calls) == 1 and sleeps == []
    assert info.value.status_code == status
    line = ledger_lines(settings)[-1]
    assert line["status"] == "error" and line["attempts"] == 1 and line["cost_usd"] == 0.0


def test_possible_unbilled_counts_timeouts_and_5xx_only(settings: Settings) -> None:
    worst = worst_case_cost_usd(MESSAGES, None, None, settings)
    sdk = FakeSDK(timeout_error(), status_error(502), status_error(429), connection_error())
    with pytest.raises(LLMUnavailable):
        make_client(settings, sdk).chat(MESSAGES, purpose="t")
    line = ledger_lines(settings)[-1]
    assert line["attempts"] == 4 and line["cost_usd"] == 0.0
    assert line["possible_unbilled_usd"] == pytest.approx(2 * worst)
    assert line["error"] == "APIConnectionError"
    # Not counted toward the budget guard.
    assert read_ledger_totals(settings.ledger_path).total_usd == 0.0


def test_possible_unbilled_recorded_on_eventual_success(settings: Settings) -> None:
    worst = worst_case_cost_usd(MESSAGES, None, None, settings)
    make_client(settings, FakeSDK(timeout_error(), make_response())).chat(MESSAGES, purpose="t")
    line = ledger_lines(settings)[-1]
    assert line["status"] == "ok" and line["possible_unbilled_usd"] == pytest.approx(worst)


def test_missing_key_raises_only_when_a_network_call_is_needed(settings: Settings) -> None:
    client = LLMClient(settings)  # constructing without a key is fine
    with pytest.raises(MissingAPIKey, match="BASETEN_API_KEY"):
        client.chat(MESSAGES, purpose="t")


# --- reasoning, usage, cost ------------------------------------------------------------


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_reasoning_extracted_from_either_field(settings: Settings, field: str) -> None:
    sdk = FakeSDK(make_response(**{field: "step by step"}))
    assert make_client(settings, sdk).chat(MESSAGES, purpose="t").reasoning == "step by step"


def test_reasoning_absent_is_none_and_structured_is_serialized() -> None:
    assert extract_reasoning({"content": "x"}) is None
    assert extract_reasoning({"reasoning_content": "", "reasoning": "fallback"}) == "fallback"
    assert extract_reasoning({"reasoning": [{"text": "a"}]}) == '[{"text": "a"}]'


def test_tool_calls_normalized(settings: Settings) -> None:
    calls = [{"id": "call_1", "type": "function", "function": {"name": "get_import_price", "arguments": '{"hour": 18}'}}]
    sdk = FakeSDK(make_response(None, finish_reason="tool_calls", tool_calls=calls))
    response = make_client(settings, sdk).chat(MESSAGES, tools=TOOLS, tool_choice="auto", purpose="t")
    assert response.finish_reason == "tool_calls"
    assert [(c.id, c.name, json.loads(c.arguments_json)) for c in response.tool_calls] == [
        ("call_1", "get_import_price", {"hour": 18})
    ]


def test_usage_nested_flat_and_missing() -> None:
    nested = {
        "prompt_tokens": 100,
        "completion_tokens": 77,
        "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 22},
    }
    flat = {"prompt_tokens": 100, "completion_tokens": 77, "cached_tokens": 40, "reasoning_tokens": 22}
    assert normalize_usage(nested) == normalize_usage(flat) == Usage(100, 77, 40, 22)
    assert normalize_usage({"prompt_tokens": 5, "completion_tokens": 1, "prompt_tokens_details": None}) == Usage(5, 1)
    assert normalize_usage({"prompt_tokens": 5, "cached_tokens": 50}).cached == 5  # clamped to prompt
    assert normalize_usage(None) is None and normalize_usage({}) is None


def test_cost_formula(settings: Settings) -> None:
    usage = Usage(prompt=1_000_000, completion=1_000_000, cached=400_000)
    # 600k·1.25 + 400k·0.15 + 1M·4.25 per million
    assert estimate_cost_usd(usage, settings) == pytest.approx(0.75 + 0.06 + 4.25)


def test_missing_usage_charges_worst_case(settings: Settings) -> None:
    raw = make_response()
    del raw["usage"]
    response = make_client(settings, FakeSDK(raw)).chat(MESSAGES, purpose="t")
    assert response.cost_usd == pytest.approx(worst_case_cost_usd(MESSAGES, None, None, settings))
    assert ledger_lines(settings)[-1]["usage_missing"] is True


def test_response_without_choices_is_an_error_but_charged(settings: Settings) -> None:
    raw = make_response()
    raw["choices"] = []
    with pytest.raises(LLMError, match="no message"):
        make_client(settings, FakeSDK(raw)).chat(MESSAGES, purpose="t")
    line = ledger_lines(settings)[-1]
    assert line["status"] == "error" and line["cost_usd"] > 0


# --- wire format and key redaction (real SDK, mocked transport, no network) ------------


def _real_client(settings: Settings, handler: Any, effort: str | None = "xhigh") -> tuple[LLMClient, Settings]:
    keyed = dataclasses.replace(settings, baseten_api_key=Secret(SENTINEL), llm_reasoning_effort=effort)
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return LLMClient(keyed, http_client=http_client, sleep=lambda _s: None), keyed


def test_wire_format_through_real_sdk(settings: Settings) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        message = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "need the tariff",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_import_price", "arguments": '{"hour":18}'}}
            ],
        }
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": settings.llm_model,
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 77,
                    "total_tokens": 177,
                    "completion_tokens_details": {"reasoning_tokens": 22},
                },
            },
        )

    client, keyed = _real_client(settings, handler)
    response = client.chat(MESSAGES, tools=TOOLS, tool_choice="auto", purpose="t")

    assert seen["auth"] == f"Bearer {SENTINEL}"  # the key really is sent ...
    assert seen["url"] == f"{keyed.llm_base_url}/chat/completions"
    body = seen["body"]
    assert body["model"] == keyed.llm_model
    assert body["reasoning_effort"] == "xhigh"
    assert body["max_tokens"] == keyed.llm_max_output_tokens
    assert "temperature" not in body and "top_p" not in body
    assert response.reasoning == "need the tariff"
    assert response.tool_calls[0].name == "get_import_price"
    assert response.usage == Usage(prompt=100, completion=77, cached=0, reasoning=22)


def test_reasoning_effort_omitted_when_empty(settings: Settings) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=make_response())

    client, _ = _real_client(settings, handler, effort=None)
    client.chat(MESSAGES, purpose="t")
    assert "reasoning_effort" not in bodies[0]


def test_key_never_appears_in_errors_reprs_logs_or_files(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"message": f"Invalid API key: {SENTINEL} (header Bearer {SENTINEL})"}}
        )

    caplog.set_level(logging.DEBUG)
    client, keyed = _real_client(settings, handler)
    with pytest.raises(LLMRequestRejected) as info:
        client.chat(MESSAGES, purpose="t")

    error = info.value
    assert error.status_code == 401
    assert error.__cause__ is None and error.__context__ is None  # no SDK exception (with headers) attached
    formatted = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    for text in (str(error), repr(error), formatted, repr(keyed), repr(client), caplog.text):
        assert SENTINEL not in text
    assert SENTINEL not in keyed.ledger_path.read_text(encoding="utf-8")


def test_key_never_appears_after_retries_exhausted(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": f"overloaded for key {SENTINEL}"}})

    caplog.set_level(logging.DEBUG)
    client, _ = _real_client(settings, handler)
    with pytest.raises(LLMUnavailable) as info:
        client.chat(MESSAGES, purpose="t")
    assert info.value.__context__ is None
    assert SENTINEL not in str(info.value) and SENTINEL not in caplog.text


def test_successful_response_repr_and_cache_have_no_key(settings: Settings) -> None:
    client, keyed = _real_client(settings, lambda request: httpx.Response(200, json=make_response()))
    response = client.chat(MESSAGES, purpose="t")
    assert SENTINEL not in repr(response)
    for path in keyed.cache_dir.rglob("*.json"):
        assert SENTINEL not in path.read_text(encoding="utf-8")


def test_redact_removes_secret_and_auth_tokens() -> None:
    text = f"key {SENTINEL}; Authorization: Bearer abcdefgh12345678; Api-Key zyxwvuts98765432"
    cleaned = redact(text, SENTINEL)
    assert SENTINEL not in cleaned
    assert "abcdefgh12345678" not in cleaned and "zyxwvuts98765432" not in cleaned
