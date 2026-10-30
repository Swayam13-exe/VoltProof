"""Credit exhaustion and auth errors stop the client immediately (fake SDK, no network)."""

from __future__ import annotations

import json

import pytest

from tests.fakes import MESSAGES, FakeSDK, make_client, make_response, status_error
from voltproof.config import Settings
from voltproof.llm.client import (
    MAX_ATTEMPTS,
    LLMAccountError,
    LLMClientStopped,
    LLMFatalError,
    LLMRateLimitPersisted,
    LLMRequestRejected,
    LLMUnavailable,
)


@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_and_credit_errors_are_fatal_and_stop_the_client(settings: Settings, status: int) -> None:
    sleeps: list[float] = []
    sdk = FakeSDK(status_error(status), make_response("never sent"))
    client = make_client(settings, sdk, sleeps)
    with pytest.raises(LLMAccountError) as info:
        client.chat(MESSAGES, purpose="t")
    assert isinstance(info.value, LLMFatalError) and isinstance(info.value, LLMRequestRejected)
    assert info.value.status_code == status and len(sdk.calls) == 1 and sleeps == []
    with pytest.raises(LLMClientStopped, match=f"LLMAccountError \\(HTTP {status}\\)"):
        client.chat([{"role": "user", "content": "another call"}], purpose="t")
    assert len(sdk.calls) == 1  # refused without touching the network
    last = json.loads(settings.ledger_path.read_text(encoding="utf-8").splitlines()[-1])
    assert last["status"] == "refused" and last["error"] == "LLMClientStopped"


def test_rate_limit_that_outlasts_the_retries_is_fatal(settings: Settings) -> None:
    sdk = FakeSDK(*[status_error(429) for _ in range(MAX_ATTEMPTS)])
    client = make_client(settings, sdk)
    with pytest.raises(LLMRateLimitPersisted) as info:
        client.chat(MESSAGES, purpose="t")
    assert isinstance(info.value, LLMUnavailable) and info.value.attempts == MAX_ATTEMPTS == len(sdk.calls)
    with pytest.raises(LLMClientStopped):
        client.chat(MESSAGES, purpose="t")
    assert len(sdk.calls) == MAX_ATTEMPTS


def test_a_429_that_recovers_is_not_fatal(settings: Settings) -> None:
    sdk = FakeSDK(status_error(429), make_response("ok"), make_response("again"))
    client = make_client(settings, sdk)
    assert client.chat(MESSAGES, purpose="t").content == "ok"
    assert client.chat([{"role": "user", "content": "next"}], purpose="t").content == "again"


def test_server_errors_after_retries_are_not_fatal(settings: Settings) -> None:
    sdk = FakeSDK(*[status_error(503) for _ in range(MAX_ATTEMPTS)], make_response("later"))
    client = make_client(settings, sdk)
    with pytest.raises(LLMUnavailable) as info:
        client.chat(MESSAGES, purpose="t")
    assert not isinstance(info.value, LLMFatalError)
    assert client.chat([{"role": "user", "content": "next"}], purpose="t").content == "later"
