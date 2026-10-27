"""Test doubles for the OpenAI SDK. No network."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import httpx
import openai

from voltproof.config import Settings
from voltproof.llm.client import LLMClient

FAKE_URL = "https://llm.example.invalid/v1/chat/completions"

MESSAGES = [
    {"role": "system", "content": "You are a test assistant."},
    {"role": "user", "content": "What is the price at hour 18?"},
]


class FakeSDK:
    """Stands in for openai.OpenAI: `.chat.completions.create(**kwargs)` plays a script.

    Each script item is a response dict (returned), an exception (raised), or a callable
    taking the kwargs (called, and its result used).
    """

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item(kwargs) if callable(item) else item


def make_response(
    content: str | None = "ok",
    *,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    **message_extra: Any,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content, **message_extra}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": usage if usage is not None else {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def status_error(status: int, *, body: Any = None, headers: dict[str, str] | None = None) -> openai.APIStatusError:
    request = httpx.Request("POST", FAKE_URL)
    body = body if body is not None else {"error": {"message": f"status {status}"}}
    response = httpx.Response(status, request=request, headers=headers or {}, json=body)
    error_class = {
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
        404: openai.NotFoundError,
        422: openai.UnprocessableEntityError,
        429: openai.RateLimitError,
    }.get(status, openai.InternalServerError if status >= 500 else openai.APIStatusError)
    return error_class(f"Error code: {status} - {body}", response=response, body=body)


def timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", FAKE_URL))


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", FAKE_URL))


def make_client(settings: Settings, sdk: FakeSDK, sleeps: list[float] | None = None) -> LLMClient:
    return LLMClient(
        settings,
        sdk_client=sdk,
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
        rng=random.Random(2026),
    )
