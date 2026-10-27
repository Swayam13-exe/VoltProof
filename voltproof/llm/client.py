"""The only module that talks to the LLM (SPEC §6.8).

`LLMClient.chat` wraps the OpenAI-compatible endpoint with a disk cache, a JSONL spend
ledger, a hard budget guard, retries with backoff, and key redaction. Every error it
raises is one of its own types, built from redacted text, so no SDK object holding the
request headers ever escapes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import openai

from voltproof.config import Settings

log = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0
RETRY_AFTER_CAP_S = 60.0
BYTES_PER_TOKEN_ESTIMATE = 3  # deliberately low, so input-token estimates err high
MAX_ERROR_CHARS = 500

_AUTH_TOKEN_RE = re.compile(r"(?i)\b(bearer|api[-_]?key)(\s*[:=]?\s*['\"]?)([A-Za-z0-9._\-]{8,})")


class LLMError(RuntimeError):
    """Base class for every error raised by the client. Messages never contain the key."""

    def __init__(self, message: str, *, status_code: int | None = None, attempts: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempts = attempts


class MissingAPIKey(LLMError):
    """A network call was needed but BASETEN_API_KEY is not set."""


class BudgetExceeded(LLMError):
    """The call could push spend over BUDGET_USD_CAP (or DEMO_BUDGET_USD_CAP for demo calls)."""


class LLMRequestRejected(LLMError):
    """The server rejected the request with a non-retryable 4xx status."""


class LLMUnavailable(LLMError):
    """Every attempt failed with a retryable error (429, 5xx, timeout, connection)."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class Usage:
    prompt: int = 0
    completion: int = 0
    cached: int = 0
    reasoning: int = 0


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    reasoning: str | None
    finish_reason: str | None
    usage: Usage
    latency_s: float
    cost_usd: float
    cache_hit: bool
    cache_key: str
    reasoning_effort_sent: str | None


@dataclass(frozen=True)
class LedgerEntry:
    """One line of runs/ledger.jsonl. Never holds message content or the key."""

    ts: str
    purpose: str
    tags: dict[str, Any]
    model: str
    cache_key: str
    cache_hit: bool
    status: str  # "ok" | "error" | "refused"
    error: str | None
    attempts: int
    finish_reason: str | None
    reasoning_effort: str | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    cost_usd: float
    possible_unbilled_usd: float  # worst case for failed attempts that may have been billed
    usage_missing: bool
    latency_s: float


@dataclass(frozen=True)
class LedgerTotals:
    total_usd: float
    demo_usd: float
    calls: int
    malformed_lines: int


@dataclass(frozen=True)
class _Failure:
    """A failed attempt reduced to redacted text, so no SDK object (with headers) is kept."""

    error_type: str
    status_code: int | None
    message: str
    retryable: bool
    maybe_billed: bool
    retry_after_s: float | None = None

    @property
    def label(self) -> str:
        return f"{self.error_type} (HTTP {self.status_code})" if self.status_code else self.error_type

    def to_error(self, attempts: int) -> LLMError:
        if self.retryable:
            return LLMUnavailable(
                f"LLM unavailable after {attempts} attempt(s); last error: {self.message}",
                status_code=self.status_code,
                attempts=attempts,
            )
        if self.status_code is not None:
            return LLMRequestRejected(
                f"LLM request rejected: {self.message}", status_code=self.status_code, attempts=attempts
            )
        return LLMError(f"LLM call failed: {self.message}", attempts=attempts)


@dataclass(frozen=True)
class _SendOutcome:
    raw: dict[str, Any] | None
    failure: _Failure | None
    attempts: int
    maybe_billed_failures: int


def redact(text: str, secret: str) -> str:
    """Remove the secret and anything that looks like an auth token from `text`."""
    if secret:
        text = text.replace(secret, "***")
    return _AUTH_TOKEN_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def cache_key(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    tool_choice: Any,
    response_format: Mapping[str, Any] | None,
    reasoning_effort: str | None,
    max_tokens: int,
) -> str:
    payload = {
        "v": CACHE_FORMAT_VERSION,
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "response_format": response_format,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def estimate_cost_usd(usage: Usage, settings: Settings) -> float:
    """(prompt - cached)·in + cached·cached_in + completion·out, prices per million tokens."""
    cached = min(usage.cached, usage.prompt)
    return (
        (usage.prompt - cached) * settings.price_input_usd_per_m
        + cached * settings.price_cached_input_usd_per_m
        + usage.completion * settings.price_output_usd_per_m
    ) / 1_000_000


def worst_case_cost_usd(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    response_format: Mapping[str, Any] | None,
    settings: Settings,
) -> float:
    """Upper bound for one call: over-estimated input at full price plus `max_tokens` of output."""
    request_text = canonical_json({"messages": messages, "tools": tools, "response_format": response_format})
    input_tokens = math.ceil(len(request_text.encode("utf-8")) / BYTES_PER_TOKEN_ESTIMATE)
    return (
        input_tokens * settings.price_input_usd_per_m
        + settings.llm_max_output_tokens * settings.price_output_usd_per_m
    ) / 1_000_000


def normalize_usage(raw_usage: Any) -> Usage | None:
    """Token counts from either nested (`*_tokens_details`) or flat fields. None if absent."""
    if not isinstance(raw_usage, Mapping) or raw_usage.get("prompt_tokens") is None:
        return None
    prompt = _as_int(raw_usage.get("prompt_tokens"))
    cached = _as_int(_nested(raw_usage, "prompt_tokens_details", "cached_tokens"), raw_usage.get("cached_tokens"))
    reasoning = _as_int(
        _nested(raw_usage, "completion_tokens_details", "reasoning_tokens"), raw_usage.get("reasoning_tokens")
    )
    return Usage(
        prompt=prompt,
        completion=_as_int(raw_usage.get("completion_tokens")),
        cached=min(cached, prompt),
        reasoning=reasoning,
    )


def extract_reasoning(message: Mapping[str, Any]) -> str | None:
    """Reasoning text from `reasoning_content` or `reasoning`, whichever is present."""
    for field_name in ("reasoning_content", "reasoning"):
        value = message.get(field_name)
        if not value:
            continue
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return None


def normalize_tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
        calls.append(ToolCall(id=raw.get("id") or "", name=function.get("name") or "", arguments_json=arguments))
    return tuple(calls)


def read_ledger_totals(path: Path) -> LedgerTotals:
    """Sum `cost_usd` over the ledger; demo spend counts lines whose tags have `demo: true`."""
    if not path.is_file():
        return LedgerTotals(total_usd=0.0, demo_usd=0.0, calls=0, malformed_lines=0)
    total = demo = 0.0
    calls = malformed = 0
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                cost = float(record["cost_usd"])
            except (ValueError, KeyError, TypeError):
                malformed += 1
                continue
            if not math.isfinite(cost) or cost < 0:
                malformed += 1
                continue
            total += cost
            calls += 1
            if _is_demo(record.get("tags")):
                demo += cost
    if malformed:
        log.warning("Ledger %s: skipped %d malformed line(s)", path.name, malformed)
    return LedgerTotals(total_usd=total, demo_usd=demo, calls=calls, malformed_lines=malformed)


class LLMClient:
    """Chat completions against Muse Spark Open with cache, ledger, budget guard and retries."""

    def __init__(
        self,
        settings: Settings,
        *,
        sdk_client: Any = None,
        http_client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._sdk = sdk_client
        self._http_client = http_client
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._clock = clock
        self._lock = threading.Lock()
        self._reserved_usd = 0.0
        self._reserved_demo_usd = 0.0

    def __repr__(self) -> str:
        return f"LLMClient(model={self._settings.llm_model!r}, base_url={self._settings.llm_base_url!r})"

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: Mapping[str, Any] | None = None,
        *,
        purpose: str,
        tags: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """One chat completion. Cache hits cost $0; misses pass the budget guard first.

        `tags` is a JSON-serializable dict (e.g. arm, episode id); `{"demo": True}` also
        applies DEMO_BUDGET_USD_CAP.
        """
        s = self._settings
        tags = dict(tags or {})
        canonical_json(tags)  # fail before any call if tags cannot be written to the ledger
        key = cache_key(
            model=s.llm_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning_effort=s.llm_reasoning_effort,
            max_tokens=s.llm_max_output_tokens,
        )
        started = self._clock()
        cached = self._load_cached(key, started)
        if cached is not None:
            self._append_ledger(
                self._entry(purpose, tags, key, status="ok", cache_hit=True, response=cached, attempts=0)
            )
            return cached

        sdk = self._sdk_client()
        worst = worst_case_cost_usd(messages, tools, response_format, s)
        demo = _is_demo(tags)
        try:
            self._reserve(worst, demo)
        except BudgetExceeded:
            self._append_ledger(self._entry(purpose, tags, key, status="refused", error="BudgetExceeded"))
            raise

        error: LLMError | None = None
        try:
            outcome = self._send(sdk, self._request_kwargs(messages, tools, tool_choice, response_format))
            latency_s = self._clock() - started
            unbilled = worst * outcome.maybe_billed_failures
            if outcome.failure is not None:
                error = outcome.failure.to_error(outcome.attempts)
                self._append_ledger(
                    self._entry(
                        purpose,
                        tags,
                        key,
                        status="error",
                        error=outcome.failure.label,
                        attempts=outcome.attempts,
                        possible_unbilled_usd=unbilled,
                        latency_s=latency_s,
                    )
                )
            else:
                response, error = self._finish(key, outcome, latency_s, worst, purpose, tags, unbilled)
        finally:
            self._release(worst, demo)
        if error is not None:
            raise error
        return response

    def _finish(
        self,
        key: str,
        outcome: _SendOutcome,
        latency_s: float,
        worst: float,
        purpose: str,
        tags: dict[str, Any],
        unbilled: float,
    ) -> tuple[LLMResponse, LLMError | None]:
        """Turn a successful raw response into an LLMResponse, then cache and ledger it."""
        raw = outcome.raw or {}
        usage = normalize_usage(raw.get("usage"))
        usage_missing = usage is None
        if usage is None:
            log.warning("LLM response has no usage block; charging the worst-case estimate")
        cost = worst if usage is None else estimate_cost_usd(usage, self._settings)
        choices = raw.get("choices") or []
        message = choices[0].get("message") if choices else None
        response = LLMResponse(
            content=(message or {}).get("content"),
            tool_calls=normalize_tool_calls(message or {}),
            reasoning=extract_reasoning(message or {}),
            finish_reason=choices[0].get("finish_reason") if choices else None,
            usage=usage or Usage(),
            latency_s=latency_s,
            cost_usd=cost,
            cache_hit=False,
            cache_key=key,
            reasoning_effort_sent=self._settings.llm_reasoning_effort,
        )
        ok = isinstance(message, Mapping)
        self._append_ledger(
            self._entry(
                purpose,
                tags,
                key,
                status="ok" if ok else "error",
                error=None if ok else "MalformedResponse",
                cache_hit=False,
                response=response,
                attempts=outcome.attempts,
                possible_unbilled_usd=unbilled,
                usage_missing=usage_missing,
            )
        )
        if not ok:
            return response, LLMError("LLM response has no message", attempts=outcome.attempts)
        self._store_cached(key, response, raw, usage_missing)
        return response, None

    def _request_kwargs(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: Any,
        response_format: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        # temperature and top_p stay unset: the model card says defaults work best.
        s = self._settings
        kwargs: dict[str, Any] = {
            "model": s.llm_model,
            "messages": list(messages),
            "max_tokens": s.llm_max_output_tokens,
        }
        if tools is not None:
            kwargs["tools"] = list(tools)
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        if s.llm_reasoning_effort is not None:
            kwargs["reasoning_effort"] = s.llm_reasoning_effort
        return kwargs

    def _sdk_client(self) -> Any:
        with self._lock:
            if self._sdk is None:
                key = self._settings.baseten_api_key.reveal()
                if not key:
                    raise MissingAPIKey(
                        "BASETEN_API_KEY is not set. Copy .env.example to .env and set the key there."
                    )
                self._sdk = openai.OpenAI(
                    api_key=key,
                    base_url=self._settings.llm_base_url,
                    timeout=self._settings.llm_timeout_s,
                    max_retries=0,  # retries happen here, with our own policy
                    http_client=self._http_client,
                )
            return self._sdk

    def _send(self, sdk: Any, kwargs: dict[str, Any]) -> _SendOutcome:
        maybe_billed = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            raw, failure = self._attempt(sdk, kwargs)
            if failure is None:
                return _SendOutcome(raw=raw, failure=None, attempts=attempt, maybe_billed_failures=maybe_billed)
            maybe_billed += failure.maybe_billed
            if not failure.retryable or attempt == MAX_ATTEMPTS:
                return _SendOutcome(raw=None, failure=failure, attempts=attempt, maybe_billed_failures=maybe_billed)
            delay = self._backoff_s(attempt, failure.retry_after_s)
            log.warning(
                "LLM attempt %d/%d failed (%s); retrying in %.1f s", attempt, MAX_ATTEMPTS, failure.label, delay
            )
            self._sleep(delay)
        raise AssertionError("unreachable")

    def _attempt(self, sdk: Any, kwargs: dict[str, Any]) -> tuple[dict[str, Any] | None, _Failure | None]:
        secret = self._settings.baseten_api_key.reveal()
        try:
            return _as_dict(sdk.chat.completions.create(**kwargs)), None
        except openai.APIStatusError as exc:
            return None, _status_failure(exc, secret)
        except openai.APITimeoutError as exc:  # subclass of APIConnectionError: keep it first
            return None, _Failure(type(exc).__name__, None, _clean(str(exc), secret), True, maybe_billed=True)
        except openai.APIConnectionError as exc:
            return None, _Failure(type(exc).__name__, None, _clean(str(exc), secret), True, maybe_billed=False)
        except openai.OpenAIError as exc:
            return None, _Failure(type(exc).__name__, None, _clean(str(exc), secret), False, maybe_billed=False)

    def _backoff_s(self, attempt: int, retry_after_s: float | None) -> float:
        """Full-jitter exponential backoff, raised to the server's Retry-After when given."""
        delay = self._rng.uniform(0, min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (attempt - 1)))
        if retry_after_s is not None:
            delay = max(delay, min(retry_after_s, RETRY_AFTER_CAP_S))
        return delay

    def _reserve(self, worst_usd: float, demo: bool) -> None:
        s = self._settings
        with self._lock:
            totals = read_ledger_totals(s.ledger_path)
            committed = totals.total_usd + self._reserved_usd
            if committed + worst_usd > s.budget_usd_cap:
                raise BudgetExceeded(
                    f"Budget cap ${s.budget_usd_cap:.2f} would be exceeded: spent ${totals.total_usd:.6f}, "
                    f"in flight ${self._reserved_usd:.6f}, this call up to ${worst_usd:.6f}"
                )
            demo_committed = totals.demo_usd + self._reserved_demo_usd
            if demo and demo_committed + worst_usd > s.demo_budget_usd_cap:
                raise BudgetExceeded(
                    f"Demo budget cap ${s.demo_budget_usd_cap:.2f} would be exceeded: demo spent "
                    f"${totals.demo_usd:.6f}, in flight ${self._reserved_demo_usd:.6f}, "
                    f"this call up to ${worst_usd:.6f}"
                )
            self._reserved_usd += worst_usd
            if demo:
                self._reserved_demo_usd += worst_usd

    def _release(self, worst_usd: float, demo: bool) -> None:
        with self._lock:
            self._reserved_usd -= worst_usd
            if demo:
                self._reserved_demo_usd -= worst_usd

    def _cache_path(self, key: str) -> Path:
        return self._settings.cache_dir / key[:2] / f"{key}.json"

    def _load_cached(self, key: str, started: float) -> LLMResponse | None:
        path = self._cache_path(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry.get("v") != CACHE_FORMAT_VERSION:
                return None
            body = entry["response"]
            return LLMResponse(
                content=body["content"],
                tool_calls=tuple(ToolCall(**call) for call in body["tool_calls"]),
                reasoning=body["reasoning"],
                finish_reason=body["finish_reason"],
                usage=Usage(**entry["usage"]),
                latency_s=self._clock() - started,
                cost_usd=0.0,
                cache_hit=True,
                cache_key=key,
                reasoning_effort_sent=entry["reasoning_effort_sent"],
            )
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            log.warning("Ignoring unreadable cache entry %s", path.name)
            return None

    def _store_cached(self, key: str, response: LLMResponse, raw: dict[str, Any], usage_missing: bool) -> None:
        entry = {
            "v": CACHE_FORMAT_VERSION,
            "created": _utc_now(),
            "response": {
                "content": response.content,
                "tool_calls": [asdict(call) for call in response.tool_calls],
                "reasoning": response.reasoning,
                "finish_reason": response.finish_reason,
            },
            "usage": asdict(response.usage),
            "usage_missing": usage_missing,
            "cost_usd": response.cost_usd,
            "latency_s": response.latency_s,
            "reasoning_effort_sent": response.reasoning_effort_sent,
            "raw": raw,
        }
        try:
            _write_json_atomic(self._cache_path(key), entry)
        except (OSError, TypeError, ValueError):
            log.warning("Could not write cache entry %s", key, exc_info=True)

    def _entry(
        self,
        purpose: str,
        tags: dict[str, Any],
        key: str,
        *,
        status: str,
        error: str | None = None,
        cache_hit: bool = False,
        response: LLMResponse | None = None,
        attempts: int = 0,
        possible_unbilled_usd: float = 0.0,
        usage_missing: bool = False,
        latency_s: float | None = None,
    ) -> LedgerEntry:
        usage = response.usage if response else Usage()
        return LedgerEntry(
            ts=_utc_now(),
            purpose=purpose,
            tags=tags,
            model=self._settings.llm_model,
            cache_key=key,
            cache_hit=cache_hit,
            status=status,
            error=error,
            attempts=attempts,
            finish_reason=response.finish_reason if response else None,
            reasoning_effort=self._settings.llm_reasoning_effort,
            prompt_tokens=usage.prompt,
            completion_tokens=usage.completion,
            cached_tokens=usage.cached,
            reasoning_tokens=usage.reasoning,
            cost_usd=response.cost_usd if response else 0.0,
            possible_unbilled_usd=possible_unbilled_usd,
            usage_missing=usage_missing,
            latency_s=round(response.latency_s if response else (latency_s or 0.0), 3),
        )

    def _append_ledger(self, entry: LedgerEntry) -> None:
        path = self._settings.ledger_path
        line = json.dumps(asdict(entry), ensure_ascii=False) + "\n"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as ledger:
                ledger.write(line)


def _status_failure(exc: openai.APIStatusError, secret: str) -> _Failure:
    status = exc.status_code
    retry_after = _retry_after_s(exc.response.headers.get("retry-after")) if status == 429 else None
    return _Failure(
        error_type=type(exc).__name__,
        status_code=status,
        message=_clean(str(exc), secret),
        retryable=status == 429 or status >= 500,
        maybe_billed=status >= 500,
        retry_after_s=retry_after,
    )


def _retry_after_s(value: str | None) -> float | None:
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None  # absent, or an HTTP date: fall back to our own backoff
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _clean(text: str, secret: str) -> str:
    return redact(text, secret)[:MAX_ERROR_CHARS]


def _as_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    return dict(response)


def _as_int(*candidates: Any) -> int:
    for value in candidates:
        if value is None:
            continue
        try:
            return max(int(value), 0)
        except (TypeError, ValueError):
            continue
    return 0


def _nested(mapping: Mapping[str, Any], outer: str, inner: str) -> Any:
    details = mapping.get(outer)
    return details.get(inner) if isinstance(details, Mapping) else None


def _is_demo(tags: Any) -> bool:
    return isinstance(tags, Mapping) and tags.get("demo") is True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(obj, tmp, ensure_ascii=False, default=str)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
