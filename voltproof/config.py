"""Settings from `.env` and the process environment (SPEC §5).

`load_settings` is a pure function of a mapping, so tests never need a real `.env`.
The API key is wrapped in `Secret` so it never appears in a repr, log line or error.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, TypeVar

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "llm"
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "runs" / "ledger.jsonl"

API_KEY_VAR = "BASETEN_API_KEY"
REASONING_EFFORTS = ("low", "medium", "high", "xhigh")

# Values used when a variable is absent; identical to .env.example.
DEFAULTS: Mapping[str, str] = MappingProxyType(
    {
        "LLM_BASE_URL": "https://inference.baseten.co/v1",
        "LLM_MODEL": "meta-models/muse-spark-open-1026",
        "LLM_REASONING_EFFORT": "medium",
        "LLM_MAX_OUTPUT_TOKENS": "8192",
        "LLM_TIMEOUT_S": "120",
        "PRICE_INPUT_USD_PER_M": "1.25",
        "PRICE_OUTPUT_USD_PER_M": "4.25",
        "PRICE_CACHED_INPUT_USD_PER_M": "0.15",
        "BUDGET_USD_CAP": "85",
        "DEMO_BUDGET_USD_CAP": "10",
        "LIVE_MODE": "false",
        "TRUSTED_PROXY_HOPS": "0",
        "CLIENT_IP_SOURCE": "xff",
        "LIVE_RUNS_PER_HOUR": "40",
        "DEBUG_CLIENT": "false",
        "DEMO_VIDEO_URL": "",
        "PUBLIC_BASE_URL": "",
        "BILL_READS_PER_HOUR": "20",
        "BILL_READS_PER_VISITOR": "5",
    }
)
CLIENT_IP_SOURCE_RE = re.compile(r"^(xff|none|header:[!#$%&'*+.^_`|~0-9A-Za-z-]+)$")  # header names are RFC 9110 tokens
ENV_VARS = (API_KEY_VAR, *DEFAULTS)

_TRUE_WORDS = frozenset({"true", "1", "yes", "on"})
_FALSE_WORDS = frozenset({"false", "0", "no", "off"})

N = TypeVar("N", int, float)


class ConfigError(ValueError):
    """Settings are invalid. The message lists every problem at once."""


class Secret:
    """A string that never shows its value in repr, str or format()."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return "Secret('***')" if self._value else "Secret('')"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Settings:
    baseten_api_key: Secret = field(repr=False)
    llm_base_url: str
    llm_model: str
    llm_reasoning_effort: str | None  # None: do not send the parameter
    llm_max_output_tokens: int
    llm_timeout_s: float
    price_input_usd_per_m: float
    price_output_usd_per_m: float
    price_cached_input_usd_per_m: float
    budget_usd_cap: float
    demo_budget_usd_cap: float
    live_mode: bool
    trusted_proxy_hops: int = 0  # demo app: X-Forwarded-For entries added by trusted proxies (0: use the peer address)
    client_ip_source: str = "xff"  # demo app: "xff", "header:<Name>" or "none" (no per-visitor limit)
    live_runs_per_hour: int = 40  # demo app: live runs per hour for all visitors together
    debug_client: bool = False  # demo app: enables GET /api/debug/client
    demo_video_url: str = ""  # demo app: the "Watch video" link (https), hidden when empty
    public_base_url: str = ""  # demo app: the public https address, for the absolute og:image link; no image tags when empty
    bill_reads_per_hour: int = 20  # demo app: bill images read per hour, all visitors together
    bill_reads_per_visitor: int = 5  # demo app: bill images read per hour, per visitor
    cache_dir: Path = DEFAULT_CACHE_DIR
    ledger_path: Path = DEFAULT_LEDGER_PATH


class _Reader:
    """Reads typed values from an environment mapping and collects every error."""

    def __init__(self, environ: Mapping[str, str]) -> None:
        self._environ = environ
        self.errors: list[str] = []

    def text(self, name: str) -> str:
        value = self._environ.get(name)
        return DEFAULTS[name] if value is None else value.strip()

    def number(
        self,
        name: str,
        kind: Callable[[str], N],
        *,
        low: float,
        high: float | None = None,
        low_inclusive: bool = True,
    ) -> N:
        text = self.text(name)
        try:
            value = kind(text)
        except ValueError:
            self.errors.append(f"{name}={text!r} is not a valid {getattr(kind, '__name__', 'number')}")
            return kind("0")
        if not math.isfinite(value):
            self.errors.append(f"{name}={text!r} must be a finite number")
        elif value < low or (value == low and not low_inclusive):
            self.errors.append(f"{name}={text} must be {'>=' if low_inclusive else '>'} {low:g}")
        elif high is not None and value > high:
            self.errors.append(f"{name}={text} must be <= {high:g}")
        return value

    def boolean(self, name: str) -> bool:
        text = self.text(name).lower()
        if text in _TRUE_WORDS:
            return True
        if text not in _FALSE_WORDS:
            self.errors.append(f"{name}={text!r} must be one of true/false/1/0/yes/no/on/off")
        return False

    def url(self, name: str) -> str:
        text = self.text(name)
        if not text.startswith(("http://", "https://")):
            self.errors.append(f"{name}={text!r} must start with http:// or https://")
        return text.rstrip("/")

    def non_empty(self, name: str) -> str:
        text = self.text(name)
        if not text:
            self.errors.append(f"{name} must not be empty")
        return text

    def client_ip_source(self, name: str) -> str:
        text = self.text(name)
        if not CLIENT_IP_SOURCE_RE.match(text):
            self.errors.append(f"{name}={text!r} must be xff, none or header:<Name>")
        return text

    def https_url_or_ignored(self, name: str) -> str:
        """An https:// URL, or "" for anything else: a bad optional link is dropped with a warning (the
        value is not logged), so it can never stop the app or reach the page."""
        text = self.text(name)
        if text and (not text.startswith("https://") or any(c.isspace() for c in text)):
            logging.getLogger("voltproof.config").warning("%s ignored: it must be an https:// URL", name)
            return ""
        return text

    def optional_https_url(self, name: str) -> str:
        text = self.text(name)
        if text and (not text.startswith("https://") or any(c.isspace() for c in text)):
            self.errors.append(f"{name}={text!r} must be empty or an https:// URL")
        return text

    def reasoning_effort(self, name: str) -> str | None:
        text = self.text(name).lower()
        if not text:
            return None
        if text not in REASONING_EFFORTS:
            self.errors.append(f"{name}={text!r} must be empty or one of {', '.join(REASONING_EFFORTS)}")
        return text


def load_settings(environ: Mapping[str, str]) -> Settings:
    """Parse and validate settings. Absent variables take the `.env.example` defaults.

    A missing API key is allowed here; the LLM client raises when a network call needs it.
    """
    read = _Reader(environ)
    settings = Settings(
        baseten_api_key=Secret((environ.get(API_KEY_VAR) or "").strip()),
        llm_base_url=read.url("LLM_BASE_URL"),
        llm_model=read.non_empty("LLM_MODEL"),
        llm_reasoning_effort=read.reasoning_effort("LLM_REASONING_EFFORT"),
        llm_max_output_tokens=read.number("LLM_MAX_OUTPUT_TOKENS", int, low=1, high=131072),
        llm_timeout_s=read.number("LLM_TIMEOUT_S", float, low=0, high=600, low_inclusive=False),
        price_input_usd_per_m=read.number("PRICE_INPUT_USD_PER_M", float, low=0),
        price_output_usd_per_m=read.number("PRICE_OUTPUT_USD_PER_M", float, low=0),
        price_cached_input_usd_per_m=read.number("PRICE_CACHED_INPUT_USD_PER_M", float, low=0),
        budget_usd_cap=read.number("BUDGET_USD_CAP", float, low=0, low_inclusive=False),
        demo_budget_usd_cap=read.number("DEMO_BUDGET_USD_CAP", float, low=0, low_inclusive=False),
        live_mode=read.boolean("LIVE_MODE"),
        trusted_proxy_hops=read.number("TRUSTED_PROXY_HOPS", int, low=0, high=5),
        client_ip_source=read.client_ip_source("CLIENT_IP_SOURCE"),
        live_runs_per_hour=read.number("LIVE_RUNS_PER_HOUR", int, low=1, high=1000),
        debug_client=read.boolean("DEBUG_CLIENT"),
        demo_video_url=read.https_url_or_ignored("DEMO_VIDEO_URL"),
        public_base_url=read.optional_https_url("PUBLIC_BASE_URL").rstrip("/"),
        bill_reads_per_hour=read.number("BILL_READS_PER_HOUR", int, low=1, high=1000),
        bill_reads_per_visitor=read.number("BILL_READS_PER_VISITOR", int, low=1, high=100),
    )
    if 0 < settings.budget_usd_cap < settings.demo_budget_usd_cap:
        read.errors.append(
            f"DEMO_BUDGET_USD_CAP={settings.demo_budget_usd_cap:g} must be <= "
            f"BUDGET_USD_CAP={settings.budget_usd_cap:g}"
        )
    if read.errors:
        raise ConfigError("Invalid settings:\n  - " + "\n  - ".join(read.errors))
    return settings


def get_settings(dotenv_path: Path | None = None) -> Settings:
    """Settings from the `.env` file (default: the repo's) overridden by the process environment.

    Reads the file with `dotenv_values`, so `os.environ` is never modified.
    """
    path = DOTENV_PATH if dotenv_path is None else dotenv_path
    file_values = dotenv_values(path) if path.is_file() else {}
    merged = {name: value for name, value in file_values.items() if value is not None}
    merged.update(os.environ)
    return load_settings(merged)
