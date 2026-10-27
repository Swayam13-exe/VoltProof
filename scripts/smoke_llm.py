"""Smoke test: one tool-calling request to Muse Spark Open through LLMClient (SPEC §6.8).

Run from the repo root (PowerShell):  python scripts\\smoke_llm.py
The first run makes one paid call; running it again must be a cache hit at $0.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltproof.config import ConfigError, Settings, get_settings  # noqa: E402
from voltproof.llm.client import (  # noqa: E402
    LLMClient,
    LLMError,
    LLMRequestRejected,
    LLMResponse,
    read_ledger_totals,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_import_price",
            "description": "Grid import price for one hour of the day, in INR per kWh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 23,
                        "description": "Hour of the day in local time, 0-23.",
                    }
                },
                "required": ["hour"],
                "additionalProperties": False,
            },
        },
    }
]

# Fixed text (no timestamps) so the second run hits the cache.
MESSAGES = [
    {
        "role": "system",
        "content": "You help operate a site with solar panels and a battery. "
        "Use the tools to look up prices; never guess them.",
    },
    {"role": "user", "content": "What does grid electricity cost at 6 PM? Look it up."},
]

PREVIEW_CHARS = 200


def _chat(settings: Settings) -> LLMResponse:
    client = LLMClient(settings)
    return client.chat(MESSAGES, tools=TOOLS, tool_choice="auto", purpose="smoke", tags={"script": "smoke_llm"})


def _effort_status(settings: Settings, response: LLMResponse) -> str:
    if settings.llm_reasoning_effort is None:
        return "not sent (LLM_REASONING_EFFORT is empty)"
    if response.cache_hit:
        return f"{settings.llm_reasoning_effort!r} accepted (cached response of an earlier accepted request)"
    return (
        f"{settings.llm_reasoning_effort!r} accepted: the server did not reject it "
        "(its effect cannot be measured from one call)"
    )


def _report(settings: Settings, response: LLMResponse, effort_status: str) -> None:
    usage = response.usage
    totals = read_ledger_totals(settings.ledger_path)
    calls = ", ".join(f"{c.name}({c.arguments_json})" for c in response.tool_calls) or "none"
    reasoning = response.reasoning or ""
    preview = " ".join(reasoning.split())[:PREVIEW_CHARS]
    rows = [
        ("finish_reason", str(response.finish_reason)),
        ("tool calls", calls),
        ("content", repr(response.content)),
        ("reasoning returned", f"yes ({len(reasoning)} chars)" if reasoning else "no"),
        ("reasoning preview", preview or "-"),
        ("reasoning_effort", effort_status),
        (
            "usage",
            f"prompt={usage.prompt} completion={usage.completion} "
            f"cached={usage.cached} reasoning={usage.reasoning}",
        ),
        ("estimated cost", f"${response.cost_usd:.6f}"),
        ("cache hit", str(response.cache_hit)),
        ("latency", f"{response.latency_s:.2f} s"),
        (
            "ledger total",
            f"${totals.total_usd:.6f} of ${settings.budget_usd_cap:.2f} cap "
            f"({totals.calls} ledger lines, {settings.ledger_path})",
        ),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}} : {value}")


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        response = _chat(settings)
        effort_status = _effort_status(settings, response)
    except LLMRequestRejected as exc:
        if settings.llm_reasoning_effort is None or "reasoning_effort" not in str(exc):
            print(f"Request rejected: {exc}", file=sys.stderr)
            return 1
        print(f"reasoning_effort={settings.llm_reasoning_effort!r} was REJECTED by the server: {exc}")
        print("Hint: set LLM_REASONING_EFFORT= (empty) in .env. Retrying once without it ...")
        settings = dataclasses.replace(settings, llm_reasoning_effort=None)
        try:
            response = _chat(settings)
        except LLMError as retry_exc:
            print(f"LLM call failed: {retry_exc}", file=sys.stderr)
            return 1
        effort_status = f"REJECTED (HTTP {exc.status_code}); this response was made without it"
    except LLMError as exc:
        print(f"LLM call failed: {exc}", file=sys.stderr)
        return 1

    _report(settings, response, effort_status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
