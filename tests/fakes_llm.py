"""A scripted stand-in for LLMClient and small synthetic episodes for the agent tests. No network."""

from __future__ import annotations

import copy
import json
from datetime import date
from typing import Any, Callable, Mapping, Sequence

from tests.fakes_core import make_day, simple_site
from tests.fakes_data import TOU_TEMPLATE
from voltproof.agents.tools import EpisodeContext
from voltproof.data.tariff import build_tariff
from voltproof.llm.client import LLMResponse, ToolCall, Usage
from voltproof.models import DayData, Site

Item = LLMResponse | BaseException | Callable[[list[dict[str, Any]]], LLMResponse]


def call(name: str, args: Mapping[str, Any] | str | None = None, call_id: str | None = None) -> ToolCall:
    arguments = args if isinstance(args, str) else json.dumps(args or {})
    return ToolCall(id=call_id or f"id_{name}", name=name, arguments_json=arguments)


def reply(
    content: str | None = None,
    calls: Sequence[ToolCall] = (),
    *,
    prompt: int = 100,
    completion: int = 50,
    cost_usd: float = 0.001,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tuple(calls),
        reasoning=None,
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(prompt=prompt, completion=completion),
        latency_s=0.1,
        cost_usd=cost_usd,
        cache_hit=False,
        cache_key="test",
        reasoning_effort_sent="medium",
    )


class ScriptedLLM:
    """`chat()` with LLMClient's signature that plays scripted items: per purpose when a script for
    that purpose was given (e.g. review=[...]), else from the default script."""

    def __init__(self, *script: Item, **by_purpose: Sequence[Item]) -> None:
        self.script = list(script)
        self.by_purpose = {purpose: list(items) for purpose, items in by_purpose.items()}
        self.calls: list[dict[str, Any]] = []

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
        snapshot = copy.deepcopy(list(messages))
        self.calls.append(
            {"messages": snapshot, "tools": tools, "tool_choice": tool_choice, "response_format": response_format, "purpose": purpose, "tags": dict(tags or {})}
        )
        queue = self.by_purpose[purpose] if purpose in self.by_purpose else self.script
        if not queue:
            raise AssertionError(f"script exhausted (purpose={purpose})")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item(snapshot) if callable(item) else item

    def purposes(self) -> list[str]:
        return [c["purpose"] for c in self.calls]


def agent_site(**overrides: Any) -> Site:
    params = {"capacity_kwh": 10.0, "power_kw": 5.0, "eta": 1.0, "degradation": 0.01, "export_max_kw": 3.0}
    params.update(overrides)
    return simple_site(**params)


def agent_day(**overrides: Any) -> DayData:
    pv = [0.0] * 8 + [4.0] * 8 + [0.0] * 8
    params: dict[str, Any] = {"pv": pv, "load": 2.0, "critical": 1.0, "price": [5.0] * 12 + [10.0] * 12, "export_price": 1.0}
    params.update(overrides)
    return make_day(**params)


def make_context(goal: str = "Keep at least 50% from 18:00 to 22:00.", *, site: Site | None = None, day: DayData | None = None) -> EpisodeContext:
    site = site or agent_site()
    day = day or agent_day()
    tariff = build_tariff(TOU_TEMPLATE, date(2024, 6, 5), seed=2026, stream=site.site_id)
    return EpisodeContext(episode_id="test_episode", site=site, day=day, tariff=tariff, goal=goal)
