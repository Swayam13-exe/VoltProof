"""A2 LLM-direct (SPEC §6.9): one call, no tools; the LLM writes the 24 setpoints itself.

Parse failures (not JSON, schema mismatch, not 24 values, a non-finite value) and non-fatal LLM
errors fall back to A0, recorded. A schedule that parses but breaks physics is NOT replaced: it is
executed as written (the executor clips it) and recorded with plan_physics_valid=False, because
measuring that is what this arm is for.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from voltproof.agents.declarations import DeclImportCap, DeclReserve, DeclWindow, controller_limits
from voltproof.agents.loop import STOP_RUN_ERRORS, ChatLLM, LLMTotals, Trace, call_data, parse_json_object, to_json
from voltproof.agents.tools import ArmRun, EpisodeContext, forecast_info, site_info, tariff_info
from voltproof.llm.client import BudgetExceeded, LLMError
from voltproof.llm.prompts import A2_SYSTEM, goal_message
from voltproof.models import HOURS

ARM = "A2"
_WINDOW = {
    "type": "object",
    "additionalProperties": False,
    "required": ["start_hour", "end_hour"],
    "properties": {"start_hour": {"type": "integer"}, "end_hour": {"type": "integer"}},
}
_RESERVE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["start_hour", "end_hour", "min_soc_pct"],
    "properties": {"start_hour": {"type": "integer"}, "end_hour": {"type": "integer"}, "min_soc_pct": {"type": "number"}},
}
_IMPORT_CAP = {
    "type": "object",
    "additionalProperties": False,
    "required": ["start_hour", "end_hour", "max_kw"],
    "properties": {"start_hour": {"type": "integer"}, "end_hour": {"type": "integer"}, "max_kw": {"type": "number"}},
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "battery_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["battery_kw", "explanation", "no_export_hours", "reserve", "import_caps"],
            "properties": {
                "battery_kw": {"type": "array", "items": {"type": "number"}, "minItems": HOURS, "maxItems": HOURS},
                "explanation": {"type": "string"},
                "no_export_hours": {"type": "array", "items": _WINDOW},
                "reserve": {"type": "array", "items": _RESERVE},
                "import_caps": {"type": "array", "items": _IMPORT_CAP},
            },
        },
    },
}


class DirectPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    battery_kw: list[float] = Field(min_length=HOURS, max_length=HOURS)
    explanation: str = ""
    no_export_hours: list[DeclWindow] = Field(default_factory=list)
    reserve: list[DeclReserve] = Field(default_factory=list)
    import_caps: list[DeclImportCap] = Field(default_factory=list)

    @field_validator("battery_kw")
    @classmethod
    def _finite(cls, values: list[float]) -> list[float]:
        bad = [t for t, v in enumerate(values) if not math.isfinite(v)]
        if bad:
            raise ValueError(f"non-finite setpoints at hours {bad}")
        return values


def direct_messages(ctx: EpisodeContext) -> list[dict[str, str]]:
    data = {"site": site_info(ctx, glossary=False), "forecast": forecast_info(ctx), "tariff": tariff_info(ctx)}
    return [
        {"role": "system", "content": A2_SYSTEM},
        {"role": "user", "content": f"Data:\n{to_json(data)}\n\n{goal_message(ctx.goal)}"},
    ]


def parse_direct(content: str | None) -> tuple[DirectPlan | None, str | None]:
    try:
        return DirectPlan.model_validate(parse_json_object(content)), None
    except ValidationError as exc:
        return None, "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    except ValueError as exc:
        return None, f"not JSON: {exc}"


def run_a2(llm: ChatLLM, ctx: EpisodeContext, *, tags: dict[str, Any] | None = None) -> ArmRun:
    tags = {"arm": ARM, "episode": ctx.episode_id, **(tags or {})}
    trace, totals = Trace(turn=1), LLMTotals()
    run = ArmRun(ARM, None, None, None, "fallback", True, "parse_failure", None, trace, totals, tool_calls=0, repair_turns=0)
    try:
        response = llm.chat(direct_messages(ctx), response_format=RESPONSE_FORMAT, purpose="llm_direct", tags=tags)
    except STOP_RUN_ERRORS:
        raise
    except BudgetExceeded:
        run.stop_reason = "budget_exceeded"
        return run
    except LLMError as exc:
        run.stop_reason = f"llm_error:{type(exc).__name__}"
        trace.add("final", "llm_error", summary=str(exc)[:200])
        return run
    totals.add(response)
    trace.add("llm_call", "llm_direct", summary=(response.content or "")[:200], data=call_data(response))
    plan, error = parse_direct(response.content)
    if plan is None:
        trace.add("final", "parse_failure", summary=(error or "")[:200], data={"error": error})
        return run
    limits = controller_limits(plan.battery_kw, plan.no_export_hours, plan.reserve, ctx.day, ctx.site, plan.import_caps)
    if limits.errors:
        trace.add("final", "declarations", summary="; ".join(limits.errors)[:200], data={"errors": limits.errors})
    run.battery_kw, run.export_limit_kw, run.protected_soc_kwh = list(plan.battery_kw), limits.export_limit_kw, limits.protected_soc_kwh
    run.import_limit_kw = limits.import_limit_kw
    run.plan_source, run.fallback_used, run.stop_reason, run.explanation = "finalized", False, "final_answer", plan.explanation
    trace.add("final", "selection", summary="plan source: finalized")
    return run
