"""A3 LLM + simulator (SPEC §6.9): the LLM writes the schedule and tests it on a physics simulator.

Plan selection (physics only; A3 checks the goal itself, nothing here does):
1. the finalized schedule, if physics-valid with its declared export limit
2. else the cheapest physics-valid simulated schedule (no declarations)
3. else the A0 fallback
"""

from __future__ import annotations

from typing import Any

from voltproof.agents.declarations import controller_limits
from voltproof.agents.loop import ChatLLM, ToolOutcome, run_tool_loop
from voltproof.agents.tools import ArmRun, EpisodeContext, ScheduleFinalizeArgs, SimBook, a3_tools, simulate_schedule
from voltproof.llm.prompts import A3_SYSTEM, goal_message

ARM = "A3"


def run_a3(llm: ChatLLM, ctx: EpisodeContext, *, tags: dict[str, Any] | None = None, max_turns: int = 10) -> ArmRun:
    tags = {"arm": ARM, "episode": ctx.episode_id, **(tags or {})}
    sims = SimBook(ctx)
    submitted: list[ScheduleFinalizeArgs] = []

    def finalize(args: ScheduleFinalizeArgs) -> ToolOutcome:
        submitted.append(args)
        return ToolOutcome({"accepted": True}, stop=True)

    loop = run_tool_loop(
        llm, A3_SYSTEM, goal_message(ctx.goal), a3_tools(ctx, sims, finalize), purpose="llm_simulate", tags=tags, max_turns=max_turns, require_tool_end=True
    )
    run = ArmRun(
        arm=ARM,
        battery_kw=None,
        export_limit_kw=None,
        protected_soc_kwh=None,
        plan_source="fallback",
        fallback_used=True,
        stop_reason=loop.stop_reason,
        explanation=None,
        trace=loop.trace,
        totals=loop.totals,
        tool_calls=loop.tool_calls,
        repair_turns=max(0, len(sims.records) - 1),
        metrics={"unknown_tool_calls": loop.unknown_tool_calls},
    )
    if submitted:
        final = submitted[-1]
        run.explanation = final.explanation
        limits = controller_limits(final.battery_kw, final.no_export_hours, final.reserve, ctx.day, ctx.site, final.import_caps)
        if limits.errors:
            loop.trace.add("final", "declarations", summary="; ".join(limits.errors), data={"errors": limits.errors})
        if simulate_schedule(final.battery_kw, ctx, limits.export_limit_kw).valid:
            run.battery_kw, run.export_limit_kw, run.protected_soc_kwh = list(final.battery_kw), limits.export_limit_kw, limits.protected_soc_kwh
            run.import_limit_kw = limits.import_limit_kw
            run.plan_source, run.fallback_used = "finalized", False
    if run.fallback_used:
        best = sims.cheapest_valid()
        if best is not None:
            run.battery_kw, run.plan_source, run.fallback_used = best.battery_kw, "log", False
    loop.trace.add("final", "selection", summary=f"plan source: {run.plan_source}", data={"stop_reason": loop.stop_reason})
    return run
