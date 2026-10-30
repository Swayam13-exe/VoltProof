"""A4 VoltProof (SPEC §6.9): the LLM writes a spec, the optimizer plans, finalize gates.

`finalize` is the only way to end. Attempt 1 is the first finalize; attempts 2 and 3 are the at
most 2 repair rounds. Each attempt runs checks 1-3 (deterministic) and returns all their reasons
together; the reviewer (check 4) runs only when 1-3 pass, and on the final attempt whenever the
hard checks pass, so every accepted plan has a review. On the final attempt:
- hard checks pass                        → accepted, soft failures flagged
- a hard check fails, a physically valid  → escalated with that plan (the finalized one, else the
  plan exists                               cheapest physically valid plan in the log); executed
- no physically valid plan exists         → escalated to the A0 fallback

Plan selection when the loop ends without a gate outcome (turn cap, budget, LLM error), in the
same order as the final attempt: the cheapest fully verified plan in the log, else the cheapest
physically valid one, else A0 (`fallback_used`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from voltproof.agents import gate
from voltproof.agents.loop import ChatLLM, LLMTotals, Tool, ToolOutcome, Trace, call_data, run_tool_loop
from voltproof.agents.tools import ArmRun, EpisodeContext, FinalizeArgs, PlanBook, PlanRecord, a4_tools
from voltproof.llm.prompts import A4_SYSTEM, goal_message
from voltproof.models import IntentSpec

ARM = "A4"
MAX_REPAIRS = 2


@dataclass(frozen=True)
class GateOutcome:
    decision: str  # accepted | escalated_with_plan | escalated_fallback
    record: PlanRecord | None
    explanation: str
    assumptions: list[str]


class A4Session:
    """State of one A4 episode: the plan book, every tool output (for grounding) and the gate."""

    def __init__(self, llm: ChatLLM, ctx: EpisodeContext, tags: dict[str, Any]) -> None:
        self.llm = llm
        self.ctx = ctx
        self.tags = tags
        self.book = PlanBook(ctx)
        self.trace = Trace()
        self.totals = LLMTotals()
        self.outputs: list[Any] = []
        self.attempts = 0
        self.outcome: GateOutcome | None = None
        self.metrics: dict[str, Any] = {}
        self._review_cache: dict[str, gate.ReviewOutcome] = {}

    def tools(self) -> list[Tool]:
        tools = a4_tools(self.ctx, self.book, self.finalize)
        return [tool if tool.name == "finalize" else _recorded(tool, self.outputs) for tool in tools]

    # --- the gate ---------------------------------------------------------------------------

    def finalize(self, args: FinalizeArgs) -> dict[str, Any] | ToolOutcome:
        self.attempts += 1
        final = self.attempts > MAX_REPAIRS
        record = self.book.records.get(args.plan_id)
        capacity = self.ctx.site.battery.capacity_kwh
        spec = record.spec if record else None

        verified = gate.check_verified_plan(args.plan_id, record, args.explanation, capacity)
        cover = gate.check_coverage(self.book.checklist or [], args.coverage, spec or IntentSpec(), args.assumptions, args.plan_id)
        if record is None:  # the coverage messages would only repeat "no such plan"
            cover = gate.CoverageResult(messages=[], dropped=list(self.book.checklist or []), coverage=0.0)
        ungrounded = gate.ungrounded_numbers(args.explanation, gate.build_pool(self.outputs, self.ctx.goal, spec))
        hard_ok = not verified and not cover.messages

        review = None
        if record is not None and hard_ok and (not ungrounded or final):
            review = self._review(record.spec, args.assumptions)

        failed = [{"check": "verified_plan", "message": m} for m in verified]
        failed += [{"check": "coverage", "message": m} for m in cover.messages]
        if ungrounded:
            failed.append({"check": "grounded_explanation", "message": gate.grounding_message(ungrounded)})
        if review is not None and review.approved is False:
            failed.append({"check": "review", "message": gate.review_message(review)})

        self._record_metrics(cover, ungrounded, review)
        self.trace.add(
            "gate",
            "finalize",
            summary=f"attempt {self.attempts}: " + ("all checks passed" if not failed else ", ".join(f["check"] for f in failed)),
            data={"attempt": self.attempts, "final": final, "plan_id": args.plan_id, "failed_checks": failed, "dropped": cover.dropped},
        )

        if not failed or (final and hard_ok):
            flags = [f"{f['check']}: not passed" for f in failed]
            self.metrics["review_disputed"] = review is not None and review.approved is False
            return self._end("accepted", record, args, {"accepted": True, "plan_id": args.plan_id, **({"flags": flags} if flags else {})})
        if final:
            return self._escalate(record, args, cover)
        return {"accepted": False, "repair_round": self.attempts, "repairs_left": MAX_REPAIRS - self.attempts, "failed_checks": failed}

    def _escalate(self, record: PlanRecord | None, args: FinalizeArgs, cover: gate.CoverageResult) -> ToolOutcome:
        plan = record if record is not None and record.physics_valid else self.book.cheapest(lambda r: r.physics_valid)
        if plan is None:
            message = "No verified plan after 2 repairs. The episode is escalated to the operator; the rule-based fallback will run."
            return self._end("escalated_fallback", None, args, {"accepted": False, "escalated": True, "message": message})
        if cover.dropped:
            items = ", ".join(f'"{text}"' for text in cover.dropped)
            message = f"Checklist items are still unmapped after 2 repairs: {items}. The episode is escalated to the operator with plan {plan.plan_id}."
        else:
            message = f"Plan {args.plan_id} did not pass verification after 2 repairs. The episode is escalated to the operator with plan {plan.plan_id}."
        return self._end("escalated_with_plan", plan, args, {"accepted": False, "escalated": True, "message": message})

    def _end(self, decision: str, record: PlanRecord | None, args: FinalizeArgs, result: dict[str, Any]) -> ToolOutcome:
        self.outcome = GateOutcome(decision, record, args.explanation, list(args.assumptions))
        self.trace.add("gate", decision, summary=result.get("message") or f"accepted plan {args.plan_id}", data=result)
        return ToolOutcome(result, stop=True)

    def _review(self, spec: IntentSpec, assumptions: list[str]) -> gate.ReviewOutcome:
        key = gate.to_json([spec.model_dump(mode="json"), assumptions])
        if key in self._review_cache:  # same input: same verdict, no second call
            return self._review_cache[key]
        outcome = gate.run_review(self.llm, self.ctx, spec, assumptions, {**self.tags, "attempt": self.attempts})
        if outcome.response is not None:
            self.totals.add(outcome.response)
            self.trace.add("llm_call", "review", data=call_data(outcome.response))
        self.trace.add(
            "review",
            "reviewer",
            summary=f"approve={outcome.approved}",
            data={"approved": outcome.approved, "issues": [i.model_dump() for i in outcome.issues], "raw": outcome.raw},
        )
        self._review_cache[key] = outcome
        return outcome

    def _record_metrics(self, cover: gate.CoverageResult, ungrounded: list[str], review: gate.ReviewOutcome | None) -> None:
        self.metrics.update(
            checklist_coverage=cover.coverage,
            dropped_requirements=cover.dropped,
            explanation_grounded=not ungrounded,
            ungrounded_numbers=ungrounded,
            repair_rounds=min(self.attempts - 1, MAX_REPAIRS),
        )
        if review is not None:
            self.metrics.update(review_approved=review.approved, review_issues=list(review.issues))


def run_a4(llm: ChatLLM, ctx: EpisodeContext, *, tags: dict[str, Any] | None = None, max_turns: int = 10) -> ArmRun:
    tags = {"arm": ARM, "episode": ctx.episode_id, **(tags or {})}
    session = A4Session(llm, ctx, tags)
    loop = run_tool_loop(
        llm,
        A4_SYSTEM,
        goal_message(ctx.goal),
        session.tools(),
        purpose="agent",
        tags=tags,
        max_turns=max_turns,
        require_tool_end=True,
        trace=session.trace,
        totals=session.totals,
    )
    return _select(session, loop.stop_reason, loop.tool_calls, loop.unknown_tool_calls)


def _select(session: A4Session, stop_reason: str, tool_calls: int, unknown: int) -> ArmRun:
    outcome = session.outcome
    metrics = {
        "unknown_tool_calls": unknown,
        "repair_rounds": session.metrics.get("repair_rounds", 0),
        "escalated": False,
        "escalation": None,
        **{k: v for k, v in session.metrics.items() if k != "repair_rounds"},
    }
    if outcome is None:
        if session.book.checklist and "checklist_coverage" not in metrics:
            metrics.update(checklist_coverage=0.0, dropped_requirements=list(session.book.checklist))
        record = session.book.cheapest(lambda r: r.fully_verified)
        source = "log_verified"
        if record is None:
            record = session.book.cheapest(lambda r: r.physics_valid)
            source = "log_physics_valid"
        if record is None:
            source = "fallback"
        explanation = None
    elif outcome.decision == "escalated_fallback":
        record, source, explanation = None, "escalated_fallback", outcome.explanation
        metrics.update(escalated=True, escalation="fallback")
    else:
        record, source, explanation = outcome.record, ("finalized" if outcome.decision == "accepted" else outcome.decision), outcome.explanation
        if outcome.decision == "escalated_with_plan":
            metrics.update(escalated=True, escalation="with_plan")
    last =session.book.last()
    spec = record.spec if record else (last.spec if last else None)
    session.trace.add("final", "selection", summary=f"plan source: {source}", data={"plan_id": record.plan_id if record else None, "stop_reason": stop_reason})
    return ArmRun(
        arm=ARM,
        battery_kw=record.plan.battery_kw if record else None,
        export_limit_kw=record.plan.export_limit_kw if record else None,
        protected_soc_kwh=record.plan.protected_soc_kwh if record else None,
        plan_source=source,
        fallback_used=record is None,
        stop_reason=stop_reason,
        explanation=explanation,
        trace=session.trace,
        totals=session.totals,
        tool_calls=tool_calls,
        repair_turns=metrics["repair_rounds"],
        spec=spec,
        metrics=metrics,
        import_limit_kw=record.plan.import_limit_kw if record else None,
    )


def _recorded(tool: Tool, outputs: list[Any]) -> Tool:
    """The same tool, with every result kept for the grounding check."""

    def run(args: Any) -> Any:
        result = tool.run(args)
        outputs.append(result.result if isinstance(result, ToolOutcome) else result)
        return result

    return replace(tool, run=run)
