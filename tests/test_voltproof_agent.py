"""A4 end to end with a scripted LLM: the finalize gate, repairs, escalation, selection, metrics."""

from __future__ import annotations

import json
from typing import Any, Sequence

import pytest

from tests.fakes_llm import ScriptedLLM, agent_site, call, make_context, reply
from voltproof.agents.voltproof import run_a4
from voltproof.llm.client import LLMAccountError
from voltproof.models import IntentSpec

GOAL = "Keep at least 50% from 18:00 to 22:00."
SPEC = {"reserve_windows": [{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}]}
CHECKLIST = ["keep 50% from 18:00 to 22:00", "keep the bill low"]
COVERAGE = [{"item": 1, "maps_to": "reserve_windows[0]"}, {"item": 2, "maps_to": "objective"}]
GOOD_TEXT = "The battery keeps at least 50% from 18:00 to 22:00 at the lowest cost."
APPROVE = reply('{"approve": true, "issues": []}')
DISPUTE = reply('{"approve": false, "issues": [{"kind": "wrong", "detail": "the window should start earlier"}]}')


def opening(items: Sequence[str] = CHECKLIST) -> Any:
    return reply(None, [call("set_checklist", {"items": list(items)}, "c1"), call("get_site_info", {}, "c2"), call("get_forecast", {}, "c3"), call("get_tariff", {}, "c4")])


def solve(spec: dict | None = None) -> Any:
    return reply(None, [call("solve_plan", {"spec": SPEC if spec is None else spec}, "s")])


def finalize(plan_id: str = "p1", explanation: str = GOOD_TEXT, coverage: list | None = None, assumptions: Sequence[str] = ()) -> Any:
    args = {"plan_id": plan_id, "explanation": explanation, "assumptions": list(assumptions), "coverage": COVERAGE if coverage is None else coverage}
    return reply(None, [call("finalize", args, "f")])


def last_tool_result(llm: ScriptedLLM, agent_turn: int) -> dict[str, Any]:
    """The last tool result the agent saw at its `agent_turn`-th call (0-based; reviewer calls skipped)."""
    agent_calls = [c for c in llm.calls if c["purpose"] == "agent"]
    tools = [m for m in agent_calls[agent_turn]["messages"] if m["role"] == "tool"]
    return json.loads(tools[-1]["content"])


def test_happy_path_accepts_the_finalized_plan() -> None:
    llm = ScriptedLLM(opening(), solve(), finalize(), review=[APPROVE])
    run = run_a4(llm, make_context(GOAL))
    assert run.plan_source == "finalized" and not run.fallback_used and run.stop_reason == "finalized"
    assert run.battery_kw is not None and run.protected_soc_kwh is not None and run.spec.reserve_windows[0].min_soc_pct == 50
    assert run.import_limit_kw == [50.0] * 24  # no caps in the spec: the site limit every hour
    assert llm.purposes() == ["agent", "agent", "agent", "review"]
    m = run.metrics
    assert m["checklist_coverage"] == 1.0 and m["dropped_requirements"] == [] and m["explanation_grounded"] is True
    assert m["review_approved"] is True and m["review_disputed"] is False and m["repair_rounds"] == 0
    assert m["escalated"] is False and m["escalation"] is None and m["unknown_tool_calls"] == 0
    assert run.totals.calls == 4 and run.explanation == GOOD_TEXT
    assert {"gate", "review", "tool_call", "tool_result", "llm_call", "final"} <= {e.kind for e in run.trace.events}


def test_coverage_failure_is_refused_then_repaired() -> None:
    llm = ScriptedLLM(opening(), solve(), finalize(coverage=COVERAGE[:1]), finalize(), review=[APPROVE])
    run = run_a4(llm, make_context(GOAL))
    refusal = last_tool_result(llm, 3)
    assert refusal == {
        "accepted": False,
        "repair_round": 1,
        "repairs_left": 1,
        "failed_checks": [
            {
                "check": "coverage",
                "message": 'checklist item 2 ("keep the bill low") is not mapped. Map every item to a spec field of plan p1 '
                '(e.g. "objective", "grid_outages[0]") or to one of your assumptions ("assumptions[0]").',
            }
        ],
    }
    assert llm.purposes().count("review") == 1  # not called while a deterministic check fails
    assert run.plan_source == "finalized" and run.metrics["repair_rounds"] == 1 and run.metrics["checklist_coverage"] == 1.0


def test_ungrounded_explanation_is_refused_then_repaired() -> None:
    llm = ScriptedLLM(opening(), solve(), finalize(explanation="This saves 42.3 kWh."), finalize(), review=[APPROVE])
    run = run_a4(llm, make_context(GOAL))
    (failed,) = last_tool_result(llm, 3)["failed_checks"]
    assert failed["check"] == "grounded_explanation" and '"42.3 kWh"' in failed["message"]
    assert run.metrics["explanation_grounded"] is True and run.metrics["ungrounded_numbers"] == []


def test_reviewer_dispute_is_refused_then_approved() -> None:
    llm = ScriptedLLM(opening(), solve(), finalize(), finalize(assumptions=["evening read as 18-22"]), review=[DISPUTE, APPROVE])
    run = run_a4(llm, make_context(GOAL))
    (failed,) = last_tool_result(llm, 3)["failed_checks"]
    assert failed == {
        "check": "review",
        "message": "the independent reviewer did not approve the spec: [wrong] the window should start earlier. If you agree, "
        "fix the spec (solve_plan again) or your assumptions; if you disagree, keep the spec and state your reading in assumptions.",
    }
    assert run.plan_source == "finalized" and run.metrics["review_approved"] is True and run.metrics["review_disputed"] is False


def test_soft_failures_after_two_repairs_are_accepted_and_flagged() -> None:
    llm = ScriptedLLM(opening(), solve(), finalize(), finalize(), finalize(), review=[DISPUTE])
    run = run_a4(llm, make_context(GOAL))
    assert llm.purposes().count("review") == 1  # unchanged input: the verdict is reused
    assert run.plan_source == "finalized" and run.metrics["review_disputed"] is True and run.metrics["review_approved"] is False
    assert run.metrics["repair_rounds"] == 2 and run.metrics["escalated"] is False


def test_ungrounded_on_every_attempt_is_accepted_flagged_with_a_final_review() -> None:
    bad = "It saves 42.3 kWh."
    llm = ScriptedLLM(opening(), solve(), finalize(explanation=bad), finalize(explanation=bad), finalize(explanation=bad), review=[APPROVE])
    run = run_a4(llm, make_context(GOAL))
    assert llm.purposes().count("review") == 1  # only on the final attempt
    assert run.plan_source == "finalized" and run.metrics["explanation_grounded"] is False
    assert run.metrics["ungrounded_numbers"] == ["42.3 kWh"] and run.metrics["review_approved"] is True


def test_hard_coverage_failure_escalates_with_the_verified_plan() -> None:
    llm = ScriptedLLM(opening(), solve(), *[finalize(coverage=COVERAGE[:1]) for _ in range(3)])
    run = run_a4(llm, make_context(GOAL))
    assert run.plan_source == "escalated_with_plan" and run.battery_kw is not None and not run.fallback_used
    assert run.metrics["escalated"] is True and run.metrics["escalation"] == "with_plan"
    assert run.metrics["dropped_requirements"] == ["keep the bill low"] and run.metrics["checklist_coverage"] == 0.5
    assert "review" not in llm.purposes()


def test_missing_plan_escalates_to_a_log_plan_when_one_is_physically_valid() -> None:
    llm = ScriptedLLM(opening(), solve(), *[finalize(plan_id="p7") for _ in range(3)])
    run = run_a4(llm, make_context(GOAL))
    assert run.plan_source == "escalated_with_plan" and run.battery_kw is not None
    assert run.metrics["escalation"] == "with_plan"


def test_no_physically_valid_plan_escalates_to_the_fallback() -> None:
    llm = ScriptedLLM(opening(), *[finalize(plan_id="p1") for _ in range(3)])
    run = run_a4(llm, make_context(GOAL))
    assert run.plan_source == "escalated_fallback" and run.battery_kw is None and run.fallback_used
    assert run.metrics["escalated"] is True and run.metrics["escalation"] == "fallback"
    assert last_tool_result(llm, 3)["repairs_left"] == 0 and len(llm.calls) == 4  # the escalation ended the loop


def test_unstated_shortfall_is_refused_then_accepted_when_stated() -> None:
    ctx = make_context("Make the battery full by 1 AM.", site=agent_site(power_kw=1.0))
    spec = {"soc_targets": [{"by_hour": 1, "min_soc_pct": 100}]}
    cover = [{"item": 1, "maps_to": "soc_targets[0]"}]
    llm = ScriptedLLM(
        opening(["full by 1 AM"]),
        solve(spec),
        finalize(explanation="The battery charges as much as it can by 1 AM.", coverage=cover),
        finalize(explanation="The battery charges as much as it can, but by 1 AM it is still 4 kWh short of full.", coverage=cover),
        review=[APPROVE],
    )
    run = run_a4(llm, ctx)
    (failed,) = last_tool_result(llm, 3)["failed_checks"]
    assert failed["check"] == "verified_plan" and "short by up to 4.000000 kWh (40.000000% of capacity)" in failed["message"]
    assert run.plan_source == "finalized" and run.metrics["repair_rounds"] == 1


def test_loop_end_without_finalize_selects_the_cheapest_verified_plan() -> None:
    llm = ScriptedLLM(opening(), solve(), solve({}), reply("done"), reply("still done"))
    run = run_a4(llm, make_context(GOAL))
    assert run.stop_reason == "final_answer" and run.plan_source == "log_verified" and run.metrics["escalated"] is False
    assert run.spec.model_dump(exclude={"assumptions"}) in (IntentSpec().model_dump(exclude={"assumptions"}), IntentSpec.model_validate(SPEC).model_dump(exclude={"assumptions"}))


SHORT_SPEC = {"soc_targets": [{"by_hour": 1, "min_soc_pct": 100}]}  # 4 kWh short on a 1 kW battery


def test_loop_end_prefers_a_fully_verified_plan_over_a_cheaper_physically_valid_one() -> None:
    ctx = make_context("Full by 1 AM, or else keep the bill low.", site=agent_site(power_kw=1.0))
    llm = ScriptedLLM(opening(), solve(SHORT_SPEC), solve({"min_soc_pct_all_day": 50}), reply("done"), reply("done"))
    run = run_a4(llm, ctx)
    assert run.plan_source == "log_verified" and run.spec.min_soc_pct_all_day == 50


def test_loop_end_falls_back_to_the_cheapest_physically_valid_plan() -> None:
    ctx = make_context("Full by 1 AM.", site=agent_site(power_kw=1.0))
    llm = ScriptedLLM(opening(), solve(SHORT_SPEC), reply("done"), reply("done"))
    run = run_a4(llm, ctx)
    assert run.plan_source == "log_physics_valid" and not run.fallback_used and run.battery_kw is not None
    assert run.spec.soc_targets[0].by_hour == 1 and run.metrics["escalated"] is False


def test_loop_end_without_any_plan_falls_back() -> None:
    llm = ScriptedLLM(opening(), reply("done"), reply("done"))
    run = run_a4(llm, make_context(GOAL))
    assert run.plan_source == "fallback" and run.fallback_used and run.battery_kw is None and run.spec is None
    assert run.metrics["checklist_coverage"] == 0.0


def test_unknown_tool_calls_are_counted_in_metrics() -> None:
    llm = ScriptedLLM(reply(None, [call("send_to_battery", {}, "x")]), opening(), solve(), finalize(), review=[APPROVE])
    run = run_a4(llm, make_context(GOAL))
    assert run.metrics["unknown_tool_calls"] == 1 and run.plan_source == "finalized"


def test_fatal_error_in_the_reviewer_stops_the_run() -> None:
    llm = ScriptedLLM(opening(), solve(), finalize(), review=[LLMAccountError("402", status_code=402)])
    with pytest.raises(LLMAccountError):
        run_a4(llm, make_context(GOAL))
