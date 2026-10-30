"""A2 (LLM-direct) and A3 (LLM + simulator): parsing, plan selection, fallback, declarations, evaluation."""

from __future__ import annotations

import json

import pytest

from tests.fakes_llm import ScriptedLLM, call, make_context, reply
from voltproof.agents.llm_direct import RESPONSE_FORMAT, run_a2
from voltproof.agents.llm_simulate import run_a3
from voltproof.bench.episode import Episode, evaluate
from voltproof.bench.intents import GoalInstance
from voltproof.llm.client import LLMAccountError, LLMRequestRejected
from voltproof.llm.prompts import A2_SYSTEM
from voltproof.models import IntentSpec

IDLE = [0.0] * 24
EVENING = [0.0] * 8 + [1.0] * 4 + [0.0] * 6 + [-1.0] * 4 + [0.0] * 2  # 5 → 9 → 5 kWh: physically valid


def direct(battery_kw: object, **extra: object) -> str:
    return json.dumps({"battery_kw": battery_kw, "explanation": "plan", "no_export_hours": [], "reserve": [], "import_caps": [], **extra})


def episode(ctx, gold: IntentSpec | None = None) -> Episode:
    goal = GoalInstance("g_p0", "g", "en", ctx.goal, gold or IntentSpec(), False)
    return Episode(ctx.episode_id, ctx.site, ctx.day, ctx.tariff, goal)


# --- A2 -----------------------------------------------------------------------------------


def test_a2_one_structured_call_with_all_data() -> None:
    llm = ScriptedLLM(reply(direct(EVENING)))
    ctx = make_context()
    run = run_a2(llm, ctx)
    (sent,) = llm.calls
    assert sent["purpose"] == "llm_direct" and sent["tools"] is None and sent["response_format"] == RESPONSE_FORMAT
    assert sent["messages"][0]["content"] == A2_SYSTEM
    data = json.loads(sent["messages"][1]["content"].split("Data:\n", 1)[1].split("\n\nOperator's goal:", 1)[0])
    assert set(data) == {"site", "forecast", "tariff"} and ctx.goal in sent["messages"][1]["content"]
    assert run.battery_kw == EVENING and run.plan_source == "finalized" and not run.fallback_used and run.explanation == "plan"


@pytest.mark.parametrize(
    "content",
    ["not json at all", direct([1.0] * 23), '{"battery_kw": [NaN' + ", 0" * 23 + '], "explanation": "x"}', json.dumps({"explanation": "x"})],
    ids=["not-json", "23-values", "nan", "missing-field"],
)
def test_a2_parse_failures_fall_back_to_a0(content: str) -> None:
    run = run_a2(ScriptedLLM(reply(content)), make_context())
    assert run.fallback_used and run.battery_kw is None and run.stop_reason == "parse_failure" and run.plan_source == "fallback"
    assert run.totals.calls == 1  # the spend is still counted


def test_a2_physics_invalid_plan_is_executed_and_flagged() -> None:
    ctx = make_context()
    run = run_a2(ScriptedLLM(reply(direct([50.0] * 24))), ctx)
    assert run.battery_kw == [50.0] * 24 and not run.fallback_used
    result = evaluate(episode(ctx), run)
    assert result.plan_physics_valid is False and result.fallback_used is False and result.arm == "A2"


def test_a2_declarations_map_to_controller_limits() -> None:
    content = direct(EVENING, no_export_hours=[{"start_hour": 10, "end_hour": 14}], reserve=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 40}])
    run = run_a2(ScriptedLLM(reply(content)), make_context())
    assert run.export_limit_kw[10:14] == [0.0] * 4 and run.export_limit_kw[9] == 3.0
    assert run.protected_soc_kwh is not None and max(run.protected_soc_kwh[17:22]) > 0


def test_a2_import_cap_declarations_map_to_the_import_limit() -> None:
    content = direct(EVENING, import_caps=[{"start_hour": 13, "end_hour": 17, "max_kw": 1.0}])
    run = run_a2(ScriptedLLM(reply(content)), make_context())
    assert run.import_limit_kw[13:17] == [1.0] * 4 and run.import_limit_kw[12] == 50.0
    assert "import_caps" in RESPONSE_FORMAT["json_schema"]["schema"]["required"]


def test_a2_llm_error_falls_back_and_fatal_errors_propagate() -> None:
    run = run_a2(ScriptedLLM(LLMRequestRejected("400", status_code=400)), make_context())
    assert run.fallback_used and run.stop_reason == "llm_error:LLMRequestRejected"
    with pytest.raises(LLMAccountError):
        run_a2(ScriptedLLM(LLMAccountError("402", status_code=402)), make_context())


# --- A3 -----------------------------------------------------------------------------------


def info_turn() -> object:
    return reply(None, [call("get_site_info", {}, "a"), call("get_forecast", {}, "b"), call("get_tariff", {}, "c")])


def simulate(schedule: list[float], cid: str = "s") -> object:
    return reply(None, [call("simulate_plan", {"battery_kw": schedule}, cid)])


def finalize(schedule: list[float], **declarations: object) -> object:
    return reply(None, [call("finalize", {"battery_kw": schedule, "explanation": "done", **declarations}, "f")])


def test_a3_finalized_plan_is_accepted_on_physics_only() -> None:
    # The goal asks for an evening reserve; an idle schedule ignores it but is physically valid.
    llm = ScriptedLLM(info_turn(), simulate(IDLE), finalize(IDLE))
    run = run_a3(llm, make_context("Keep 90% from 18:00 to 22:00."))
    assert run.plan_source == "finalized" and run.battery_kw == IDLE and run.stop_reason == "finalized"
    assert run.repair_turns == 0 and llm.purposes() == ["llm_simulate"] * 3


def test_a3_invalid_final_plan_falls_back_to_the_cheapest_valid_simulation() -> None:
    llm = ScriptedLLM(info_turn(), simulate(IDLE, "s1"), simulate(EVENING, "s2"), finalize([50.0] * 24))
    run = run_a3(llm, make_context())
    assert run.plan_source == "log_physics_valid" and run.battery_kw in (IDLE, EVENING) and run.export_limit_kw is None
    assert run.repair_turns == 1


def test_a3_nothing_valid_falls_back_to_a0() -> None:
    llm = ScriptedLLM(info_turn(), simulate([50.0] * 24), finalize([50.0] * 24))
    run = run_a3(llm, make_context())
    assert run.plan_source == "fallback" and run.fallback_used and run.battery_kw is None


def test_a3_declarations_pass_through() -> None:
    llm = ScriptedLLM(info_turn(), finalize(EVENING, no_export_hours=[{"start_hour": 10, "end_hour": 14}], reserve=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 40}]))
    run = run_a3(llm, make_context())
    assert run.plan_source == "finalized" and run.export_limit_kw[10:14] == [0.0] * 4 and run.protected_soc_kwh is not None


def test_evaluate_records_llm_fields_and_runs_a0_on_fallback() -> None:
    ctx = make_context()
    run = run_a3(ScriptedLLM(info_turn(), reply("I cannot do this"), reply("really")), ctx)
    result = evaluate(episode(ctx), run)
    assert result.fallback_used and result.plan_source == "fallback" and result.savings_vs_rule_pct in (0.0, None)
    assert result.tokens_in == 300 and result.cost_usd == pytest.approx(0.003) and result.episode_id == ctx.episode_id
    assert result.spec_precision is None and result.language == "en" and result.unknown_tool_calls == 0


def test_a3_import_cap_declarations_map_to_the_import_limit() -> None:
    llm = ScriptedLLM(info_turn(), finalize(EVENING, import_caps=[{"start_hour": 13, "end_hour": 17, "max_kw": 1.0}]))
    run = run_a3(llm, make_context())
    assert run.plan_source == "finalized" and run.import_limit_kw[13:17] == [1.0] * 4 and run.import_limit_kw[0] == 50.0


def test_a3_simulated_fallback_plan_carries_no_declarations() -> None:
    caps = [{"start_hour": 13, "end_hour": 17, "max_kw": 1.0}]
    llm = ScriptedLLM(info_turn(), simulate(IDLE), finalize([50.0] * 24, import_caps=caps))
    run = run_a3(llm, make_context())
    assert run.plan_source == "log_physics_valid" and run.import_limit_kw is None
