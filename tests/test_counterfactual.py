"""The E1 executor flag, plan recovery from traces, the regression check and E1 (SPEC §6.15).

Offline: a synthetic site and day; the LLM arms run for real on a scripted fake LLM, so their traces
are genuine, then the plans are recovered from those traces and re-evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.fakes_core import make_day, random_day, random_site, random_spec
from tests.fakes_llm import ScriptedLLM, agent_day, agent_site, call, make_context, reply
from tests.test_executor import SITE, balance_residual
from voltproof.agents.llm_direct import run_a2
from voltproof.agents.llm_simulate import run_a3
from voltproof.agents.tools import ArmRun
from voltproof.agents.voltproof import run_a4
from voltproof.bench import counterfactual as cf
from voltproof.bench.episode import Episode, baseline_results, evaluate
from voltproof.bench.intents import GoalInstance
from voltproof.core.executor import execute
from voltproof.models import HourWindow, IntentSpec

OUTAGE = {"grid_outages": [{"start_hour": 17, "end_hour": 20}]}  # PV is 0 then; load 2 kW, critical 1 kW
GOAL = "Power cut from 17:00 to 20:00. Keep the critical load running."


# --- the executor flag -------------------------------------------------------------------------------


def test_default_and_explicit_override_are_identical_on_random_inputs() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        site = random_site(rng)
        day = random_day(site, rng)
        spec = random_spec(rng)
        p_max = max(site.battery.p_charge_max_kw, site.battery.p_discharge_max_kw)
        setpoints = rng.uniform(-1.5 * p_max, 1.5 * p_max, 24)
        outages = [HourWindow(start_hour=a, end_hour=a + int(rng.integers(1, 6))) for a in [int(rng.integers(0, 19))]]
        kwargs: dict[str, Any] = {"import_limit_kw": [float(rng.uniform(1, 80))] * 24} if spec.import_caps else {}
        assert execute(setpoints, day, site, outages, **kwargs) == execute(setpoints, day, site, outages, outage_override=True, **kwargs)


def test_without_override_energy_balance_holds_and_the_grid_stays_off() -> None:
    rng = np.random.default_rng(11)
    for _ in range(200):
        site = random_site(rng)
        day = random_day(site, rng)
        p_max = max(site.battery.p_charge_max_kw, site.battery.p_discharge_max_kw)
        outages = [HourWindow(start_hour=a, end_hour=a + int(rng.integers(1, 6))) for a in [int(rng.integers(0, 19))]]
        result = execute(rng.uniform(-1.5 * p_max, 1.5 * p_max, 24), day, site, outages, outage_override=False)
        assert np.abs(balance_residual(result, day)).max() <= 1e-6
        assert not set(result.override_hours) & set(result.outage_hours)  # no outage override without the flag
        for t in result.outage_hours:
            assert result.flows.grid_import_kw[t] == 0 and result.flows.grid_export_kw[t] == 0


def test_without_override_an_idle_plan_sheds_non_critical_load_first() -> None:
    outage = [HourWindow(start_hour=3, end_hour=5)]
    result = execute([0.0] * 24, make_day(load=3, critical=1), SITE, outage, outage_override=False)
    assert result.battery_kw[3:5] == [0.0, 0.0] and result.overrides == 0
    assert result.unserved_noncritical_kwh == pytest.approx(4.0) and result.unserved_critical_kwh == pytest.approx(2.0)
    covered = execute([0.0] * 24, make_day(load=3, critical=1), SITE, outage)
    assert covered.unserved_critical_kwh == 0  # the frozen override covers it


def test_without_override_a_partial_discharge_loses_only_the_rest() -> None:
    plan = [0.0] * 3 + [-1.5, -1.5] + [0.0] * 19
    result = execute(plan, make_day(load=3, critical=1), SITE, [HourWindow(start_hour=3, end_hour=5)], outage_override=False)
    assert result.unserved_noncritical_kwh == pytest.approx(3.0) and result.unserved_critical_kwh == 0


def test_without_override_surplus_is_stored_and_planned_grid_charging_is_a_deficit() -> None:
    sunny = make_day(pv=[8.0] * 24, load=1)
    stored = execute([-1.0] * 24, sunny, SITE, [HourWindow(start_hour=0, end_hour=2)], outage_override=False)
    assert stored.battery_kw[:2] == pytest.approx([5.0, 0.0]) and stored.solar_absorbed_hours[:2] == [0, 1]
    assert stored.flows.curtail_kw[:2] == pytest.approx([2.0, 7.0])
    charging = execute([4.0] + [0.0] * 23, make_day(pv=1.0, load=2, critical=1), SITE, [HourWindow(start_hour=0, end_hour=1)], outage_override=False)
    assert charging.battery_kw[0] == pytest.approx(4.0)
    assert charging.unserved_noncritical_kwh == pytest.approx(1.0) and charging.unserved_critical_kwh == pytest.approx(4.0)


# --- a tiny run made by the real arms -------------------------------------------------------------


def episode(gold: dict[str, Any] = OUTAGE, template_id: str = "outage_critical") -> Episode:
    ctx = make_context(GOAL, site=agent_site(), day=agent_day())
    goal = GoalInstance(f"{template_id}_p0", template_id, "en", GOAL, IntentSpec.model_validate(gold), False)
    return Episode(f"{ctx.site.site_id}_2024-06-05_{template_id}_p0", ctx.site, ctx.day, ctx.tariff, goal)


A3_SIM = [0.0] * 8 + [2.0] * 2 + [0.0] * 14  # SOC 5 -> 9 kWh of 10
A3_FINAL = [0.0] * 8 + [2.0] * 2 + [0.0] * 7 + [-1.0] * 3 + [0.0] * 4


def a2_llm(content: str | None = None) -> ScriptedLLM:
    plan = {"battery_kw": [0.0] * 24, "explanation": "idle", "no_export_hours": [], "reserve": [{"start_hour": 17, "end_hour": 20, "min_soc_pct": 30}], "import_caps": []}
    return ScriptedLLM(reply(json.dumps(plan) if content is None else content))


def a3_llm(final: list[float] = A3_FINAL) -> ScriptedLLM:
    return ScriptedLLM(
        reply(None, [call("get_site_info", {}, "i1"), call("get_forecast", {}, "i2")]),
        reply(None, [call("simulate_plan", {"battery_kw": A3_SIM}, "s1")]),
        reply(None, [call("simulate_plan", {"battery_kw": [9.0] * 24}, "s2")]),
        reply(None, [call("finalize", {"battery_kw": final, "explanation": "charge then cover the cut"}, "f1")]),
    )


def a4_llm() -> ScriptedLLM:
    coverage = [{"item": 1, "maps_to": "grid_outages[0]"}, {"item": 2, "maps_to": "objective"}]
    return ScriptedLLM(
        reply(None, [call("set_checklist", {"items": ["grid off 17:00-20:00", "keep the bill low"]}, "c1"), call("get_forecast", {}, "c2")]),
        reply(None, [call("solve_plan", {"spec": {"reserve_windows": [{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}]}}, "s1")]),
        reply(None, [call("solve_plan", {"spec": {"grid_outages": "bad"}}, "s2")]),
        reply(None, [call("solve_plan", {"spec": OUTAGE}, "s3")]),
        reply(None, [call("finalize", {"plan_id": "p2", "explanation": "Grid off 17:00 to 20:00.", "assumptions": [], "coverage": coverage}, "f1")]),
        review=[reply('{"approve": true, "issues": []}')],
    )


def write_run(run_dir: Path, ep: Episode, runs: dict[str, ArmRun]) -> list[dict[str, Any]]:
    """A run directory as the runner writes it: episodes.jsonl rows and one trace per LLM arm."""
    (run_dir / "traces").mkdir(parents=True)
    rows = [r.model_dump(mode="json") for r in baseline_results(ep)]
    for arm, run in runs.items():
        rows.append(evaluate(ep, run).model_dump(mode="json"))
        path = cf.trace_path(run_dir, arm, ep.episode_id)
        path.write_text("".join(json.dumps(e.model_dump(mode="json")) + "\n" for e in run.trace.events), encoding="utf-8")
    (run_dir / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


@pytest.fixture()
def tiny_run(tmp_path: Path) -> tuple[Path, Episode, dict[str, ArmRun]]:
    ep = episode()
    ctx = ep.context()
    runs = {"A2": run_a2(a2_llm(), ctx), "A3": run_a3(a3_llm(), ctx), "A4": run_a4(a4_llm(), ctx)}
    write_run(tmp_path / "run", ep, runs)
    return tmp_path / "run", ep, runs


def test_recovered_plans_equal_what_the_arms_executed(tiny_run) -> None:
    run_dir, ep, runs = tiny_run
    for arm, run in runs.items():
        plan = cf.recover(arm, ep, run_dir)
        assert plan.battery_kw == run.battery_kw and plan.plan_source == run.plan_source
        assert plan.export_limit_kw == run.export_limit_kw and plan.import_limit_kw == run.import_limit_kw
        assert plan.protected_soc_kwh == run.protected_soc_kwh
    assert runs["A4"].plan_source == "finalized" and cf.recover("A4", ep, run_dir).replayed_calls == 4  # checklist + 3 solves
    assert cf.recover("A2", ep, run_dir).protected_soc_kwh is not None  # the declared reserve


def test_a2_parse_failure_and_a3_invalid_final_fall_back_like_the_arms(tmp_path: Path) -> None:
    ep = episode()
    ctx = ep.context()
    runs = {"A2": run_a2(a2_llm("not json"), ctx), "A3": run_a3(a3_llm(final=[9.0] * 24), ctx)}
    write_run(tmp_path / "run", ep, runs)
    a2, a3 = cf.recover("A2", ep, tmp_path / "run"), cf.recover("A3", ep, tmp_path / "run")
    assert (a2.battery_kw, a2.plan_source) == (None, "fallback") == (runs["A2"].battery_kw, runs["A2"].plan_source)
    assert (a3.battery_kw, a3.plan_source) == (A3_SIM, "log_physics_valid") == (runs["A3"].battery_kw, runs["A3"].plan_source)


def test_a4_replay_that_differs_from_the_trace_is_refused(tiny_run) -> None:
    run_dir, ep, _ = tiny_run
    path = cf.trace_path(run_dir, "A4", ep.episode_id)
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event["kind"] == "tool_result" and event["name"] == "solve_plan" and "plan_id" in event["data"]["result"]:
            event["data"]["result"]["energy_cost"] += 0.01
            break
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    with pytest.raises(cf.RecoveryError, match="differs"):
        cf.recover("A4", ep, run_dir)


def test_regression_check_passes_and_catches_a_changed_result(tiny_run) -> None:
    run_dir, ep, _ = tiny_run
    report = cf.regression_check(run_dir, [ep], progress=False)
    assert report["passed"] and report["rows_checked"] == 7 and report["max_abs_diff"]["realized_cost"] == 0.0
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    next(r for r in rows if r["arm"] == "A3")["realized_total_cost"] += 1e-7
    (run_dir / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    failed = cf.regression_check(run_dir, [ep], progress=False)
    assert not failed["passed"] and [(m["arm"], m["field"]) for m in failed["gate_mismatches"]] == [("A3", "realized_total_cost")]


def test_e1_rows_reproduce_the_run_and_remove_only_the_deficit_cover(tiny_run, tmp_path: Path) -> None:
    run_dir, ep, _ = tiny_run
    rows = cf.e1_rows(run_dir, [ep], progress=False)
    by_arm = {r["arm"]: r for r in rows}
    assert set(by_arm) == {"A0", "A1", "A2", "A3", "A4", "A5", "A6"}
    assert by_arm["A2"]["unserved_critical_kwh"] == 0 and by_arm["A2"]["cf_unserved_critical_kwh"] == pytest.approx(3.0)  # idle plan
    assert by_arm["A2"]["cf_unserved_noncritical_kwh"] == pytest.approx(3.0) and not by_arm["A2"]["cf_gold_satisfied"]
    assert by_arm["A3"]["cf_unserved_critical_kwh"] == 0 and by_arm["A3"]["cf_unserved_noncritical_kwh"] == pytest.approx(3.0)
    assert by_arm["A4"]["cf_unserved_critical_kwh"] == 0 and by_arm["A6"]["cf_unserved_critical_kwh"] == 0
    summary = cf.write_e1(rows, tmp_path / "e1", "tiny")
    a2 = next(r for r in summary["per_arm"] if r["arm"] == "A2")
    assert a2["episodes"] == 1 and a2["cf_episodes_any_critical_loss"] == 1 and a2["episodes_any_critical_loss"] == 0
    assert [r["arm"] for r in summary["per_arm"]] == ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    assert (tmp_path / "e1" / "e1_episodes.csv").is_file() and "conservative" in summary["caption"]


def test_e1_skips_templates_without_an_outage(tiny_run) -> None:
    run_dir, _, _ = tiny_run
    assert cf.e1_rows(run_dir, [episode({}, "cost_min")], progress=False) == []
