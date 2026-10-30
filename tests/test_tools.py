"""Agent tools: schemas, compact outputs, checklist-first, plan book, simulator, no actuals."""

from __future__ import annotations


import numpy as np

from tests.fakes_llm import agent_day, agent_site, make_context
from voltproof.agents.gate import NumberPool
from voltproof.agents.tools import (
    ChecklistArgs,
    PlanBook,
    SimBook,
    SimulateArgs,
    SolvePlanArgs,
    forecast_info,
    intent_spec_schema,
    site_info,
    tariff_info,
)
from voltproof.llm.prompts import GLOSSARY
from voltproof.models import IntentSpec

RESERVE = {"reserve_windows": [{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}]}


def _keys(obj: object) -> list[str]:
    if isinstance(obj, dict):
        return [str(k) for k in obj] + [k for v in obj.values() for k in _keys(v)]
    if isinstance(obj, list):
        return [k for item in obj for k in _keys(item)]
    return []


def solved_book(spec: dict | None = None) -> tuple[PlanBook, dict]:
    book = PlanBook(make_context())
    book.set_checklist(ChecklistArgs(items=["keep 50% in the evening"]))
    return book, book.solve(SolvePlanArgs(spec=spec if spec is not None else RESERVE))


def test_intent_spec_schema_covers_every_field_with_a_description() -> None:
    schema = intent_spec_schema()
    assert set(schema["properties"]) == set(IntentSpec.model_fields)
    assert all("description" in prop for prop in schema["properties"].values())
    assert schema["additionalProperties"] is False


def test_site_info_has_battery_grid_critical_load_and_glossary() -> None:
    info = site_info(make_context())
    assert info["battery"]["capacity_kwh"] == 10 and info["battery"]["min_soc_pct"] == 10.0
    assert info["critical_load"] == {"name": "test fridge", "kw": 1.0}
    assert info["currency"] == "XTS" and info["glossary"] == list(GLOSSARY)
    assert "glossary" not in site_info(make_context(), glossary=False)


def test_tariff_periods_are_half_open_windows() -> None:
    info = tariff_info(make_context())
    assert info["periods"]["peak"] == [[17, 21]] and info["periods"]["cheap"] == [[0, 6]]
    assert len(info["import_price_per_kwh"]) == 24


def test_tool_outputs_never_contain_actuals() -> None:
    pv_actual, load_actual = 4.321, 2.789  # distinctive values that differ from the forecast
    day = agent_day(pv_actual=[0.0] * 8 + [pv_actual] * 8 + [0.0] * 8, load_actual=load_actual)
    ctx = make_context(day=day)
    book = PlanBook(ctx)
    book.set_checklist(ChecklistArgs(items=["x"]))
    outputs = [
        site_info(ctx),
        forecast_info(ctx),
        tariff_info(ctx),
        book.solve(SolvePlanArgs(spec=RESERVE)),
        SimBook(ctx).simulate(SimulateArgs(battery_kw=[0.0] * 24)),
    ]
    assert not [key for key in _keys(outputs) if "actual" in key]
    pool = NumberPool()
    pool.add(outputs)
    for value in (pv_actual, load_actual, pv_actual * 8, load_actual * 24):
        assert round(value, 3) not in pool.values


def test_solve_plan_requires_the_checklist_first() -> None:
    book = PlanBook(make_context())
    assert book.solve(SolvePlanArgs(spec={})) == {"error": "call set_checklist first: list what the goal requires before solving"}
    assert book.records == {}


def test_invalid_spec_returns_every_error() -> None:
    book = PlanBook(make_context())
    book.set_checklist(ChecklistArgs(items=["x"]))
    out = book.solve(SolvePlanArgs(spec={"reserve_windows": [{"start_hour": 18, "end_hour": 22, "min_soc_pct": 120}], "min_soc_pct_all_day": -5, "bogus": 1}))
    assert out["error"] == "invalid spec" and len(out["spec_errors"]) == 3
    assert any("reserve_windows[0].min_soc_pct=120" in e for e in out["spec_errors"])
    assert any("bogus" in e for e in out["spec_errors"])


def test_solve_plan_summary_ids_and_spec_relevant_hours() -> None:
    book, out = solved_book()
    assert out["plan_id"] == "p1" and out["valid"] is True and out["shortfalls"] == []
    assert out["currency"] == "XTS"
    soc = out["soc_pct_end_of_hour"]
    assert {"start", "2", "5", "17", "18", "21", "23"} <= set(soc)  # every 3rd hour + reserve hours 17-21
    assert all(soc[h] >= 50.0 - 1e-6 for h in ("17", "18", "19", "20", "21"))
    assert book.solve(SolvePlanArgs(spec={}))["plan_id"] == "p2"
    assert book.records["p1"].physics_valid and book.records["p1"].fully_verified


def test_shortfalls_are_reported_in_kwh_and_percent() -> None:
    ctx = make_context(site=agent_site(power_kw=1.0))
    book = PlanBook(ctx)
    book.set_checklist(ChecklistArgs(items=["full by 1 AM"]))
    out = book.solve(SolvePlanArgs(spec={"soc_targets": [{"by_hour": 1, "min_soc_pct": 100}]}))
    assert out["valid"] is False
    (short,) = out["shortfalls"]
    assert short["goal"] == "soc_targets[0]" and short["short_by_kwh"] == 4.0 and short["short_by_pct"] == 40.0
    assert book.records["p1"].physics_valid and not book.records["p1"].fully_verified


def test_checklist_is_locked_after_the_first_solve() -> None:
    book, _ = solved_book()
    assert "fixed" in book.set_checklist(ChecklistArgs(items=["something else"]))["error"]
    assert book.checklist == ["keep 50% in the evening"]


def test_simulate_plan_is_physics_only() -> None:
    sims = SimBook(make_context())
    ok = sims.simulate(SimulateArgs(battery_kw=[0.0] * 24))
    assert ok["valid"] is True and len(ok["soc_pct_end_of_hour"]) == 24  # no goal check at all
    bad = sims.simulate(SimulateArgs(battery_kw=[9.0] + [0.0] * 23))
    assert bad["valid"] is False and bad["violations"][0].startswith("CHARGE_LIMIT at hour 0")
    assert sims.cheapest_valid().sim_id == "s1" and [r.sim_id for r in sims.records] == ["s1", "s2"]
    assert np.isclose(ok["soc_pct_end_of_hour"][-1], 50.0)
