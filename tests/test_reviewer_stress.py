"""E3 reviewer stress test: injection rules, selection, input reconstruction, the run loop, summary.

Offline: a synthetic site; the A4 run that the reviewer inputs come from is made by the real agent on
the scripted fake LLM (see test_counterfactual), and the stress run plays scripted verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.fakes_core import simple_site
from tests.fakes_llm import ScriptedLLM, reply
from tests.test_counterfactual import a4_llm, episode, write_run
from voltproof.agents.gate import review_messages, review_assumptions
from voltproof.agents.voltproof import run_a4
from voltproof.bench import reviewer_stress as rs
from voltproof.bench.runner import EXIT_BUDGET, EXIT_OK
from voltproof.core.metrics import spec_accuracy
from voltproof.llm.client import BudgetExceeded
from voltproof.models import IntentSpec

SITE = simple_site(capacity_kwh=10, power_kw=5, import_max_kw=20, soc_min_frac=0.1)
APPROVE = reply('{"approve": true, "issues": []}')
REJECT = reply('{"approve": false, "issues": [{"kind": "wrong", "detail": "hours are off"}]}')


def spec(**fields: Any) -> IntentSpec:
    return IntentSpec.model_validate(fields)


def by_type(s: IntentSpec, eid: str = "ep") -> dict[str, rs.Injection]:
    return {i.error_type: i for i in rs.injections(s, SITE, eid)}


def changed_once(variant: IntentSpec, original: IntentSpec) -> bool:
    precision, recall = spec_accuracy(variant, original)
    return (precision, recall) != (1.0, 1.0)


# --- injection rules -------------------------------------------------------------------------------


def test_reserve_spec_gets_all_five_errors() -> None:
    original = spec(reserve_windows=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}])
    found = by_type(original)
    assert set(found) == set(rs.ERROR_TYPES)
    shifted = found["shift_hours"].spec.reserve_windows[0]
    assert (shifted.start_hour, shifted.end_hour) in ((16, 20), (20, 24)) and found["shift_hours"].removed == frozenset({18.0, 22.0})
    assert found["wrong_level"].spec.reserve_windows[0].min_soc_pct in (30, 70)
    assert found["missing_item"].spec.reserve_windows == []
    assert found["unneeded_item"].spec.min_soc_pct_all_day == 50 and found["unneeded_item"].removed is None
    wrong = found["wrong_field"].spec
    assert wrong.reserve_windows == [] and [(t.by_hour, t.min_soc_pct) for t in wrong.soc_targets] == [(18, 50)]
    assert found["wrong_field"].removed == frozenset({22.0})
    assert all(changed_once(i.spec, original) for i in found.values())


def test_rules_that_do_not_apply_are_skipped() -> None:
    outage = by_type(spec(grid_outages=[{"start_hour": 14, "end_hour": 17}]))
    assert "wrong_level" not in outage
    assert [(c.start_hour, c.end_hour, c.max_kw) for c in outage["wrong_field"].spec.import_caps] == [(14, 17, 0.0)]
    cycles = by_type(spec(max_throughput_kwh=10))
    assert set(cycles) == {"wrong_level", "missing_item", "unneeded_item"}
    assert cycles["wrong_level"].spec.max_throughput_kwh in (5.0, 15.0)


def test_levels_stay_within_the_battery_and_grid_limits() -> None:
    low = by_type(spec(min_soc_pct_all_day=20))  # 20 - 20 = 0 is below the 10 % minimum: only +20 is valid
    assert low["wrong_level"].spec.min_soc_pct_all_day == 40
    assert low["wrong_field"].spec.end_of_day_min_soc_pct == 20 and low["wrong_field"].spec.min_soc_pct_all_day is None
    assert low["unneeded_item"].spec.reserve_windows[0].min_soc_pct == 60  # all-day already used
    cap = by_type(spec(import_caps=[{"start_hour": 13, "end_hour": 17, "max_kw": 16}]))  # 24 kW is above the 20 kW grid
    assert cap["wrong_level"].spec.import_caps[0].max_kw == 8.0
    assert [(w.start_hour, w.end_hour) for w in cap["wrong_field"].spec.no_export_windows] == [(13, 17)]


def test_discharge_only_windows_are_one_item() -> None:
    original = spec(discharge_only_windows=[{"start_hour": 6, "end_hour": 8}, {"start_hour": 17, "end_hour": 23}])
    found = by_type(original)
    assert found["missing_item"].spec.discharge_only_windows is None
    wrong = found["wrong_field"].spec
    assert wrong.discharge_only_windows is None and [(w.start_hour, w.end_hour) for w in wrong.no_export_windows] == [(6, 8), (17, 23)]
    assert found["wrong_field"].removed is None  # same numbers, different field
    assert len(found["shift_hours"].spec.discharge_only_windows) == 2


def test_injections_are_seeded_per_episode_and_type() -> None:
    original = spec(grid_outages=[{"start_hour": 9, "end_hour": 12}], reserve_windows=[{"start_hour": 12, "end_hour": 20, "min_soc_pct": 40}])
    assert rs.injections(original, SITE, "a") == rs.injections(original, SITE, "a")
    picks = {rs.injections(original, SITE, eid)[2].item for eid in ("a", "b", "c", "d", "e", "f")}  # missing_item
    assert picks == {"grid_outages[0]", "reserve_windows[0]"}


def test_assumptions_stating_original_numbers() -> None:
    assert rs.states_original(frozenset({18.0, 22.0}), ["window read as [18,22)"]) is True
    assert rs.states_original(frozenset({18.0}), ["from 6 PM"]) is True
    assert rs.states_original(frozenset({18.0}), ["evening"]) is False
    assert rs.states_original(None, ["[18,22)"]) is None


# --- selection ---------------------------------------------------------------------------------------


def pool_row(i: int, site: str, template: str, language: str) -> dict[str, Any]:
    return {"episode_id": f"{site}_{i:03d}_{template}_{language}", "site_id": site, "spec_name": template, "language": language}


def test_selection_respects_quotas_caps_languages_and_distinct_goals() -> None:
    templates = [f"t{k}" for k in range(12)]
    rows = [pool_row(i, site, templates[i % 12], ("en", "es", "hi", "pt", "de")[i % 5]) for site in ("s_a", "s_b", "s_c") for i in range(30)]
    goals = {r["episode_id"]: f"{r['spec_name']}:{r['language']}:{int(r['episode_id'].split('_')[2]) % 3}" for r in rows}
    chosen = rs.select_stress(rows, goals, 40, 2026)
    assert len(chosen) == 40 and chosen == rs.select_stress(rows, goals, 40, 2026)
    sites = [r["site_id"] for r in chosen]
    assert (sites.count("s_a"), sites.count("s_b"), sites.count("s_c")) == (14, 13, 13)
    assert max(sum(r["spec_name"] == t for r in chosen) for t in templates) <= rs.TEMPLATE_CAP
    assert all(sum(r["language"] == lang for r in chosen) >= 2 for lang in ("es", "hi", "pt", "de"))
    keys = [(r["site_id"], goals[r["episode_id"]]) for r in chosen]
    assert len(set(keys)) == len(keys)


def test_the_pool_keeps_exact_accepted_reviewed_a4_specs_only() -> None:
    base = {"arm": "A4", "spec_precision": 1.0, "spec_recall": 1.0, "plan_source": "finalized", "review_approved": True, "spec_name": "evening_reserve", "episode_id": "e"}
    rows = [base, {**base, "spec_recall": 0.5}, {**base, "spec_name": "cost_min"}, {**base, "review_approved": None}, {**base, "arm": "A3"}, {**base, "plan_source": "escalated_with_plan"}]
    assert rs.stress_pool(rows) == [base]


# --- reconstruction and the run ------------------------------------------------------------------------


@pytest.fixture()
def a4_run(tmp_path: Path):
    ep = episode()
    llm = a4_llm()
    run = run_a4(llm, ep.context())
    write_run(tmp_path / "run", ep, {"A4": run})
    return tmp_path / "run", ep, llm


def test_control_reproduces_the_runs_review_call_exactly(a4_run) -> None:
    run_dir, ep, llm = a4_run
    item = rs.review_input(ep, run_dir)
    variants = rs.build_variants([item])
    control = next(v for v in variants if v["error_type"] == rs.CONTROL and v["condition"] == "with_assumptions")
    shown_spec, shown_assumptions = rs.variant_inputs(control)
    sent = next(c for c in llm.calls if c["purpose"] == "review")["messages"]
    assert review_messages(ep.context(), shown_spec, review_assumptions(shown_spec, shown_assumptions)) == sent
    blind = next(v for v in variants if v["error_type"] == rs.CONTROL and v["condition"] == "no_assumptions")
    assert blind["assumptions"] == [] and blind["spec"]["assumptions"] == []
    assert [v["condition"] for v in variants] == sorted((v["condition"] for v in variants), key=rs.CONDITIONS.index)


def test_run_variants_records_verdicts_resumes_and_summarizes(a4_run, tmp_path: Path) -> None:
    run_dir, ep, _ = a4_run
    variants = rs.build_variants([rs.review_input(ep, run_dir)])
    out = tmp_path / "results.jsonl"
    first = variants[:3]
    code, _ = rs.run_variants(ScriptedLLM(review=[APPROVE, REJECT, reply("not json")]), first, {ep.episode_id: ep}, out, concurrency=1, progress=False)
    assert code == EXIT_OK
    verdicts = {r["variant_id"]: r["approved"] for r in rs.read_jsonl(out)}
    assert list(verdicts.values()) == [True, False, None]
    code, _ = rs.run_variants(ScriptedLLM(review=[APPROVE]), variants[:4], {ep.episode_id: ep}, out, concurrency=1, progress=False)
    assert code == EXIT_OK and len(rs.read_jsonl(out)) == 4  # only the new one ran
    summary = {(r["condition"], r["error_type"]): r for r in rs.summarize(variants, rs.read_jsonl(out))}
    control = summary[("with_assumptions", rs.CONTROL)]
    assert control["flagged"] == 0 and control["variants"] == 1 and control["rate"] == 0.0


def test_budget_exceeded_stops_the_run(a4_run, tmp_path: Path) -> None:
    run_dir, ep, _ = a4_run
    variants = rs.build_variants([rs.review_input(ep, run_dir)])
    code, message = rs.run_variants(ScriptedLLM(review=[BudgetExceeded("cap")] * 3), variants[:3], {ep.episode_id: ep}, tmp_path / "r.jsonl", concurrency=1, progress=False)
    assert code == EXIT_BUDGET and "BudgetExceeded" in message and rs.read_jsonl(tmp_path / "r.jsonl") == []


def test_wilson_interval() -> None:
    p, lo, hi = rs.wilson(0, 10)
    assert p == 0 and lo == 0 and hi == pytest.approx(0.2775, abs=1e-4)
    p, lo, hi = rs.wilson(8, 10)
    assert p == 0.8 and lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    assert rs.wilson(0, 0) == (None, None, None)


def test_issue_kind_match_counts_only_flagged_variants() -> None:
    variants = [
        {"variant_id": "a", "condition": "no_assumptions", "error_type": "missing_item", "expected_kind": "missing", "assumptions_state_original": None},
        {"variant_id": "b", "condition": "no_assumptions", "error_type": "missing_item", "expected_kind": "missing", "assumptions_state_original": None},
    ]
    results = [
        {"variant_id": "a", "approved": False, "issues": [{"kind": "missing", "detail": "x"}]},
        {"variant_id": "b", "approved": False, "issues": [{"kind": "wrong", "detail": "y"}]},
    ]
    row = next(r for r in rs.summarize(variants, results) if r["condition"] == "no_assumptions" and r["error_type"] == "missing_item")
    assert row["flagged"] == 2 and row["kind_match"] == 1 and json.dumps(row)
