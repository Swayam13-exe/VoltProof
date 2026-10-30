"""VoltProof Bench report on a small synthetic run directory (SPEC §6.11 aggregation rules).

Two fake sites in different currencies, three episodes each, all seven arms, hand-picked numbers so
every pooled ratio, median, win/loss/tie count and decomposition can be checked by hand. No data
file, no LLM, no held-out site.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voltproof.bench import report as rp

ARMS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
SITES = {"fake_store": "XAA", "fake_clinic": "XBB"}
BOUND = 90.0
P_HAT = 10.0
# per arm: (realized_cost, realized_total_cost) for every episode; A0's total is 100 → base Σ|A0_total| = 300 per site
COSTS = {
    "A0": [(100.0, 100.0)] * 3,
    "A1": [(95.0, 200.0), (105.0, 110.0), (100.0, 100.0)],
    "A2": [(98.0, 99.0), (99.0, 99.0), (101.0, 101.0)],
    "A3": [(97.0, 97.0), (97.0, 97.0), (97.0, 97.0)],
    "A4": [(96.0, 96.0), (96.0, 96.0), (96.0, 96.0)],
    "A5": [(95.0, 95.0), (95.0, 95.0), (95.0, 95.0)],
    "A6": [(91.0, 91.0), (91.0, 91.0), (91.0, 91.0)],
}
SATISFIED = {"A0": [False, True, False], "A1": [True, False, False], "A2": [True, True, False], "A3": [True, True, True]}
TEMPLATES = ("cost_min", "outage_critical", "evening_reserve")


def episode_ids() -> list[tuple[str, str, int]]:
    return [(f"{site}_2025-01-0{k + 1}_{TEMPLATES[k]}_p0", site, k) for site in SITES for k in range(3)]


def make_rows() -> list[dict[str, Any]]:
    rows = []
    for eid, site, k in episode_ids():
        for arm in ARMS:
            money, total = COSTS[arm][k]
            gap = (total - BOUND) / 100.0 * 100.0
            rows.append(
                {
                    "episode_id": eid,
                    "site_id": site,
                    "arm": arm,
                    "spec_name": TEMPLATES[k],
                    "currency": SITES[site],
                    "realized_cost": money,
                    "realized_total_cost": total,
                    "savings_vs_rule_pct": 1e6,  # must never be averaged into anything
                    "gap_to_oracle_pct": gap if arm == "A0" else -1e6,
                    "gold_satisfied": SATISFIED.get(arm, [True] * 3)[k],
                    "plan_physics_valid": arm not in ("A1", "A2") or k == 0,
                    "unserved_critical_kwh": 0.5 if arm == "A0" and k == 1 else 0.0,
                    "overrides": 2 if k == 1 and arm != "A6" else 0,
                    "plan_source": "baseline" if arm in ("A0", "A1", "A5", "A6") else "finalized",
                    "fallback_used": False,
                    "cost_usd": {"A2": 0.02, "A3": 0.05, "A4": 0.01}.get(arm),
                    "latency_s": {"A2": 10.0, "A3": 40.0, "A4": 8.0 + k}.get(arm),
                    "tokens_in": 1000 if arm in ("A2", "A3", "A4") else None,
                    "tokens_out": 100 if arm in ("A2", "A3", "A4") else None,
                    "spec_precision": (0.5 if k == 2 and site == "fake_clinic" else 1.0) if arm == "A4" else None,
                    "spec_recall": 1.0 if arm == "A4" else None,
                    "review_approved": True if arm == "A4" else None,
                    "repair_rounds": (1 if k == 1 else 0) if arm == "A4" else None,
                    "language": "en",
                }
            )
    return rows


def gate_trace(repairs: list[list[str]]) -> list[dict[str, Any]]:
    events = []
    for attempt, checks in enumerate(repairs + [[]], start=1):
        failed = [{"check": c, "message": "x"} for c in checks]
        events.append({"kind": "gate", "name": "finalize", "turn": attempt, "data": {"attempt": attempt, "final": False, "failed_checks": failed}})
    return events


def make_run(run_dir: Path) -> None:
    (run_dir / "traces").mkdir(parents=True)
    entries = [
        {
            "episode_id": eid,
            "site_id": site,
            "date": f"2025-01-0{k + 1}",
            "template_id": TEMPLATES[k],
            "goal_id": f"{TEMPLATES[k]}_p0",
            "language": "en",
            "held_out_template": TEMPLATES[k] == "evening_reserve",
            "held_out_site": site == "fake_clinic",
            "goal": "g",
            "gold": {},
        }
        for eid, site, k in episode_ids()
    ]
    plan = {"split": "test", "seed": 2026, "n_episodes": len(entries), "plan_sha256": "abc", "counts": {}, "episodes": entries}
    config = {"fingerprint": "f", "code_fingerprint": "c", "plan_sha256": "abc", "args": {"seed": 2026}, "settings": {"budget_usd_cap": "85", "ledger_path": str(run_dir.parent / "ledger.jsonl")}}
    (run_dir / "episodes_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in make_rows()), encoding="utf-8")
    for eid, _, k in episode_ids():
        events = gate_trace([["grounded_explanation"]] if k == 1 else [])
        if k == 2:
            events = gate_trace([["coverage", "grounded_explanation"], ["review"]])
        (run_dir / "traces" / f"A4__{eid}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def fake_references(run_dir: Path, progress: bool) -> tuple[dict[str, float], dict[str, float]]:
    ids = [eid for eid, _, _ in episode_ids()]
    return {e: P_HAT for e in ids}, {e: BOUND for e in ids}


@pytest.fixture()
def data() -> tuple[dict, dict]:
    plan_entries = {"episodes": []}
    rows = make_rows()
    for eid, site, k in episode_ids():
        plan_entries["episodes"].append(
            {"episode_id": eid, "site_id": site, "template_id": TEMPLATES[k], "goal_id": f"{TEMPLATES[k]}_p0", "language": "en", "held_out_site": site == "fake_clinic", "held_out_template": k == 2}
        )
    p_hats, bounds = fake_references(Path("."), False)
    return rp.paired(rows), rp.episode_infos(plan_entries, p_hats, bounds)


def by_arm(table: rp.Table) -> dict[str, dict[str, Any]]:
    return {r["arm"]: r for r in table.rows}


def test_money_is_pooled_per_site_with_medians_and_paired_records(data) -> None:
    by_ep, infos = data
    table = by_arm(rp.money_table(by_ep, infos, "fake_store", ARMS))
    assert table["A1"]["pooled savings vs A0"] == "+0.00%"  # (5 - 5 + 0) / 300
    assert table["A1"]["median (A0 - X) per day"] == "+0.00" and table["A1"]["W/L/T vs A0"] == "1/1/1"
    assert table["A2"]["pooled savings vs A0"] == "+0.67%" and table["A2"]["W/L/T vs A0"] == "2/1/0"
    assert table["A1"]["cost_min: pooled"] == "+5.00%" and table["A1"]["cost_min: W/L/T"] == "1/0/0"
    assert "XAA" in rp.money_table(by_ep, infos, "fake_store", ARMS).title


def test_total_cost_gap_uses_the_oracle_bound_and_a4_records(data) -> None:
    by_ep, infos = data
    table = by_arm(rp.total_cost_table(by_ep, infos, "fake_clinic", ARMS))
    assert table["A0"]["pooled gap to oracle bound"] == "+10.00%"  # 3 x 10 / 300
    assert table["A1"]["pooled gap to oracle bound"] == "+46.67%" and table["A1"]["median gap per day"] == "+20.00"  # (110 + 20 + 10) / 300
    assert table["A1"]["A4 vs arm W/L/T"] == "3/0/0" and table["A5"]["A4 vs arm W/L/T"] == "0/3/0" and table["A4"]["A4 vs arm W/L/T"] == "-"


def test_ties_use_the_currency_invariant_tolerance() -> None:
    assert rp.wlt([1.0, 1.0 + 5e-6, 1.0 + 2e-5], [1.0, 1.0, 1.0], [rp.tie_tol(10.0)] * 3) == "0/1/2"


def test_goal_bootstrap_is_paired_deterministic_and_brackets_the_estimate(data) -> None:
    by_ep, infos = data
    first = rp.goal_bootstrap(by_ep, infos)
    second = rp.goal_bootstrap(by_ep, infos)
    assert first.rows == second.rows
    row = {r["comparison"]: r for r in first.rows}["A4 - A0"]
    assert row["overall"].startswith("+66.7 pts")  # A0 meets 2 of 6
    mean, lo, hi = rp.bootstrap_mean_ci([1, 0, 1, 1, 0, 1])
    assert lo <= mean <= hi and mean == pytest.approx(4 / 6)


def test_goal_rates_cover_sites_and_held_out_groups(data) -> None:
    by_ep, infos = data
    rates = by_arm(rp.goal_rates(by_ep, infos, ARMS))
    assert rates["A0"]["overall"] == "33.3% (2/6)" and rates["A0"]["fake_clinic (held out)"] == "33.3% (1/3)"
    assert rates["A3"]["held-out templates"] == "100.0% (2/2)"


def test_error_decomposition_adds_up_to_the_a4_gap(data) -> None:
    by_ep, infos = data
    row = rp.decomposition_table(by_ep, infos).rows[0]
    assert row["A4 pooled gap"] == "+6.000%"  # 3 x (96 - 90) / 300
    assert row["translation (A4 - A5)"] == "+1.000%" and row["forecast (A5 - A6 executed)"] == "+4.000%"
    assert row["remainder (A6 executed - bound)"] == "+1.000%"
    assert sum(row["_values"].values()) == pytest.approx(6.0)


def test_outage_table_marks_a6_validity_not_applicable_and_merges_e1(data) -> None:
    by_ep, infos = data
    e1 = {"per_arm": [{"arm": a, "cf_unserved_critical_kwh": 1.5, "cf_episodes_any_critical_loss": 1, "cf_unserved_noncritical_kwh": 2.0} for a in ARMS], "per_site_arm": [], "caption": "cap"}
    table = by_arm(rp.outage_table(by_ep, infos, ARMS, e1))
    assert table["A6"]["plan physics valid"] == "n/a" and table["A1"]["plan physics valid"] == "0.0%"
    assert table["A0"]["unserved critical kWh"] == "1.000" and table["A0"]["executor overrides (hours / episodes)"] == "4 / 2"
    assert table["A4"]["E1: unserved critical kWh"] == "1.500"


def test_repair_triggers_are_read_from_the_gate_events() -> None:
    traces = {"e1": gate_trace([["grounded_explanation"]]), "e2": gate_trace([["coverage", "grounded_explanation"], ["review"]]), "e3": gate_trace([])}
    traces["e3"].append({"kind": "gate", "name": "finalize", "data": {"attempt": 3, "final": True, "failed_checks": [{"check": "review"}]}})
    repairs = rp.repair_events(traces)
    assert [(r["episode_id"], r["attempt"], r["checks"]) for r in repairs] == [("e1", 1, "grounded_explanation"), ("e2", 1, "coverage+grounded_explanation"), ("e2", 2, "review")]
    triggers = {r["check"]: r["repairs triggered (attempts failing it)"] for r in rp.gate_tables([], repairs)[1].rows}
    assert triggers == {"verified_plan": 0, "coverage": 1, "grounded_explanation": 2, "review": 1}


def test_bound_check_flags_a_mismatch_and_a_result_below_the_bound(data) -> None:
    by_ep, infos = data
    assert rp.bound_check(by_ep, infos)["passed"]
    shifted = {e: rp.EpisodeInfo(**{**i.__dict__, "bound": i.bound + 1.0}) for e, i in infos.items()}
    result = rp.bound_check(by_ep, shifted)
    assert not result["passed"] and len(result["mismatches"]) == 6 and result["below_bound"] == []
    lowered = {e: rp.EpisodeInfo(**{**i.__dict__, "bound": 95.5}) for e, i in infos.items()}
    assert {b["arm"] for b in rp.bound_check(by_ep, lowered)["below_bound"]} == {"A5", "A6"}


def test_report_writes_everything_under_out_and_leaves_the_run_untouched(tmp_path: Path) -> None:
    run_dir, out = tmp_path / "run", tmp_path / "results"
    make_run(run_dir)
    before = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))
    summary = rp.report(run_dir, out, publish=True, progress=False, references=fake_references)
    assert sorted(p.relative_to(run_dir) for p in run_dir.rglob("*")) == before
    assert summary["bound_check"]["passed"] and len(summary["figures"]) == 5
    text = (out / "report" / "REPORT.md").read_text(encoding="utf-8")
    assert "Money (realized_cost, XAA)" in text and "1000000" not in text and "1e+06" not in text  # per-episode % never used
    assert (out / "report" / "tables" / "money_fake_store.csv").is_file() and (out / "report" / "tables" / "a4_repairs.csv").is_file()
    assert all((out / f).is_file() for f in summary["figures"])
    for name in ("config.json", "episodes_plan.json", "episodes.jsonl", "traces.zip", "README.md"):
        assert (out / "test-run" / name).is_file()
    readme = (out / "test-run" / "README.md").read_text(encoding="utf-8")
    assert "Aggregation rules" in readme and summary["run_digest"] in readme
