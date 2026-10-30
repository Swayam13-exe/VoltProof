from __future__ import annotations

import numpy as np
import pytest

from tests.fakes_core import make_day, random_day, random_site, random_spec, simple_site
from voltproof.baselines.optimizer_arms import a6_plan
from voltproof.core.executor import execute
from voltproof.core.metrics import (
    critical_served_pct,
    episode_result,
    gold_check,
    oracle_bound,
    oracle_gate,
    pct_of_base,
    spec_accuracy,
    trajectory_total_cost,
)
from voltproof.core.verifier import verify
from voltproof.models import HourWindow, IntentSpec

SITE = simple_site(capacity_kwh=10, power_kw=5, eta=1.0, export_max_kw=3)  # E0 = 5


def spec(**fields) -> IntentSpec:
    return IntentSpec.model_validate(fields)


def test_pct_of_base_near_zero_and_negative_costs() -> None:
    assert pct_of_base(10.0, 100.0, 10.0) == pytest.approx(10.0)
    assert pct_of_base(1.0, 0.05, 10.0) is None  # |base| < 1 % of a kWh at p̂ = 10 → undefined, not huge
    assert pct_of_base(1.0, 0.1, 10.0) == pytest.approx(1000.0)  # just above the floor is still defined
    # Export-earning days: A0 earns 50 (cost −50), the arm earns 80 (cost −80) → saves 60 % of |A0|.
    assert pct_of_base(-50.0 - (-80.0), -50.0, 10.0) == pytest.approx(60.0)
    assert pct_of_base(-50.0 - (-20.0), -50.0, 10.0) == pytest.approx(-60.0)


def test_total_cost_components_and_battery_energy_change() -> None:
    day = make_day(load=2, price=10)
    result = execute([-1.0] * 4 + [0.0] * 20, day, SITE)  # drains 4 kWh, serves part of the load
    cost = trajectory_total_cost(result, IntentSpec(), day, SITE)
    assert cost.money == pytest.approx(10 * (48 - 4))
    assert cost.energy_change == pytest.approx(10 * 1.0 * 4)  # V = mean price · η_d = 10
    assert cost.net_money == pytest.approx(480.0)  # draining is not free: same as idle
    assert cost.total == pytest.approx(cost.net_money)


def test_unserved_and_goal_penalties_use_the_optimizer_weights() -> None:
    day = make_day(load=2, critical=1, price=10)
    goal = spec(grid_outages=[{"start_hour": 0, "end_hour": 6}], reserve_windows=[{"start_hour": 10, "end_hour": 12, "min_soc_pct": 80}])
    result = execute([0.0] * 24, day, SITE, goal.grid_outages)
    cost = trajectory_total_cost(result, goal, day, SITE)
    assert cost.unserved_penalty == pytest.approx(10 * (1000 * 2.0 + 100 * 6.0))
    # Reserve rows soc[9], soc[10], soc[11] all sit at 1 kWh vs 8 kWh: worst hour 7 kWh at 20p̂,
    # plus 0.2p̂ per kWh per hour over the 3 rows.
    assert cost.goal_penalty == pytest.approx(20 * 10 * 7.0 + 0.2 * 10 * 7.0 * 3)


def test_gold_check_reads_the_executed_trajectory_not_the_setpoints() -> None:
    # An idle plan during an announced outage: simulating its setpoints keeps SOC at 5 kWh, so the
    # reserve looks met; in execution the safety override discharges 4 kWh, and the reserve is missed.
    day = make_day(load=2, critical=1)
    goal = spec(
        grid_outages=[{"start_hour": 16, "end_hour": 18}],
        reserve_windows=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}],
    )
    simulated = verify([0.0] * 24, day, SITE, goal, use="actual")
    assert "RESERVE" not in {v.rule for v in simulated.violations}
    executed = execute([0.0] * 24, day, SITE, goal.grid_outages)
    assert executed.override_hours == [16, 17] and executed.soc_kwh[17] == pytest.approx(1.0)
    satisfied, violations = gold_check(executed, goal, SITE)
    assert not satisfied and {v.rule for v in violations} == {"RESERVE"}


def test_gold_check_flags_unserved_critical_during_announced_outage() -> None:
    day = make_day(load=2, critical=1)
    goal = spec(grid_outages=[{"start_hour": 0, "end_hour": 8}])
    result = execute([0.0] * 24, day, SITE, goal.grid_outages)
    satisfied, violations = gold_check(result, goal, SITE)
    assert not satisfied and "OUTAGE_DEFICIT" in {v.rule for v in violations}
    assert critical_served_pct(result, day) == pytest.approx(100 * (24 - 4) / 24)


def test_no_executed_trajectory_beats_the_oracle_lp_bound() -> None:
    """LP optimality: A6's LP objective on actual data (+V·E0) bounds every arm's realized_total_cost,
    A6's own executed total included."""
    rng = np.random.default_rng(2026)
    for _ in range(40):
        site = random_site(rng)
        day, goal = random_day(site, rng), random_spec(rng)
        a6 = a6_plan(day, site, goal)
        bound = oracle_bound(a6)
        a6_exec = execute(
            a6.battery_kw, day, site, goal.grid_outages, export_limit_kw=a6.export_limit_kw, plan_basis="actual", protected_soc_kwh=a6.protected_soc_kwh
        )
        a6_total = trajectory_total_cost(a6_exec, goal, day, site).total
        adjusted = a6_exec.overrides or a6_exec.clipped_hours or a6_exec.grid_charge_limited_hours or a6_exec.solar_absorbed_hours
        if not adjusted and not a6_exec.goal_topup_hours:
            assert a6_total == pytest.approx(bound, rel=1e-6, abs=1e-6 * max(day.import_price_per_kwh))
        p_max = max(site.battery.p_charge_max_kw, site.battery.p_discharge_max_kw)
        totals = [("A6", a6_total)]
        for _ in range(10):
            setpoints = rng.uniform(-p_max, p_max, 24)
            result = execute(setpoints, day, site, goal.grid_outages)
            totals.append(("random", trajectory_total_cost(result, goal, day, site).total))
        assert oracle_gate(totals, bound, max(day.import_price_per_kwh)) == []


def test_gap_to_oracle_uses_the_lp_bound() -> None:
    site = simple_site(capacity_kwh=10, power_kw=5)
    day = make_day(load=2, price=[1.0] * 12 + [10.0] * 12)
    a6 = a6_plan(day, site, IntentSpec())
    bound = oracle_bound(a6)
    idle = execute([0.0] * 24, day, site)
    idle_cost = trajectory_total_cost(idle, IntentSpec(), day, site)
    result = episode_result(
        arm="A0",
        spec_name="no_goal",
        day=day,
        site=site,
        gold_spec=IntentSpec(),
        execution=idle,
        plan_check=verify([0.0] * 24, day, site, physics_only=True),
        a0_money=idle_cost.net_money,
        a0_total=idle_cost.total,
        oracle_total=bound,
    )
    assert result.gap_to_oracle_pct == pytest.approx((idle_cost.total - bound) / abs(idle_cost.total) * 100)
    assert result.gap_to_oracle_pct > 0
    with pytest.raises(ValueError, match="actual data"):
        oracle_bound(a6.model_copy(update={"use": "forecast"}))


def test_episode_result_fields() -> None:
    day = make_day(load=2, price=10)
    goal = IntentSpec()
    a0 = execute([0.0] * 24, day, SITE)
    a0_cost = trajectory_total_cost(a0, goal, day, SITE)
    result = episode_result(
        arm="A0",
        spec_name="no_goal",
        day=day,
        site=SITE,
        gold_spec=goal,
        execution=a0,
        plan_check=verify([0.0] * 24, day, SITE, goal, physics_only=True),
        a0_money=a0_cost.net_money,
        a0_total=a0_cost.total,
        oracle_total=a0_cost.total,
    )
    assert result.savings_vs_rule_pct == pytest.approx(0.0) and result.gap_to_oracle_pct == pytest.approx(0.0)
    assert result.gold_satisfied and result.plan_physics_valid and result.critical_served_pct == 100.0
    assert result.currency == "XTS" and result.realized_cost == pytest.approx(480.0)


def test_spec_accuracy() -> None:
    gold = spec(
        grid_outages=[{"start_hour": 14, "end_hour": 17}],
        reserve_windows=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}],
        import_caps=[{"start_hour": 12, "end_hour": 15, "max_kw": 5}],
        max_throughput_kwh=40,
    )
    exact = spec(
        grid_outages=[{"start_hour": 14, "end_hour": 15}, {"start_hour": 15, "end_hour": 17}],  # merged → match
        reserve_windows=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50.9}],  # within ±1 pp
        import_caps=[{"start_hour": 12, "end_hour": 15, "max_kw": 5.1}],  # within ±0.1 kW
        max_throughput_kwh=40.3,  # within 1 %
    )
    assert spec_accuracy(exact, gold) == (1.0, 1.0)
    off = spec(
        grid_outages=[{"start_hour": 14, "end_hour": 18}],  # wrong hours
        reserve_windows=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 52}],  # 2 pp off
        no_export_windows=[{"start_hour": 10, "end_hour": 12}],  # not in gold
    )
    assert spec_accuracy(off, gold) == (0.0, 0.0)
    assert spec_accuracy(IntentSpec(), IntentSpec()) == (1.0, 1.0)
    assert spec_accuracy(IntentSpec(), gold) == (1.0, 0.0)
    half = spec(grid_outages=[{"start_hour": 14, "end_hour": 17}], end_of_day_min_soc_pct=30)
    assert spec_accuracy(half, gold) == (0.5, 0.25)


def test_executed_outage_is_feasible_for_a6() -> None:
    site = simple_site(capacity_kwh=20, power_kw=5)
    day = make_day(load=2, critical=1, price=10)
    goal = spec(grid_outages=[{"start_hour": 14, "end_hour": 17}])
    plan = a6_plan(day, site, goal)
    result = execute(plan.battery_kw, day, site, [HourWindow(start_hour=14, end_hour=17)], export_limit_kw=plan.export_limit_kw)
    assert result.unserved_critical_kwh == 0 and result.unserved_noncritical_kwh == 0
