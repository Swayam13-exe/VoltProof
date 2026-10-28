from __future__ import annotations

import numpy as np
import pytest

from tests.fakes_core import make_day, random_day, random_site, random_spec, simple_site
from voltproof.core.optimizer import solve
from voltproof.core.verifier import verify
from voltproof.models import IntentSpec

CHEAP_THEN_EXPENSIVE = [1.0] * 12 + [10.0] * 12


def spec(**fields) -> IntentSpec:
    return IntentSpec.model_validate(fields)


def test_flat_prices_no_pv_battery_idle() -> None:
    plan = solve(make_day(pv=0, load=2, price=10), simple_site(degradation=0.1))
    assert np.allclose(plan.battery_kw, 0.0, atol=1e-9)
    assert plan.energy_cost == pytest.approx(24 * 2 * 10)
    assert plan.shortfalls == [] and plan.status == "optimal"


def test_cheap_then_expensive_charges_then_discharges_limited_by_energy() -> None:
    site = simple_site(capacity_kwh=10, power_kw=5, degradation=0.01)  # E0 = 5, E_min = 1, E_max = 10
    plan = solve(make_day(pv=0, load=2, price=CHEAP_THEN_EXPENSIVE), site)
    b = np.array(plan.battery_kw)
    assert b[:12].min() >= -1e-9 and b[12:].max() <= 1e-9
    assert b[:12].sum() == pytest.approx(5.0)  # fills the remaining 5 kWh of capacity
    assert -b[12:].sum() == pytest.approx(9.0)  # empties down to the 1 kWh minimum
    assert -b.min() <= 2.0 + 1e-9  # no export allowed: discharge cannot exceed the 2 kW load
    assert max(plan.soc_kwh) == pytest.approx(10.0) and plan.soc_kwh[-1] == pytest.approx(1.0)


def test_cheap_then_expensive_limited_by_power() -> None:
    prices = [10.0] * 10 + [1.0] * 2 + [10.0] * 6 + [30.0] * 6
    site = simple_site(capacity_kwh=10, power_kw=1.5, degradation=0.01)
    plan = solve(make_day(pv=0, load=2, price=prices), site)
    b = np.array(plan.battery_kw)
    assert b[10:12].tolist() == pytest.approx([1.5, 1.5])  # only 2 cheap hours at 1.5 kW
    assert b.max() <= 1.5 + 1e-9 and -b.min() <= 1.5 + 1e-9


def test_outage_with_too_little_energy_reports_unserved_critical() -> None:
    site = simple_site(capacity_kwh=10, power_kw=5, degradation=0.0)  # usable 4 kWh from E0 = 5
    plan = solve(make_day(pv=0, load=2, critical=1), site, spec(grid_outages=[{"start_hour": 0, "end_hour": 6}]))
    assert -sum(plan.battery_kw[:6]) == pytest.approx(4.0)
    assert plan.unserved_noncritical_kwh == pytest.approx(6.0)  # all 6 h of non-critical load shed first
    assert plan.unserved_critical_kwh == pytest.approx(2.0)
    assert all(plan.flows.grid_import_kw[t] == 0 for t in range(6))


def test_reserve_window_met_exactly_at_the_bound() -> None:
    prices = [10.0] * 18 + [20.0] * 4 + [10.0] * 2
    site = simple_site(capacity_kwh=10, power_kw=5, degradation=0.01)
    plan = solve(make_day(pv=0, load=2, price=prices), site, spec(reserve_windows=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}]))
    assert plan.shortfalls == []
    assert plan.soc_kwh[21] == pytest.approx(5.0, abs=1e-6)  # discharges right down to the reserve
    assert min(plan.soc_kwh[17:22]) >= 5.0 - 1e-6
    assert verify(plan.battery_kw, make_day(pv=0, load=2, price=prices), site, plan.spec, export_limit_kw=plan.export_limit_kw).valid


def test_impossible_goal_reports_shortfall_without_crashing() -> None:
    site = simple_site(capacity_kwh=10, power_kw=1, degradation=0.0)  # E0 = 5 kWh
    goal = spec(
        reserve_windows=[{"start_hour": 0, "end_hour": 2, "min_soc_pct": 100}],
        soc_targets=[{"by_hour": 2, "min_soc_pct": 100}],
    )
    plan = solve(make_day(pv=0, load=2), site, goal)
    by_path = {s.path: s for s in plan.shortfalls}
    reserve = by_path["reserve_windows[0]"]
    assert reserve.rule == "RESERVE" and reserve.amount == pytest.approx(5.0) and -1 in reserve.hours
    target = by_path["soc_targets[0]"]
    assert target.amount == pytest.approx(3.0)  # 5 + 2 h × 1 kW = 7 kWh of the 10 required
    assert "cannot be fully met" in target.message


def test_no_simultaneous_charge_and_discharge_on_random_days() -> None:
    rng = np.random.default_rng(2026)
    for _ in range(100):
        site = random_site(rng)
        plan = solve(random_day(site, rng), site, random_spec(rng))
        assert plan.simultaneous_hours == [], plan.status
        c, d = np.array(plan.flows.charge_kw), np.array(plan.flows.discharge_kw)
        assert np.all(np.minimum(c, d) <= 1e-6)


def _scaled(day, site, factor):
    day2 = day.model_copy(
        update={
            "import_price_per_kwh": [p * factor for p in day.import_price_per_kwh],
            "export_price_per_kwh": [p * factor for p in day.export_price_per_kwh],
        }
    )
    battery = site.battery.model_copy(update={"degradation_cost_per_kwh": site.battery.degradation_cost_per_kwh * factor})
    return day2, site.model_copy(update={"battery": battery})


@pytest.mark.parametrize("factor", [0.001, 1000.0])
def test_currency_invariance(factor: float) -> None:
    rng = np.random.default_rng(7)
    for _ in range(25):
        site = random_site(rng)
        day, goal = random_day(site, rng), random_spec(rng)
        base = solve(day, site, goal)
        scaled = solve(*_scaled(day, site, factor), goal)
        np.testing.assert_allclose(scaled.battery_kw, base.battery_kw, atol=1e-6)
        assert scaled.energy_cost == pytest.approx(base.energy_cost * factor, rel=1e-6, abs=1e-6 * factor)
        assert [s.path for s in scaled.shortfalls] == [s.path for s in base.shortfalls]


# --- priority order: critical > non-critical > goals > throughput > money -------------------


def test_a_tight_import_cap_is_exceeded_rather_than_shedding_load() -> None:
    site = simple_site(capacity_kwh=10, power_kw=5, soc_min_frac=0.1, initial_soc_frac=0.1)
    load = [2.0] * 12 + [10.0] * 3 + [2.0] * 9
    plan = solve(make_day(pv=0, load=load, critical=1), site, spec(import_caps=[{"start_hour": 12, "end_hour": 15, "max_kw": 2}]))
    assert plan.unserved_noncritical_kwh == pytest.approx(0.0, abs=1e-6)
    assert plan.unserved_critical_kwh == pytest.approx(0.0, abs=1e-6)
    caps = [s for s in plan.shortfalls if s.rule == "IMPORT_CAP"]
    assert caps and caps[0].amount > 1.0
    assert sum(plan.battery_kw[:12]) > 0  # it still pre-charges to shrink the excess


WHOLE_DAY_RESERVE = {"start_hour": 0, "end_hour": 24, "min_soc_pct": 50}


def test_b_outage_against_a_whole_day_reserve_serves_all_load_first() -> None:
    # 24-hour reserve at 50 % (10 kWh) + outage 0–4 with 2 kW load (1 kW critical): the battery has 8 kWh
    # above its minimum, exactly the 8 kWh deficit. Load (critical AND non-critical) beats the reserve.
    site = simple_site(capacity_kwh=20, power_kw=5, degradation=0.0)  # E0 = 10, E_min = 2
    goal = spec(grid_outages=[{"start_hour": 0, "end_hour": 4}], reserve_windows=[WHOLE_DAY_RESERVE])
    plan = solve(make_day(pv=0, load=2, critical=1), site, goal)
    assert plan.unserved_critical_kwh == pytest.approx(0.0, abs=1e-6)
    assert plan.unserved_noncritical_kwh == pytest.approx(0.0, abs=1e-6)
    assert -sum(plan.battery_kw[:4]) == pytest.approx(8.0)
    (reserve,) = [s for s in plan.shortfalls if s.rule == "RESERVE"]
    assert reserve.amount == pytest.approx(8.0)  # worst hour: 2 kWh left at the end of hour 3


def test_b_critical_first_when_energy_is_short() -> None:
    site = simple_site(capacity_kwh=10, power_kw=5, degradation=0.0)  # E0 = 5, E_min = 1: only 4 kWh usable
    goal = spec(grid_outages=[{"start_hour": 0, "end_hour": 4}], reserve_windows=[WHOLE_DAY_RESERVE])
    plan = solve(make_day(pv=0, load=2, critical=1), site, goal)
    assert plan.unserved_critical_kwh == pytest.approx(0.0, abs=1e-6)  # 4 kWh go to the 4 kWh of critical load
    assert plan.unserved_noncritical_kwh == pytest.approx(4.0)


def test_soc_shortfall_costs_the_same_over_short_and_long_windows() -> None:
    from voltproof.core.rules import Trajectory, goal_penalty
    from voltproof.core.spec import soft_rows

    site = simple_site(capacity_kwh=10, power_kw=5, degradation=0.0)  # E0 = 5 kWh
    short = spec(reserve_windows=[{"start_hour": 0, "end_hour": 2, "min_soc_pct": 60}])
    long = spec(reserve_windows=[{"start_hour": 0, "end_hour": 24, "min_soc_pct": 60}])
    # Optimizer: the initial SOC is 1 kWh under the 6 kWh reserve whatever it does; it recharges in
    # hour 0, so only the initial row is short: 20·p̂·1 (worst hour) + 0.2·p̂·1 (that one hour), either way.
    day = make_day(pv=0, load=2, price=10)
    for goal in (short, long):
        plan = solve(day, site, goal)
        assert plan.penalty_cost == pytest.approx(20 * 10 * 1.0 + 0.2 * 10 * 1.0, abs=0.01)
        assert plan.soc_kwh[0] >= 6.0 - 1e-6
    # Metrics: a trajectory 1 kWh short throughout costs the same worst-hour 20·p̂ over 2 or 24 hours;
    # only the small hourly term grows (3 rows vs 25 rows, initial SOC included).
    flat = np.full(24, 5.0)
    zeros = np.zeros(24)
    trajectory = Trajectory(5.0, flat, zeros, zeros, zeros, zeros)
    assert goal_penalty(soft_rows(short, site), trajectory, 10.0) == pytest.approx(200.0 + 0.2 * 10 * 3)
    assert goal_penalty(soft_rows(long, site), trajectory, 10.0) == pytest.approx(200.0 + 0.2 * 10 * 25)


def test_after_an_outage_the_battery_recovers_to_the_reserve_sooner(monkeypatch: pytest.MonkeyPatch) -> None:
    import voltproof.core.rules as rules

    site = simple_site(capacity_kwh=20, power_kw=5, degradation=0.0)  # E0 = 10, E_min = 2
    goal = spec(grid_outages=[{"start_hour": 0, "end_hour": 4}], reserve_windows=[WHOLE_DAY_RESERVE])
    day = make_day(pv=0, load=2, critical=1, price=10)

    def first_hour_back_at_reserve(plan) -> int | None:
        return next((t for t in range(4, 24) if plan.soc_kwh[t] >= 10.0 - 1e-6), None)

    with_term = solve(day, site, goal)
    monkeypatch.setattr(rules, "HOURLY_SOC_WEIGHT", 0.0)
    without_term = solve(day, site, goal)
    assert first_hour_back_at_reserve(with_term) == 5  # 2 kWh → +5 → +3: back at 10 kWh after hour 5
    recovered_without = first_hour_back_at_reserve(without_term)
    assert recovered_without is None or recovered_without > 5


def test_flow_goal_shortfall_is_summed_over_hours() -> None:
    from voltproof.core.rules import Trajectory, goal_penalty
    from voltproof.core.spec import soft_rows

    site = simple_site()
    goal = spec(import_caps=[{"start_hour": 10, "end_hour": 14, "max_kw": 1}])
    imports = np.zeros(24)
    imports[10:14] = 2.0  # 1 kW over the cap for 4 hours = 4 kWh of excess
    zeros = np.zeros(24)
    trajectory = Trajectory(5.0, np.full(24, 5.0), zeros, zeros, imports, zeros)
    assert goal_penalty(soft_rows(goal, site), trajectory, 10.0) == pytest.approx(20 * 10 * 4.0)


def test_c_goals_are_never_broken_to_save_money() -> None:
    rng = np.random.default_rng(99)
    for _ in range(60):
        capacity = float(rng.uniform(5, 50))
        site = simple_site(
            capacity_kwh=capacity, power_kw=capacity / 3, eta=0.95, degradation=float(rng.uniform(0, 2)), import_max_kw=500
        )
        prices = rng.uniform(1, 10, 24)
        prices[int(rng.integers(6, 24)) :] *= rng.uniform(1, 20)  # tempt discharge late in the day
        day = make_day(pv=rng.uniform(0, 3, 24), load=rng.uniform(1, 4, 24), critical=0.5, price=prices)
        a = int(rng.integers(6, 23))
        goal = spec(
            reserve_windows=[{"start_hour": a, "end_hour": int(rng.integers(a + 1, 25)), "min_soc_pct": float(rng.uniform(20, 95))}],
            soc_targets=[{"by_hour": int(rng.integers(6, 25)), "min_soc_pct": float(rng.uniform(20, 95))}],
            end_of_day_min_soc_pct=float(rng.uniform(10, 95)),
        )
        plan = solve(day, site, goal)
        assert plan.shortfalls == [], [s.message for s in plan.shortfalls]


def test_throughput_cap_respected() -> None:
    prices = [1.0, 30.0] * 12
    site = simple_site(capacity_kwh=10, power_kw=5, degradation=0.0)
    free = solve(make_day(pv=0, load=5, price=prices), site)
    capped = solve(make_day(pv=0, load=5, price=prices), site, spec(max_throughput_kwh=3))
    assert sum(free.flows.charge_kw) + sum(free.flows.discharge_kw) > 3
    assert sum(capped.flows.charge_kw) + sum(capped.flows.discharge_kw) <= 3 + 1e-6
    assert capped.shortfalls == []


def test_import_cap_met_by_discharging() -> None:
    load = [1.0] * 18 + [4.0] * 3 + [1.0] * 3
    plan = solve(make_day(pv=0, load=load), simple_site(degradation=0.01), spec(import_caps=[{"start_hour": 18, "end_hour": 21, "max_kw": 2}]))
    assert plan.shortfalls == []
    assert max(plan.flows.grid_import_kw[18:21]) <= 2 + 1e-6


def test_discharge_only_windows() -> None:
    plan = solve(make_day(pv=0, load=2, price=[1.0, 20.0] * 12), simple_site(degradation=0.0), spec(discharge_only_windows=[{"start_hour": 17, "end_hour": 21}]))
    discharge = np.array(plan.flows.discharge_kw)
    assert np.all(discharge[[t for t in range(24) if not 17 <= t < 21]] <= 1e-6) and plan.shortfalls == []


def test_no_export_sets_zero_export_limit_and_curtails() -> None:
    site = simple_site(capacity_kwh=5, power_kw=1, export_max_kw=10, initial_soc_frac=1.0)
    day = make_day(pv=[0.0] * 8 + [8.0] * 8 + [0.0] * 8, load=1, price=10, export_price=2)
    plan = solve(day, site, spec(no_export_windows=[{"start_hour": 10, "end_hour": 14}]))
    assert plan.export_limit_kw[10:14] == [0.0] * 4 and plan.export_limit_kw[9] == 10.0
    assert max(plan.flows.grid_export_kw[10:14]) <= 1e-9 and min(plan.flows.curtail_kw[10:14]) >= 6 - 1e-6
    assert plan.shortfalls == []


def test_protected_soc_follows_goals_and_never_exceeds_the_plan() -> None:
    site = simple_site(capacity_kwh=10, power_kw=5, degradation=0.01)
    day = make_day(pv=0, load=2, price=[1.0] * 12 + [10.0] * 12)
    assert solve(day, site).protected_soc_kwh is None  # no SOC goals → no top-up allowed
    plan = solve(day, site, spec(reserve_windows=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}]))
    protected, soc = np.array(plan.protected_soc_kwh), np.array(plan.soc_kwh)
    assert np.all(protected <= soc + 1e-9)
    assert protected[17:22] == pytest.approx([5.0] * 5)  # the reserve level in the window
    assert np.all(protected[22:] == 0.0)  # nothing to protect after the goal
    charging = [t for t in range(17) if plan.flows.charge_kw[t] > 1e-6]
    assert charging and all(protected[t] == pytest.approx(min(soc[t], 5.0)) for t in range(charging[-1], 17))
    # Power floor: with a 1 kW battery, protecting 9 kWh at 18–22 must start ≥ 8 h earlier.
    slow = simple_site(capacity_kwh=10, power_kw=1, degradation=0.01)
    slow_day = make_day(pv=0, load=0.5, critical=0.2, price=[1.0] * 12 + [10.0] * 12)
    slow_plan = solve(slow_day, slow, spec(reserve_windows=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 90}]))
    floor_hours = [t for t in range(17) if slow_plan.protected_soc_kwh[t] > 1e-9]
    assert floor_hours and floor_hours[0] <= 17 - 8 and np.all(np.array(slow_plan.protected_soc_kwh) <= np.array(slow_plan.soc_kwh) + 1e-9)


def test_export_price_above_import_price_rejected() -> None:
    with pytest.raises(ValueError, match="export price exceeds import price"):
        solve(make_day(price=5, export_price=6), simple_site())


def test_solve_under_one_second_with_every_constraint_type() -> None:
    goal = spec(
        grid_outages=[{"start_hour": 14, "end_hour": 17}],
        reserve_windows=[{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}],
        soc_targets=[{"by_hour": 7, "min_soc_pct": 80}],
        min_soc_pct_all_day=20,
        end_of_day_min_soc_pct=30,
        import_caps=[{"start_hour": 9, "end_hour": 12, "max_kw": 3}],
        no_export_windows=[{"start_hour": 11, "end_hour": 15}],
        discharge_only_windows=[{"start_hour": 17, "end_hour": 23}],
        max_throughput_kwh=40,
    )
    rng = np.random.default_rng(3)
    for _ in range(20):
        site = random_site(rng)
        plan = solve(random_day(site, rng), site, goal)
        assert plan.solve_time_s < 1.0


def test_energy_cost_matches_flows() -> None:
    site = simple_site(degradation=0.5, export_max_kw=5)
    day = make_day(pv=[0.0] * 9 + [6.0] * 6 + [0.0] * 9, load=2, price=CHEAP_THEN_EXPENSIVE, export_price=0.5)
    plan = solve(day, site)
    f = plan.flows
    expected = (
        np.dot(day.import_price_per_kwh, f.grid_import_kw)
        - np.dot(day.export_price_per_kwh, f.grid_export_kw)
        + 0.5 * (sum(f.charge_kw) + sum(f.discharge_kw))
    )
    assert plan.energy_cost == pytest.approx(expected)
    assert plan.objective == pytest.approx(plan.energy_cost + plan.penalty_cost - plan.terminal_value)
    assert plan.terminal_value_per_kwh == pytest.approx(np.mean(CHEAP_THEN_EXPENSIVE) * site.battery.eta_discharge)
    assert plan.terminal_value == pytest.approx(plan.terminal_value_per_kwh * plan.soc_kwh[-1])


def test_terminal_value_keeps_cheap_energy_for_tomorrow() -> None:
    # Expensive morning (10), cheap evening (2): V = mean · η_d = 6. The battery drains in the morning
    # (10 > 6), then refills in the evening because a stored kWh is worth 6 > 2 + β tomorrow.
    prices = [10.0] * 12 + [2.0] * 12
    plan = solve(make_day(pv=0, load=1, price=prices), simple_site(capacity_kwh=10, power_kw=10, degradation=0.1))
    assert plan.terminal_value_per_kwh == pytest.approx(6.0)
    assert plan.soc_kwh[11] == pytest.approx(1.0)
    assert plan.soc_kwh[-1] == pytest.approx(10.0)
    # Reversed (cheap morning, expensive evening): evening energy is worth 10 > V = 6, so it ends empty.
    reversed_plan = solve(make_day(pv=0, load=1, price=[2.0] * 12 + [10.0] * 12), simple_site(capacity_kwh=10, power_kw=10, degradation=0.1))
    assert reversed_plan.soc_kwh[-1] == pytest.approx(1.0)
