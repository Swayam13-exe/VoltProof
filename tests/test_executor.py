from __future__ import annotations

import numpy as np
import pytest

from tests.fakes_core import make_day, random_day, random_site, random_spec, simple_site
from voltproof.core.executor import execute
from voltproof.core.optimizer import solve
from voltproof.models import TOL, HourWindow, IntentSpec

SITE = simple_site(capacity_kwh=10, power_kw=5, eta=1.0, export_max_kw=3)  # E0 = 5, E_min = 1, E_max = 10


def balance_residual(result, day) -> np.ndarray:
    f = result.flows
    pv, load = np.array(day.pv_actual_kw), np.array(day.load_actual_kw)
    supply = pv - np.array(f.curtail_kw) + np.array(f.grid_import_kw) + np.array(f.discharge_kw)
    supply += np.array(f.unserved_noncritical_kw) + np.array(f.unserved_critical_kw)
    demand = load + np.array(f.charge_kw) + np.array(f.grid_export_kw)
    return supply - demand


def test_energy_balance_holds_every_hour_on_random_inputs() -> None:
    rng = np.random.default_rng(2026)
    for _ in range(300):
        site = random_site(rng)
        day = random_day(site, rng)
        p_max = max(site.battery.p_charge_max_kw, site.battery.p_discharge_max_kw)
        setpoints = rng.uniform(-1.5 * p_max, 1.5 * p_max, 24)
        outages = [HourWindow(start_hour=a, end_hour=a + int(rng.integers(1, 6))) for a in [int(rng.integers(0, 19))]] if rng.random() < 0.5 else []
        result = execute(setpoints, day, site, outages)
        assert np.abs(balance_residual(result, day)).max() <= TOL
        battery = site.battery
        soc = np.array(result.soc_kwh)
        assert soc.min() >= battery.soc_min_frac * battery.capacity_kwh - TOL
        assert soc.max() <= battery.soc_max_frac * battery.capacity_kwh + TOL
        applied = np.array(result.battery_kw)
        assert applied.max() <= battery.p_charge_max_kw + TOL and -applied.min() <= battery.p_discharge_max_kw + TOL
        for t in result.outage_hours:
            assert result.flows.grid_import_kw[t] == 0 and result.flows.grid_export_kw[t] == 0
        for name in ("curtail_kw", "grid_import_kw", "grid_export_kw", "unserved_noncritical_kw", "unserved_critical_kw"):
            assert min(getattr(result.flows, name)) >= 0


def test_setpoints_are_clipped_to_power_and_soc_headroom() -> None:
    result = execute([7.0, 7.0] + [0.0] * 22, make_day(load=2), SITE)
    assert result.battery_kw[:2] == pytest.approx([5.0, 0.0])  # power limit, then the battery is full
    assert result.clipped_hours == [0, 1] and result.soc_kwh[1] == pytest.approx(10.0)
    drained = execute([-5.0, -5.0] + [0.0] * 22, make_day(load=10), SITE)
    assert drained.battery_kw[:2] == pytest.approx([-4.0, 0.0])  # only 4 kWh above the minimum
    assert drained.clipped_hours == [0, 1]


def test_outage_override_covers_the_deficit() -> None:
    result = execute([0.0] * 24, make_day(load=2), SITE, [HourWindow(start_hour=3, end_hour=5)])
    assert result.battery_kw[3:5] == pytest.approx([-2.0, -2.0])
    assert result.overrides == 2 and result.override_hours == [3, 4]
    assert result.unserved_critical_kwh == 0 and result.unserved_noncritical_kwh == 0


def test_outage_override_protects_critical_load_first() -> None:
    result = execute([0.0] * 24, make_day(load=2, critical=1), SITE, [HourWindow(start_hour=0, end_hour=6)])
    assert -sum(result.battery_kw[:6]) == pytest.approx(4.0)  # all 4 usable kWh
    assert result.unserved_noncritical_kwh == pytest.approx(6.0)  # non-critical shed first
    assert result.unserved_critical_kwh == pytest.approx(2.0)


def test_outage_override_stores_surplus_and_curtails_the_rest() -> None:
    day = make_day(pv=[8.0] * 24, load=1)
    result = execute([-1.0] * 24, day, SITE, [HourWindow(start_hour=0, end_hour=2)])
    assert result.battery_kw[:2] == pytest.approx([5.0, 0.0])  # charges at full power, then full
    assert result.flows.curtail_kw[:2] == pytest.approx([2.0, 7.0])
    assert result.flows.grid_export_kw[:2] == [0.0, 0.0]
    assert result.override_hours == [0, 1]


LOW_IMPORT = simple_site(capacity_kwh=10, power_kw=5, eta=1.0, import_max_kw=5)  # E0 = 5, E_min = 1


def test_import_limit_first_reduces_charging() -> None:
    result = execute([4.0] + [0.0] * 23, make_day(load=[4.0] + [1.0] * 23), LOW_IMPORT)
    assert result.battery_kw[0] == pytest.approx(1.0)  # 4 + 4 = 8 > 5 → charging cut by 3
    assert result.flows.grid_import_kw[0] == pytest.approx(5.0) and result.unserved_noncritical_kwh == 0
    assert result.override_hours == [0]


def test_import_limit_then_discharges_before_shedding() -> None:
    result = execute([0.0] * 24, make_day(load=[7.0] + [1.0] * 23, critical=1), LOW_IMPORT)
    assert result.battery_kw[0] == pytest.approx(-2.0)
    assert result.unserved_noncritical_kwh == 0 and result.override_hours == [0]


def test_import_limit_sheds_non_critical_only_when_the_battery_cannot_help() -> None:
    empty = simple_site(capacity_kwh=10, power_kw=5, eta=1.0, import_max_kw=5, initial_soc_frac=0.1)
    result = execute([0.0] * 24, make_day(load=[7.0] + [1.0] * 23, critical=1), empty)
    assert result.battery_kw[0] == 0.0 and result.overrides == 0  # nothing to change: battery at its minimum
    assert result.flows.unserved_noncritical_kw[0] == pytest.approx(2.0) and result.unserved_critical_kwh == 0


def test_no_grid_charging_for_expected_sun_that_did_not_come() -> None:
    # Forecast: 6 kW PV, 1 kW load at hour 10 → the plan charges 5 kW from PV surplus.
    # Actual PV is 2 kW → only 1 kW of surplus, and the battery must not import the other 4 kW.
    pv_f = [0.0] * 10 + [6.0] + [0.0] * 13
    pv_a = [0.0] * 10 + [2.0] + [0.0] * 13
    day = make_day(pv=pv_f, pv_actual=pv_a, load=1)
    result = execute([0.0] * 10 + [5.0] + [0.0] * 13, day, SITE)
    assert result.battery_kw[10] == pytest.approx(1.0) and result.flows.grid_import_kw[10] == pytest.approx(0.0)
    assert result.grid_charge_limited_hours == [10] and result.overrides == 0


def test_intended_grid_charging_still_happens() -> None:
    day = make_day(pv=0, load=1)  # no expected surplus: a 3 kW charge was meant to come from the grid
    result = execute([3.0] + [0.0] * 23, day, SITE)
    assert result.battery_kw[0] == pytest.approx(3.0) and result.flows.grid_import_kw[0] == pytest.approx(4.0)
    assert result.grid_charge_limited_hours == []


def test_partly_intended_grid_charging_is_capped_at_the_intended_part() -> None:
    # Expected surplus 2 kW, planned charge 5 kW → 3 kW intended from the grid; actual surplus 0.
    pv_f = [3.0] + [0.0] * 23
    day = make_day(pv=pv_f, pv_actual=[0.0] * 24, load=1)
    result = execute([5.0] + [0.0] * 23, day, SITE)
    assert result.battery_kw[0] == pytest.approx(3.0) and result.flows.grid_import_kw[0] == pytest.approx(4.0)


def _pv_shortfall_reserve_case():
    """Forecast: 8 kW of PV 09–14 to fill the battery for an 80 % evening reserve; actual: no sun."""
    site = simple_site(capacity_kwh=10, power_kw=5, eta=1.0, degradation=0.1)  # E0 = 5, no export
    pv_f = [0.0] * 9 + [8.0] * 6 + [0.0] * 9
    day = make_day(pv=pv_f, pv_actual=[0.0] * 24, load=1, price=[10.0] * 17 + [20.0] * 7)
    gold = IntentSpec.model_validate({"reserve_windows": [{"start_hour": 18, "end_hour": 22, "min_soc_pct": 80}]})
    return site, day, gold, solve(day, site, gold, use="forecast")


def test_goal_topup_lets_a_forecast_plan_meet_its_reserve_when_the_sun_fails() -> None:
    from voltproof.core.metrics import gold_check

    site, day, gold, plan = _pv_shortfall_reserve_case()
    assert plan.protected_soc_kwh is not None and plan.shortfalls == []
    result = execute(plan.battery_kw, day, site, [], export_limit_kw=plan.export_limit_kw, protected_soc_kwh=plan.protected_soc_kwh)
    assert result.grid_charge_limited_hours  # the PV charging it counted on is not taken from the grid ...
    assert result.goal_topup_hours  # ... except what the reserve needs
    assert min(result.soc_kwh[17:22]) >= 8.0 - 1e-6
    assert gold_check(result, gold, site)[0]
    # The top-up never goes beyond the protected level the plan asked for.
    assert all(result.soc_kwh[t] <= max(plan.protected_soc_kwh[t], plan.soc_kwh[t]) + 1e-6 for t in result.goal_topup_hours)


def test_a_plan_without_protected_soc_gets_no_topup() -> None:
    from voltproof.core.metrics import gold_check

    site, day, gold, plan = _pv_shortfall_reserve_case()
    result = execute(plan.battery_kw, day, site, [], export_limit_kw=plan.export_limit_kw)  # battery-only plan (like A0, A2, A3)
    assert result.goal_topup_hours == []
    assert min(result.soc_kwh[17:22]) < 8.0 - 1e-6
    assert not gold_check(result, gold, site)[0]


def test_topup_respects_the_import_limit() -> None:
    site = simple_site(capacity_kwh=10, power_kw=5, eta=1.0, import_max_kw=2)
    day = make_day(load=1)
    result = execute([0.0] * 24, day, site, protected_soc_kwh=[10.0] * 24)
    assert result.battery_kw[0] == pytest.approx(1.0)  # only 2 − 1 = 1 kW of import headroom
    assert max(result.flows.grid_import_kw) <= 2.0 + 1e-9 and result.goal_topup_hours[:5] == [0, 1, 2, 3, 4]


def test_no_wasted_solar_when_export_is_not_allowed() -> None:
    no_export = simple_site(capacity_kwh=10, power_kw=5, eta=1.0, export_max_kw=0)
    day = make_day(pv=[0.0] * 10 + [8.0] * 2 + [0.0] * 12, load=1)
    result = execute([0.0] * 24, day, no_export)  # an idle plan would curtail 7 kW twice
    assert result.battery_kw[10:12] == pytest.approx([5.0, 0.0])  # 5 → 10 kWh, then full
    assert result.flows.curtail_kw[10:12] == pytest.approx([2.0, 7.0])
    assert result.solar_absorbed_hours == [10] and result.overrides == 0


def test_no_wasted_solar_beyond_the_export_limit() -> None:
    day = make_day(pv=[6.0] * 24, load=1, export_price=1)
    result = execute([0.0] * 24, day, SITE)  # surplus 5 kW, export limit 3 kW → 2 kW into the battery
    assert result.flows.grid_export_kw[0] == pytest.approx(3.0)
    assert result.battery_kw[0] == pytest.approx(2.0) and result.flows.curtail_kw[0] == pytest.approx(0.0)
    zero = execute([0.0] * 24, day, SITE, export_limit_kw=[0.0] * 24)
    assert zero.flows.grid_export_kw[0] == 0 and zero.battery_kw[0] == pytest.approx(5.0)
    assert zero.flows.curtail_kw[0] == pytest.approx(0.0) and zero.flows.curtail_kw[1] == pytest.approx(5.0)


def test_realized_cost_is_grid_plus_degradation() -> None:
    site = simple_site(degradation=0.5)
    day = make_day(load=2, price=10)
    result = execute([1.0] + [-1.0] + [0.0] * 22, day, site)
    assert result.grid_cost == pytest.approx(10 * (48 + 1 - 1))
    assert result.degradation_cost == pytest.approx(0.5 * 2)
    assert result.realized_cost == pytest.approx(result.grid_cost + result.degradation_cost)


def test_executing_an_optimizer_plan_on_its_own_data_reproduces_it() -> None:
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(60):
        site = random_site(rng)
        day = random_day(site, rng)
        day = day.model_copy(update={"pv_actual_kw": day.pv_forecast_kw, "load_actual_kw": day.load_forecast_kw})
        goal = random_spec(rng).model_copy(update={"grid_outages": []})
        plan = solve(day, site, goal, use="actual")
        result = execute(plan.battery_kw, day, site, [], export_limit_kw=plan.export_limit_kw, plan_basis="actual")
        assert result.grid_charge_limited_hours == []  # the plan's own data: every charge is as intended
        if result.overrides:
            # Allowed deviation: the LP shed load at the import limit to keep energy for later, and the
            # executor's import-limit rule served that load from the battery instead.
            first = result.override_hours[0]
            assert plan.flows.unserved_noncritical_kw[first] + plan.flows.unserved_critical_kw[first] > 1e-6
            continue
        if result.solar_absorbed_hours:
            # The only allowed deviation: the LP curtailed PV with battery headroom (storing was not
            # worth the degradation), and the executor's no-wasted-solar rule stored it anyway.
            first = result.solar_absorbed_hours[0]
            assert plan.flows.curtail_kw[first] > 1e-6 and plan.soc_kwh[first] < site.battery.capacity_kwh * site.battery.soc_max_frac
            continue
        assert result.clipped_hours == []
        np.testing.assert_allclose(result.soc_kwh, plan.soc_kwh, atol=1e-6)
        np.testing.assert_allclose(result.flows.grid_import_kw, plan.flows.grid_import_kw, atol=1e-6)
        np.testing.assert_allclose(result.flows.grid_export_kw, plan.flows.grid_export_kw, atol=1e-6)
        assert result.realized_cost == pytest.approx(plan.energy_cost, rel=1e-6, abs=1e-6)
        checked += 1
    assert checked >= 25  # the rest are the two explained deviations above (random β is often >> prices)


def test_rejects_bad_setpoints() -> None:
    with pytest.raises(ValueError, match="24 finite setpoints"):
        execute([0.0] * 23, make_day(), SITE)
    with pytest.raises(ValueError):
        execute([float("nan")] * 24, make_day(), SITE)


# --- import caps (SPEC 6.6) ---------------------------------------------------------------------

CAP_SITE = simple_site(capacity_kwh=10, power_kw=5, eta=1.0, import_max_kw=50)  # E0 = 5, E_min = 1


def capped(hour: int, cap: float) -> list[float]:
    return [cap if t == hour else 50.0 for t in range(24)]


def test_cloudy_day_holds_the_cap_by_reducing_charging() -> None:
    # The plan expected 3 kW of PV at hour 0 (load 5: import 5 = the cap, plus 3 kW of charging still
    # meant to come from the grid). Actual PV is 1 kW, so without the cap import would be 7 kW.
    day = make_day(pv=[3.0] + [0.0] * 23, pv_actual=[1.0] + [0.0] * 23, load=5.0, critical=1.0)
    free = execute([3.0] + [0.0] * 23, day, CAP_SITE)
    assert free.flows.grid_import_kw[0] == pytest.approx(7.0)
    held = execute([3.0] + [0.0] * 23, day, CAP_SITE, import_limit_kw=capped(0, 5.0))
    assert held.flows.grid_import_kw[0] == pytest.approx(5.0) and held.battery_kw[0] == pytest.approx(1.0)
    assert held.import_cap_hours == [0] and held.overrides == 0 and held.unserved_noncritical_kwh == 0


def test_cap_then_discharges_within_limits() -> None:
    day = make_day(pv=[1.0] + [0.0] * 23, load=5.0, critical=1.0)
    result = execute([0.0] * 24, day, CAP_SITE, import_limit_kw=capped(0, 3.0))
    assert result.battery_kw[0] == pytest.approx(-1.0) and result.flows.grid_import_kw[0] == pytest.approx(3.0)
    assert result.import_cap_hours == [0]


def test_cap_is_exceeded_rather_than_shedding_load_and_counts_as_a_shortfall() -> None:
    from voltproof.core.metrics import gold_check, trajectory_total_cost

    low = simple_site(capacity_kwh=10, power_kw=5, eta=1.0, import_max_kw=50, initial_soc_frac=0.2)  # 1 kWh above E_min
    day = make_day(load=[7.0] + [1.0] * 23, critical=1.0)
    gold = IntentSpec.model_validate({"import_caps": [{"start_hour": 0, "end_hour": 1, "max_kw": 3.0}]})
    result = execute([0.0] * 24, day, low, import_limit_kw=capped(0, 3.0))
    assert result.battery_kw[0] == pytest.approx(-1.0)  # all the battery can give
    assert result.flows.grid_import_kw[0] == pytest.approx(6.0)  # above the cap of 3 kW
    assert result.unserved_noncritical_kwh == 0 and result.unserved_critical_kwh == 0
    assert result.import_cap_hours == [0] and result.overrides == 0
    ok, violations = gold_check(result, gold, low)
    assert not ok and violations[0].rule == "IMPORT_CAP" and violations[0].value == pytest.approx(6.0)
    assert trajectory_total_cost(result, gold, day, low).goal_penalty == pytest.approx(20 * 10.0 * 3.0)


def test_goal_topup_stays_under_the_cap() -> None:
    day = make_day(load=1.0, critical=1.0)
    protected = [9.0] + [0.0] * 23
    free = execute([0.0] * 24, day, CAP_SITE, protected_soc_kwh=protected)
    held = execute([0.0] * 24, day, CAP_SITE, protected_soc_kwh=protected, import_limit_kw=capped(0, 3.0))
    assert free.battery_kw[0] == pytest.approx(4.0) and held.battery_kw[0] == pytest.approx(2.0)
    assert held.flows.grid_import_kw[0] == pytest.approx(3.0)


def test_optimizer_plans_carry_their_import_caps() -> None:
    day = make_day(load=2.0, price=10.0)
    caps = [{"start_hour": 13, "end_hour": 17, "max_kw": 1.5}, {"start_hour": 15, "end_hour": 18, "max_kw": 1.0}]
    plan = solve(day, CAP_SITE, IntentSpec.model_validate({"import_caps": caps}))
    assert plan.import_limit_kw[13:15] == [1.5, 1.5] and plan.import_limit_kw[15:18] == [1.0] * 3
    assert plan.import_limit_kw[12] == 50.0 and solve(day, CAP_SITE).import_limit_kw == [50.0] * 24
