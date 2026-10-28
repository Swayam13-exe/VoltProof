from __future__ import annotations

import numpy as np
import pytest

from tests.fakes_core import make_day, random_day, random_site, random_spec, simple_site
from voltproof.core.optimizer import solve
from voltproof.core.rules import fmt_pair
from voltproof.core.verifier import MAX_LISTED, verify
from voltproof.models import IntentSpec

SITE = simple_site(capacity_kwh=10, power_kw=5, eta=1.0, export_max_kw=3)  # E0 = 5, E_min = 1, E_max = 10
IDLE = [0.0] * 24


def spec(**fields) -> IntentSpec:
    return IntentSpec.model_validate(fields)


def rules(result) -> set[str]:
    return {v.rule for v in result.violations}


def test_idle_plan_on_simple_day_is_valid() -> None:
    result = verify(IDLE, make_day(pv=0, load=2), SITE)
    assert result.valid and result.total_violations == 0
    assert result.energy_cost == pytest.approx(24 * 2 * 10)


@pytest.mark.parametrize(
    ("battery", "expected"),
    [
        ([0.0] * 23, "LENGTH"),
        ([0.0] * 23 + [float("nan")], "NOT_FINITE"),
        ([0.0] * 23 + [float("inf")], "NOT_FINITE"),
    ],
)
def test_input_rules_stop_early(battery: list[float], expected: str) -> None:
    result = verify(battery, make_day(), SITE)
    assert not result.valid and rules(result) == {expected} and result.soc_kwh is None


def test_charge_and_discharge_limits() -> None:
    result = verify([5.5, -6.0] + [0.0] * 22, make_day(load=10), SITE)
    assert {"CHARGE_LIMIT", "DISCHARGE_LIMIT"} <= rules(result)
    msg = next(v.message for v in result.violations if v.rule == "CHARGE_LIMIT")
    assert msg == "CHARGE_LIMIT at hour 0: charge setpoint 5.500000 kW exceeds the maximum charge power 5.000000 kW"


def test_soc_min_and_max() -> None:
    low = verify([-5.0] + [0.0] * 23, make_day(load=10), SITE)  # 5 − 5 = 0 < 1
    assert "SOC_MIN" in rules(low)
    high = verify([5.0, 5.0] + [0.0] * 22, make_day(load=10), SITE)  # 15 > 10
    assert "SOC_MAX" in rules(high)


def test_import_limit_and_unserved_load() -> None:
    site = simple_site(import_max_kw=3)
    result = verify(IDLE, make_day(load=[2.0] * 23 + [5.0], critical=1), site)
    assert rules(result) == {"IMPORT_LIMIT", "UNSERVED_LOAD"}
    assert result.unserved_noncritical_kwh == pytest.approx(2.0) and result.unserved_critical_kwh == 0


def test_outage_deficit() -> None:
    result = verify(IDLE, make_day(load=2), SITE, spec(grid_outages=[{"start_hour": 3, "end_hour": 5}]))
    deficits = [v for v in result.violations if v.rule == "OUTAGE_DEFICIT"]
    assert [v.hour for v in deficits] == [3, 4] and result.unserved_critical_kwh == pytest.approx(2.0)
    covered = verify([0.0] * 3 + [-2.0, -2.0] + [0.0] * 19, make_day(load=2), SITE, spec(grid_outages=[{"start_hour": 3, "end_hour": 5}]))
    assert covered.valid


GOAL_DAY = make_day(pv=[0.0] * 10 + [6.0] * 4 + [0.0] * 10, load=1)
GOALS = dict(
    reserve_windows=[{"start_hour": 18, "end_hour": 20, "min_soc_pct": 80}],
    soc_targets=[{"by_hour": 6, "min_soc_pct": 90}],
    end_of_day_min_soc_pct=70,
    import_caps=[{"start_hour": 0, "end_hour": 2, "max_kw": 0.5}],
    no_export_windows=[{"start_hour": 10, "end_hour": 12}],
    discharge_only_windows=[{"start_hour": 20, "end_hour": 24}],
    max_throughput_kwh=0.5,
)
DISCHARGE_AT_4 = [0.0] * 4 + [-1.0] + [0.0] * 19


def test_all_day_minimum() -> None:
    result = verify(DISCHARGE_AT_4, GOAL_DAY, SITE, spec(min_soc_pct_all_day=45))
    assert rules(result) == {"ALL_DAY_MIN"} and result.violations[0].hour == 4 and result.total_violations == 20


def test_goal_rules() -> None:
    day = GOAL_DAY
    goal = spec(
        reserve_windows=[{"start_hour": 18, "end_hour": 20, "min_soc_pct": 80}],
        soc_targets=[{"by_hour": 6, "min_soc_pct": 90}],
        end_of_day_min_soc_pct=70,
        import_caps=[{"start_hour": 0, "end_hour": 2, "max_kw": 0.5}],
        no_export_windows=[{"start_hour": 10, "end_hour": 12}],
        discharge_only_windows=[{"start_hour": 20, "end_hour": 24}],
        max_throughput_kwh=0.5,
    )
    battery = [0.0] * 4 + [-1.0] + [0.0] * 19  # discharge 1 kWh at hour 4
    result = verify(battery, day, SITE, goal)
    assert rules(result) == {
        "RESERVE", "SOC_TARGET", "END_SOC", "IMPORT_CAP", "NO_EXPORT", "DISCHARGE_WINDOW", "THROUGHPUT",
    }  # fmt: skip
    physics = verify(battery, day, SITE, goal, physics_only=True)
    assert physics.valid and physics.physics_only


def test_all_seventeen_rule_codes_are_reachable() -> None:
    seen = set()
    seen |= rules(verify([0.0] * 5, make_day(), SITE))
    seen |= rules(verify([float("nan")] * 24, make_day(), SITE))
    seen |= rules(verify([6.0, -6.0, -6.0] + [0.0] * 21, make_day(load=10), simple_site(import_max_kw=3, power_kw=5)))
    seen |= rules(verify([5.0, 5.0] + [0.0] * 22, make_day(load=10), SITE))
    seen |= rules(verify(IDLE, make_day(load=2), SITE, spec(grid_outages=[{"start_hour": 0, "end_hour": 1}])))
    seen |= rules(verify(DISCHARGE_AT_4, GOAL_DAY, SITE, spec(**GOALS)))
    seen |= rules(verify(DISCHARGE_AT_4, GOAL_DAY, SITE, spec(min_soc_pct_all_day=45)))
    assert seen == {
        "LENGTH", "NOT_FINITE", "CHARGE_LIMIT", "DISCHARGE_LIMIT", "SOC_MIN", "SOC_MAX", "IMPORT_LIMIT",
        "UNSERVED_LOAD", "OUTAGE_DEFICIT", "RESERVE", "SOC_TARGET", "ALL_DAY_MIN", "END_SOC", "IMPORT_CAP",
        "NO_EXPORT", "DISCHARGE_WINDOW", "THROUGHPUT",
    }  # fmt: skip


def test_soc_landing_on_minimum_after_float_arithmetic_passes() -> None:
    site = simple_site(capacity_kwh=1.0, power_kw=1.0, eta=0.95, soc_min_frac=0.1, initial_soc_frac=0.7)
    # Discharge exactly the usable 0.6 kWh in three uneven steps; the float sum lands within 1e-16 of 0.1.
    steps = np.array([0.1, 0.2, 0.3]) * 0.95
    battery = [-steps[0], -steps[1], -steps[2]] + [0.0] * 21
    result = verify(battery, make_day(load=1, critical=0.1), site)
    assert abs(result.soc_kwh[2] - 0.1) < 1e-12  # on the minimum up to float noise, either side
    assert "SOC_MIN" not in rules(result)
    assert result.valid


def test_messages_never_print_a_value_equal_to_its_limit() -> None:
    rng = np.random.default_rng(5)
    for _ in range(100_000):
        limit = float(rng.uniform(-1000, 1000))
        value = limit + float(rng.choice([-1, 1])) * (1e-6 + float(rng.uniform(0, 1e-5)))
        v, lim = fmt_pair(value, limit)
        assert v != lim
    v, lim = fmt_pair(0.9999995, 1.0000004)  # rounds equal at 6 decimals → more decimals
    assert v != lim and len(v.split(".")[1]) > 6


def test_violation_message_near_limit_differs() -> None:
    import re

    battery = [-4.0 - 2.5e-6] + [0.0] * 23  # SOC 0.9999975 vs minimum 1.0
    message = next(v.message for v in verify(battery, make_day(load=10), SITE).violations if v.rule == "SOC_MIN")
    value, limit = re.findall(r"(\d+\.\d+) kWh", message)
    assert value.startswith("0.99999") and limit == "1.000000" and value != limit


def test_at_most_twenty_listed_with_total_count() -> None:
    result = verify([6.0] * 24, make_day(load=10), SITE)
    assert len(result.violations) == MAX_LISTED and result.total_violations > MAX_LISTED


def test_export_limit_controls_no_export_rule() -> None:
    day = make_day(pv=[0.0] * 10 + [6.0] * 4 + [0.0] * 10, load=1)
    goal = spec(no_export_windows=[{"start_hour": 10, "end_hour": 14}])
    defaulted = verify(IDLE, day, SITE, goal)  # battery-only plan: the site's own export limit applies
    assert rules(defaulted) == {"NO_EXPORT"}
    zero_export = verify(IDLE, day, SITE, goal, export_limit_kw=[3.0] * 10 + [0.0] * 4 + [3.0] * 10)
    assert zero_export.valid and zero_export.flows.curtail_kw[10] == pytest.approx(5.0)


def test_optimizer_plans_without_shortfall_verify_clean() -> None:
    """SPEC §6.5 cross-check on 200 seeded random days and specs."""
    rng = np.random.default_rng(2026)
    checked = 0
    for _ in range(200):
        site = random_site(rng)
        day, goal = random_day(site, rng), random_spec(rng)
        plan = solve(day, site, goal)
        result = verify(plan.battery_kw, day, site, goal, export_limit_kw=plan.export_limit_kw)
        physics = verify(plan.battery_kw, day, site, goal, physics_only=True, export_limit_kw=plan.export_limit_kw)
        assert not ({"CHARGE_LIMIT", "DISCHARGE_LIMIT", "SOC_MIN", "SOC_MAX"} & rules(physics))
        if plan.shortfalls or plan.unserved_critical_kwh + plan.unserved_noncritical_kwh > 1e-6:
            continue
        checked += 1
        assert result.valid, [v.message for v in result.violations]
        assert result.energy_cost == pytest.approx(plan.energy_cost, rel=1e-6, abs=1e-6)
    assert checked > 100
