from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from tests.fakes_core import make_day, random_day, random_site, simple_site
from tests.fakes_data import make_site
from voltproof.baselines.optimizer_arms import a1_plan, a5_plan, a6_plan
from voltproof.baselines.rule_based import SOC_FLOOR_FRAC, a0_setpoints, rule_based_setpoints
from voltproof.data.days import dev_days
from voltproof.models import IntentSpec

SITE = simple_site(capacity_kwh=10, power_kw=5, eta=1.0)  # E0 = 5, floor = 20 % = 2 kWh


def simulate_soc(setpoints, site) -> np.ndarray:
    b = site.battery
    return b.initial_soc_frac * b.capacity_kwh + np.cumsum(
        b.eta_charge * np.maximum(setpoints, 0) - np.maximum(-np.asarray(setpoints), 0) / b.eta_discharge
    )


def test_a0_charges_only_from_pv_surplus() -> None:
    rng = np.random.default_rng(1)
    for _ in range(100):
        site = random_site(rng)
        day = random_day(site, rng)
        setpoints = a0_setpoints(day, site, "actual")
        surplus = np.maximum(np.array(day.pv_actual_kw) - np.array(day.load_actual_kw), 0)
        assert np.all(setpoints <= surplus + 1e-9)  # never charges from the grid
        deficit = np.maximum(np.array(day.load_actual_kw) - np.array(day.pv_actual_kw), 0)
        assert np.all(-setpoints <= deficit + 1e-9)  # never discharges to export


def test_a0_stops_discharging_at_20_percent() -> None:
    setpoints = rule_based_setpoints(np.zeros(24), np.full(24, 2.0), SITE)
    soc = simulate_soc(setpoints, SITE)
    assert soc.min() == pytest.approx(SOC_FLOOR_FRAC * 10)  # 5 → 2 kWh, then idle
    assert -setpoints.sum() == pytest.approx(3.0)


def test_a0_respects_power_and_capacity() -> None:
    setpoints = rule_based_setpoints(np.full(24, 20.0), np.zeros(24), SITE)
    assert setpoints[0] == pytest.approx(5.0) and setpoints[1] == pytest.approx(0.0)  # 5 → 10 kWh, full


def test_a0_is_causal() -> None:
    pv, load = np.linspace(0, 8, 24), np.full(24, 3.0)
    base = rule_based_setpoints(pv, load, SITE)
    changed_future = pv.copy()
    changed_future[15:] = 0
    assert rule_based_setpoints(changed_future, load, SITE)[:15] == pytest.approx(base[:15])


def test_a0_forecast_vs_actual() -> None:
    day = make_day(pv=[0.0] * 10 + [6.0] * 4 + [0.0] * 10, pv_actual=[0.0] * 24, load=1)
    assert a0_setpoints(day, SITE, "forecast")[10] == pytest.approx(5.0)  # expected PV surplus → charge
    assert a0_setpoints(day, SITE, "actual")[10] == pytest.approx(0.0)  # no PV; already at its 20 % floor


def test_optimizer_arms_wiring() -> None:
    day = make_day(pv=0, load=2, price=[1.0] * 12 + [10.0] * 12, load_actual=[2.0] * 12 + [4.0] * 12)
    gold = IntentSpec.model_validate({"reserve_windows": [{"start_hour": 18, "end_hour": 22, "min_soc_pct": 50}]})
    a1, a5, a6 = a1_plan(day, SITE), a5_plan(day, SITE, gold), a6_plan(day, SITE, gold)
    assert (a1.use, a5.use, a6.use) == ("forecast", "forecast", "actual")
    assert a1.spec == IntentSpec() and a5.spec == gold and a6.spec == gold
    assert min(a5.soc_kwh[17:22]) >= 5 - 1e-6 and min(a1.soc_kwh[17:22]) < 5


def test_dev_days_refuse_held_out_sites() -> None:
    dev_site, held = make_site("dev_site"), make_site("held_site", held_out=True)
    splits = {"dev": [{"site_id": "dev_site", "date": "2024-05-01"}], "test": []}
    assert dev_days(splits, [dev_site, held]) == [(dev_site, date(2024, 5, 1))]
    bad = {"dev": [{"site_id": "held_site", "date": "2024-05-01"}], "test": []}
    with pytest.raises(ValueError, match="held-out site"):
        dev_days(bad, [dev_site, held])
