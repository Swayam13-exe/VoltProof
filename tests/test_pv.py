from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voltproof.data.pv import (
    cell_temperature_c,
    clear_sky_ghi_w_per_m2,
    cos_zenith,
    pv_power_kw,
    solar_elevation_deg,
    solar_noon_utc_min,
)
from voltproof.models import PVSpec

PV40 = PVSpec(capacity_kwp=40)


def test_hand_computed_point() -> None:
    # T_cell = 30 + (45 − 20)/800 · 800 = 55 °C; P = 40 · 0.8 · 0.8 · (1 − 0.004 · 30) = 22.528 kW
    assert cell_temperature_c(800, 30, 45) == pytest.approx(55.0)
    assert pv_power_kw([800.0], [30.0], PV40)[0] == pytest.approx(22.528)


def test_second_hand_computed_point() -> None:
    # T_cell = 10 + 25/800 · 400 = 22.5 °C; P = 40 · 0.4 · 0.8 · (1 + 0.004 · 2.5) = 12.928 kW
    assert pv_power_kw([400.0], [10.0], PV40)[0] == pytest.approx(12.928)


def test_zero_and_negative_ghi_give_zero_power() -> None:
    assert pv_power_kw([0.0, -5.0], [20.0, 20.0], PV40).tolist() == [0.0, 0.0]


def test_clipped_at_capacity() -> None:
    # 1400 W/m² at −20 °C: 40 · 1.4 · 0.8 · (1 − 0.004 · (−20 + 43.75 − 25)) ≈ 45.0 kW → clipped to 40
    small_pr = PVSpec(capacity_kwp=40, performance_ratio=1.0)
    assert pv_power_kw([1400.0], [-20.0], PV40)[0] == pytest.approx(40.0)
    assert pv_power_kw([1400.0], [-20.0], small_pr)[0] == pytest.approx(40.0)


def test_hotter_means_less_power() -> None:
    cool, hot = pv_power_kw([800.0, 800.0], [10.0, 40.0], PV40)
    assert hot < cool


def test_vectorized_shape() -> None:
    ghi = np.linspace(0, 1000, 24)
    assert pv_power_kw(ghi, np.full(24, 25.0), PV40).shape == (24,)


def test_solar_noon_moves_four_minutes_per_degree() -> None:
    days = pd.DatetimeIndex(["2024-03-20", "2024-11-03"], tz="UTC")
    at_0 = solar_noon_utc_min(days, 0.0)
    at_15 = solar_noon_utc_min(days, 15.0)
    assert np.allclose(at_0 - at_15, 60.0)
    # Equation of time: about −7 min near the March equinox (noon later), about +16 min early November.
    assert at_0[0] == pytest.approx(727, abs=2)
    assert at_0[1] == pytest.approx(704, abs=2)


def test_elevation_at_equinox_noon_on_equator_is_near_zenith() -> None:
    noon = pd.DatetimeIndex([pd.Timestamp("2024-03-20 12:07", tz="UTC")])
    assert solar_elevation_deg(noon, 0.0, 0.0)[0] > 89.0
    midnight = pd.DatetimeIndex([pd.Timestamp("2024-03-20 00:00", tz="UTC")])
    assert cos_zenith(midnight, 0.0, 0.0)[0] < 0


def test_clear_sky_ghi() -> None:
    times = pd.date_range("2024-06-21", periods=24, freq="1h", tz="UTC")
    ghi = clear_sky_ghi_w_per_m2(times, 0.0, 0.0)
    assert ghi[0] == 0.0 and ghi.max() < 1100.0
    assert int(np.argmax(ghi)) == 12
