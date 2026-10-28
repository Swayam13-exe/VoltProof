from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from tests.fakes_data import make_site
from voltproof.data.load import activity_schedule, ar1_noise, generate_load, noncritical_load_kw

SITE_TYPES = ["commercial_cold_store", "health_clinic", "household"]
WEDNESDAY, SATURDAY, SUNDAY = date(2024, 6, 5), date(2024, 6, 8), date(2024, 6, 9)
TEMP_F = 28.0 + 4.0 * np.sin(np.arange(24) / 24 * 2 * np.pi)
TEMP_A = TEMP_F + 1.5


@pytest.mark.parametrize("site_type", SITE_TYPES)
def test_same_seed_same_load_different_seed_differs(site_type: str) -> None:
    site = make_site(site_type=site_type)
    first = generate_load(site, WEDNESDAY, TEMP_F, TEMP_A, seed=2026)
    again = generate_load(site, WEDNESDAY, TEMP_F, TEMP_A, seed=2026)
    other = generate_load(site, WEDNESDAY, TEMP_F, TEMP_A, seed=7)
    np.testing.assert_array_equal(first.actual_kw, again.actual_kw)
    np.testing.assert_array_equal(first.forecast_kw, other.forecast_kw)  # forecast has no noise
    assert not np.array_equal(first.actual_kw, other.actual_kw)


@pytest.mark.parametrize("site_type", SITE_TYPES)
def test_forecast_is_deterministic_part_with_forecast_temperature(site_type: str) -> None:
    site = make_site(site_type=site_type)
    series = generate_load(site, WEDNESDAY, TEMP_F, TEMP_A, seed=2026)
    expected = site.critical_load_kw + noncritical_load_kw(site, WEDNESDAY, TEMP_F)
    np.testing.assert_allclose(series.forecast_kw, expected)
    assert np.all(series.critical_kw == site.critical_load_kw)
    assert np.all(series.actual_kw >= series.critical_kw) and np.all(series.forecast_kw >= series.critical_kw)


@pytest.mark.parametrize("site_type", SITE_TYPES)
def test_different_days_have_different_noise(site_type: str) -> None:
    site = make_site(site_type=site_type)
    a = generate_load(site, WEDNESDAY, TEMP_F, TEMP_F, seed=2026)
    b = generate_load(site, date(2024, 6, 12), TEMP_F, TEMP_F, seed=2026)  # also a Wednesday
    np.testing.assert_array_equal(a.forecast_kw, b.forecast_kw)
    assert not np.array_equal(a.actual_kw, b.actual_kw)


def test_cold_store_schedule_and_cooling() -> None:
    site = make_site(site_type="commercial_cold_store")
    params = site.load_params
    weekday = activity_schedule(params, WEDNESDAY.weekday())
    assert weekday[9:19].tolist() == [1.0] * 10 and weekday[:9].sum() == 0 and weekday[19:].sum() == 0
    assert activity_schedule(params, SATURDAY.weekday()).sum() == weekday.sum()
    assert activity_schedule(params, SUNDAY.weekday()).sum() == pytest.approx(0.25 * weekday.sum())
    hot, mild = noncritical_load_kw(site, WEDNESDAY, np.full(24, 34.0)), noncritical_load_kw(site, WEDNESDAY, np.full(24, 20.0))
    # 10 °C above threshold: 0.5 kW/°C · 10 = 5 kW at night, × 1.5 during activity
    assert hot[3] - mild[3] == pytest.approx(5.0)
    assert hot[12] - mild[12] == pytest.approx(7.5)


def test_clinic_weekends_are_lower() -> None:
    site = make_site(site_type="health_clinic")
    weekday = noncritical_load_kw(site, WEDNESDAY, np.full(24, 20.0))
    weekend = noncritical_load_kw(site, SUNDAY, np.full(24, 20.0))
    assert weekend.sum() < weekday.sum()
    assert weekday[8:17].min() > weekday[:8].max()


def test_household_peaks_weekend_and_heating() -> None:
    site = make_site(site_type="household")
    weekday = noncritical_load_kw(site, WEDNESDAY, np.full(24, 18.0))
    weekend = noncritical_load_kw(site, SATURDAY, np.full(24, 18.0))
    assert weekday[7] > weekday[12] and weekday[19] > weekday[12]  # morning and evening peaks
    assert weekend.sum() > weekday.sum() and weekend[12] > weekday[12]  # higher on weekends, incl. daytime
    cold = noncritical_load_kw(site, WEDNESDAY, np.full(24, 5.0))
    assert cold[3] - weekday[3] == pytest.approx(0.05 * 10)  # heating below 15 °C
    hot = noncritical_load_kw(site, WEDNESDAY, np.full(24, 35.0))
    np.testing.assert_allclose(hot, weekday)  # no cooling term for households


def test_ar1_noise_statistics() -> None:
    rng = np.random.default_rng(0)
    noise = ar1_noise(rng, 200_000, phi=0.7, sigma=0.08)
    assert noise.std() == pytest.approx(0.08, rel=0.03)
    assert np.corrcoef(noise[:-1], noise[1:])[0, 1] == pytest.approx(0.7, abs=0.02)


def test_temperature_array_must_have_24_values() -> None:
    with pytest.raises(ValueError, match="24 hourly values"):
        noncritical_load_kw(make_site(), WEDNESDAY, np.ones(23))
