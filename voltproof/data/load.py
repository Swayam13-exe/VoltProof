"""Synthetic site load: one generator per site type, all sharing one structure (SPEC §6.2).

    load = critical + base + activity_kw · schedule(weekday, hour) + temperature_term(T)

- critical load: constant, all day, never noised
- schedule: 0–1 activity from the site's windows, scaled on Saturday/Sunday, plus weekend-only windows
- temperature term, by site type:
    commercial_cold_store: cooling above a threshold, amplified during activity (door openings, loading)
    health_clinic:         cooling (fans, air conditioning) above a threshold, amplified during clinic hours
    household:             auxiliary heating below a threshold (circulation pump, electric top-up)
- forecast = deterministic part with the forecast temperature, no noise
- actual   = critical + non-critical part with the actual temperature · (1 + AR(1) noise), seeded per day

So the load forecast error is real and correlated with the weather forecast error.
Public holidays are not modeled (they are country-specific).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import numpy.typing as npt

from voltproof.data import day_rng
from voltproof.models import HOURS, LoadParams, Site, SiteType, window_hours

FloatArray = npt.NDArray[np.float64]
TemperatureTerm = Callable[[FloatArray, FloatArray, LoadParams], FloatArray]

SATURDAY, SUNDAY = 5, 6


@dataclass(frozen=True)
class LoadSeries:
    forecast_kw: FloatArray
    actual_kw: FloatArray
    critical_kw: FloatArray


def activity_schedule(params: LoadParams, weekday: int) -> FloatArray:
    """Activity level 0–1+ per local hour for a weekday (Monday = 0)."""
    schedule = np.zeros(HOURS)
    schedule[window_hours(params.activity_windows)] = 1.0
    if weekday == SATURDAY:
        schedule *= params.saturday_factor
    elif weekday == SUNDAY:
        schedule *= params.sunday_factor
    if weekday in (SATURDAY, SUNDAY) and params.weekend_extra_windows:
        extra = window_hours(params.weekend_extra_windows)
        schedule[extra] = np.maximum(schedule[extra], params.weekend_extra_factor)
    return schedule


def cooling_term(temp_c: FloatArray, activity: FloatArray, params: LoadParams) -> FloatArray:
    amplification = 1.0 + (params.cooling_activity_factor - 1.0) * np.clip(activity, 0.0, 1.0)
    return params.cooling_kw_per_c * np.maximum(0.0, temp_c - params.cooling_threshold_c) * amplification


def heating_term(temp_c: FloatArray, activity: FloatArray, params: LoadParams) -> FloatArray:
    return params.heating_kw_per_c * np.maximum(0.0, params.heating_threshold_c - temp_c)


TEMPERATURE_TERMS: dict[SiteType, TemperatureTerm] = {
    "commercial_cold_store": cooling_term,
    "health_clinic": cooling_term,
    "household": heating_term,
}


def noncritical_load_kw(site: Site, day: date, temp_c: npt.ArrayLike) -> FloatArray:
    """Deterministic non-critical load for one local day."""
    temp = _hourly(temp_c, "temp_c")
    params = site.load_params
    activity = activity_schedule(params, day.weekday())
    temperature = TEMPERATURE_TERMS[site.site_type](temp, activity, params)
    return params.base_kw + params.activity_kw * activity + temperature


def ar1_noise(rng: np.random.Generator, n: int, phi: float, sigma: float) -> FloatArray:
    """Stationary AR(1) series with standard deviation `sigma`."""
    noise = np.empty(n)
    noise[0] = rng.normal(0.0, sigma)
    innovation_sigma = sigma * np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + rng.normal(0.0, innovation_sigma)
    return noise


def generate_load(
    site: Site, day: date, temp_forecast_c: npt.ArrayLike, temp_actual_c: npt.ArrayLike, *, seed: int
) -> LoadSeries:
    critical = np.full(HOURS, site.critical_load_kw)
    forecast = critical + noncritical_load_kw(site, day, temp_forecast_c)
    params = site.load_params
    rng = day_rng(seed, f"load:{site.site_id}", day)
    noise = ar1_noise(rng, HOURS, params.ar1_phi, params.noise_sigma_frac)
    noncritical_actual = np.maximum(0.0, noncritical_load_kw(site, day, temp_actual_c) * (1.0 + noise))
    return LoadSeries(forecast_kw=forecast, actual_kw=critical + noncritical_actual, critical_kw=critical)


def _hourly(values: npt.ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.shape != (HOURS,):
        raise ValueError(f"{name} must have {HOURS} hourly values, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array
