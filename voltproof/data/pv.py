"""PV power model and solar geometry (SPEC §6.2).

PV: a GHI-based model with no transposition to the panel plane (documented simplification):
    T_cell = T_air + (NOCT − 20) / 800 · GHI
    P_ac   = capacity_kwp · (GHI / 1000) · PR · (1 + γ · (T_cell − 25)), clipped to [0, capacity_kwp]

Solar geometry uses the NOAA fractional-year approximations (declination and equation of
time), accurate to about a minute, which is plenty for alignment checks. Clear-sky GHI uses
the Haurwitz model, GHI_cs = 1098 · cos z · exp(−0.057 / cos z).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from voltproof.models import PVSpec

FloatArray = npt.NDArray[np.float64]


def cell_temperature_c(ghi_w_per_m2: npt.ArrayLike, temp_air_c: npt.ArrayLike, noct_c: float) -> FloatArray:
    ghi = np.asarray(ghi_w_per_m2, dtype=float)
    return np.asarray(temp_air_c, dtype=float) + (noct_c - 20.0) / 800.0 * ghi


def pv_power_kw(ghi_w_per_m2: npt.ArrayLike, temp_air_c: npt.ArrayLike, pv: PVSpec) -> FloatArray:
    """AC power for hourly-mean GHI (W/m²) and air temperature (°C)."""
    ghi = np.clip(np.asarray(ghi_w_per_m2, dtype=float), 0.0, None)
    t_cell = cell_temperature_c(ghi, temp_air_c, pv.noct_c)
    power = pv.capacity_kwp * (ghi / 1000.0) * pv.performance_ratio * (1.0 + pv.temp_coeff_per_c * (t_cell - 25.0))
    return np.clip(power, 0.0, pv.capacity_kwp)


def _fractional_year_rad(utc: pd.DatetimeIndex) -> FloatArray:
    hours = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
    days_in_year = np.where(utc.is_leap_year, 366.0, 365.0)
    return 2.0 * np.pi / days_in_year * (utc.dayofyear - 1 + (hours - 12.0) / 24.0)


def equation_of_time_min(utc: pd.DatetimeIndex) -> FloatArray:
    g = _fractional_year_rad(utc)
    return 229.18 * (
        0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
    )


def declination_rad(utc: pd.DatetimeIndex) -> FloatArray:
    g = _fractional_year_rad(utc)
    return (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )


def cos_zenith(utc: pd.DatetimeIndex, lat: float, lon: float) -> FloatArray:
    """Cosine of the solar zenith angle at UTC instants (negative when the sun is below the horizon)."""
    utc = _as_utc(utc)
    minutes = utc.hour * 60.0 + utc.minute + utc.second / 60.0
    true_solar_min = minutes + equation_of_time_min(utc) + 4.0 * lon
    hour_angle = np.radians(true_solar_min / 4.0 - 180.0)
    lat_rad = np.radians(lat)
    decl = declination_rad(utc)
    return np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)


def solar_elevation_deg(utc: pd.DatetimeIndex, lat: float, lon: float) -> FloatArray:
    return np.degrees(np.arcsin(np.clip(cos_zenith(utc, lat, lon), -1.0, 1.0)))


def solar_noon_utc_min(utc_dates: pd.DatetimeIndex, lon: float) -> FloatArray:
    """Minutes after 00:00 UTC of solar noon on each date."""
    noon_guess = _as_utc(utc_dates).normalize() + pd.Timedelta(hours=12)
    return 720.0 - 4.0 * lon - equation_of_time_min(noon_guess)


def clear_sky_ghi_w_per_m2(utc: pd.DatetimeIndex, lat: float, lon: float) -> FloatArray:
    """Haurwitz clear-sky GHI at UTC instants (0 when the sun is down)."""
    cz = cos_zenith(utc, lat, lon)
    safe = np.where(cz > 0, cz, 1.0)
    return np.where(cz > 0, 1098.0 * cz * np.exp(-0.057 / safe), 0.0)


def _as_utc(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(times)
    return times.tz_localize("UTC") if times.tz is None else times.tz_convert("UTC")
