"""Synthetic sites, weather and HTTP doubles for the data-layer tests. No network.

Test sites are built here from scratch; they never reuse held-out site configuration.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from voltproof.data.pv import clear_sky_ghi_w_per_m2
from voltproof.models import (
    BatterySpec,
    FlatTemplate,
    GridSpec,
    HourlyDynamicTemplate,
    HourWindow,
    LoadParams,
    OutageProfile,
    PVSpec,
    Site,
    TariffTemplate,
    TimeOfUseTemplate,
)

TOU_TEMPLATE = TimeOfUseTemplate(
    name="Test ToU",
    currency="XTS",  # ISO 4217 code reserved for testing
    base_price_per_kwh=10.0,
    default_period="normal",
    windows={"peak": [HourWindow(start_hour=17, end_hour=21)], "cheap": [HourWindow(start_hour=0, end_hour=6)]},
    multipliers={"peak": 1.5, "cheap": 0.5},
    export_price_per_kwh=2.0,
)

FLAT_TEMPLATE = FlatTemplate(name="Test flat", currency="XTS", price_per_kwh=20.0, export_price_per_kwh=0.0)

DYNAMIC_TEMPLATE = HourlyDynamicTemplate(
    name="Test dynamic",
    currency="XTS",
    mean_price_per_kwh=1.0,
    export_price_per_kwh=0.2,
    shape=[0.8] * 6 + [1.2] * 4 + [1.0] * 7 + [1.4] * 4 + [0.9] * 3,
    solar_dip_windows=[HourWindow(start_hour=10, end_hour=16)],
    solar_dip_frac=0.3,
    day_sigma_frac=0.1,
    hour_sigma_frac=0.05,
    floor_frac=0.3,
)

LOAD_PARAMS = {
    "commercial_cold_store": LoadParams(
        base_kw=2.0,
        activity_kw=10.0,
        activity_windows=[HourWindow(start_hour=9, end_hour=19)],
        saturday_factor=1.0,
        sunday_factor=0.25,
        cooling_kw_per_c=0.5,
        cooling_threshold_c=24.0,
        cooling_activity_factor=1.5,
    ),
    "health_clinic": LoadParams(
        base_kw=0.5,
        activity_kw=3.0,
        activity_windows=[HourWindow(start_hour=8, end_hour=17)],
        saturday_factor=0.5,
        sunday_factor=0.5,
        cooling_kw_per_c=0.1,
        cooling_threshold_c=25.0,
    ),
    "household": LoadParams(
        base_kw=0.1,
        activity_kw=1.0,
        activity_windows=[HourWindow(start_hour=6, end_hour=9), HourWindow(start_hour=17, end_hour=22)],
        saturday_factor=1.2,
        sunday_factor=1.2,
        weekend_extra_windows=[HourWindow(start_hour=10, end_hour=16)],
        weekend_extra_factor=0.5,
        heating_kw_per_c=0.05,
        heating_threshold_c=15.0,
    ),
}


def make_site(
    site_id: str = "test_site",
    *,
    site_type: str = "commercial_cold_store",
    timezone: str = "Asia/Kolkata",
    lat: float = 20.0,
    lon: float = 75.0,
    tariff_template: TariffTemplate = TOU_TEMPLATE,
    held_out: bool = False,
) -> Site:
    return Site(
        site_id=site_id,
        name=f"Test {site_id}",
        site_type=site_type,
        country="Testland",
        lat=lat,
        lon=lon,
        timezone=timezone,
        pv=PVSpec(capacity_kwp=20),
        battery=BatterySpec(
            capacity_kwh=40,
            soc_min_frac=0.1,
            soc_max_frac=1.0,
            p_charge_max_kw=20,
            p_discharge_max_kw=20,
            eta_charge=0.95,
            eta_discharge=0.95,
            initial_soc_frac=0.5,
            degradation_cost_per_kwh=1.0,
        ),
        grid=GridSpec(import_max_kw=50, export_max_kw=10, export_allowed=True),
        tariff_template=tariff_template,
        critical_load_name="test fridge",
        critical_load_kw=3.0,
        load_params=LOAD_PARAMS[site_type],
        outage_profile=OutageProfile(outages_per_month=1, duration_hours_min=1, duration_hours_max=3),
        held_out=held_out,
    )


def synthetic_utc_frame(site: Site, start: date, end: date, *, seed: int = 0, shift_hours: int = 0) -> pd.DataFrame:
    """Hourly UTC frame for UTC dates [start, end] in parse_response's column layout.

    GHI is a clear-sky hourly mean (labelled at the end of its hour) scaled by a random daily
    cloudiness factor; the forecast scales by a noisy copy of the same factor. `shift_hours`
    moves the GHI series later in time to simulate a misaligned source.
    """
    index = pd.date_range(
        pd.Timestamp(start, tz="UTC"), pd.Timestamp(end + timedelta(days=1), tz="UTC"), freq="1h", inclusive="left"
    )
    quarter_means = [
        clear_sky_ghi_w_per_m2(index - pd.Timedelta(hours=shift_hours + 1) + pd.Timedelta(minutes=7.5 + 15 * k), site.lat, site.lon)
        for k in range(4)
    ]
    clear = np.mean(quarter_means, axis=0)
    rng = np.random.default_rng(seed)
    day_codes, days = pd.factorize(index.date)
    factor = rng.uniform(0.2, 1.0, len(days))
    forecast_factor = np.clip(factor + rng.normal(0.0, 0.08, len(days)), 0.05, 1.0)
    hours = np.asarray(index.hour, dtype=float)
    temp = 25.0 + 6.0 * np.sin((hours - 9.0) / 24.0 * 2 * np.pi)
    return pd.DataFrame(
        {
            "ghi_actual_w_per_m2": clear * factor[day_codes],
            "ghi_forecast_w_per_m2": clear * forecast_factor[day_codes],
            "temp_actual_c": temp,
            "temp_forecast_c": temp + 0.5,
            "cloud_actual_pct": 100.0 * (1.0 - factor[day_codes]),
            "cloud_forecast_pct": 100.0 * (1.0 - forecast_factor[day_codes]),
        },
        index=pd.DatetimeIndex(index, name="time_utc"),
    )


def open_meteo_payload(
    times: list[str], values: dict[str, list[Any]], *, model: str = "icon_global", suffix: bool = False
) -> dict[str, Any]:
    """A response shaped like Open-Meteo's, for the six variables (keys optionally model-suffixed)."""
    units = {"shortwave_radiation": "W/m²", "temperature_2m": "°C", "cloud_cover": "%"}
    hourly: dict[str, Any] = {"time": times}
    hourly_units: dict[str, str] = {"time": "iso8601"}
    for name, series in values.items():
        key = f"{name}_{model}" if suffix else name
        hourly[key] = series
        hourly_units[key] = units[name.removesuffix("_previous_day1")]
    return {
        "latitude": 20.0,
        "longitude": 75.0,
        "elevation": 300.0,
        "utc_offset_seconds": 0,
        "timezone": "GMT",
        "hourly_units": hourly_units,
        "hourly": hourly,
    }


class FakeResponse:
    def __init__(self, status_code: int, body: Any = None, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = "" if body is None else str(body)

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body


class FakeGet:
    """Plays a script of FakeResponse objects or exceptions; records every call."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, params: dict[str, str], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item
