"""Seeded random days and specs for the core tests (synthetic sites only, no real data)."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from tests.fakes_data import TOU_TEMPLATE, make_site
from voltproof.models import (
    BatterySpec,
    DayData,
    GridSpec,
    HourWindow,
    ImportCap,
    IntentSpec,
    ReserveWindow,
    Site,
    SocTarget,
    WeatherSummary,
)

DAY = date(2024, 6, 5)
SUMMARY = WeatherSummary(
    ghi_forecast_kwh_per_m2=5,
    ghi_actual_kwh_per_m2=5,
    clear_sky_index_forecast=0.8,
    clear_sky_index_actual=0.8,
    cloud_cover_forecast_pct=20,
    cloud_cover_actual_pct=20,
    temp_forecast_mean_c=25,
    temp_actual_mean_c=25,
    pv_forecast_kwh=0,
    pv_actual_kwh=0,
    pv_error_kwh=0,
    pv_hourly_mae_kw=0,
    pv_hourly_bias_kw=0,
)


def simple_site(
    *,
    capacity_kwh: float = 10.0,
    power_kw: float = 5.0,
    eta: float = 1.0,
    soc_min_frac: float = 0.1,
    initial_soc_frac: float = 0.5,
    degradation: float = 0.0,
    import_max_kw: float = 50.0,
    export_max_kw: float = 0.0,
    critical_kw: float = 1.0,
) -> Site:
    """A small synthetic site whose numbers are easy to reason about in known-answer tests."""
    base = make_site("core_test_site", tariff_template=TOU_TEMPLATE)
    return base.model_copy(
        update={
            "battery": BatterySpec(
                capacity_kwh=capacity_kwh,
                soc_min_frac=soc_min_frac,
                soc_max_frac=1.0,
                p_charge_max_kw=power_kw,
                p_discharge_max_kw=power_kw,
                eta_charge=eta,
                eta_discharge=eta,
                initial_soc_frac=initial_soc_frac,
                degradation_cost_per_kwh=degradation,
            ),
            "grid": GridSpec(import_max_kw=import_max_kw, export_max_kw=export_max_kw, export_allowed=export_max_kw > 0),
            "critical_load_kw": critical_kw,
        }
    )


def make_day(
    *,
    pv: Any = 0.0,
    load: Any = 2.0,
    critical: Any = 1.0,
    price: Any = 10.0,
    export_price: Any = 0.0,
    pv_actual: Any = None,
    load_actual: Any = None,
) -> DayData:
    def arr(value: Any) -> list[float]:
        return (np.broadcast_to(np.asarray(value, dtype=float), (24,))).tolist()

    return DayData(
        site_id="core_test_site",
        date=DAY,
        pv_forecast_kw=arr(pv),
        pv_actual_kw=arr(pv if pv_actual is None else pv_actual),
        load_forecast_kw=arr(load),
        load_actual_kw=arr(load if load_actual is None else load_actual),
        critical_load_kw=arr(critical),
        import_price_per_kwh=arr(price),
        export_price_per_kwh=arr(export_price),
        weather=SUMMARY,
    )


def random_site(rng: np.random.Generator) -> Site:
    capacity = float(rng.uniform(5, 120))
    exporting = bool(rng.random() < 0.6)
    base = make_site("core_random_site")
    return base.model_copy(
        update={
            "battery": BatterySpec(
                capacity_kwh=capacity,
                soc_min_frac=float(rng.uniform(0.05, 0.2)),
                soc_max_frac=float(rng.uniform(0.9, 1.0)),
                p_charge_max_kw=float(capacity * rng.uniform(0.2, 0.8)),
                p_discharge_max_kw=float(capacity * rng.uniform(0.2, 0.8)),
                eta_charge=float(rng.uniform(0.85, 1.0)),
                eta_discharge=float(rng.uniform(0.85, 1.0)),
                initial_soc_frac=0.5,
                degradation_cost_per_kwh=float(rng.uniform(0, 1.5)),
            ),
            "grid": GridSpec(
                import_max_kw=float(rng.uniform(5, 80)),
                export_max_kw=float(rng.uniform(1, 30)) if exporting else 0.0,
                export_allowed=exporting,
            ),
            "critical_load_kw": float(rng.uniform(0.2, 5)),
        }
    )


def random_day(site: Site, rng: np.random.Generator) -> DayData:
    hours = np.arange(24)
    bell = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None)
    pv_peak = float(rng.uniform(0, 3)) * site.battery.capacity_kwh / 4
    pv_f = pv_peak * bell * rng.uniform(0.6, 1.0)
    pv_a = np.clip(pv_f * rng.normal(1.0, 0.25, 24), 0, None)
    critical = np.full(24, site.critical_load_kw)
    load_f = critical + rng.uniform(0, 2, 24) * site.battery.capacity_kwh / 10
    load_a = np.clip(load_f * rng.normal(1.0, 0.1, 24), critical, None)
    price = rng.uniform(5, 15, 24) * float(rng.choice([0.01, 1.0, 100.0]))
    export = price * rng.uniform(0, 0.6, 24)
    return make_day(pv=pv_f, load=load_f, critical=critical, price=price, export_price=export, pv_actual=pv_a, load_actual=load_a)


def random_window(rng: np.random.Generator, min_len: int = 1, max_len: int = 6) -> tuple[int, int]:
    length = int(rng.integers(min_len, max_len + 1))
    start = int(rng.integers(0, 24 - length + 1))
    return start, start + length


def random_spec(rng: np.random.Generator) -> IntentSpec:
    fields: dict[str, Any] = {}
    if rng.random() < 0.3:
        fields["grid_outages"] = [HourWindow(start_hour=a, end_hour=b) for a, b in [random_window(rng, 1, 4)]]
    if rng.random() < 0.4:
        a, b = random_window(rng)
        fields["reserve_windows"] = [ReserveWindow(start_hour=a, end_hour=b, min_soc_pct=float(rng.uniform(15, 90)))]
    if rng.random() < 0.3:
        fields["soc_targets"] = [SocTarget(by_hour=int(rng.integers(1, 25)), min_soc_pct=float(rng.uniform(20, 95)))]
    if rng.random() < 0.2:
        fields["min_soc_pct_all_day"] = float(rng.uniform(10, 40))
    if rng.random() < 0.2:
        fields["end_of_day_min_soc_pct"] = float(rng.uniform(10, 80))
    if rng.random() < 0.3:
        a, b = random_window(rng)
        fields["import_caps"] = [ImportCap(start_hour=a, end_hour=b, max_kw=float(rng.uniform(0, 10)))]
    if rng.random() < 0.3:
        a, b = random_window(rng)
        fields["no_export_windows"] = [HourWindow(start_hour=a, end_hour=b)]
    if rng.random() < 0.2:
        a, b = random_window(rng, 2, 8)
        fields["discharge_only_windows"] = [HourWindow(start_hour=a, end_hour=b)]
    if rng.random() < 0.2:
        fields["max_throughput_kwh"] = float(rng.uniform(0, 60))
    return IntentSpec(**fields)
