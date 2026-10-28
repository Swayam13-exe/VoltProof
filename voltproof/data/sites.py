"""The three reference sites (SPEC §6.1). All parameters are illustrative.

This is the only module that names countries, currencies, timezones and price levels;
everything else reads them from a `Site`.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

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
    TimeOfUseTemplate,
)


def _battery(capacity_kwh: float, power_kw: float, degradation_cost_per_kwh: float) -> BatterySpec:
    # Shared across the reference sites: η = 0.95 each way, SOC 10–100 %, start at 50 %.
    return BatterySpec(
        capacity_kwh=capacity_kwh,
        soc_min_frac=0.10,
        soc_max_frac=1.00,
        p_charge_max_kw=power_kw,
        p_discharge_max_kw=power_kw,
        eta_charge=0.95,
        eta_discharge=0.95,
        initial_soc_frac=0.50,
        degradation_cost_per_kwh=degradation_cost_per_kwh,
    )


def _w(start_hour: int, end_hour: int) -> HourWindow:
    return HourWindow(start_hour=start_hour, end_hour=end_hour)


AHMEDABAD_COLD_STORE = Site(
    site_id="ahmedabad_cold_store",
    name="Ahmedabad cold store",
    site_type="commercial_cold_store",
    country="India",
    lat=23.03,
    lon=72.58,
    timezone="Asia/Kolkata",
    pv=PVSpec(capacity_kwp=40),
    battery=_battery(capacity_kwh=100, power_kw=50, degradation_cost_per_kwh=1.00),
    grid=GridSpec(import_max_kw=60, export_max_kw=30, export_allowed=True),
    tariff_template=TimeOfUseTemplate(
        name="Illustrative commercial time-of-use tariff",
        currency="INR",
        base_price_per_kwh=8.00,
        default_period="normal",
        windows={"peak": [_w(6, 8), _w(17, 23)], "solar": [_w(11, 15)]},
        multipliers={"peak": 1.2, "solar": 0.8},
        export_price_per_kwh=3.00,
        notes="Illustrative. Ratios follow India's 2023 time-of-day rules (peak 1.2x, solar hours 0.8x); "
        "not any utility's actual tariff.",
    ),
    critical_load_name="cold room",
    critical_load_kw=8.0,
    load_params=LoadParams(
        base_kw=3.0,
        activity_kw=12.0,
        activity_windows=[_w(9, 19)],
        saturday_factor=1.0,
        sunday_factor=0.25,
        cooling_kw_per_c=0.6,
        cooling_threshold_c=24.0,
        cooling_activity_factor=1.5,
    ),
    outage_profile=OutageProfile(
        outages_per_month=1.0,
        duration_hours_min=2,
        duration_hours_max=4,
        notes="Illustrative: occasional scheduled maintenance outages, usually announced a day ahead.",
    ),
)

NAIROBI_CLINIC = Site(
    site_id="nairobi_clinic",
    name="Rural health clinic near Nairobi",
    site_type="health_clinic",
    country="Kenya",
    lat=-1.29,
    lon=36.82,
    timezone="Africa/Nairobi",
    pv=PVSpec(capacity_kwp=15),
    battery=_battery(capacity_kwh=40, power_kw=15, degradation_cost_per_kwh=3.00),
    grid=GridSpec(import_max_kw=20, export_max_kw=0, export_allowed=False),
    tariff_template=FlatTemplate(
        name="Illustrative flat commercial tariff",
        currency="KES",
        price_per_kwh=25.00,
        export_price_per_kwh=0.0,
        notes="Illustrative single rate; export not allowed.",
    ),
    critical_load_name="vaccine fridge, lights and essential equipment",
    critical_load_kw=0.8,
    load_params=LoadParams(
        base_kw=0.3,
        activity_kw=3.0,
        activity_windows=[_w(8, 17)],
        saturday_factor=0.5,
        sunday_factor=0.5,
        cooling_kw_per_c=0.15,
        cooling_threshold_c=25.0,
        cooling_activity_factor=1.2,
    ),
    outage_profile=OutageProfile(
        outages_per_month=6.0,
        duration_hours_min=1,
        duration_hours_max=6,
        notes="Illustrative: frequent unannounced and announced outages on a rural feeder.",
    ),
)

MUNICH_HOME = Site(
    site_id="munich_home",
    name="Munich household",
    site_type="household",
    country="Germany",
    lat=48.14,
    lon=11.58,
    timezone="Europe/Berlin",
    pv=PVSpec(capacity_kwp=10),
    battery=_battery(capacity_kwh=15, power_kw=5, degradation_cost_per_kwh=0.04),
    grid=GridSpec(import_max_kw=15, export_max_kw=7, export_allowed=True),
    tariff_template=HourlyDynamicTemplate(
        name="Illustrative hourly dynamic household tariff",
        currency="EUR",
        mean_price_per_kwh=0.30,
        export_price_per_kwh=0.08,
        shape=[
            0.85, 0.80, 0.78, 0.78, 0.80, 0.88, 1.05, 1.20, 1.20, 1.10, 1.00, 0.98,
            0.95, 0.95, 0.97, 1.00, 1.05, 1.20, 1.30, 1.30, 1.20, 1.08, 0.98, 0.90,
        ],  # fmt: skip
        solar_dip_windows=[_w(10, 16)],
        solar_dip_frac=0.25,
        day_sigma_frac=0.10,
        hour_sigma_frac=0.05,
        floor_frac=0.3,
        cheap_hours=6,
        peak_hours=4,
        notes="Illustrative retail price following a day-ahead-style hourly shape; not a real market feed.",
    ),
    critical_load_name="fridge, router and lights",
    critical_load_kw=0.15,
    load_params=LoadParams(
        base_kw=0.05,
        activity_kw=0.6,
        activity_windows=[_w(6, 9), _w(17, 22)],
        saturday_factor=1.2,
        sunday_factor=1.2,
        weekend_extra_windows=[_w(10, 16)],
        weekend_extra_factor=0.5,
        heating_kw_per_c=0.01,
        heating_threshold_c=15.0,
    ),
    outage_profile=OutageProfile(
        outages_per_month=0.05,
        duration_hours_min=1,
        duration_hours_max=2,
        notes="Illustrative: outages are rare; mostly announced network works.",
    ),
    held_out=True,
)

SITES: Mapping[str, Site] = MappingProxyType(
    {site.site_id: site for site in (AHMEDABAD_COLD_STORE, NAIROBI_CLINIC, MUNICH_HOME)}
)


def get_site(site_id: str) -> Site:
    try:
        return SITES[site_id]
    except KeyError:
        raise KeyError(f"unknown site_id {site_id!r}; known: {', '.join(SITES)}") from None


def dev_sites() -> list[Site]:
    """Sites usable during development (held-out sites excluded)."""
    return [site for site in SITES.values() if not site.held_out]
