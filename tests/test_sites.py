from __future__ import annotations

import re
from pathlib import Path

import pytest

from voltproof.config import PROJECT_ROOT
from voltproof.data.sites import SITES, dev_sites, get_site
from voltproof.models import Site

# Expected values from the SPEC §6.1 table for the development sites.
DEV_TABLE = {
    "ahmedabad_cold_store": dict(
        site_type="commercial_cold_store", timezone="Asia/Kolkata", pv=40, battery=(100, 50),
        grid=(60, 30, True), kind="time_of_use", currency="INR", critical=8.0,
    ),
    "nairobi_clinic": dict(
        site_type="health_clinic", timezone="Africa/Nairobi", pv=15, battery=(40, 15),
        grid=(20, 0, False), kind="flat", currency="KES", critical=0.8,
    ),
}  # fmt: skip


def test_three_valid_sites_with_unique_ids() -> None:
    assert len(SITES) == 3
    for site_id, site in SITES.items():
        assert site.site_id == site_id
        assert Site.model_validate(site.model_dump()) == site  # round-trips through validation


@pytest.mark.parametrize("site_id", sorted(DEV_TABLE))
def test_dev_sites_match_spec_table(site_id: str) -> None:
    site, expected = get_site(site_id), DEV_TABLE[site_id]
    assert site.site_type == expected["site_type"] and site.timezone == expected["timezone"]
    assert site.pv.capacity_kwp == expected["pv"]
    assert (site.battery.capacity_kwh, site.battery.p_charge_max_kw, site.battery.p_discharge_max_kw) == (
        expected["battery"][0],
        expected["battery"][1],
        expected["battery"][1],
    )
    assert (site.grid.import_max_kw, site.grid.export_max_kw, site.grid.export_allowed) == expected["grid"]
    assert site.tariff_template.kind == expected["kind"] and site.currency == expected["currency"]
    assert site.critical_load_kw == expected["critical"]
    assert not site.held_out


def test_exactly_one_held_out_site_and_it_is_a_household_with_dynamic_prices() -> None:
    held_out = [site for site in SITES.values() if site.held_out]
    assert len(held_out) == 1
    site = held_out[0]
    assert site.site_type == "household" and site.tariff_template.kind == "hourly_dynamic"
    assert (site.pv.capacity_kwp, site.battery.capacity_kwh, site.battery.p_charge_max_kw) == (10, 15, 5)
    assert (site.grid.import_max_kw, site.grid.export_max_kw, site.critical_load_kw) == (15, 7, 0.15)
    assert site not in dev_sites()


def test_shared_battery_parameters() -> None:
    for site in SITES.values():
        b = site.battery
        assert (b.eta_charge, b.eta_discharge, b.soc_min_frac, b.soc_max_frac, b.initial_soc_frac) == (
            0.95,
            0.95,
            0.10,
            1.00,
            0.50,
        )


def test_degradation_cost_is_10_to_15_pct_of_import_price() -> None:
    from datetime import date

    from voltproof.data.tariff import build_tariff

    for site in SITES.values():
        tariff = build_tariff(site.tariff_template, date(2024, 6, 1), seed=2026, stream=site.site_id)
        mean_price = sum(tariff.import_price_per_kwh) / 24
        assert 0.10 <= site.battery.degradation_cost_per_kwh / mean_price <= 0.16, site.site_id


def _weekly_mean_daily_kwh(site: Site, temp_c: float) -> float:
    from datetime import date, timedelta

    import numpy as np

    from voltproof.data.load import noncritical_load_kw

    week = [date(2024, 6, 3) + timedelta(days=i) for i in range(7)]
    return float(
        np.mean([24 * site.critical_load_kw + noncritical_load_kw(site, day, np.full(24, temp_c)).sum() for day in week])
    )


def test_clinic_daily_load_matches_spec_target() -> None:
    # SPEC §6.1: total load about 45–55 kWh/day (22 °C: below the cooling threshold).
    assert 45 <= _weekly_mean_daily_kwh(get_site("nairobi_clinic"), 22.0) <= 55


def test_held_out_household_daily_load_is_household_sized() -> None:
    # SPEC §6.1: about 11–12 kWh/day over a year. At 15 °C there is no heating term (≈ 10.4 kWh);
    # at 5 °C heating adds 0.01 kW/°C · 10 °C · 24 h = 2.4 kWh.
    site = next(site for site in SITES.values() if site.held_out)
    assert 10.0 <= _weekly_mean_daily_kwh(site, 15.0) <= 11.0
    assert _weekly_mean_daily_kwh(site, 5.0) - _weekly_mean_daily_kwh(site, 15.0) == pytest.approx(2.4)


def test_unknown_site_id() -> None:
    with pytest.raises(KeyError, match="unknown site_id"):
        get_site("nowhere")


def test_no_country_currency_or_timezone_outside_sites_py() -> None:
    """Global by design: locations, currencies and timezones live only in voltproof/data/sites.py."""
    banned = set()
    for site in SITES.values():
        banned |= {site.country, site.currency, site.timezone}
    pattern = re.compile("|".join(re.escape(token) for token in banned))
    offenders = []
    for path in (PROJECT_ROOT / "voltproof").rglob("*.py"):
        if path == PROJECT_ROOT / "voltproof" / "data" / "sites.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{Path(path).relative_to(PROJECT_ROOT)}:{number}: {line.strip()}")
    assert offenders == []
