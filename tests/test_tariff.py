from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from tests.fakes_data import DYNAMIC_TEMPLATE, FLAT_TEMPLATE, TOU_TEMPLATE
from voltproof.data.sites import get_site
from voltproof.data.tariff import build_tariff

DAY = date(2024, 6, 5)


def test_reference_time_of_use_ratios_and_windows() -> None:
    template = get_site("ahmedabad_cold_store").tariff_template
    tariff = build_tariff(template, DAY, seed=2026, stream="s")
    prices = tariff.import_price_per_kwh
    assert tariff.kind == "time_of_use" and tariff.currency == "INR"
    assert tariff.periods == {
        "normal": [0, 1, 2, 3, 4, 5, 8, 9, 10, 15, 16, 23],
        "peak": [6, 7, 17, 18, 19, 20, 21, 22],
        "solar": [11, 12, 13, 14],
    }
    normal = prices[0]
    assert normal == pytest.approx(8.00)
    assert all(prices[h] / normal == pytest.approx(1.2) for h in tariff.periods["peak"])
    assert all(prices[h] / normal == pytest.approx(0.8) for h in tariff.periods["solar"])
    assert set(tariff.export_price_per_kwh) == {3.00}


def test_time_of_use_covers_all_hours_and_is_date_independent() -> None:
    a = build_tariff(TOU_TEMPLATE, DAY, seed=1, stream="s")
    b = build_tariff(TOU_TEMPLATE, date(2025, 1, 1), seed=99, stream="t")
    assert a == b
    assert sorted(h for hours in a.periods.values() for h in hours) == list(range(24))
    assert a.import_price_per_kwh[18] == 15.0 and a.import_price_per_kwh[2] == 5.0 and a.import_price_per_kwh[12] == 10.0


def test_flat_tariff() -> None:
    tariff = build_tariff(FLAT_TEMPLATE, DAY, seed=1, stream="s")
    assert tariff.kind == "flat" and tariff.periods == {}
    assert set(tariff.import_price_per_kwh) == {20.0} and set(tariff.export_price_per_kwh) == {0.0}


def test_reference_flat_tariff_has_no_export() -> None:
    tariff = build_tariff(get_site("nairobi_clinic").tariff_template, DAY, seed=2026, stream="s")
    assert set(tariff.import_price_per_kwh) == {25.00} and set(tariff.export_price_per_kwh) == {0.0}


def test_hourly_dynamic_is_deterministic_per_seed_and_date() -> None:
    a = build_tariff(DYNAMIC_TEMPLATE, DAY, seed=2026, stream="site_a", solar_frac=0.5)
    b = build_tariff(DYNAMIC_TEMPLATE, DAY, seed=2026, stream="site_a", solar_frac=0.5)
    assert a == b
    assert build_tariff(DYNAMIC_TEMPLATE, date(2024, 6, 6), seed=2026, stream="site_a", solar_frac=0.5) != a
    assert build_tariff(DYNAMIC_TEMPLATE, DAY, seed=2027, stream="site_a", solar_frac=0.5) != a
    assert build_tariff(DYNAMIC_TEMPLATE, DAY, seed=2026, stream="site_b", solar_frac=0.5) != a


def test_hourly_dynamic_named_periods() -> None:
    tariff = build_tariff(DYNAMIC_TEMPLATE, DAY, seed=2026, stream="s", solar_frac=0.3)
    prices = np.array(tariff.import_price_per_kwh)
    cheap, peak = tariff.periods["cheap"], tariff.periods["peak"]
    assert len(cheap) == 6 and len(peak) == 4 and not set(cheap) & set(peak)
    others = [h for h in range(24) if h not in cheap]
    assert prices[cheap].max() <= prices[others].min()
    rest = [h for h in range(24) if h not in peak]
    assert prices[peak].min() >= prices[rest].max()
    assert set(tariff.export_price_per_kwh) == {0.2}


def test_hourly_dynamic_level_floor_and_solar_dip() -> None:
    means = [
        np.mean(build_tariff(DYNAMIC_TEMPLATE, date(2024, 1, 1 + i), seed=2026, stream="s").import_price_per_kwh)
        for i in range(30)
    ]
    assert np.mean(means) == pytest.approx(1.0, abs=0.06)
    sunny = build_tariff(DYNAMIC_TEMPLATE, DAY, seed=2026, stream="s", solar_frac=1.0).import_price_per_kwh
    dull = build_tariff(DYNAMIC_TEMPLATE, DAY, seed=2026, stream="s", solar_frac=0.0).import_price_per_kwh
    assert sunny[12] == pytest.approx(dull[12] * 0.7)  # 30 % dip at full sun, same random draws
    assert sunny[20] == dull[20]
    assert min(sunny) >= 0.3


def test_flat_prices_tie_free_periods_stay_disjoint() -> None:
    flat_shape = DYNAMIC_TEMPLATE.model_copy(update={"shape": [1.0] * 24, "day_sigma_frac": 0.0, "hour_sigma_frac": 0.0})
    tariff = build_tariff(flat_shape, DAY, seed=1, stream="s")
    assert tariff.periods["cheap"] == [0, 1, 2, 3, 4, 5] and tariff.periods["peak"] == [6, 7, 8, 9]
