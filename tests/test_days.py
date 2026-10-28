from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.fakes_data import DYNAMIC_TEMPLATE, FLAT_TEMPLATE, TOU_TEMPLATE, make_site, synthetic_utc_frame
from voltproof.config import PROJECT_ROOT
from voltproof.data.days import (
    DEV_PERIOD,
    SPLITS_PATH,
    TEST_PERIOD,
    build_day,
    day_stats,
    eligible_dates,
    load_splits,
    make_splits,
    site_stats,
    split_dates,
)
from voltproof.data.sites import SITES
from voltproof.data.weather import PERIOD_END, PERIOD_START, align_to_local, utc_fetch_range
from voltproof.models import DayData

SITE_A = make_site("site_a", site_type="commercial_cold_store", timezone="Asia/Kolkata", lat=23, lon=72.6, tariff_template=TOU_TEMPLATE)
SITE_B = make_site("site_b", site_type="health_clinic", timezone="Africa/Nairobi", lat=-1.3, lon=36.8, tariff_template=FLAT_TEMPLATE)
HELD = make_site(
    "held_site", site_type="household", timezone="America/Chicago", lat=41.9, lon=-87.6,
    tariff_template=DYNAMIC_TEMPLATE, held_out=True,
)  # fmt: skip
TEST_SITES = [SITE_A, SITE_B, HELD]


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    out = {}
    for seed, site in enumerate(TEST_SITES):
        utc = synthetic_utc_frame(site, *utc_fetch_range(PERIOD_START, PERIOD_END), seed=seed)
        out[site.site_id] = align_to_local(utc, site, PERIOD_START, PERIOD_END)
    return out


def test_build_day_shapes_and_consistency(frames: dict[str, pd.DataFrame]) -> None:
    day = build_day(SITE_A, date(2024, 6, 5), frames["site_a"])
    assert isinstance(day, DayData) and day.date == date(2024, 6, 5) and day.site_id == "site_a"
    for name in ("pv_forecast_kw", "pv_actual_kw", "load_forecast_kw", "load_actual_kw", "critical_load_kw"):
        assert len(getattr(day, name)) == 24
    assert max(day.pv_actual_kw) <= SITE_A.pv.capacity_kwp and min(day.pv_actual_kw) == 0.0
    assert set(day.critical_load_kw) == {SITE_A.critical_load_kw}
    assert day.weather.pv_error_kwh == pytest.approx(day.weather.pv_forecast_kwh - day.weather.pv_actual_kwh)
    assert day.import_price_per_kwh[18] == 15.0  # test ToU peak
    assert 0 < day.weather.clear_sky_index_actual <= 1.05


def test_build_day_is_deterministic(frames: dict[str, pd.DataFrame]) -> None:
    a = build_day(HELD, date(2025, 2, 3), frames["held_site"])
    b = build_day(HELD, date(2025, 2, 3), frames["held_site"])
    assert a == b
    assert build_day(HELD, date(2025, 2, 3), frames["held_site"], seed=1) != a


def test_dst_days_are_not_eligible_and_cannot_be_built(frames: dict[str, pd.DataFrame]) -> None:
    dates = eligible_dates(frames["held_site"])
    for dst_day in (date(2024, 3, 10), date(2024, 11, 3), date(2025, 3, 9), date(2025, 11, 2)):
        assert dst_day not in dates
        with pytest.raises(ValueError, match="24 complete local hours"):
            build_day(HELD, dst_day, frames["held_site"])
    assert len(eligible_dates(frames["site_a"])) == (PERIOD_END - PERIOD_START).days + 1


def test_day_stats(frames: dict[str, pd.DataFrame]) -> None:
    stats = day_stats(SITE_B, frames["site_b"])
    assert len(stats) == (PERIOD_END - PERIOD_START).days + 1
    day = build_day(SITE_B, date(2024, 7, 1), frames["site_b"])
    row = stats.loc[date(2024, 7, 1)]
    assert row["pv_actual_kwh"] == pytest.approx(day.weather.pv_actual_kwh)
    assert row["csi_actual"] == pytest.approx(day.weather.clear_sky_index_actual)


def test_site_stats_keys(frames: dict[str, pd.DataFrame]) -> None:
    dates = eligible_dates(frames["site_a"])[:10]
    stats = site_stats(SITE_A, frames["site_a"], dates)
    assert stats["n_days"] == 10 and stats["pv_capacity_kwp"] == 20 and stats["grid_import_max_kw"] == 50
    assert stats["pv_actual_max_kw"] <= 20 and stats["load_actual_max_kw"] > stats["pv_hourly_mae_kw"]


# --- splits ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def splits(frames: dict[str, pd.DataFrame]) -> dict:
    return make_splits(TEST_SITES, frames, seed=2026)


def test_split_sizes_and_categories(splits: dict) -> None:
    for site in TEST_SITES:
        test = [e for e in splits["test"] if e["site_id"] == site.site_id]
        assert sorted(e["category"] for e in test) == ["clear", "clear", "cloudy", "large_error", "large_error"]
    for site in (SITE_A, SITE_B):
        dev = [e for e in splits["dev"] if e["site_id"] == site.site_id]
        assert sorted(e["category"] for e in dev) == ["clear", "cloudy", "large_error"]
    assert len(splits["dev"]) == 6 and len(splits["test"]) == 15


def test_held_out_site_only_in_test(splits: dict) -> None:
    assert all(e["site_id"] != HELD.site_id for e in splits["dev"])
    assert len(split_dates(splits, "test", HELD.site_id)) == 5


def test_no_overlap_and_temporal_split(splits: dict) -> None:
    dev = {(e["site_id"], e["date"]) for e in splits["dev"]}
    test = {(e["site_id"], e["date"]) for e in splits["test"]}
    assert not dev & test
    assert all(DEV_PERIOD[0] <= date.fromisoformat(d) <= DEV_PERIOD[1] for _, d in dev)
    assert all(TEST_PERIOD[0] <= date.fromisoformat(d) <= TEST_PERIOD[1] for _, d in test)
    for site in TEST_SITES:
        assert len(split_dates(splits, "test", site.site_id)) == len(set(split_dates(splits, "test", site.site_id)))


def test_categories_follow_their_rules(splits: dict, frames: dict[str, pd.DataFrame]) -> None:
    for site in TEST_SITES:
        thresholds = splits["thresholds"][site.site_id]["test"]
        stats = day_stats(site, frames[site.site_id])
        test_stats = stats[[TEST_PERIOD[0] <= d <= TEST_PERIOD[1] for d in stats.index]]
        entries = [e for e in splits["test"] if e["site_id"] == site.site_id]
        for entry in entries:
            if entry["category"] == "clear":
                assert entry["clear_sky_index_actual"] >= thresholds["clear_csi_min"] - 1e-4
            if entry["category"] == "cloudy":
                assert entry["clear_sky_index_actual"] <= thresholds["cloudy_csi_max"] + 1e-4
        largest = test_stats["pv_abs_error_kwh"].max()
        error_days = [e for e in entries if e["category"] == "large_error"]
        picked = {date.fromisoformat(e["date"]) for e in entries if e["category"] != "large_error"}
        if test_stats["pv_abs_error_kwh"].idxmax() not in picked:
            assert max(e["pv_abs_error_kwh"] for e in error_days) == pytest.approx(largest, abs=1e-3)


def test_splits_are_reproducible_and_seed_dependent(splits: dict, frames: dict[str, pd.DataFrame]) -> None:
    assert make_splits(TEST_SITES, frames, seed=2026) == splits
    other = make_splits(TEST_SITES, frames, seed=7)
    assert [e["date"] for e in other["test"]] != [e["date"] for e in splits["test"]]


def test_too_few_days_raises(frames: dict[str, pd.DataFrame]) -> None:
    short = {site_id: frame[frame["date_local"] < date(2024, 1, 23)] for site_id, frame in frames.items()}
    with pytest.raises(ValueError, match="eligible days"):
        make_splits(TEST_SITES, short)


def test_committed_splits_file_respects_held_out_sites() -> None:
    if not SPLITS_PATH.exists():
        pytest.skip("data/splits.json not generated yet")
    committed = load_splits()
    held_out = {site.site_id for site in SITES.values() if site.held_out}
    assert held_out and not {e["site_id"] for e in committed["dev"]} & held_out
    assert {e["site_id"] for e in committed["test"]} == set(SITES)
    assert committed["seed"] == 2026


def test_no_held_out_site_name_in_tests_or_fixtures() -> None:
    """SPEC §7: the held-out site must not appear in dev fixtures or tests."""
    word = re.compile("mun" + "ich", re.IGNORECASE)  # split so this file does not match itself
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "tests").rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".csv", ".txt"} and word.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []


def test_synthetic_frames_have_variety(frames: dict[str, pd.DataFrame]) -> None:
    stats = day_stats(SITE_A, frames["site_a"])
    assert np.ptp(stats["csi_actual"]) > 0.5
