"""DayData assembly and the dev/test day split (SPEC §6.2).

A day is eligible when it has exactly 24 complete local hours (DST-change dates are not).
Splits are temporal: dev days come from 2024, test days from 2025, so no test day is near
a dev day. Held-out sites (`Site.held_out`) appear only in the test split.
"""

from __future__ import annotations

import json
import zlib
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from voltproof.config import PROJECT_ROOT
from voltproof.data import GLOBAL_SEED
from voltproof.data.load import generate_load
from voltproof.data.pv import pv_power_kw
from voltproof.data.tariff import build_tariff
from voltproof.data.weather import CLEAR_SKY_COLUMN, MODEL, PERIOD_END, PERIOD_START, daily_quality
from voltproof.models import HOURS, DayData, Site, Tariff, WeatherSummary

SPLITS_PATH = PROJECT_ROOT / "data" / "splits.json"
DEV_PERIOD = (PERIOD_START, date(2024, 12, 31))
TEST_PERIOD = (date(2025, 1, 1), PERIOD_END)
CLEAR_QUANTILE = 0.8
CLOUDY_QUANTILE = 0.2
TEST_PICKS = {"clear": 2, "cloudy": 1, "large_error": 2}
DEV_PICKS = {"clear": 1, "cloudy": 1, "large_error": 1}


def eligible_dates(frame: pd.DataFrame) -> list[date]:
    quality = daily_quality(frame)
    return sorted(quality.index[quality["eligible"]])


def day_weather(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    """The 24 local hours of `day`, ordered by hour; raises if the day is not complete."""
    rows = frame[frame["date_local"] == day].sort_values("hour_local")
    if rows["hour_local"].tolist() != list(range(HOURS)) or rows.drop(columns=["time_utc", "time_local"]).isna().any().any():
        raise ValueError(f"{day} does not have 24 complete local hours")
    return rows.reset_index(drop=True)


def clear_sky_index(ghi: np.ndarray, clear_sky: np.ndarray) -> float:
    total = float(np.sum(clear_sky))
    return float(np.sum(ghi)) / total if total > 0 else 0.0


def day_tariff(site: Site, day: date, frame: pd.DataFrame, *, seed: int = GLOBAL_SEED) -> Tariff:
    weather = day_weather(frame, day)
    solar_frac = clear_sky_index(weather["ghi_forecast_w_per_m2"].to_numpy(), weather[CLEAR_SKY_COLUMN].to_numpy())
    return build_tariff(site.tariff_template, day, seed=seed, stream=site.site_id, solar_frac=solar_frac)


def build_day(site: Site, day: date, frame: pd.DataFrame, *, seed: int = GLOBAL_SEED) -> DayData:
    """DayData for one local date from the site's processed weather frame."""
    w = day_weather(frame, day)
    ghi_f, ghi_a = w["ghi_forecast_w_per_m2"].to_numpy(), w["ghi_actual_w_per_m2"].to_numpy()
    temp_f, temp_a = w["temp_forecast_c"].to_numpy(), w["temp_actual_c"].to_numpy()
    clear = w[CLEAR_SKY_COLUMN].to_numpy()
    pv_f, pv_a = pv_power_kw(ghi_f, temp_f, site.pv), pv_power_kw(ghi_a, temp_a, site.pv)
    load = generate_load(site, day, temp_f, temp_a, seed=seed)
    tariff = build_tariff(
        site.tariff_template, day, seed=seed, stream=site.site_id, solar_frac=clear_sky_index(ghi_f, clear)
    )
    error = pv_f - pv_a
    summary = WeatherSummary(
        ghi_forecast_kwh_per_m2=float(ghi_f.sum()) / 1000.0,
        ghi_actual_kwh_per_m2=float(ghi_a.sum()) / 1000.0,
        clear_sky_index_forecast=clear_sky_index(ghi_f, clear),
        clear_sky_index_actual=clear_sky_index(ghi_a, clear),
        cloud_cover_forecast_pct=float(w["cloud_forecast_pct"].mean()),
        cloud_cover_actual_pct=float(w["cloud_actual_pct"].mean()),
        temp_forecast_mean_c=float(temp_f.mean()),
        temp_actual_mean_c=float(temp_a.mean()),
        pv_forecast_kwh=float(pv_f.sum()),
        pv_actual_kwh=float(pv_a.sum()),
        pv_error_kwh=float(error.sum()),
        pv_hourly_mae_kw=float(np.abs(error).mean()),
        pv_hourly_bias_kw=float(error.mean()),
    )
    return DayData(
        site_id=site.site_id,
        date=day,
        pv_forecast_kw=pv_f.tolist(),
        pv_actual_kw=pv_a.tolist(),
        load_forecast_kw=load.forecast_kw.tolist(),
        load_actual_kw=load.actual_kw.tolist(),
        critical_load_kw=load.critical_kw.tolist(),
        import_price_per_kwh=tariff.import_price_per_kwh,
        export_price_per_kwh=tariff.export_price_per_kwh,
        weather=summary,
    )


def day_stats(site: Site, frame: pd.DataFrame, dates: Sequence[date] | None = None) -> pd.DataFrame:
    """Per eligible date: clear-sky index (actual, forecast) and daily PV energy forecast vs actual."""
    chosen = set(eligible_dates(frame) if dates is None else dates)
    rows = frame[frame["date_local"].isin(chosen)].copy()
    rows["pv_forecast_kw"] = pv_power_kw(rows["ghi_forecast_w_per_m2"], rows["temp_forecast_c"], site.pv)
    rows["pv_actual_kw"] = pv_power_kw(rows["ghi_actual_w_per_m2"], rows["temp_actual_c"], site.pv)
    daily = rows.groupby("date_local").agg(
        ghi_actual=("ghi_actual_w_per_m2", "sum"),
        ghi_forecast=("ghi_forecast_w_per_m2", "sum"),
        ghi_clear=(CLEAR_SKY_COLUMN, "sum"),
        pv_forecast_kwh=("pv_forecast_kw", "sum"),
        pv_actual_kwh=("pv_actual_kw", "sum"),
    )
    clear = daily["ghi_clear"].where(daily["ghi_clear"] > 0)
    daily["csi_actual"] = (daily["ghi_actual"] / clear).fillna(0.0)
    daily["csi_forecast"] = (daily["ghi_forecast"] / clear).fillna(0.0)
    daily["pv_error_kwh"] = daily["pv_forecast_kwh"] - daily["pv_actual_kwh"]
    daily["pv_abs_error_kwh"] = daily["pv_error_kwh"].abs()
    return daily[["csi_actual", "csi_forecast", "pv_forecast_kwh", "pv_actual_kwh", "pv_error_kwh", "pv_abs_error_kwh"]]


def site_stats(
    site: Site, frame: pd.DataFrame, dates: Sequence[date] | None = None, *, seed: int = GLOBAL_SEED
) -> dict[str, float]:
    """Forecast-error statistics over the given (default: all eligible) dates, from full DayData."""
    days = [build_day(site, day, frame, seed=seed) for day in (eligible_dates(frame) if dates is None else dates)]
    if not days:
        raise ValueError(f"no days to summarise for {site.site_id}")
    pv_f = np.array([d.pv_forecast_kw for d in days])
    pv_a = np.array([d.pv_actual_kw for d in days])
    load_f = np.array([d.load_forecast_kw for d in days])
    load_a = np.array([d.load_actual_kw for d in days])
    daily_error = pv_f.sum(axis=1) - pv_a.sum(axis=1)
    return {
        "n_days": float(len(days)),
        "pv_daily_forecast_kwh_mean": float(pv_f.sum(axis=1).mean()),
        "pv_daily_actual_kwh_mean": float(pv_a.sum(axis=1).mean()),
        "pv_daily_mae_kwh": float(np.abs(daily_error).mean()),
        "pv_daily_bias_kwh": float(daily_error.mean()),
        "pv_hourly_mae_kw": float(np.abs(pv_f - pv_a).mean()),
        "pv_hourly_bias_kw": float((pv_f - pv_a).mean()),
        "pv_forecast_max_kw": float(pv_f.max()),
        "pv_actual_max_kw": float(pv_a.max()),
        "pv_capacity_kwp": site.pv.capacity_kwp,
        "load_daily_forecast_kwh_mean": float(load_f.sum(axis=1).mean()),
        "load_hourly_mae_kw": float(np.abs(load_f - load_a).mean()),
        "load_hourly_bias_kw": float((load_f - load_a).mean()),
        "load_forecast_max_kw": float(load_f.max()),
        "load_actual_max_kw": float(load_a.max()),
        "grid_import_max_kw": site.grid.import_max_kw,
    }


# --- splits ----------------------------------------------------------------------------


def make_splits(sites: Sequence[Site], frames: Mapping[str, pd.DataFrame], *, seed: int = GLOBAL_SEED) -> dict[str, Any]:
    """Deterministic dev/test days per site. Test: all sites, 2025. Dev: non-held-out sites, 2024.

    Within a pool: `clear` days are sampled from the top 20 % of the actual clear-sky index,
    `cloudy` days from the bottom 20 %, and `large_error` days are the remaining days with the
    largest absolute daily PV forecast error (ties broken by date).
    """
    splits: dict[str, Any] = {
        "seed": seed,
        "weather_model": MODEL,
        "period": [PERIOD_START.isoformat(), PERIOD_END.isoformat()],
        "dev_period": [d.isoformat() for d in DEV_PERIOD],
        "test_period": [d.isoformat() for d in TEST_PERIOD],
        "rules": {
            "clear": f"sampled from days with clear-sky index >= the {CLEAR_QUANTILE:.0%} quantile",
            "cloudy": f"sampled from days with clear-sky index <= the {CLOUDY_QUANTILE:.0%} quantile",
            "large_error": "largest |daily PV forecast - actual| among remaining days",
            "held_out_sites": "test split only",
        },
        "thresholds": {},
        "dev": [],
        "test": [],
    }
    for site in sorted(sites, key=lambda s: s.site_id):
        stats = day_stats(site, frames[site.site_id])
        stream = zlib.crc32(site.site_id.encode("utf-8"))
        thresholds: dict[str, Any] = {}
        test, thresholds["test"] = _pick(
            site.site_id, _in_period(stats, TEST_PERIOD), TEST_PICKS, np.random.default_rng([seed, stream, 1])
        )
        splits["test"].extend(test)
        if not site.held_out:
            dev, thresholds["dev"] = _pick(
                site.site_id, _in_period(stats, DEV_PERIOD), DEV_PICKS, np.random.default_rng([seed, stream, 0])
            )
            splits["dev"].extend(dev)
        splits["thresholds"][site.site_id] = thresholds
    return splits


def write_splits(splits: Mapping[str, Any], path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_splits(path: Path = SPLITS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_dates(splits: Mapping[str, Any], split: str, site_id: str) -> list[date]:
    return sorted(date.fromisoformat(entry["date"]) for entry in splits[split] if entry["site_id"] == site_id)


def _in_period(stats: pd.DataFrame, period: tuple[date, date]) -> pd.DataFrame:
    start, end = period
    return stats[[start <= day <= end for day in stats.index]]


def _pick(
    site_id: str, stats: pd.DataFrame, counts: Mapping[str, int], rng: np.random.Generator
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if len(stats) < sum(counts.values()):
        raise ValueError(f"{site_id}: only {len(stats)} eligible days, need {sum(counts.values())}")
    clear_threshold = float(stats["csi_actual"].quantile(CLEAR_QUANTILE))
    cloudy_threshold = float(stats["csi_actual"].quantile(CLOUDY_QUANTILE))
    chosen: list[tuple[date, str]] = []

    def sample(pool: list[date], n: int, category: str) -> None:
        pool = sorted(day for day in pool if day not in {d for d, _ in chosen})
        if len(pool) < n:
            raise ValueError(f"{site_id}: only {len(pool)} {category} candidates, need {n}")
        for index in sorted(rng.choice(len(pool), size=n, replace=False)):
            chosen.append((pool[index], category))

    sample(list(stats.index[stats["csi_actual"] >= clear_threshold]), counts["clear"], "clear")
    sample(list(stats.index[stats["csi_actual"] <= cloudy_threshold]), counts["cloudy"], "cloudy")
    taken = {d for d, _ in chosen}
    remaining = stats[[day not in taken for day in stats.index]]
    by_error = sorted(remaining.index, key=lambda day: (-remaining.at[day, "pv_abs_error_kwh"], day))
    chosen.extend((day, "large_error") for day in by_error[: counts["large_error"]])

    entries = [
        {
            "site_id": site_id,
            "date": day.isoformat(),
            "category": category,
            "clear_sky_index_actual": round(float(stats.at[day, "csi_actual"]), 4),
            "pv_abs_error_kwh": round(float(stats.at[day, "pv_abs_error_kwh"]), 3),
        }
        for day, category in chosen
    ]
    return entries, {"clear_csi_min": round(clear_threshold, 4), "cloudy_csi_max": round(cloudy_threshold, 4)}
