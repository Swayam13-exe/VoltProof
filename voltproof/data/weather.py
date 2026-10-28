"""Open-Meteo Previous Runs weather: download, cache, UTC→local alignment, sanity checks (SPEC §6.2).

Source: DWD ICON global via Open-Meteo's Previous Runs API (CC BY 4.0, "Weather data by Open-Meteo.com").
- actual   = the base hourly variable: ICON's same-day analysis (shortest lead time). Not a measurement.
- forecast = the same variable with suffix `_previous_day1`: the value predicted 24 h before valid time.
So the forecast error here is day-1 vs day-0 model error, smaller than the error against real meters.

Open-Meteo's local-time output is not used. It applies one fixed UTC offset to a whole response
(wrong after a DST change) and labels +5:30 zones on a 5-hour shift. We always request UTC and
convert to each site's local hours here (`align_to_local`), which handles any offset that is a
multiple of 15 minutes and DST days of 23 or 25 hours.

Radiation is the mean over the preceding hour; temperature and cloud cover are instantaneous.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from voltproof.config import PROJECT_ROOT
from voltproof.data.pv import clear_sky_ghi_w_per_m2
from voltproof.models import HOURS, Site

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "icon_global"
MODEL_DESCRIPTION = "DWD ICON global"
PERIOD_START = date(2024, 1, 20)  # radiation _previous_day1 starts 2024-01-19 12:00 UTC for every global model
PERIOD_END = date(2025, 12, 31)

BASE_VARIABLES = ("shortwave_radiation", "temperature_2m", "cloud_cover")
FORECAST_SUFFIX = "_previous_day1"
EXPECTED_UNITS = {"shortwave_radiation": "W/m²", "temperature_2m": "°C", "cloud_cover": "%"}
COLUMNS = {
    "shortwave_radiation": "ghi_actual_w_per_m2",
    "shortwave_radiation_previous_day1": "ghi_forecast_w_per_m2",
    "temperature_2m": "temp_actual_c",
    "temperature_2m_previous_day1": "temp_forecast_c",
    "cloud_cover": "cloud_actual_pct",
    "cloud_cover_previous_day1": "cloud_forecast_pct",
}
HOUR_MEAN_COLUMNS = ("ghi_actual_w_per_m2", "ghi_forecast_w_per_m2")
INSTANT_COLUMNS = ("temp_actual_c", "temp_forecast_c", "cloud_actual_pct", "cloud_forecast_pct")
WEATHER_COLUMNS = tuple(COLUMNS.values())
CLEAR_SKY_COLUMN = "ghi_clear_sky_w_per_m2"
VALUE_COLUMNS = (*WEATHER_COLUMNS, CLEAR_SKY_COLUMN)
CSV_DECIMALS = {
    "ghi_actual_w_per_m2": 1,
    "ghi_forecast_w_per_m2": 1,
    "temp_actual_c": 2,
    "temp_forecast_c": 2,
    "cloud_actual_pct": 1,
    "cloud_forecast_pct": 1,
    CLEAR_SKY_COLUMN: 1,
}

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "open_meteo"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SLOT_MINUTES = 15
SLOTS_PER_HOUR = 60 // SLOT_MINUTES
MAX_GRID_DISTANCE_DEG = 0.3
NIGHT_GHI_LIMIT_W_PER_M2 = 10.0
NOON_OFFSET_LIMIT_MIN = 20.0
MIN_DAILY_CORRELATION = 0.6
RANGES = {
    "ghi_actual_w_per_m2": (0.0, 1400.0),
    "ghi_forecast_w_per_m2": (0.0, 1400.0),
    "temp_actual_c": (-40.0, 55.0),
    "temp_forecast_c": (-40.0, 55.0),
    "cloud_actual_pct": (0.0, 100.0),
    "cloud_forecast_pct": (0.0, 100.0),
}


class WeatherError(RuntimeError):
    """Weather data is missing, malformed or fails a check."""


class WeatherAPIError(WeatherError):
    """Open-Meteo returned an error or could not be reached."""


class HttpResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]
    text: str

    def json(self) -> Any: ...


HttpGet = Callable[..., HttpResponse]  # get(url, params=..., timeout=...)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    backoff_base_s: float = 2.0
    backoff_cap_s: float = 60.0
    timeout_s: float = 60.0


@dataclass(frozen=True)
class GridPoint:
    latitude: float
    longitude: float
    elevation_m: float


# --- download --------------------------------------------------------------------------


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Calendar-month pieces of [start, end], inclusive; the first and last may be partial."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    chunks = []
    current = start
    while current <= end:
        next_month = date(current.year + current.month // 12, current.month % 12 + 1, 1)
        chunk_end = min(next_month - timedelta(days=1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def utc_fetch_range(start_local: date, end_local: date) -> tuple[date, date]:
    """UTC dates covering every local hour of [start_local, end_local] in any timezone (|offset| < 24 h),
    including the hour after the period (an hourly mean is labelled at the end of its hour)."""
    return start_local - timedelta(days=1), end_local + timedelta(days=1)


def covered_local_range(utc_index: pd.DatetimeIndex, timezone: str) -> tuple[date, date]:
    """First and last local dates whose every hour can be built from hourly UTC data on `utc_index`."""
    tz = ZoneInfo(timezone)
    first_local = utc_index[0].tz_convert(tz)
    start = first_local.date()
    if (first_local.hour, first_local.minute) != (0, 0):
        start += timedelta(days=1)
    end = utc_index[-1].tz_convert(tz).date() - timedelta(days=1)
    if start > end:
        raise WeatherError(f"UTC data {utc_index[0]} … {utc_index[-1]} covers no full local day in {timezone}")
    return start, end


def request_params(site: Site, start: date, end: date, model: str = MODEL) -> dict[str, str]:
    hourly = [*BASE_VARIABLES, *(variable + FORECAST_SUFFIX for variable in BASE_VARIABLES)]
    return {
        "latitude": f"{site.lat}",
        "longitude": f"{site.lon}",
        "hourly": ",".join(hourly),
        "models": model,
        "timezone": "GMT",  # always UTC; local alignment happens in align_to_local
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }


def fetch_json(
    get: HttpGet,
    url: str,
    params: Mapping[str, str],
    *,
    sleep: Callable[[float], None],
    policy: RetryPolicy = RetryPolicy(),
) -> dict[str, Any]:
    """GET JSON with retries on 429, 5xx, timeouts and connection errors; other errors raise at once."""
    last_error = ""
    for attempt in range(1, policy.max_attempts + 1):
        retry_after: float | None = None
        try:
            response = get(url, params=dict(params), timeout=policy.timeout_s)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = type(exc).__name__
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise WeatherAPIError("Open-Meteo returned invalid JSON") from exc
            reason = _error_reason(response)
            if response.status_code != 429 and response.status_code < 500:
                raise WeatherAPIError(f"Open-Meteo rejected the request (HTTP {response.status_code}): {reason}")
            last_error = f"HTTP {response.status_code}: {reason}"
            retry_after = _retry_after_s(response)
        if attempt == policy.max_attempts:
            break
        delay = min(policy.backoff_cap_s, policy.backoff_base_s * 2 ** (attempt - 1))
        if retry_after is not None:
            delay = max(delay, min(retry_after, policy.backoff_cap_s))
        sleep(delay)
    raise WeatherAPIError(f"Open-Meteo request failed after {policy.max_attempts} attempts; last error: {last_error}")


def raw_chunk_path(raw_dir: Path, model: str, site_id: str, chunk_start: date) -> Path:
    return raw_dir / model / site_id / f"{chunk_start:%Y-%m}.json"


def download_site(
    site: Site,
    chunks: Sequence[tuple[date, date]],
    *,
    get: HttpGet,
    sleep: Callable[[float], None],
    raw_dir: Path = RAW_DIR,
    model: str = MODEL,
    polite_delay_s: float = 1.0,
    policy: RetryPolicy = RetryPolicy(),
    on_chunk: Callable[[], None] | None = None,
) -> list[Path]:
    """Download each chunk unless an identical request is already cached; returns the cache files."""
    paths = []
    for start, end in chunks:
        params = request_params(site, start, end, model)
        path = raw_chunk_path(raw_dir, model, site.site_id, start)
        if _read_cached(path, params) is None:
            payload = fetch_json(get, API_URL, params, sleep=sleep, policy=policy)
            parse_response(payload, model)  # validate before caching
            _write_json_atomic(path, {"request": params, "response": payload})
            sleep(polite_delay_s)
        paths.append(path)
        if on_chunk is not None:
            on_chunk()
    return paths


# --- parsing ---------------------------------------------------------------------------


def parse_response(payload: Mapping[str, Any], model: str = MODEL) -> tuple[pd.DataFrame, GridPoint]:
    """Open-Meteo JSON (UTC) → frame indexed by UTC hour with our column names, plus the grid point."""
    if payload.get("error"):
        raise WeatherAPIError(f"Open-Meteo error: {payload.get('reason', 'unknown')}")
    if payload.get("utc_offset_seconds", 0) != 0:
        raise WeatherError("expected a UTC response (timezone=GMT)")
    try:
        hourly = payload["hourly"]
        units = payload.get("hourly_units", {})
        times = pd.DatetimeIndex(pd.to_datetime(hourly["time"])).tz_localize("UTC")
        grid = GridPoint(
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            elevation_m=float(payload.get("elevation", float("nan"))),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherError(f"malformed Open-Meteo response: {exc}") from exc
    data = {}
    for api_name, column in COLUMNS.items():
        key = _hourly_key(hourly, api_name, model)
        expected = EXPECTED_UNITS[api_name.removesuffix(FORECAST_SUFFIX)]
        if units.get(key) != expected:
            raise WeatherError(f"{key}: unit {units.get(key)!r}, expected {expected!r}")
        values = np.array(hourly[key], dtype=float)  # None → NaN
        if values.shape != (len(times),):
            raise WeatherError(f"{key}: {values.shape[0]} values for {len(times)} timestamps")
        data[column] = values
    frame = pd.DataFrame(data, index=times)
    frame.index.name = "time_utc"
    return frame, grid


def load_raw_frame(paths: Sequence[Path], model: str = MODEL) -> tuple[pd.DataFrame, GridPoint]:
    """Concatenate cached chunks into one regular hourly UTC frame (gaps become NaN rows)."""
    frames, grids = [], []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))["response"]
        frame, grid = parse_response(payload, model)
        frames.append(frame)
        grids.append(grid)
    if not frames:
        raise WeatherError("no weather chunks to load")
    if len({(g.latitude, g.longitude) for g in grids}) > 1:
        raise WeatherError(f"chunks come from different grid points: {sorted(set(grids), key=str)}")
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    full_index = pd.date_range(combined.index[0], combined.index[-1], freq="1h", name="time_utc")
    return combined.reindex(full_index), grids[0]


# --- UTC → local alignment -------------------------------------------------------------


def align_to_local(utc_frame: pd.DataFrame, site: Site, start_local: date, end_local: date) -> pd.DataFrame:
    """Hourly rows in the site's local time for local dates [start_local, end_local].

    Each local hour [t:00, t+1:00) is split into 15-minute slots. Radiation (a mean over the
    preceding UTC hour) is constant within its hour, so the local-hour mean is exact for any
    offset. Instantaneous variables are interpolated linearly to slot midpoints. A local hour
    with any missing input is NaN. DST dates yield 23 or 25 rows.
    """
    tz = ZoneInfo(site.timezone)
    period_start = pd.Timestamp(datetime.combine(start_local, time()), tz=tz).tz_convert("UTC")
    period_end = pd.Timestamp(datetime.combine(end_local + timedelta(days=1), time()), tz=tz).tz_convert("UTC")
    slots = pd.date_range(period_start, period_end, freq=f"{SLOT_MINUTES}min", inclusive="left")

    hourly = utc_frame.sort_index()
    t0 = hourly.index[0]
    if (hourly.index != pd.date_range(t0, periods=len(hourly), freq="1h")).any():
        raise WeatherError("UTC frame must be a regular hourly series")
    position_h = np.asarray((slots - t0) / pd.Timedelta(hours=1), dtype=float)
    mean_index = np.floor(position_h).astype(int) + 1  # hour mean labelled at the end of its hour
    mid = position_h + SLOT_MINUTES / 120.0
    left = np.floor(mid).astype(int)
    weight = mid - left
    if mean_index.min() < 0 or left.min() < 0 or max(mean_index.max(), left.max() + 1) >= len(hourly):
        raise WeatherError(
            f"UTC data {hourly.index[0]} … {hourly.index[-1]} does not cover local dates "
            f"{start_local} … {end_local} in {site.timezone}"
        )

    slot_values = {}
    for column in HOUR_MEAN_COLUMNS:
        slot_values[column] = hourly[column].to_numpy(dtype=float)[mean_index]
    for column in INSTANT_COLUMNS:
        values = hourly[column].to_numpy(dtype=float)
        slot_values[column] = (1.0 - weight) * values[left] + weight * values[left + 1]
    midpoints = slots + pd.Timedelta(minutes=SLOT_MINUTES / 2)
    slot_values[CLEAR_SKY_COLUMN] = clear_sky_ghi_w_per_m2(midpoints, site.lat, site.lon)

    local_minutes = np.asarray(slots.tz_convert(tz).minute)
    slot_frame = pd.DataFrame(slot_values)
    slot_frame["hour_start_utc"] = slots - pd.to_timedelta(local_minutes, unit="min")
    grouped = slot_frame.groupby("hour_start_utc", sort=True)
    means = grouped[list(VALUE_COLUMNS)].mean()
    complete = grouped[list(VALUE_COLUMNS)].count() == SLOTS_PER_HOUR
    means = means.where(complete)

    time_utc = pd.DatetimeIndex(means.index).tz_convert("UTC")
    time_local = time_utc.tz_convert(tz)
    out = pd.DataFrame(
        {
            "time_utc": time_utc,
            "time_local": time_local,
            "date_local": time_local.date,
            "hour_local": time_local.hour,
        }
    )
    for column in VALUE_COLUMNS:
        out[column] = means[column].to_numpy()
    return out.reset_index(drop=True)


# --- quality and sanity ----------------------------------------------------------------


def daily_quality(local_frame: pd.DataFrame) -> pd.DataFrame:
    """Per local date: hour count, completeness, eligibility for DayData and the reason if not."""
    rows = []
    for day, group in local_frame.groupby("date_local", sort=True):
        hours = group["hour_local"].tolist()
        missing = int(group[list(VALUE_COLUMNS)].isna().any(axis=1).sum())
        if len(hours) != HOURS:
            reason = f"DST day: {len(hours)} local hours"
        elif sorted(hours) != list(range(HOURS)):
            reason = "irregular local hours"
        elif missing:
            reason = f"{missing} hour(s) with missing values"
        else:
            reason = ""
        rows.append({"date_local": day, "n_hours": len(hours), "missing_hours": missing, "eligible": not reason, "reason": reason})
    return pd.DataFrame(rows).set_index("date_local")


def expected_dst_dates(timezone: str, start_local: date, end_local: date) -> list[date]:
    """Local dates whose length is not 24 h (UTC offset differs at the start and end of the day)."""
    tz = ZoneInfo(timezone)
    dates = []
    day = start_local
    while day <= end_local:
        start = datetime.combine(day, time(), tzinfo=tz)
        end = datetime.combine(day + timedelta(days=1), time(), tzinfo=tz)
        if start.utcoffset() != end.utcoffset():
            dates.append(day)
        day += timedelta(days=1)
    return dates


@dataclass(frozen=True)
class SanityReport:
    site_id: str
    grid: GridPoint
    grid_distance_deg: float
    n_local_hours: int
    missing_hours: dict[str, int]
    n_days: int
    n_eligible_days: int
    ineligible: dict[str, list[str]]
    dst_dates: list[str]
    night_ghi_max_w_per_m2: float
    noon_offset_actual_min: float
    noon_offset_forecast_min: float
    daily_ghi_correlation: dict[int, float]
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def sanity_report(local_frame: pd.DataFrame, site: Site, grid: GridPoint) -> SanityReport:
    problems: list[str] = []
    quality = daily_quality(local_frame)

    distance = max(abs(grid.latitude - site.lat), abs(grid.longitude - site.lon))
    if distance > MAX_GRID_DISTANCE_DEG:
        problems.append(f"grid point {grid.latitude},{grid.longitude} is {distance:.2f} deg from the site")

    missing = {column: int(local_frame[column].isna().sum()) for column in WEATHER_COLUMNS}
    for column, (low, high) in RANGES.items():
        values = local_frame[column].dropna()
        if len(values) and (values.min() < low or values.max() > high):
            problems.append(f"{column} outside [{low}, {high}]: min {values.min():.1f}, max {values.max():.1f}")

    # Deep night: the sun is down for this local hour and both neighbours. Hours next to sunrise or
    # sunset are skipped because an hourly mean spread over a :30 offset legitimately leaks into them.
    by_time = local_frame.sort_values("time_utc")
    dark = by_time[CLEAR_SKY_COLUMN] == 0
    night = (dark & dark.shift(1, fill_value=False) & dark.shift(-1, fill_value=False)).reindex(local_frame.index)
    night_max = float(local_frame.loc[night, list(HOUR_MEAN_COLUMNS)].max().max()) if night.any() else 0.0
    if night_max > NIGHT_GHI_LIMIT_W_PER_M2:
        problems.append(f"GHI up to {night_max:.1f} W/m2 while the sun is down: time alignment is off")

    eligible = quality.index[quality["eligible"]]
    days = local_frame[local_frame["date_local"].isin(eligible)]
    noon_actual = _noon_offset_min(days, "ghi_actual_w_per_m2")
    noon_forecast = _noon_offset_min(days, "ghi_forecast_w_per_m2")
    for name, offset in (("actual", noon_actual), ("forecast", noon_forecast)):
        if np.isfinite(offset) and abs(offset) > NOON_OFFSET_LIMIT_MIN:
            problems.append(f"{name} GHI peaks {offset:+.1f} min from clear-sky noon: time alignment is off")

    correlation = _lagged_daily_correlation(days)
    if correlation and np.isfinite(correlation.get(0, np.nan)):
        if correlation[0] < MIN_DAILY_CORRELATION:
            problems.append(f"daily GHI forecast/actual correlation {correlation[0]:.2f} < {MIN_DAILY_CORRELATION}")
        for lag in (-1, 1):
            if correlation.get(lag, -1.0) >= correlation[0]:
                problems.append(f"daily GHI correlation at lag {lag:+d} day >= lag 0: forecast may be shifted")

    first, last = min(quality.index), max(quality.index)
    expected_dst = expected_dst_dates(site.timezone, first, last)
    odd_length = sorted(quality.index[quality["n_hours"] != HOURS])
    if odd_length != expected_dst:
        problems.append(f"days with != 24 local hours {odd_length} differ from DST dates {expected_dst}")

    ineligible: dict[str, list[str]] = {}
    for day, row in quality[~quality["eligible"]].iterrows():
        ineligible.setdefault(row["reason"], []).append(day.isoformat())
    if not len(eligible):
        problems.append("no eligible days")

    return SanityReport(
        site_id=site.site_id,
        grid=grid,
        grid_distance_deg=distance,
        n_local_hours=len(local_frame),
        missing_hours=missing,
        n_days=len(quality),
        n_eligible_days=len(eligible),
        ineligible=ineligible,
        dst_dates=[day.isoformat() for day in expected_dst],
        night_ghi_max_w_per_m2=night_max,
        noon_offset_actual_min=noon_actual,
        noon_offset_forecast_min=noon_forecast,
        daily_ghi_correlation=correlation,
        problems=problems,
    )


def _noon_offset_min(days: pd.DataFrame, column: str, min_clear_sky_index: float = 0.5) -> float:
    """Mean minutes between a day's GHI centroid and its clear-sky centroid, over clear-ish days."""
    offsets = []
    for _, day in days.groupby("date_local"):
        ghi, clear = day[column].to_numpy(), day[CLEAR_SKY_COLUMN].to_numpy()
        if clear.sum() <= 0 or ghi.sum() <= 0 or ghi.sum() / clear.sum() < min_clear_sky_index:
            continue
        hours = day["hour_local"].to_numpy() + 0.5
        offsets.append((np.dot(hours, ghi) / ghi.sum() - np.dot(hours, clear) / clear.sum()) * 60.0)
    return float(np.mean(offsets)) if offsets else float("nan")


def _lagged_daily_correlation(days: pd.DataFrame) -> dict[int, float]:
    daily = days.groupby("date_local")[list(HOUR_MEAN_COLUMNS)].sum()
    if len(daily) < 5:
        return {}
    daily.index = pd.DatetimeIndex(daily.index)
    daily = daily.asfreq("D")
    actual, forecast = daily["ghi_actual_w_per_m2"], daily["ghi_forecast_w_per_m2"]
    return {lag: float(actual.corr(forecast.shift(lag))) for lag in (-1, 0, 1)}


# --- processed CSV and attribution -----------------------------------------------------


def processed_csv_path(site_id: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    return processed_dir / f"weather_{site_id}.csv"


def write_processed_csv(local_frame: pd.DataFrame, path: Path) -> None:
    out = local_frame.copy()
    out["time_utc"] = pd.DatetimeIndex(out["time_utc"]).strftime("%Y-%m-%dT%H:%MZ")
    out["time_local"] = [ts.isoformat(timespec="minutes") for ts in out["time_local"]]
    out["date_local"] = [day.isoformat() for day in out["date_local"]]
    out = out.round(CSV_DECIMALS)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n")


def read_processed_csv(path: Path, timezone: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_utc = pd.to_datetime(frame["time_utc"], utc=True)
    frame["time_utc"] = time_utc
    frame["time_local"] = time_utc.dt.tz_convert(ZoneInfo(timezone))
    frame["date_local"] = pd.to_datetime(frame["date_local"]).dt.date
    frame["hour_local"] = frame["hour_local"].astype(int)
    return frame


def write_attribution(path: Path, sites: Sequence[Site], model_description: str = MODEL_DESCRIPTION) -> None:
    site_lines = "\n".join(
        f"- `weather_{site.site_id}.csv`: {site.name}, {site.country} "
        f"({site.lat}, {site.lon}), local time {site.timezone}"
        for site in sites
    )
    text = f"""# Data attribution

## Weather

[Weather data by Open-Meteo.com](https://open-meteo.com/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- Source: Open-Meteo Previous Runs API (`{API_URL}`), model `{MODEL}`
  ({model_description}, provided by Deutscher Wetterdienst, via Open-Meteo).
- Variables: `shortwave_radiation` (GHI, W/m²), `temperature_2m` (°C), `cloud_cover` (%).
- Period: local dates {PERIOD_START} to {PERIOD_END}.

**What "actual" and "forecast" mean here.** "Actual" is the model's same-day analysis
(the shortest-lead-time value), **not a measurement**. "Forecast" is the same variable as
predicted one day earlier (`_previous_day1`). The forecast error in these files is therefore
day-1 vs day-0 model error, which is smaller than the error against real meters.

**Changes made by VoltProof** (as CC BY 4.0 requires us to state): values were requested in
UTC and re-aggregated to each site's local hours (radiation as exact hourly means,
temperature and cloud cover interpolated linearly), rounded, and a clear-sky GHI column
(Haurwitz model, computed by our code) was added. Dates with missing values or a daylight-
saving change are kept but not used as benchmark days.

Files:
{site_lines}

## Sites, loads and tariffs

Site parameters, load profiles and electricity prices are illustrative and generated by
VoltProof's own code with fixed seeds. They are not real measurements or any utility's tariff.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# --- helpers ---------------------------------------------------------------------------


def _hourly_key(hourly: Mapping[str, Any], api_name: str, model: str) -> str:
    for key in (api_name, f"{api_name}_{model}"):
        if key in hourly:
            return key
    raise WeatherError(f"response has no {api_name!r} values")


def _error_reason(response: HttpResponse) -> str:
    try:
        body = response.json()
        if isinstance(body, Mapping) and body.get("reason"):
            return str(body["reason"])[:300]
    except ValueError:
        pass
    return (response.text or "")[:300]


def _retry_after_s(response: HttpResponse) -> float | None:
    try:
        value = float(response.headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _read_cached(path: Path, params: Mapping[str, str]) -> dict[str, Any] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return None
    if not isinstance(cached, dict) or cached.get("request") != dict(params):
        return None
    return cached.get("response")


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            json.dump(obj, tmp, ensure_ascii=False)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
