from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from tests.fakes_data import FakeGet, FakeResponse, make_site, open_meteo_payload, synthetic_utc_frame
from voltproof.data.weather import (
    API_URL,
    CLEAR_SKY_COLUMN,
    GridPoint,
    RetryPolicy,
    WeatherAPIError,
    WeatherError,
    align_to_local,
    covered_local_range,
    daily_quality,
    download_site,
    expected_dst_dates,
    fetch_json,
    load_raw_frame,
    month_chunks,
    parse_response,
    raw_chunk_path,
    read_processed_csv,
    request_params,
    sanity_report,
    utc_fetch_range,
    write_attribution,
    write_processed_csv,
)

VARIABLES = [
    "shortwave_radiation",
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation_previous_day1",
    "temperature_2m_previous_day1",
    "cloud_cover_previous_day1",
]


def step_frame(start: str, hours: int) -> pd.DataFrame:
    """UTC frame whose every column at label k equals k (handy for checking alignment arithmetic)."""
    index = pd.date_range(start, periods=hours, freq="1h", tz="UTC", name="time_utc")
    values = np.arange(hours, dtype=float)
    columns = [
        "ghi_actual_w_per_m2",
        "ghi_forecast_w_per_m2",
        "temp_actual_c",
        "temp_forecast_c",
        "cloud_actual_pct",
        "cloud_forecast_pct",
    ]
    return pd.DataFrame({column: values for column in columns}, index=index)


# --- chunking and requests ---------------------------------------------------------------


def test_month_chunks_partial_ends_and_year_rollover() -> None:
    assert month_chunks(date(2024, 1, 19), date(2024, 3, 2)) == [
        (date(2024, 1, 19), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 2)),
    ]
    chunks = month_chunks(date(2024, 12, 15), date(2026, 1, 1))
    assert chunks[0] == (date(2024, 12, 15), date(2024, 12, 31))
    assert chunks[1] == (date(2025, 1, 1), date(2025, 1, 31))
    assert chunks[-1] == (date(2026, 1, 1), date(2026, 1, 1))
    assert len(chunks) == 14
    with pytest.raises(ValueError):
        month_chunks(date(2024, 2, 1), date(2024, 1, 1))


def test_full_period_is_25_monthly_requests_per_site() -> None:
    from voltproof.data.weather import PERIOD_END, PERIOD_START

    start, end = utc_fetch_range(PERIOD_START, PERIOD_END)
    assert (start, end) == (date(2024, 1, 19), date(2026, 1, 1))
    assert len(month_chunks(start, end)) == 25


def test_request_params_ask_for_utc_and_one_model() -> None:
    params = request_params(make_site(), date(2024, 3, 1), date(2024, 3, 31))
    assert params["timezone"] == "GMT" and params["models"] == "icon_global"
    assert params["hourly"].split(",") == VARIABLES
    assert (params["start_date"], params["end_date"]) == ("2024-03-01", "2024-03-31")


# --- fetch with retries --------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [FakeResponse(429), FakeResponse(500), FakeResponse(503), requests.Timeout(), requests.ConnectionError()],
    ids=["429", "500", "503", "timeout", "connection"],
)
def test_fetch_retries_transient_errors(failure: object) -> None:
    sleeps: list[float] = []
    get = FakeGet(failure, FakeResponse(200, {"ok": True}))
    assert fetch_json(get, API_URL, {"a": "1"}, sleep=sleeps.append) == {"ok": True}
    assert len(get.calls) == 2 and sleeps == [2.0]
    assert get.calls[0]["timeout"] == RetryPolicy().timeout_s


def test_fetch_gives_up_after_max_attempts_with_backoff() -> None:
    sleeps: list[float] = []
    get = FakeGet(*[FakeResponse(502, {"reason": "bad gateway"}) for _ in range(4)])
    with pytest.raises(WeatherAPIError, match="after 4 attempts; last error: HTTP 502: bad gateway"):
        fetch_json(get, API_URL, {}, sleep=sleeps.append)
    assert sleeps == [2.0, 4.0, 8.0]


def test_fetch_honors_retry_after() -> None:
    sleeps: list[float] = []
    get = FakeGet(FakeResponse(429, headers={"Retry-After": "30"}), FakeResponse(200, {}))
    fetch_json(get, API_URL, {}, sleep=sleeps.append)
    assert sleeps == [30.0]


def test_fetch_does_not_retry_400() -> None:
    get = FakeGet(FakeResponse(400, {"error": True, "reason": "Cannot initialize WeatherVariable"}))
    with pytest.raises(WeatherAPIError, match="HTTP 400\\): Cannot initialize"):
        fetch_json(get, API_URL, {}, sleep=lambda _s: None)
    assert len(get.calls) == 1


def _payload_for(start: date, end: date) -> dict:
    times = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), freq="1h")
    n = len(times)
    values = {name: [1.0] * n for name in VARIABLES}
    return open_meteo_payload([t.strftime("%Y-%m-%dT%H:%M") for t in times], values)


def test_download_caches_and_skips_identical_requests(tmp_path: Path) -> None:
    site = make_site()
    chunks = month_chunks(date(2024, 1, 30), date(2024, 2, 2))
    sleeps: list[float] = []
    get = FakeGet(*[FakeResponse(200, _payload_for(a, b)) for a, b in chunks])
    paths = download_site(site, chunks, get=get, sleep=sleeps.append, raw_dir=tmp_path, polite_delay_s=1.5)
    assert len(get.calls) == 2 and sleeps == [1.5, 1.5]
    assert paths[0] == raw_chunk_path(tmp_path, "icon_global", site.site_id, date(2024, 1, 30))
    assert json.loads(paths[0].read_text(encoding="utf-8"))["request"] == request_params(site, *chunks[0])

    again = FakeGet()  # any HTTP call would fail: the script is empty
    assert download_site(site, chunks, get=again, sleep=sleeps.append, raw_dir=tmp_path) == paths
    assert again.calls == []

    frame, grid = load_raw_frame(paths)
    assert len(frame) == 4 * 24 and grid.elevation_m == 300.0


def test_download_refetches_when_request_differs(tmp_path: Path) -> None:
    site = make_site()
    chunk = [(date(2024, 2, 1), date(2024, 2, 2))]
    download_site(site, chunk, get=FakeGet(FakeResponse(200, _payload_for(*chunk[0]))), sleep=lambda _s: None, raw_dir=tmp_path)
    moved = make_site(lat=21.0)
    get = FakeGet(FakeResponse(200, _payload_for(*chunk[0])))
    download_site(moved, chunk, get=get, sleep=lambda _s: None, raw_dir=tmp_path)
    assert len(get.calls) == 1


# --- parsing -----------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [False, True])
def test_parse_response_with_or_without_model_suffix(suffix: bool) -> None:
    values = {name: [1.0, None, 3.0] for name in VARIABLES}
    payload = open_meteo_payload(["2024-03-01T00:00", "2024-03-01T01:00", "2024-03-01T02:00"], values, suffix=suffix)
    frame, grid = parse_response(payload)
    assert str(frame.index.tz) == "UTC" and len(frame) == 3
    assert np.isnan(frame["ghi_actual_w_per_m2"].iloc[1])
    assert frame["temp_forecast_c"].iloc[2] == 3.0
    assert (grid.latitude, grid.longitude) == (20.0, 75.0)


def test_parse_response_rejects_wrong_units_and_local_time() -> None:
    values = {name: [1.0] for name in VARIABLES}
    payload = open_meteo_payload(["2024-03-01T00:00"], values)
    payload["hourly_units"]["temperature_2m"] = "°F"
    with pytest.raises(WeatherError, match="unit '°F', expected '°C'"):
        parse_response(payload)
    payload = open_meteo_payload(["2024-03-01T00:00"], values)
    payload["utc_offset_seconds"] = 19800
    with pytest.raises(WeatherError, match="UTC response"):
        parse_response(payload)


def test_parse_recorded_fixture() -> None:
    path = Path(__file__).parent / "fixtures" / "open_meteo_icon_global_2days.json"
    frame, grid = parse_response(json.loads(path.read_text(encoding="utf-8")))
    assert len(frame) == 48 and not frame.isna().any().any()
    assert frame["ghi_actual_w_per_m2"].min() >= 0 and frame["ghi_actual_w_per_m2"].max() > 300
    assert abs(grid.latitude - (-1.29)) < 0.3 and abs(grid.longitude - 36.82) < 0.3


# --- UTC → local alignment ---------------------------------------------------------------


def test_half_hour_offset_splits_hour_means_exactly() -> None:
    site = make_site(timezone="Asia/Kolkata")  # UTC+5:30
    frame = step_frame("2024-03-01", 72)  # value at label k is k
    local = align_to_local(frame, site, date(2024, 3, 2), date(2024, 3, 2))
    assert len(local) == 24 and local["hour_local"].tolist() == list(range(24))
    # Local 00:00–01:00 IST = 18:30–19:30 UTC (Mar 1): half of hour [18,19) (label 19) and half of [19,20) (label 20).
    assert local["ghi_actual_w_per_m2"].iloc[0] == pytest.approx((19 + 20) / 2)
    # Instantaneous: mean of the linear interpolation over 18:30–19:30 UTC = value at 19:00 = 19.
    assert local["temp_actual_c"].iloc[0] == pytest.approx(19.0)
    assert str(local["time_local"].iloc[0]) == "2024-03-02 00:00:00+05:30"


def test_quarter_hour_offset() -> None:
    site = make_site(timezone="Asia/Kathmandu")  # UTC+5:45
    local = align_to_local(step_frame("2024-03-01", 72), site, date(2024, 3, 2), date(2024, 3, 2))
    # Local 00:00–01:00 = 18:15–19:15 UTC: 3 slots in hour [18,19) (label 19), 1 slot in [19,20) (label 20).
    assert local["ghi_actual_w_per_m2"].iloc[0] == pytest.approx((3 * 19 + 20) / 4)


def test_dst_days_have_23_and_25_hours_and_are_ineligible() -> None:
    site = make_site(timezone="America/Chicago", lat=41.9, lon=-87.6)
    spring = align_to_local(step_frame("2024-03-08", 5 * 24), site, date(2024, 3, 9), date(2024, 3, 11))
    autumn = align_to_local(step_frame("2024-11-01", 5 * 24), site, date(2024, 11, 2), date(2024, 11, 4))
    assert spring.groupby("date_local").size().tolist() == [24, 23, 24]
    assert autumn.groupby("date_local").size().tolist() == [24, 25, 24]
    quality = daily_quality(autumn)
    assert quality["eligible"].tolist() == [True, False, True]
    assert quality.loc[date(2024, 11, 3), "reason"] == "DST day: 25 local hours"
    assert expected_dst_dates("America/Chicago", date(2024, 1, 1), date(2024, 12, 31)) == [
        date(2024, 3, 10),
        date(2024, 11, 3),
    ]
    # The repeated 01:00 hour appears twice, with different UTC instants.
    repeated = autumn[(autumn["date_local"] == date(2024, 11, 3)) & (autumn["hour_local"] == 1)]
    assert len(repeated) == 2 and repeated["time_utc"].nunique() == 2


def test_missing_input_makes_the_hour_nan_and_the_day_ineligible() -> None:
    site = make_site(timezone="Asia/Kolkata")
    frame = step_frame("2024-03-01", 72)
    frame.iloc[30, frame.columns.get_loc("temp_actual_c")] = np.nan
    local = align_to_local(frame, site, date(2024, 3, 2), date(2024, 3, 2))
    # Label 06:00Z is an interpolation endpoint for slot midpoints in (05:00Z, 07:00Z): local hours 10, 11, 12.
    assert local.loc[local["temp_actual_c"].isna(), "hour_local"].tolist() == [10, 11, 12]
    assert not daily_quality(local)["eligible"].iloc[0]


def test_alignment_requires_coverage() -> None:
    with pytest.raises(WeatherError, match="does not cover"):
        align_to_local(step_frame("2024-03-02", 24), make_site(timezone="Asia/Kolkata"), date(2024, 3, 2), date(2024, 3, 2))


def test_covered_local_range() -> None:
    index = pd.date_range("2024-03-01", "2024-03-31 23:00", freq="1h", tz="UTC")
    assert covered_local_range(index, "Asia/Kolkata") == (date(2024, 3, 2), date(2024, 3, 31))
    assert covered_local_range(index, "America/Chicago") == (date(2024, 3, 1), date(2024, 3, 30))


# --- sanity checks -----------------------------------------------------------------------


def test_sanity_report_passes_on_well_aligned_data() -> None:
    site = make_site(timezone="Asia/Kolkata", lat=23.0, lon=72.6)
    utc = synthetic_utc_frame(site, date(2024, 4, 1), date(2024, 4, 30), seed=3)
    local = align_to_local(utc, site, *covered_local_range(utc.index, site.timezone))
    report = sanity_report(local, site, GridPoint(latitude=23.0, longitude=72.5, elevation_m=50))
    assert report.problems == []
    assert abs(report.noon_offset_actual_min) < 5 and report.night_ghi_max_w_per_m2 < 1
    assert report.daily_ghi_correlation[0] > 0.9
    assert local[CLEAR_SKY_COLUMN].max() > 900


def test_sanity_report_catches_a_one_hour_shift() -> None:
    site = make_site(timezone="Asia/Kolkata", lat=23.0, lon=72.6)
    utc = synthetic_utc_frame(site, date(2024, 4, 1), date(2024, 4, 30), seed=3, shift_hours=1)
    local = align_to_local(utc, site, *covered_local_range(utc.index, site.timezone))
    report = sanity_report(local, site, GridPoint(latitude=23.0, longitude=72.5, elevation_m=50))
    assert any("time alignment is off" in problem for problem in report.problems)


def test_sanity_report_flags_far_grid_point_and_ranges() -> None:
    site = make_site(timezone="Asia/Kolkata", lat=23.0, lon=72.6)
    utc = synthetic_utc_frame(site, date(2024, 4, 1), date(2024, 4, 10), seed=3)
    utc.iloc[100, utc.columns.get_loc("cloud_actual_pct")] = 140.0
    local = align_to_local(utc, site, *covered_local_range(utc.index, site.timezone))
    report = sanity_report(local, site, GridPoint(latitude=24.0, longitude=72.5, elevation_m=50))
    assert any("deg from the site" in p for p in report.problems)
    assert any("cloud_actual_pct outside" in p for p in report.problems)


# --- CSV and attribution ---------------------------------------------------------------


def test_processed_csv_round_trip(tmp_path: Path) -> None:
    site = make_site(timezone="America/Chicago", lat=41.9, lon=-87.6)
    local = align_to_local(step_frame("2024-11-01", 5 * 24), site, date(2024, 11, 2), date(2024, 11, 4))
    path = tmp_path / "weather.csv"
    write_processed_csv(local, path)
    back = read_processed_csv(path, site.timezone)
    assert len(back) == len(local) and back["hour_local"].tolist() == local["hour_local"].tolist()
    assert back["date_local"].tolist() == local["date_local"].tolist()
    assert (back["time_utc"] == local["time_utc"]).all()
    np.testing.assert_allclose(back["ghi_actual_w_per_m2"], local["ghi_actual_w_per_m2"].round(1))
    assert path.read_text(encoding="utf-8").splitlines()[0].startswith("time_utc,time_local,date_local,hour_local")


def test_attribution_credits_open_meteo_and_model(tmp_path: Path) -> None:
    path = tmp_path / "ATTRIBUTION.md"
    write_attribution(path, [make_site()])
    text = path.read_text(encoding="utf-8")
    assert "[Weather data by Open-Meteo.com](https://open-meteo.com/)" in text
    assert "CC BY 4.0" in text and "DWD ICON global" in text and "icon_global" in text
    assert "not a measurement" in text and "day-1 vs day-0" in text
    assert "weather_test_site.csv" in text


@pytest.mark.live
def test_live_one_day_download(tmp_path: Path) -> None:
    from voltproof.data.sites import dev_sites

    site = dev_sites()[0]
    session = requests.Session()
    paths = download_site(site, [(date(2024, 6, 1), date(2024, 6, 2))], get=session.get, sleep=lambda _s: None, raw_dir=tmp_path)
    frame, _ = load_raw_frame(paths)
    assert len(frame) == 48 and not frame.isna().any().any()
