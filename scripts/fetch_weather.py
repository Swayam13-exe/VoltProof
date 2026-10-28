"""Download Open-Meteo weather for the reference sites and write processed CSVs (SPEC §6.2).

PowerShell, from the repo root:
    python scripts\\fetch_weather.py --month 2024-03    # one-month test: download, check, report; writes no CSV
    python scripts\\fetch_weather.py                    # full period: download (resumable), CSVs, ATTRIBUTION.md

Raw responses are cached in data/raw/ (gitignored); re-running only downloads missing chunks.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

from voltproof.data.days import site_stats  # noqa: E402
from voltproof.data.sites import SITES, get_site  # noqa: E402
from voltproof.data.weather import (  # noqa: E402
    MODEL,
    PERIOD_END,
    PERIOD_START,
    PROCESSED_DIR,
    SanityReport,
    align_to_local,
    covered_local_range,
    download_site,
    load_raw_frame,
    month_chunks,
    processed_csv_path,
    sanity_report,
    utc_fetch_range,
    write_attribution,
    write_processed_csv,
)
from voltproof.models import Site  # noqa: E402

USER_AGENT = "VoltProof/0.1 (hackathon research; non-commercial weather download)"


def _month_range(text: str) -> tuple[date, date]:
    first = date.fromisoformat(f"{text}-01")
    next_month = date(first.year + first.month // 12, first.month % 12 + 1, 1)
    return first, next_month - timedelta(days=1)


def _print_report(site: Site, local: pd.DataFrame, report: SanityReport, stats: dict[str, float]) -> None:
    def line(label: str, value: str) -> None:
        print(f"  {label:<34} {value}")

    print(f"\n=== {site.site_id}  ({site.name}, {site.country}; {site.timezone}) ===")
    first, last = local["date_local"].min(), local["date_local"].max()
    line("local dates", f"{first} .. {last}  ({report.n_days} days, {report.n_local_hours} local hours)")
    line(
        "grid point / elevation",
        f"{report.grid.latitude}, {report.grid.longitude}  (site {site.lat}, {site.lon}; "
        f"{report.grid_distance_deg:.2f} deg away), {report.grid.elevation_m:.0f} m",
    )
    line("missing hours per variable", ", ".join(f"{k}={v}" for k, v in report.missing_hours.items()))
    line("eligible days", f"{report.n_eligible_days} of {report.n_days}")
    for reason, dates in report.ineligible.items():
        line("  not eligible", f"{reason}: {', '.join(dates)}")
    line("DST dates expected in range", ", ".join(report.dst_dates) or "none")
    line("max GHI in deep night", f"{report.night_ghi_max_w_per_m2:.1f} W/m2 (limit 10)")
    line(
        "GHI centroid vs clear-sky noon",
        f"actual {report.noon_offset_actual_min:+.1f} min, forecast {report.noon_offset_forecast_min:+.1f} min "
        "(limit +-20)",
    )
    corr = report.daily_ghi_correlation
    if corr:
        line(
            "daily GHI corr (fcst lag -1/0/+1 d)",
            f"{corr.get(-1, float('nan')):.3f} / {corr.get(0, float('nan')):.3f} / {corr.get(1, float('nan')):.3f}",
        )
    print("  -- PV and load over eligible days --")
    line(
        "daily PV energy forecast / actual",
        f"{stats['pv_daily_forecast_kwh_mean']:.1f} / {stats['pv_daily_actual_kwh_mean']:.1f} kWh (mean)",
    )
    line("daily PV energy MAE / bias", f"{stats['pv_daily_mae_kwh']:.2f} / {stats['pv_daily_bias_kwh']:+.2f} kWh")
    line("hourly PV MAE / bias", f"{stats['pv_hourly_mae_kw']:.3f} / {stats['pv_hourly_bias_kw']:+.3f} kW")
    line(
        "max PV forecast / actual vs capacity",
        f"{stats['pv_forecast_max_kw']:.2f} / {stats['pv_actual_max_kw']:.2f} kW vs {stats['pv_capacity_kwp']:g} kWp",
    )
    line(
        "max load forecast / actual vs import",
        f"{stats['load_forecast_max_kw']:.2f} / {stats['load_actual_max_kw']:.2f} kW "
        f"vs import limit {stats['grid_import_max_kw']:g} kW",
    )
    line("daily load (forecast, mean)", f"{stats['load_daily_forecast_kwh_mean']:.1f} kWh")
    line("hourly load MAE / bias", f"{stats['load_hourly_mae_kw']:.3f} / {stats['load_hourly_bias_kw']:+.3f} kW")
    print("  checks:", "OK" if report.ok else "PROBLEMS")
    for problem in report.problems:
        print(f"    - {problem}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month", help="one-month test, e.g. 2024-03 (UTC month); writes no CSV")
    parser.add_argument("--sites", default=",".join(SITES), help="comma-separated site ids (default: all)")
    parser.add_argument("--polite-delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]

    sites = [get_site(site_id.strip()) for site_id in args.sites.split(",") if site_id.strip()]
    utc_start, utc_end = _month_range(args.month) if args.month else utc_fetch_range(PERIOD_START, PERIOD_END)
    chunks = month_chunks(utc_start, utc_end)
    print(f"Model {MODEL}; UTC {utc_start} .. {utc_end}; {len(chunks)} chunk(s) x {len(sites)} site(s)")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    paths = {}
    with tqdm(total=len(chunks) * len(sites), desc="Open-Meteo chunks", unit="chunk") as bar:
        for site in sites:
            paths[site.site_id] = download_site(
                site,
                chunks,
                get=session.get,
                sleep=time.sleep,
                polite_delay_s=args.polite_delay,
                on_chunk=lambda: bar.update(1),
            )

    exit_code = 0
    for site in sites:
        utc_frame, grid = load_raw_frame(paths[site.site_id])
        local_start, local_end = (
            covered_local_range(utc_frame.index, site.timezone) if args.month else (PERIOD_START, PERIOD_END)
        )
        local = align_to_local(utc_frame, site, local_start, local_end)
        report = sanity_report(local, site, grid)
        _print_report(site, local, report, site_stats(site, local))
        if not report.ok:
            exit_code = 1
        if not args.month:
            path = processed_csv_path(site.site_id)
            write_processed_csv(local, path)
            print(f"  wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")

    if not args.month:
        write_attribution(PROCESSED_DIR / "ATTRIBUTION.md", list(SITES.values()))
        print(f"\nwrote {PROCESSED_DIR / 'ATTRIBUTION.md'}")
    else:
        print("\nOne-month test only: no processed CSV written.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
