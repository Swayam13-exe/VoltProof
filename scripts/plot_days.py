"""Plot a clear and a cloudy NON-TEST day per site and print forecast-error statistics (P1 gate).

PowerShell, from the repo root (after fetch_weather.py and make_splits.py):
    python scripts\\plot_days.py

Days are picked from the dev period (2024) only, so no test-year day is plotted for any site.
Figures go to results/figures/.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from voltproof.config import PROJECT_ROOT  # noqa: E402
from voltproof.data.days import (  # noqa: E402
    DEV_PERIOD,
    build_day,
    day_stats,
    day_tariff,
    load_splits,
    site_stats,
    split_dates,
)
from voltproof.data.sites import SITES  # noqa: E402
from voltproof.data.weather import processed_csv_path, read_processed_csv  # noqa: E402
from voltproof.models import HOURS, Site  # noqa: E402

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
CHEAP_COLOR, MIDDLE_COLOR, EXPENSIVE_COLOR = "#2ca02c", "#9e9e9e", "#d62728"
CLEAR_TARGET_QUANTILE = 0.9
CLOUDY_TARGET_QUANTILE = 0.1


def pick_days(stats: pd.DataFrame, excluded: set[date]) -> dict[str, date]:
    """A clear and a cloudy day closest to the 90th / 10th percentile of the clear-sky index,
    taken from the dev period only (the test year is never looked at), never a test day."""
    start, end = DEV_PERIOD
    pool = stats[[start <= day <= end and day not in excluded for day in stats.index]]
    picks = {}
    for kind, quantile in (("clear", CLEAR_TARGET_QUANTILE), ("cloudy", CLOUDY_TARGET_QUANTILE)):
        target = pool["csi_actual"].quantile(quantile)
        picks[kind] = min(pool.index, key=lambda day: (abs(pool.at[day, "csi_actual"] - target), day))
    return picks


def _period_colors(periods: dict[str, list[int]], prices: list[float]) -> list[tuple[str, list[int], str]]:
    """Color named periods by price rank: cheapest green, most expensive red, others grey."""
    ranked = sorted(periods.items(), key=lambda item: float(np.mean([prices[h] for h in item[1]])))
    colors = [CHEAP_COLOR] + [MIDDLE_COLOR] * (len(ranked) - 2) + [EXPENSIVE_COLOR] if len(ranked) > 1 else [MIDDLE_COLOR]
    return [(name, hours, color) for (name, hours), color in zip(ranked, colors[: len(ranked)])]


def _steps(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(HOURS + 1), np.append(values, values[-1])


def plot_day(site: Site, day: date, kind: str, frame: pd.DataFrame, path: Path) -> None:
    data = build_day(site, day, frame)
    tariff = day_tariff(site, day, frame)
    fig, (ax_pv, ax_load) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    handles = []
    for name, hours, color in _period_colors(tariff.periods, data.import_price_per_kwh):
        for ax in (ax_pv, ax_load):
            for hour in hours:
                ax.axvspan(hour, hour + 1, color=color, alpha=0.10, linewidth=0)
        handles.append(Patch(color=color, alpha=0.25, label=f"{name} hours"))

    ax_pv.step(*_steps(data.pv_forecast_kw), where="post", label="PV forecast (day ahead)", color="#1f77b4", linestyle="--")
    ax_pv.step(*_steps(data.pv_actual_kw), where="post", label="PV actual (same-day analysis)", color="#1f77b4")
    ax_pv.set_ylabel("PV power (kW)")
    ax_pv.set_title(
        f"{site.name} ({site.country}) - {kind} day {day}\n"
        f"clear-sky index actual {data.weather.clear_sky_index_actual:.2f}, "
        f"PV {data.weather.pv_forecast_kwh:.1f} kWh forecast vs {data.weather.pv_actual_kwh:.1f} kWh actual"
    )
    ax_pv.legend(handles=ax_pv.get_legend_handles_labels()[0] + handles, loc="upper left", fontsize=8)

    ax_load.step(*_steps(data.load_forecast_kw), where="post", label="load forecast", color="#e377c2", linestyle="--")
    ax_load.step(*_steps(data.load_actual_kw), where="post", label="load actual", color="#e377c2")
    ax_load.step(*_steps(data.critical_load_kw), where="post", label=site.critical_load_name, color="#7f7f7f", linestyle=":")
    ax_load.set_ylabel("Load (kW)")
    ax_load.set_xlabel(f"Local hour ({site.timezone})")
    ax_load.set_xlim(0, HOURS)
    ax_load.set_xticks(range(0, HOURS + 1, 3))
    ax_price = ax_load.twinx()
    ax_price.step(*_steps(data.import_price_per_kwh), where="post", color="black", linewidth=1, label="import price")
    ax_price.set_ylabel(f"Import price ({tariff.currency}/kWh, {tariff.kind})")
    ax_price.set_ylim(0, max(data.import_price_per_kwh) * 1.3)
    lines = ax_load.get_legend_handles_labels()
    price_lines = ax_price.get_legend_handles_labels()
    ax_load.legend(lines[0] + price_lines[0], lines[1] + price_lines[1], loc="upper left", fontsize=8)

    fig.text(
        0.01,
        0.005,
        "Weather data by Open-Meteo.com (CC BY 4.0), DWD ICON global. Site, load and prices are illustrative/synthetic.",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    splits = load_splits()
    for site in SITES.values():
        frame = read_processed_csv(processed_csv_path(site.site_id), site.timezone)
        test_days = set(split_dates(splits, "test", site.site_id))
        stats = day_stats(site, frame)
        print(f"\n{site.site_id}{'  [held out: plotting non-test days only]' if site.held_out else ''}")
        for kind, day in pick_days(stats, test_days).items():
            assert day not in test_days
            path = FIGURES_DIR / f"p1_{site.site_id}_{kind}_{day.isoformat()}.png"
            plot_day(site, day, kind, frame, path)
            print(f"  {kind:<6} {day}  clear-sky index {stats.at[day, 'csi_actual']:.2f}  -> {path.relative_to(PROJECT_ROOT)}")
        s = site_stats(site, frame)
        print(
            f"  forecast error over {int(s['n_days'])} eligible days: daily PV forecast/actual "
            f"{s['pv_daily_forecast_kwh_mean']:.1f}/{s['pv_daily_actual_kwh_mean']:.1f} kWh, "
            f"daily MAE {s['pv_daily_mae_kwh']:.2f} kWh, bias {s['pv_daily_bias_kwh']:+.2f} kWh, "
            f"hourly MAE {s['pv_hourly_mae_kw']:.3f} kW, bias {s['pv_hourly_bias_kw']:+.3f} kW"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
