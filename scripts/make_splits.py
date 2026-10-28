"""Select dev/test days (seed 2026) from the processed weather and write data/splits.json (SPEC §6.2).

PowerShell, from the repo root (after scripts\\fetch_weather.py):
    python scripts\\make_splits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltproof.data import GLOBAL_SEED  # noqa: E402
from voltproof.data.days import SPLITS_PATH, make_splits, site_stats, write_splits  # noqa: E402
from voltproof.data.sites import SITES  # noqa: E402
from voltproof.data.weather import processed_csv_path, read_processed_csv  # noqa: E402


def main() -> int:
    frames = {
        site.site_id: read_processed_csv(processed_csv_path(site.site_id), site.timezone) for site in SITES.values()
    }
    splits = make_splits(list(SITES.values()), frames, seed=GLOBAL_SEED)
    write_splits(splits)
    print(f"wrote {SPLITS_PATH}  (seed {splits['seed']}, dev {len(splits['dev'])} days, test {len(splits['test'])} days)")

    for site in SITES.values():
        stats = site_stats(site, frames[site.site_id])
        print(f"\n{site.site_id}{'  [held out: test only]' if site.held_out else ''}")
        print(
            f"  {int(stats['n_days'])} eligible days; daily PV forecast/actual "
            f"{stats['pv_daily_forecast_kwh_mean']:.1f}/{stats['pv_daily_actual_kwh_mean']:.1f} kWh; "
            f"daily MAE {stats['pv_daily_mae_kwh']:.2f} kWh, bias {stats['pv_daily_bias_kwh']:+.2f} kWh; "
            f"hourly MAE {stats['pv_hourly_mae_kw']:.3f} kW, bias {stats['pv_hourly_bias_kw']:+.3f} kW"
        )
        for split in ("dev", "test"):
            entries = [e for e in splits[split] if e["site_id"] == site.site_id]
            if entries:
                days = ", ".join(f"{e['date']} ({e['category']})" for e in entries)
                print(f"  {split}: {days}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
