"""Tariff templates → a `Tariff` for one day (SPEC §6.2).

Every number comes from the site's template (`voltproof.data.sites`); this module holds
the rules only:
- time_of_use:    base price · multiplier of the hour's named window; other hours are the default period
- flat:           one import price all day, no named periods
- hourly_dynamic: mean level · typical daily shape · (1 − dip on sunny middays) · seeded day and hour
                  variation, floored; named periods per day: `cheap` = cheapest N hours,
                  `peak` = most expensive M hours (ties go to the earlier hour)

`solar_frac` (0–1, the day's forecast clear-sky index) drives the midday dip: dynamic
prices are set a day ahead, so they follow the forecast, not the actual weather.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from voltproof.data import day_rng
from voltproof.models import (
    HOURS,
    FlatTemplate,
    HourlyDynamicTemplate,
    Tariff,
    TariffTemplate,
    TimeOfUseTemplate,
    window_hours,
)


def build_tariff(
    template: TariffTemplate, day: date, *, seed: int, stream: str, solar_frac: float = 0.0
) -> Tariff:
    """Tariff for `day`. `stream` names the random stream (use the site id) for dynamic prices."""
    if isinstance(template, TimeOfUseTemplate):
        return _time_of_use(template)
    if isinstance(template, FlatTemplate):
        return _flat(template)
    if isinstance(template, HourlyDynamicTemplate):
        return _hourly_dynamic(template, day, seed=seed, stream=stream, solar_frac=solar_frac)
    raise TypeError(f"unsupported tariff template {type(template).__name__}")


def _time_of_use(template: TimeOfUseTemplate) -> Tariff:
    period_of_hour = [template.default_period] * HOURS
    for name, windows in template.windows.items():
        for hour in window_hours(windows):
            period_of_hour[hour] = name
    multipliers = {template.default_period: 1.0, **template.multipliers}
    prices = [template.base_price_per_kwh * multipliers[period] for period in period_of_hour]
    periods = {
        name: [hour for hour, period in enumerate(period_of_hour) if period == name]
        for name in multipliers
        if name in period_of_hour
    }
    return Tariff(
        name=template.name,
        kind="time_of_use",
        currency=template.currency,
        import_price_per_kwh=prices,
        export_price_per_kwh=[template.export_price_per_kwh] * HOURS,
        periods=periods,
        notes=template.notes,
    )


def _flat(template: FlatTemplate) -> Tariff:
    return Tariff(
        name=template.name,
        kind="flat",
        currency=template.currency,
        import_price_per_kwh=[template.price_per_kwh] * HOURS,
        export_price_per_kwh=[template.export_price_per_kwh] * HOURS,
        periods={},
        notes=template.notes,
    )


def _hourly_dynamic(
    template: HourlyDynamicTemplate, day: date, *, seed: int, stream: str, solar_frac: float
) -> Tariff:
    shape = np.asarray(template.shape, dtype=float)
    shape = shape / shape.mean()
    dip = np.ones(HOURS)
    dip[window_hours(template.solar_dip_windows)] -= template.solar_dip_frac * float(np.clip(solar_frac, 0.0, 1.0))
    rng = day_rng(seed, f"tariff:{stream}", day)
    day_factor = 1.0 + rng.normal(0.0, template.day_sigma_frac)
    hour_factor = 1.0 + rng.normal(0.0, template.hour_sigma_frac, HOURS)
    prices = template.mean_price_per_kwh * shape * dip * day_factor * hour_factor
    prices = np.maximum(prices, template.floor_frac * template.mean_price_per_kwh)
    hours = np.arange(HOURS)
    cheap = sorted(np.lexsort((hours, prices))[: template.cheap_hours].tolist())
    by_price_desc = [hour for hour in np.lexsort((hours, -prices)).tolist() if hour not in cheap]
    peak = sorted(by_price_desc[: template.peak_hours])
    return Tariff(
        name=template.name,
        kind="hourly_dynamic",
        currency=template.currency,
        import_price_per_kwh=prices.tolist(),
        export_price_per_kwh=[template.export_price_per_kwh] * HOURS,
        periods={"cheap": cheap, "peak": peak},
        notes=template.notes,
    )
