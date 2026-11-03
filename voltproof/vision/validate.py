"""Check a tariff read from a bill before anything is planned with it (SPEC §6.12). Pure functions.

The same checks run on the model's reading, on every edit the user makes in the confirm table, and
again on the server when a run is started with the confirmed tariff (the client is never trusted):
- currency must be the site's currency (blocks)
- only time-of-use and flat tariffs (blocks otherwise)
- whole hours only (a half-hour slot blocks: "half-hour slots are not supported yet"); the periods cover
  hours 0-24 exactly once, a period may wrap past midnight (22 -> 6); gaps and overlaps block
- rates above zero (blocks); a rate below 0.1x or above 10x the site's own average import price is
  flagged "check this rate" but not blocked
- export rate: null (none) or >= 0; the fixed charge is shown read-only and never used

`apply_bill` turns a valid check into the day's prices: a copy of the DayData with its two price arrays
replaced, and a matching Tariff. The frozen agent, optimizer, executor and rule-based battery read prices
only from these two objects, so they all use the bill's tariff, unchanged (no frozen code is touched).
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from voltproof.models import DayData, Tariff

KINDS = {"time_of_use": "time_of_use", "tou": "time_of_use", "flat": "flat"}
PLAUSIBLE_LOW, PLAUSIBLE_HIGH = 0.1, 10.0
HALF_HOUR = "half-hour slots are not supported yet"


def _hour(value: Any) -> tuple[int | None, str | None]:
    """A whole hour 0-24 from a number or "HH" / "HH:MM"; (None, why) otherwise."""
    if isinstance(value, bool) or value is None:
        return None, "missing"
    if isinstance(value, (int, float)):
        if value != int(value):
            return None, HALF_HOUR
        hour = int(value)
    else:
        match = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*", str(value))
        if not match:
            return None, "missing"
        if match.group(2) and match.group(2) != "00":
            return None, HALF_HOUR
        hour = int(match.group(1))
    return (hour, None) if 0 <= hour <= 24 else (None, "missing")


def _rate(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _span_text(hours: list[int]) -> str:
    runs: list[list[int]] = []
    for hour in sorted(hours):
        if runs and runs[-1][1] == hour:
            runs[-1][1] = hour + 1
        else:
            runs.append([hour, hour + 1])
    return ", ".join(f"{a}:00-{b}:00" for a, b in runs)


def check_bill(answer: Mapping[str, Any] | None, site_currency: str, average_import_price: float) -> dict[str, Any]:
    """Validate one reading (or the user's edit of it). Returns ok, errors (block), warnings (flag),
    the normalized tariff (kind, currency, periods, export_rate, fixed_charge) and the 24 hourly rates."""
    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {"ok": False, "unreadable": False, "errors": errors, "warnings": warnings, "kind": None, "currency": site_currency, "periods": [], "hourly": [None] * 24, "export_rate": None, "fixed_charge": None}
    if not isinstance(answer, Mapping):
        errors.append("The reply could not be understood. Try again or use a sample bill.")
        return result
    if answer.get("unreadable") is True:
        result["unreadable"] = True
        errors.append("The bill could not be read clearly. Try a sharper, straighter photo or a sample bill.")
        return result

    currency = str(answer.get("currency") or "").strip().upper()
    if not currency:
        errors.append("The bill's currency could not be read.")
    elif currency != site_currency:
        errors.append(f"This bill is in {currency}, but the site uses {site_currency}. Pick the {site_currency} site or another bill.")

    kind = KINDS.get(str(answer.get("tariff_kind") or answer.get("kind") or "").strip().lower())
    if kind is None:
        errors.append("Only time-of-use and flat tariffs are supported.")
    result["kind"] = kind

    periods = answer.get("periods")
    counts = [0] * 24
    normalized: list[dict[str, Any]] = []
    if not isinstance(periods, list) or not periods:
        errors.append("No import rates were found.")
    else:
        half_hour = False
        for index, period in enumerate(periods, start=1):
            if not isinstance(period, Mapping):
                errors.append(f"Row {index} is not a period.")
                continue
            start, why_start = _hour(period.get("start_hour"))
            end, why_end = _hour(period.get("end_hour"))
            if HALF_HOUR in (why_start, why_end):
                half_hour = True
                continue
            if start is None or end is None or start == 24:
                errors.append(f"Row {index}: the hours must be whole hours from 0 to 24.")
                continue
            if start == end:
                errors.append(f"Row {index}: {start}:00 to {end}:00 covers no hours.")
                continue
            rate = _rate(period.get("import_rate"))
            if rate is None or rate <= 0:
                errors.append(f"Row {index}: the rate must be a number above zero.")
                continue
            hours = list(range(start, end)) if start < end else [*range(start, 24), *range(0, end)]  # wraps past midnight
            for hour in hours:
                counts[hour] += 1
                result["hourly"][hour] = rate
            normalized.append({"start_hour": start, "end_hour": end, "import_rate": rate})
            if average_import_price > 0 and not (PLAUSIBLE_LOW * average_import_price <= rate <= PLAUSIBLE_HIGH * average_import_price):
                warnings.append(f"Row {index}: check this rate. {rate:g} {site_currency}/kWh is far from this site's usual price (about {average_import_price:.2f} {site_currency}/kWh).")
        if half_hour:
            errors.append(f"A period starts or ends at a half hour: {HALF_HOUR}.")
        gaps = [h for h, c in enumerate(counts) if c == 0]
        overlaps = [h for h, c in enumerate(counts) if c > 1]
        if normalized and gaps and not half_hour:
            errors.append(f"No rate for {_span_text(gaps)}. The periods must cover the whole day.")
        if overlaps:
            errors.append(f"Two rates for {_span_text(overlaps)}. Each hour needs exactly one rate.")
        if kind == "flat" and len({p["import_rate"] for p in normalized}) > 1:
            errors.append("A flat tariff has one rate, but this one has several. Choose time-of-use or fix the rates.")
    result["periods"] = normalized

    export = answer.get("export_rate")
    if export is not None:
        value = _rate(export)
        if value is None or value < 0:
            errors.append("The export rate must be empty (none) or a number of zero or more.")
        else:
            result["export_rate"] = value
    fixed = _rate(answer.get("fixed_charge"))
    result["fixed_charge"] = fixed  # shown read-only; never used in planning

    result["ok"] = not errors
    return result


def period_names(hourly: list[float]) -> dict[str, list[int]]:
    """Named periods for the agent's tariff view: the cheapest rate's hours "cheap", the dearest "peak",
    anything between "normal" (no names for a flat tariff)."""
    rates = sorted(set(hourly))
    if len(rates) < 2:
        return {}
    names: dict[str, list[int]] = {"cheap": [], "peak": []}
    for hour, rate in enumerate(hourly):
        key = "cheap" if rate == rates[0] else "peak" if rate == rates[-1] else "normal"
        names.setdefault(key, []).append(hour)
    return names


def apply_bill(day: DayData, tariff: Tariff, check: Mapping[str, Any]) -> tuple[DayData, Tariff]:
    """The day and tariff with the bill's prices in place of the site's (in memory only)."""
    if not check.get("ok"):
        raise ValueError("the bill tariff is not valid")
    imports = [float(r) for r in check["hourly"]]
    exports = [float(check["export_rate"] or 0.0)] * 24  # no export rate on the bill: export earns nothing
    bill_tariff = Tariff(
        name="Tariff from your bill",
        kind=check["kind"],
        currency=tariff.currency,
        import_price_per_kwh=imports,
        export_price_per_kwh=exports,
        periods=period_names(imports),
        notes="Read from a bill image and confirmed by the user.",
    )
    return day.model_copy(update={"import_price_per_kwh": imports, "export_price_per_kwh": exports}), bill_tariff
