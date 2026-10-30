"""IntentSpec parsing, validation and helpers (SPEC §6.3).

`soft_rows` is the single definition of what each soft goal constrains, hour by hour. The
optimizer turns each row into an LP constraint with its own slack; `core.rules` evaluates the
same rows on a simulated or executed trajectory. So an LP slack and a measured shortfall always
mean the same thing.

Window semantics (SPEC §3): "SOC ≥ r during [a, b)" constrains soc[a−1] (the initial SOC when
a = 0) and soc[a] … soc[b−1]; "by hour h" constrains soc[h−1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pydantic import ValidationError

from voltproof.models import HOURS, TOL, HourWindow, IntentSpec, RuleCode, Site

INITIAL = -1  # SOC index of the initial state of charge

RowKind = Literal["soc_min", "import_max", "export_max", "discharge_max", "throughput_max"]


@dataclass(frozen=True)
class SpecError:
    path: str
    value: Any
    rule: str
    message: str


@dataclass(frozen=True)
class SoftRow:
    """One soft constraint at one hour: `kind` bounds a trajectory quantity by `bound`.

    soc_min:        soc[hour] ≥ bound (kWh); hour = INITIAL means the initial SOC
    import_max:     grid_import[hour] ≤ bound (kW)
    export_max:     grid_export[hour] ≤ 0
    discharge_max:  discharge[hour] ≤ 0
    throughput_max: Σ (charge + discharge) ≤ bound (kWh); hour is None

    Rows with the same `slack_key` share one slack (SPEC §6.4): an SOC goal has ONE slack for all its
    hours, equal to its worst-hour shortfall (a shortfall in stored energy persists across hours, so
    per-hour slacks would multiply the penalty by the window length); a flow goal has one slack per
    hour, because each hour's excess is separate energy; throughput has one slack.
    """

    rule: RuleCode
    path: str
    kind: RowKind
    hour: int | None
    bound: float

    @property
    def slack_key(self) -> str:
        return self.path if self.kind in ("soc_min", "throughput_max") else f"{self.path}@{self.hour}"


@dataclass(frozen=True)
class SpecItem:
    """A comparable constraint for spec-accuracy scoring (SPEC §6.7)."""

    kind: str
    start: int | None
    end: int | None
    value: float | None


# --- parsing -----------------------------------------------------------------------------


def parse_spec(data: Mapping[str, Any] | IntentSpec, site: Site | None = None) -> tuple[IntentSpec | None, list[SpecError]]:
    """Validate a spec and return it with every error found (all at once), or (None, errors)."""
    if isinstance(data, IntentSpec):
        spec, errors = data, []
    else:
        try:
            spec, errors = IntentSpec.model_validate(data), []
        except ValidationError as exc:
            spec, errors = None, [_to_spec_error(error) for error in exc.errors()]
    if spec is not None and site is not None:
        errors.extend(_site_errors(spec, site))
    return (None, errors) if errors else (spec, [])


def format_errors(errors: list[SpecError]) -> str:
    return "\n".join(f"- {error.message}" for error in errors)


def _to_spec_error(error: Mapping[str, Any]) -> SpecError:
    path = _path(error.get("loc", ()))
    value = error.get("input")
    rule = _rule_text(error)
    if isinstance(value, (dict, list)) or error.get("type") in ("missing", "value_error"):
        message = f"{path}: {rule}"
    else:
        message = f"{path}={value!r} {rule}" if isinstance(value, str) else f"{path}={value} {rule}"
    return SpecError(path=path, value=value, rule=rule, message=message)


def _path(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "spec"


def _rule_text(error: Mapping[str, Any]) -> str:
    kind, ctx = error.get("type", ""), error.get("ctx") or {}
    texts = {
        "less_than_equal": f"must be <= {_num(ctx.get('le'))}",
        "greater_than_equal": f"must be >= {_num(ctx.get('ge'))}",
        "less_than": f"must be < {_num(ctx.get('lt'))}",
        "greater_than": f"must be > {_num(ctx.get('gt'))}",
        "extra_forbidden": "is not a recognised field",
        "missing": "is required",
        "int_type": "must be an integer",
        "int_parsing": "must be an integer",
        "int_from_float": "must be a whole number",
        "float_type": "must be a number",
        "float_parsing": "must be a number",
        "finite_number": "must be a finite number",
        "list_type": "must be a list",
        "string_type": "must be a string",
        "model_type": "must be an object",
        "dict_type": "must be an object",
        "literal_error": f"must be {ctx.get('expected')}",
    }
    if kind == "value_error":
        return str(error.get("msg", "")).removeprefix("Value error, ")
    return texts.get(kind, str(error.get("msg", "is invalid")))


def _num(value: Any) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _site_errors(spec: IntentSpec, site: Site) -> list[SpecError]:
    max_pct = site.battery.soc_max_frac * 100.0
    errors = []
    for path, value in _soc_pct_fields(spec):
        if value > max_pct + TOL:
            rule = f"must be <= {max_pct:g} (the battery's maximum state of charge, %)"
            errors.append(SpecError(path=path, value=value, rule=rule, message=f"{path}={value} {rule}"))
    return errors


def _soc_pct_fields(spec: IntentSpec) -> list[tuple[str, float]]:
    fields = [(f"reserve_windows[{i}].min_soc_pct", w.min_soc_pct) for i, w in enumerate(spec.reserve_windows)]
    fields += [(f"soc_targets[{i}].min_soc_pct", t.min_soc_pct) for i, t in enumerate(spec.soc_targets)]
    if spec.min_soc_pct_all_day is not None:
        fields.append(("min_soc_pct_all_day", spec.min_soc_pct_all_day))
    if spec.end_of_day_min_soc_pct is not None:
        fields.append(("end_of_day_min_soc_pct", spec.end_of_day_min_soc_pct))
    return fields


# --- hour helpers ------------------------------------------------------------------------


def window_soc_indices(start_hour: int, end_hour: int) -> list[int]:
    """SOC indices constrained by a window [start, end) (SPEC §3); INITIAL stands for E0."""
    return [start_hour - 1, *range(start_hour, end_hour)]


def outage_hours(spec: IntentSpec | None) -> set[int]:
    if spec is None:
        return set()
    return {hour for window in spec.grid_outages for hour in window.hours()}


def no_export_hours(spec: IntentSpec | None) -> set[int]:
    if spec is None:
        return set()
    return {hour for window in spec.no_export_windows for hour in window.hours()}


def soft_rows(spec: IntentSpec, site: Site) -> list[SoftRow]:
    """Every soft goal of `spec`, one row per constrained hour (see SoftRow)."""
    cap = site.battery.capacity_kwh
    rows: list[SoftRow] = []
    for i, window in enumerate(spec.reserve_windows):
        bound = window.min_soc_pct / 100.0 * cap
        for index in window_soc_indices(window.start_hour, window.end_hour):
            rows.append(SoftRow("RESERVE", f"reserve_windows[{i}]", "soc_min", index, bound))
    for i, target in enumerate(spec.soc_targets):
        rows.append(SoftRow("SOC_TARGET", f"soc_targets[{i}]", "soc_min", target.by_hour - 1, target.min_soc_pct / 100.0 * cap))
    if spec.min_soc_pct_all_day is not None:
        bound = spec.min_soc_pct_all_day / 100.0 * cap
        rows.extend(SoftRow("ALL_DAY_MIN", "min_soc_pct_all_day", "soc_min", t, bound) for t in range(HOURS))
    if spec.end_of_day_min_soc_pct is not None:
        rows.append(
            SoftRow("END_SOC", "end_of_day_min_soc_pct", "soc_min", HOURS - 1, spec.end_of_day_min_soc_pct / 100.0 * cap)
        )
    for i, cap_window in enumerate(spec.import_caps):
        rows.extend(
            SoftRow("IMPORT_CAP", f"import_caps[{i}]", "import_max", t, cap_window.max_kw) for t in cap_window.hours()
        )
    for i, window in enumerate(spec.no_export_windows):
        rows.extend(SoftRow("NO_EXPORT", f"no_export_windows[{i}]", "export_max", t, 0.0) for t in window.hours())
    if spec.discharge_only_windows is not None:
        allowed = {hour for window in spec.discharge_only_windows for hour in window.hours()}
        rows.extend(
            SoftRow("DISCHARGE_WINDOW", "discharge_only_windows", "discharge_max", t, 0.0)
            for t in range(HOURS)
            if t not in allowed
        )
    if spec.max_throughput_kwh is not None:
        rows.append(SoftRow("THROUGHPUT", "max_throughput_kwh", "throughput_max", None, spec.max_throughput_kwh))
    return rows


# --- spec items ----------------------------------------------------------------------------


def spec_items(spec: IntentSpec) -> list[SpecItem]:
    """Flatten a spec into comparable items; touching or overlapping windows of the same kind and
    value are merged first, so 14–15 + 15–17 equals 14–17."""
    items: list[SpecItem] = []
    items += _merged("grid_outage", [(w.start_hour, w.end_hour, None) for w in spec.grid_outages])
    items += _merged("reserve", [(w.start_hour, w.end_hour, w.min_soc_pct) for w in spec.reserve_windows])
    items += [SpecItem("soc_target", None, t.by_hour, t.min_soc_pct) for t in sorted(spec.soc_targets, key=lambda t: t.by_hour)]
    if spec.min_soc_pct_all_day is not None:
        items.append(SpecItem("min_soc_all_day", None, None, spec.min_soc_pct_all_day))
    if spec.end_of_day_min_soc_pct is not None:
        items.append(SpecItem("end_of_day_min_soc", None, None, spec.end_of_day_min_soc_pct))
    items += _merged("import_cap", [(w.start_hour, w.end_hour, w.max_kw) for w in spec.import_caps])
    items += _merged("no_export", [(w.start_hour, w.end_hour, None) for w in spec.no_export_windows])
    if spec.discharge_only_windows is not None:
        items += _merged("discharge_only", [(w.start_hour, w.end_hour, None) for w in spec.discharge_only_windows])
    if spec.max_throughput_kwh is not None:
        items.append(SpecItem("max_throughput", None, None, spec.max_throughput_kwh))
    return items


def _merged(kind: str, windows: list[tuple[int, int, float | None]]) -> list[SpecItem]:
    merged: list[list[Any]] = []
    for start, end, value in sorted(windows, key=lambda w: (w[2] if w[2] is not None else -1.0, w[0])):
        if merged and merged[-1][2] == value and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, value])
    return [SpecItem(kind, start, end, value) for start, end, value in sorted(merged, key=lambda m: (m[0], m[1]))]


def windows_hours(windows: list[HourWindow]) -> set[int]:
    return {hour for window in windows for hour in window.hours()}


def hours_to_windows(hours: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of hours as half-open windows: [6, 7, 17, 18] → [(6, 8), (17, 19)]."""
    windows: list[tuple[int, int]] = []
    for hour in sorted(set(hours)):
        if windows and windows[-1][1] == hour:
            windows[-1] = (windows[-1][0], hour + 1)
        else:
            windows.append((hour, hour + 1))
    return windows
