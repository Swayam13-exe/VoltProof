"""E3 reviewer stress test (SPEC §6.15): does the independent reviewer catch wrong specs?

40 A4 episodes of the frozen test run whose spec matched gold exactly (seeded; mixed sites,
templates, languages; cost_min excluded because its empty spec admits only one kind of error). For
each, one-error variants of the spec the reviewer approved in the run, plus that spec unchanged as a
control. Each variant is sent to the frozen reviewer (`gate.run_review`: same prompt, schema and
client) with the episode's goal and site summary, in two conditions:
- with_assumptions: the assumptions the reviewer saw in the run (as SPEC §6.15 states). The controls
  are then byte-identical to the run's review call and replay its cached verdict at $0 (disclosed)
- no_assumptions:   the assumptions removed (spec.assumptions and the finalize assumptions), so the
  reviewer must judge from the goal alone; every call is fresh
Detection = approve is False. A verdict that could not be parsed (approved None) is counted apart.

Injection rules, one error per variant; every variant must pass `parse_spec(site)` and draws its
choices from rng [seed, crc32("<episode_id>:<type>")]:
- shift_hours:   one window (or a target's by_hour) moves by ±2 h (random sign; the other sign if
                 the first leaves 0-24 or makes the spec invalid)
- wrong_level:   one SOC % ±20 points (kept within the battery's limits), or an import cap / the
                 throughput cap x0.5 or x1.5 (rounded to 0.1, at most the grid limit)
- missing_item:  one spec item dropped (discharge_only_windows counts as one item)
- unneeded_item: the first field the spec does not use of: min_soc_pct_all_day 50 → reserve
                 [6, 10) 60% → end_of_day_min_soc_pct 80
- wrong_field:   reserve [a, b) r → target by_hour a at r; target → end-of-day minimum at its level;
                 all-day minimum ↔ end-of-day minimum; outage → import cap 0 kW over its hours;
                 import cap → no-export window; no-export window → import cap 0 kW;
                 discharge_only windows → no-export windows (throughput: not applicable)
"Assumptions state the original numbers" (with_assumptions only): some number in the assumptions
the reviewer sees (hours included) equals a value the injection changed or removed; None when the
injection removed no number (unneeded_item; a wrong_field that keeps every number).

PowerShell, from the repo root:
    python -m voltproof.bench.reviewer_stress                       # plan only: variants, cache check, projected cost
    python -m voltproof.bench.reviewer_stress --run --run-id e3-... # spend (after approval)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from voltproof.agents.gate import REVIEW_FORMAT, extract_numbers, normalize_digits, review_assumptions, review_messages, run_review
from voltproof.agents.loop import STOP_RUN_ERRORS, ChatLLM
from voltproof.bench.counterfactual import FROZEN_RUN, RecoveryError, read_trace, recover_a4, rows_by_key, run_episodes
from voltproof.bench.episode import Episode
from voltproof.bench.intents import LANGUAGES
from voltproof.bench.runner import EXIT_BUDGET, EXIT_FATAL, EXIT_OK, EXIT_REFUSED, REGRESSION_PATH, RUNS_DIR, budget_check, read_jsonl, regression_refusal, write_line
from voltproof.config import Settings, get_settings
from voltproof.core.metrics import KW_TOLERANCE
from voltproof.core.spec import parse_spec
from voltproof.data import GLOBAL_SEED
from voltproof.llm.client import BudgetExceeded, LLMClient, cache_key
from voltproof.models import IntentSpec, Site

N_EPISODES = 40
TEMPLATE_CAP = 4
MIN_PER_LANGUAGE = 2
EXCLUDED_TEMPLATES = ("cost_min",)
ERROR_TYPES = ("shift_hours", "wrong_level", "missing_item", "unneeded_item", "wrong_field")
EXPECTED_KIND = {"shift_hours": "wrong", "wrong_level": "wrong", "wrong_field": "wrong", "missing_item": "missing", "unneeded_item": "unneeded"}
CONDITIONS = ("with_assumptions", "no_assumptions")
CONTROL = "control"
WINDOW_FIELDS = ("grid_outages", "reserve_windows", "import_caps", "no_export_windows")
SCALAR_SOC_FIELDS = ("min_soc_pct_all_day", "end_of_day_min_soc_pct")
SHIFT_H = 2
LEVEL_PTS = 20.0
FACTORS = (0.5, 1.5)
UNNEEDED = (
    ("min_soc_pct_all_day", 50.0),
    ("reserve_windows", {"start_hour": 6, "end_hour": 10, "min_soc_pct": 60.0}),
    ("end_of_day_min_soc_pct", 80.0),
)
WILSON_Z = 1.959964

Spec = dict[str, Any]


# --- the episodes --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewInput:
    """What the reviewer saw in the run's last review call of an episode."""

    episode: Episode
    spec: IntentSpec  # the approved plan's spec (with its own assumptions)
    assumptions: list[str]  # the finalize assumptions of that attempt


def stress_pool(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """A4 results whose spec matched gold exactly, accepted through the gate, with a review."""
    return sorted(
        (
            r
            for r in rows
            if r["arm"] == "A4"
            and r.get("spec_precision") == 1
            and r.get("spec_recall") == 1
            and r.get("plan_source") == "finalized"
            and r.get("review_approved") is not None
            and r["spec_name"] not in EXCLUDED_TEMPLATES
        ),
        key=lambda r: r["episode_id"],
    )


def site_quotas(site_ids: Sequence[str], n: int) -> dict[str, int]:
    """n split as evenly as possible over the sites in sorted order (the first sites get the rest)."""
    ordered = sorted(site_ids)
    base, extra = divmod(n, len(ordered))
    return {site: base + (1 if i < extra else 0) for i, site in enumerate(ordered)}


def select_stress(
    pool: Sequence[Mapping[str, Any]], goals: Mapping[str, str], n: int = N_EPISODES, seed: int = GLOBAL_SEED
) -> list[Mapping[str, Any]]:
    """Seeded order; site quotas; at most TEMPLATE_CAP per template; never the same goal text twice at
    one site (the reviewer does not see the day, so such episodes would send it identical inputs);
    first MIN_PER_LANGUAGE of each non-English language, then the rest in order."""
    order = [pool[int(i)] for i in np.random.default_rng(seed).permutation(len(pool))]
    quotas = site_quotas(sorted({r["site_id"] for r in pool}), n)
    chosen: list[Mapping[str, Any]] = []

    def fits(row: Mapping[str, Any]) -> bool:
        return (
            row not in chosen
            and sum(c["site_id"] == row["site_id"] for c in chosen) < quotas[row["site_id"]]
            and sum(c["spec_name"] == row["spec_name"] for c in chosen) < TEMPLATE_CAP
            and not any(c["site_id"] == row["site_id"] and goals[c["episode_id"]] == goals[row["episode_id"]] for c in chosen)
        )

    for language in LANGUAGES:
        for row in order:
            if sum(c["language"] == language for c in chosen) >= MIN_PER_LANGUAGE:
                break
            if row["language"] == language and fits(row):
                chosen.append(row)
    for row in order:
        if len(chosen) == n:
            break
        if fits(row):
            chosen.append(row)
    return sorted(chosen, key=lambda r: r["episode_id"])


def review_input(episode: Episode, run_dir: Path) -> ReviewInput:
    """The spec and assumptions of the run's last review call (the finalize attempt that ran it)."""
    events = read_trace(run_dir, "A4", episode.episode_id)
    finalize_args, reviewed = None, None
    for event in events:
        if event["kind"] == "tool_call" and event["name"] == "finalize":
            finalize_args = event["data"].get("args")
        elif event["kind"] == "review":
            reviewed = finalize_args
    if not isinstance(reviewed, dict):
        raise RecoveryError(f"{episode.episode_id}: no review call in the trace")
    plan = recover_a4(episode, events)
    selection = next(e for e in reversed(events) if e["kind"] == "final" and e["name"] == "selection")
    if plan.spec is None or reviewed.get("plan_id") != selection["data"].get("plan_id"):
        raise RecoveryError(f"{episode.episode_id}: the last review was not on the accepted plan")
    return ReviewInput(episode, plan.spec, list(reviewed.get("assumptions") or []))


# --- injections ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Injection:
    error_type: str
    item: str
    description: str
    spec: IntentSpec
    removed: frozenset[float] | None  # numbers the injection changed or removed (None: none)


def _rng(seed: int, episode_id: str, error_type: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(f"{episode_id}:{error_type}".encode("utf-8"))])


def spec_items(data: Spec) -> list[tuple[str, int | None]]:
    """(field, index) of every constraint; discharge_only_windows and the scalars are one item each."""
    items: list[tuple[str, int | None]] = []
    for field in ("grid_outages", "reserve_windows", "soc_targets", "import_caps", "no_export_windows"):
        items += [(field, i) for i in range(len(data.get(field) or []))]
    if data.get("discharge_only_windows") is not None:
        items.append(("discharge_only_windows", None))
    items += [(f, None) for f in (*SCALAR_SOC_FIELDS, "max_throughput_kwh") if data.get(f) is not None]
    return items


def item_value(data: Spec, field: str, index: int | None) -> Any:
    return data[field] if index is None else data[field][index]


def item_numbers(value: Any) -> set[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {float(value)}
    if isinstance(value, dict):
        return {float(v) for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if isinstance(value, list):
        return set().union(*(item_numbers(v) for v in value)) if value else set()
    return set()


def _label(field: str, index: int | None) -> str:
    return field if index is None else f"{field}[{index}]"


def _valid(data: Spec, site: Site) -> IntentSpec | None:
    spec, _ = parse_spec(data, site)
    return spec


def _shifted(window: Mapping[str, Any], delta: int) -> dict[str, Any] | None:
    start, end = window["start_hour"] + delta, window["end_hour"] + delta
    return {**window, "start_hour": start, "end_hour": end} if 0 <= start < end <= 24 else None


def inject_shift(data: Spec, site: Site, rng: np.random.Generator) -> Injection | None:
    candidates: list[tuple[str, int]] = [(f, i) for f in (*WINDOW_FIELDS, "soc_targets") for i in range(len(data.get(f) or []))]
    candidates += [("discharge_only_windows", i) for i in range(len(data.get("discharge_only_windows") or []))]
    for k in rng.permutation(len(candidates)):
        field, index = candidates[int(k)]
        first = int(rng.choice([-SHIFT_H, SHIFT_H]))
        for delta in (first, -first):
            new = copy.deepcopy(data)
            old = new[field][index]
            if field == "soc_targets":
                hour = old["by_hour"] + delta
                if not 1 <= hour <= 24:
                    continue
                new[field][index] = {**old, "by_hour": hour}
                removed = {float(old["by_hour"])}
            else:
                moved = _shifted(old, delta)
                if moved is None:
                    continue
                new[field][index] = moved
                removed = {float(old["start_hour"]), float(old["end_hour"])} - {float(moved["start_hour"]), float(moved["end_hour"])}
            spec = _valid(new, site)
            if spec is not None:
                return Injection("shift_hours", _label(field, index), f"{_label(field, index)} moved by {delta:+d} h", spec, frozenset(removed))
    return None


def inject_level(data: Spec, site: Site, rng: np.random.Generator) -> Injection | None:
    lo, hi = site.battery.soc_min_frac * 100.0, site.battery.soc_max_frac * 100.0
    candidates: list[tuple[str, int | None, str]] = []
    for field in ("reserve_windows", "soc_targets"):
        candidates += [(field, i, "min_soc_pct") for i in range(len(data.get(field) or []))]
    candidates += [(f, None, "") for f in SCALAR_SOC_FIELDS if data.get(f) is not None]
    candidates += [("import_caps", i, "max_kw") for i in range(len(data.get("import_caps") or []))]
    if data.get("max_throughput_kwh") is not None:
        candidates.append(("max_throughput_kwh", None, ""))
    for k in rng.permutation(len(candidates)):
        field, index, key = candidates[int(k)]
        old = item_value(data, field, index)
        value = float(old[key] if key else old)
        if key == "max_kw" or field == "max_throughput_kwh":
            first = float(rng.choice(FACTORS))
            options = [round(value * f, 1) for f in (first, FACTORS[0] + FACTORS[1] - first)]
            limit = site.grid.import_max_kw if key == "max_kw" else math.inf
            options = [v for v in options if 0 <= v <= limit and abs(v - value) > KW_TOLERANCE + 1e-9]
        else:
            first = float(rng.choice([-LEVEL_PTS, LEVEL_PTS]))
            options = [value + d for d in (first, -first) if lo - 1e-9 <= value + d <= hi + 1e-9]
        for new_value in options:
            new = copy.deepcopy(data)
            if key:
                new[field][index] = {**old, key: new_value}
            else:
                new[field] = new_value
            spec = _valid(new, site)
            if spec is not None:
                return Injection("wrong_level", _label(field, index), f"{_label(field, index)} {value:g} -> {new_value:g}", spec, frozenset({value}))
    return None


def inject_missing(data: Spec, site: Site, rng: np.random.Generator) -> Injection | None:
    items = spec_items(data)
    if not items:
        return None
    field, index = items[int(rng.integers(len(items)))]
    new = copy.deepcopy(data)
    old = item_value(data, field, index)
    if index is None:
        new[field] = None
    else:
        del new[field][index]
    spec = _valid(new, site)
    return None if spec is None else Injection("missing_item", _label(field, index), f"{_label(field, index)} dropped", spec, frozenset(item_numbers(old)))


def inject_unneeded(data: Spec, site: Site, rng: np.random.Generator) -> Injection | None:
    for field, value in UNNEEDED:
        if data.get(field):
            continue
        new = copy.deepcopy(data)
        new[field] = [value] if isinstance(value, dict) else value
        spec = _valid(new, site)
        if spec is not None:
            return Injection("unneeded_item", field, f"{field} = {json.dumps(new[field])} added", spec, None)
    return None


def _wrong_field(data: Spec, field: str, index: int | None) -> tuple[Spec, Any] | None:
    """The item moved to the wrong field, and the new item (None when the mapping does not apply)."""
    new = copy.deepcopy(data)
    old = item_value(data, field, index)

    def add(target: str, item: Any) -> Any:
        new[target] = [*(new.get(target) or []), item] if isinstance(item, dict) else item
        return item

    if field == "reserve_windows":
        if old["start_hour"] < 1:
            return None
        del new[field][index]
        return new, add("soc_targets", {"by_hour": old["start_hour"], "min_soc_pct": old["min_soc_pct"]})
    if field == "soc_targets":
        if data.get("end_of_day_min_soc_pct") is not None:
            return None
        del new[field][index]
        return new, add("end_of_day_min_soc_pct", old["min_soc_pct"])
    if field in SCALAR_SOC_FIELDS:
        other = SCALAR_SOC_FIELDS[1] if field == SCALAR_SOC_FIELDS[0] else SCALAR_SOC_FIELDS[0]
        if data.get(other) is not None:
            return None
        new[field] = None
        return new, add(other, old)
    if field in ("grid_outages", "no_export_windows"):
        del new[field][index]
        return new, add("import_caps", {"start_hour": old["start_hour"], "end_hour": old["end_hour"], "max_kw": 0.0})
    if field == "import_caps":
        del new[field][index]
        return new, add("no_export_windows", {"start_hour": old["start_hour"], "end_hour": old["end_hour"]})
    if field == "discharge_only_windows":
        new[field] = None
        new["no_export_windows"] = [*(new.get("no_export_windows") or []), *copy.deepcopy(old)]
        return new, old
    return None  # max_throughput_kwh


def inject_wrong_field(data: Spec, site: Site, rng: np.random.Generator) -> Injection | None:
    items = spec_items(data)
    for k in rng.permutation(len(items)):
        field, index = items[int(k)]
        moved = _wrong_field(data, field, index)
        if moved is None:
            continue
        new, new_item = moved
        spec = _valid(new, site)
        if spec is not None:
            removed = item_numbers(item_value(data, field, index)) - item_numbers(new_item)
            return Injection("wrong_field", _label(field, index), f"{_label(field, index)} written in the wrong field", spec, frozenset(removed) if removed else None)
    return None


INJECT: Mapping[str, Callable[[Spec, Site, np.random.Generator], Injection | None]] = {
    "shift_hours": inject_shift,
    "wrong_level": inject_level,
    "missing_item": inject_missing,
    "unneeded_item": inject_unneeded,
    "wrong_field": inject_wrong_field,
}


def injections(spec: IntentSpec, site: Site, episode_id: str, seed: int = GLOBAL_SEED) -> list[Injection]:
    data = spec.model_dump(mode="json")
    out = []
    for error_type in ERROR_TYPES:
        injection = INJECT[error_type](data, site, _rng(seed, episode_id, error_type))
        if injection is not None:
            out.append(injection)
    return out


# --- variants ----------------------------------------------------------------------------------------


_DIGITS_RE = re.compile(r"\d+(?:\.\d+)?")


def states_original(numbers: frozenset[float] | None, texts: Sequence[str]) -> bool | None:
    """Whether any number in the texts equals one of `numbers`: the gate's parser (times such as
    "6 PM" read as hour 18) plus every plain digit run (so "[18,22)" reads as 18 and 22)."""
    if not numbers:
        return None
    found = {c for text in texts for n in extract_numbers(text) for c in n.candidates}
    found |= {float(m) for text in texts for m in _DIGITS_RE.findall(normalize_digits(text))}
    return any(abs(a - b) < 1e-9 for a in found for b in numbers)


def condition_inputs(spec: IntentSpec, assumptions: Sequence[str], condition: str) -> tuple[IntentSpec, list[str]]:
    if condition == "no_assumptions":
        return spec.model_copy(update={"assumptions": []}), []
    return spec, list(assumptions)


def build_variants(inputs: Sequence[ReviewInput], seed: int = GLOBAL_SEED) -> list[dict[str, Any]]:
    """Every (episode, control or injection, condition) row; the spec is the one the reviewer sees."""
    variants = []
    for item in inputs:
        episode = item.episode
        cases: list[tuple[str, str, str, IntentSpec, frozenset[float] | None]] = [(CONTROL, "", "the approved spec, unchanged", item.spec, None)]
        cases += [(i.error_type, i.item, i.description, i.spec, i.removed) for i in injections(item.spec, episode.site, episode.episode_id, seed)]
        for error_type, path, description, spec, removed in cases:
            for condition in CONDITIONS:
                shown_spec, shown_assumptions = condition_inputs(spec, item.assumptions, condition)
                texts = review_assumptions(shown_spec, shown_assumptions)
                variants.append(
                    {
                        "variant_id": f"{episode.episode_id}__{error_type}__{condition}",
                        "episode_id": episode.episode_id,
                        "site_id": episode.site.site_id,
                        "template_id": episode.goal.template_id,
                        "language": episode.goal.language,
                        "error_type": error_type,
                        "condition": condition,
                        "item": path,
                        "description": description,
                        "expected_kind": EXPECTED_KIND.get(error_type),
                        "assumptions_state_original": states_original(removed, texts) if condition == "with_assumptions" else None,
                        "original_spec": item.spec.model_dump(mode="json", exclude_defaults=True),
                        "spec": shown_spec.model_dump(mode="json"),
                        "assumptions": shown_assumptions,
                    }
                )
    # every with_assumptions call first: a no_assumptions variant whose input is identical (the run's
    # reviewer saw no assumptions) then replays that verdict from the cache instead of a second call
    return sorted(variants, key=lambda v: CONDITIONS.index(v["condition"]))


def variant_inputs(variant: Mapping[str, Any]) -> tuple[IntentSpec, list[str]]:
    return IntentSpec.model_validate(variant["spec"]), list(variant["assumptions"])


def variant_cache_key(variant: Mapping[str, Any], episode: Episode, settings: Settings) -> str:
    """The client's cache key of this variant's review call (the same messages run_review sends)."""
    spec, assumptions = variant_inputs(variant)
    return cache_key(
        model=settings.llm_model,
        messages=review_messages(episode.context(), spec, review_assumptions(spec, assumptions)),
        tools=None,
        tool_choice=None,
        response_format=REVIEW_FORMAT,
        reasoning_effort=settings.llm_reasoning_effort,
        max_tokens=settings.llm_max_output_tokens,
    )


def is_cached(key: str, settings: Settings) -> bool:
    return (settings.cache_dir / key[:2] / f"{key}.json").is_file()


def review_call_costs(run_dir: Path) -> list[float]:
    """The billed cost of every uncached reviewer call in the run's A4 traces."""
    costs = []
    for path in sorted((run_dir / "traces").glob("A4__*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["kind"] == "llm_call" and event["name"] == "review" and not event["data"].get("cache_hit"):
                costs.append(float(event["data"].get("cost_usd") or 0.0))
    return costs


# --- running -------------------------------------------------------------------------------------------


def run_variant(llm: ChatLLM, variant: Mapping[str, Any], episode: Episode) -> dict[str, Any]:
    spec, assumptions = variant_inputs(variant)
    outcome = run_review(llm, episode.context(), spec, assumptions, {"experiment": "E3", "variant": variant["variant_id"]})
    response = outcome.response
    return {
        "variant_id": variant["variant_id"],
        "approved": outcome.approved,
        "issues": [i.model_dump() for i in outcome.issues],
        "raw": outcome.raw,
        "cost_usd": response.cost_usd if response is not None else 0.0,
        "cache_hit": response.cache_hit if response is not None else None,
        "tokens_in": response.usage.prompt if response is not None else None,
        "tokens_out": response.usage.completion if response is not None else None,
    }


def run_variants(
    llm: ChatLLM, variants: Sequence[Mapping[str, Any]], episodes: Mapping[str, Episode], out: Path, *, concurrency: int = 2, progress: bool = True
) -> tuple[int, str | None]:
    """Review every variant not yet in `out` (resumable); only the main thread writes. Returns the
    exit code and a stop message."""
    done = {r["variant_id"] for r in read_jsonl(out)}
    queue = [v for v in variants if v["variant_id"] not in done]
    code, message = EXIT_OK, None
    bar = tqdm(total=len(queue), desc="E3 reviews", unit="call", disable=not progress)
    in_flight: dict[Future[dict[str, Any]], Mapping[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        while queue or in_flight:
            while queue and code == EXIT_OK and len(in_flight) < max(1, concurrency):
                variant = queue.pop(0)
                in_flight[pool.submit(run_variant, llm, variant, episodes[variant["episode_id"]])] = variant
            if not in_flight:
                break
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                in_flight.pop(future)
                bar.update(1)
                try:
                    write_line(out, future.result())
                except BudgetExceeded:
                    if code == EXIT_OK:
                        code, message = EXIT_BUDGET, "STOPPED: BudgetExceeded (partial results saved; resume reruns the rest)"
                except STOP_RUN_ERRORS as exc:
                    if code == EXIT_OK:
                        code, message = EXIT_FATAL, f"STOPPED: {type(exc).__name__}: {exc}"
    bar.close()
    return code, message


# --- summary -------------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float | None, float | None, float | None]:
    if n == 0:
        return None, None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarize(variants: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per condition and error type: detection rate (controls: false-alarm rate) with Wilson 95% CI,
    the issue-kind match, unavailable verdicts, and (with assumptions) the split by whether the
    assumptions state the original numbers."""
    by_id = {r["variant_id"]: r for r in results}
    out = []
    for condition in CONDITIONS:
        for error_type in (CONTROL, *ERROR_TYPES):
            mine = [(v, by_id[v["variant_id"]]) for v in variants if v["condition"] == condition and v["error_type"] == error_type and v["variant_id"] in by_id]
            judged = [(v, r) for v, r in mine if r["approved"] is not None]
            flagged = [(v, r) for v, r in judged if r["approved"] is False]
            rate, lo, hi = wilson(len(flagged), len(judged))
            row: dict[str, Any] = {
                "condition": condition,
                "error_type": error_type,
                "variants": len(mine),
                "unavailable": len(mine) - len(judged),
                "flagged": len(flagged),
                "rate": rate,
                "ci_low": lo,
                "ci_high": hi,
                "kind_match": sum(any(i["kind"] == v["expected_kind"] for i in r["issues"]) for v, r in flagged) if error_type != CONTROL else None,
                "cached": sum(bool(r.get("cache_hit")) for _, r in mine),
            }
            if condition == "with_assumptions" and error_type != CONTROL:
                for flag in (True, False):
                    sub = [(v, r) for v, r in judged if v["assumptions_state_original"] is flag]
                    row[f"rate_if_assumptions_state_original_{str(flag).lower()}"] = wilson(sum(r["approved"] is False for _, r in sub), len(sub))[0]
                    row[f"n_assumptions_state_original_{str(flag).lower()}"] = len(sub)
            out.append(row)
    return out


# --- CLI -------------------------------------------------------------------------------------------------


def prepare(run_dir: Path, seed: int = GLOBAL_SEED) -> tuple[list[Mapping[str, Any]], list[ReviewInput], dict[str, Episode]]:
    episodes = {e.episode_id: e for e in run_episodes(run_dir)}
    goals = {eid: e.goal.text for eid, e in episodes.items()}
    chosen = select_stress(stress_pool(rows_by_key(run_dir).values()), goals, N_EPISODES, seed)
    inputs = [review_input(episodes[r["episode_id"]], run_dir) for r in chosen]
    return chosen, inputs, episodes


def plan_only(run_dir: Path, runs_dir: Path, settings: Settings, seed: int = GLOBAL_SEED) -> Path:
    chosen, inputs, episodes = prepare(run_dir, seed)
    variants = build_variants(inputs, seed)
    for variant in variants:
        key = variant_cache_key(variant, episodes[variant["episode_id"]], settings)
        variant["cache_key"], variant["cached"] = key, is_cached(key, settings)
    misses = [v["variant_id"] for v in variants if v["error_type"] == CONTROL and v["condition"] == "with_assumptions" and not v["cached"]]
    keys = [(v["condition"], v["cache_key"]) for v in variants]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        raise ValueError(f"{duplicates} variants would send the reviewer the same input twice within a condition")
    first_condition = {v["cache_key"] for v in variants if v["condition"] == CONDITIONS[0]}
    for v in variants:
        v["same_input_as_with_assumptions"] = v["condition"] == CONDITIONS[1] and v["cache_key"] in first_condition
    costs = review_call_costs(run_dir)
    uncached = sum(not v["cached"] and not v["same_input_as_with_assumptions"] for v in variants)
    projected = uncached * float(np.mean(costs)) if costs else float("nan")
    ok, budget = budget_check(projected, settings)
    out = runs_dir / f"e3-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "variants.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in variants), encoding="utf-8")
    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_run": run_dir.name,
        "seed": seed,
        "episodes": [r["episode_id"] for r in chosen],
        "variants": len(variants),
        "uncached_calls": uncached,
        "no_assumptions_identical_to_with_assumptions": sum(v["same_input_as_with_assumptions"] for v in variants),
        "control_cache_misses": misses,
        "mean_review_call_usd": float(np.mean(costs)) if costs else None,
        "max_review_call_usd": float(np.max(costs)) if costs else None,
        "projected_usd": projected,
        "pessimistic_usd": uncached * float(np.max(costs)) if costs else None,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    counts: dict[tuple[str, str], int] = {}
    for v in variants:
        counts[(v["condition"], v["error_type"])] = counts.get((v["condition"], v["error_type"]), 0) + 1
    print(f"E3 plan: {len(chosen)} episodes, {len(variants)} review calls ({uncached} not cached)")
    for (condition, error_type), n in sorted(counts.items()):
        print(f"  {condition:<17} {error_type:<14} {n:>3}")
    stated = [v for v in variants if v["condition"] == "with_assumptions" and v["assumptions_state_original"]]
    print(f"with_assumptions variants whose assumptions state the original numbers: {len(stated)}")
    print(f"condition-A controls not in the cache: {len(misses)} {misses[:3]}")
    same = [v for v in variants if v["same_input_as_with_assumptions"]]
    print(f"no_assumptions variants identical to their with_assumptions input (the run's reviewer saw no assumptions): {len(same)} in {len({v['episode_id'] for v in same})} episodes")
    print(f"projected ${projected:.2f} (pessimistic ${config['pessimistic_usd']:.2f}); {budget}")
    print(f"written to {out}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(prog="python -m voltproof.bench.reviewer_stress", description="E3 reviewer stress test")
    parser.add_argument("--source-run", type=Path, default=FROZEN_RUN, help="the frozen test run (read only)")
    parser.add_argument("--run", action="store_true", help="spend: review the variants of --run-id")
    parser.add_argument("--run-id", help="an e3-* directory written by the plan-only step")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args(argv)
    settings = get_settings()
    if not args.run:
        plan_only(args.source_run, RUNS_DIR, settings)
        return EXIT_OK
    if not args.run_id:
        print("REFUSED: --run needs --run-id (the plan-only directory you approved)")
        return EXIT_REFUSED
    refusal = regression_refusal(REGRESSION_PATH)
    if refusal:
        print(f"REFUSED: {refusal}")
        return EXIT_REFUSED
    out_dir = RUNS_DIR / args.run_id
    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    if config["control_cache_misses"]:
        print(f"REFUSED: {len(config['control_cache_misses'])} controls do not reproduce the run's review call")
        return EXIT_REFUSED
    variants = read_jsonl(out_dir / "variants.jsonl")
    episodes = {e.episode_id: e for e in run_episodes(args.source_run)}
    code, message = run_variants(LLMClient(settings), variants, episodes, out_dir / "results.jsonl", concurrency=args.concurrency)
    if message:
        print(message)
    summary = summarize(variants, read_jsonl(out_dir / "results.jsonl"))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for row in summary:
        print(f"{row['condition']:<17} {row['error_type']:<14} {row['flagged']:>3}/{row['variants'] - row['unavailable']:<3} {row['rate']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
