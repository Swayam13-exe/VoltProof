"""VoltProof Bench "hard" goals (SPEC §6.15 E2): load `intents_hard.yaml`, fill it per site and day,
and build the E2 episode set.

Same file schema as `intents.yaml`, with two more derivations and one more requirement (see the YAML
header); the frozen `intents.py` is reused for text filling, number formatting, goal instances and
the multilingual model, never changed.

E2 design (seed 2026, fixed before any run):
- English set: 2 seeded test days per site x every applicable template x one phrasing;
  phrasing = (site offset + day index + template index) % 2, as in the main test set
- multilingual set: the 3 multilingual goals x (en source sentence, es, hi, pt, de) on one seeded day
  out of those two, per site
Episode ids carry the `hard_` template ids, so they never collide with the frozen test run's.

PowerShell, from the repo root:
    python -m voltproof.bench.intents_hard --show      # resolved goal text and gold spec per site and day
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zlib
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from voltproof.bench.episode import DayBuilder, Episode, episode_id, processed_day_builder
from voltproof.bench.intents import (
    BUILTIN_PLACEHOLDERS,
    LANGUAGES,
    PENDING,
    Derivation,
    GoalInstance,
    MultilingualGoal,
    TranslationPendingError,
    derive,
    fill_text,
)
from voltproof.core.spec import format_errors, hours_to_windows, parse_spec
from voltproof.data import GLOBAL_SEED
from voltproof.data.days import split_dates
from voltproof.models import IntentSpec, Site, SiteType, Tariff

HARD_INTENTS_PATH = Path(__file__).with_name("intents_hard.yaml")
DAYS_PER_SITE = 2
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_WHOLE_PLACEHOLDER_RE = re.compile(r"^\{(\w+)\}$")
_FROZEN_DERIVATIONS = ("import_limit_frac_kw", "capacity_frac_kwh", "cover_hours_soc_pct", "tariff_period_windows")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HardDerivation(_Model):
    derive: Literal[
        "import_limit_frac_kw",
        "capacity_frac_kwh",
        "cover_hours_soc_pct",
        "tariff_period_windows",
        "import_limit_frac_exact_kw",
        "tariff_period_reserves",
    ]
    frac: float | None = Field(default=None, gt=0, le=1)
    hours: float | None = Field(default=None, gt=0, le=24)
    period: str | None = None
    pct: float | None = Field(default=None, gt=0, le=100)

    @model_validator(mode="after")
    def _arguments(self) -> HardDerivation:
        needed = {
            "import_limit_frac_kw": ("frac",),
            "capacity_frac_kwh": ("frac",),
            "cover_hours_soc_pct": ("hours",),
            "tariff_period_windows": ("period",),
            "import_limit_frac_exact_kw": ("frac",),
            "tariff_period_reserves": ("period", "pct"),
        }[self.derive]
        missing = [name for name in needed if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.derive} needs {', '.join(f'`{m}`' for m in missing)}")
        return self


class HardTemplate(_Model):
    id: str = Field(pattern=r"^hard_[a-z0-9_]+$")
    category: str
    held_out: bool
    site_types: list[SiteType] = Field(min_length=1)
    requires: list[str] = Field(default_factory=list)
    params: dict[str, HardDerivation] = Field(default_factory=dict)
    phrasings: list[str] = Field(min_length=2, max_length=2)
    gold: dict[str, Any]

    @model_validator(mode="after")
    def _placeholders_known(self) -> HardTemplate:
        known = set(BUILTIN_PLACEHOLDERS) | set(self.params)
        for text in self.phrasings:
            unknown = set(_PLACEHOLDER_RE.findall(text)) - known
            if unknown:
                raise ValueError(f"template {self.id}: unknown placeholders {sorted(unknown)}")
        for requirement in self.requires:
            if requirement not in ("export_allowed", "unreachable_first_hour") and not requirement.startswith("tariff_period:"):
                raise ValueError(f"template {self.id}: unknown requirement {requirement!r}")
        if "unreachable_first_hour" in self.requires and not isinstance(self.gold.get("min_soc_pct_all_day"), (int, float)):
            raise ValueError(f"template {self.id}: unreachable_first_hour needs a numeric gold min_soc_pct_all_day")
        return self


class HardIntentSet(_Model):
    version: int
    templates: list[HardTemplate]
    multilingual: list[MultilingualGoal] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> HardIntentSet:
        ids = [t.id for t in self.templates] + [m.id for m in self.multilingual]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate ids: {duplicates}")
        by_id = {t.id: t for t in self.templates}
        for goal in self.multilingual:
            template = by_id.get(goal.template)
            if template is None:
                raise ValueError(f"multilingual goal {goal.id}: unknown template {goal.template!r}")
            known = set(BUILTIN_PLACEHOLDERS) | set(template.params)
            for language in ("en", *LANGUAGES):
                unknown = set(_PLACEHOLDER_RE.findall(goal.text(language))) - known
                if unknown:
                    raise ValueError(f"multilingual goal {goal.id} ({language}): unknown placeholders {sorted(unknown)}")
        return self

    def template(self, template_id: str) -> HardTemplate:
        return next(t for t in self.templates if t.id == template_id)


def load_hard_intents(path: Path = HARD_INTENTS_PATH) -> HardIntentSet:
    return HardIntentSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def intents_sha256(path: Path = HARD_INTENTS_PATH) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def pending_translations(intents: HardIntentSet) -> list[str]:
    return [f"{goal.id}:{language}" for goal in intents.multilingual for language in goal.pending()]


# --- filling a template for a site and day --------------------------------------------------


def first_hour_max_soc_pct(site: Site) -> float:
    """The highest SOC % the battery can hold at the end of hour 0 (charging at full power from E0)."""
    battery = site.battery
    reachable = min(battery.initial_soc_frac * battery.capacity_kwh + battery.eta_charge * battery.p_charge_max_kw, battery.soc_max_frac * battery.capacity_kwh)
    return 100.0 * reachable / battery.capacity_kwh


def applicable(template: HardTemplate, site: Site, tariff: Tariff) -> bool:
    if site.site_type not in template.site_types:
        return False
    for requirement in template.requires:
        if requirement == "export_allowed" and not site.grid.export_allowed:
            return False
        if requirement.startswith("tariff_period:") and not tariff.periods.get(requirement.split(":", 1)[1]):
            return False
        if requirement == "unreachable_first_hour" and first_hour_max_soc_pct(site) >= float(template.gold["min_soc_pct_all_day"]) - 1e-9:
            return False
    return True


def derive_hard(derivation: HardDerivation, site: Site, tariff: Tariff) -> Any:
    if derivation.derive in _FROZEN_DERIVATIONS:
        return derive(Derivation(derive=derivation.derive, frac=derivation.frac, hours=derivation.hours, period=derivation.period), site, tariff)
    if derivation.derive == "import_limit_frac_exact_kw":
        return round(derivation.frac * site.grid.import_max_kw, 1)
    hours = tariff.periods.get(derivation.period or "")
    if not hours:
        raise ValueError(f"tariff {tariff.name!r} has no period {derivation.period!r}")
    return [{"start_hour": a, "end_hour": b, "min_soc_pct": derivation.pct} for a, b in hours_to_windows(hours)]


def placeholder_values(template: HardTemplate, site: Site, tariff: Tariff) -> dict[str, Any]:
    values: dict[str, Any] = {"critical_load_name": site.critical_load_name, "critical_load_kw": site.critical_load_kw}
    for name, derivation in template.params.items():
        values[name] = derive_hard(derivation, site, tariff)
    return values


def _substitute(node: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(node, dict):
        return {key: _substitute(value, values) for key, value in node.items()}
    if isinstance(node, list):
        return [_substitute(item, values) for item in node]
    if isinstance(node, str):
        match = _WHOLE_PLACEHOLDER_RE.match(node)
        if match is None or match.group(1) not in values:
            raise ValueError(f"gold values must be numbers, lists or one known placeholder, got {node!r}")
        return values[match.group(1)]
    return node


def gold_spec(template: HardTemplate, site: Site, tariff: Tariff) -> IntentSpec:
    spec, errors = parse_spec(_substitute(template.gold, placeholder_values(template, site, tariff)), site)
    if spec is None:
        raise ValueError(f"gold spec of {template.id} is invalid for {site.site_id}:\n{format_errors(errors)}")
    return spec


def english_goal(template: HardTemplate, site: Site, tariff: Tariff, phrasing: int) -> GoalInstance:
    text = fill_text(template.phrasings[phrasing], placeholder_values(template, site, tariff))
    return GoalInstance(f"{template.id}_p{phrasing}", template.id, "en", text, gold_spec(template, site, tariff), template.held_out)


def multilingual_goal(intents: HardIntentSet, goal: MultilingualGoal, language: str, site: Site, tariff: Tariff) -> GoalInstance:
    template = intents.template(goal.template)
    raw = goal.text(language)
    text = raw if raw.strip() == PENDING else fill_text(raw, placeholder_values(template, site, tariff))
    return GoalInstance(f"{goal.id}_{language}", template.id, language, text, gold_spec(template, site, tariff), template.held_out)


# --- the E2 episode set --------------------------------------------------------------------------


def _rng(seed: int, purpose: str, site_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(f"{purpose}:{site_id}".encode("utf-8"))])


def e2_dates(splits: Mapping[str, Any], site_id: str, seed: int = GLOBAL_SEED) -> list[Any]:
    """The 2 seeded test days of a site, in date order."""
    dates = split_dates(splits, "test", site_id)
    picks = _rng(seed, "e2_days", site_id).permutation(len(dates))[: min(DAYS_PER_SITE, len(dates))]
    return sorted(dates[int(i)] for i in picks)


def build_hard_episodes(
    intents: HardIntentSet,
    splits: Mapping[str, Any],
    sites: Mapping[str, Site],
    day_builder: DayBuilder | None = None,
    seed: int = GLOBAL_SEED,
    *,
    allow_pending: bool = False,
) -> list[Episode]:
    """English set, then the multilingual set, each in (site, date, template) order. With
    `allow_pending` (plan-only), untranslated goals are kept with the TRANSLATION_PENDING text."""
    pending = pending_translations(intents)
    if pending and not allow_pending:
        raise TranslationPendingError(f"{len(pending)} hard multilingual goals still {PENDING}: {', '.join(pending)}")
    build = day_builder or processed_day_builder()
    english: list[Episode] = []
    multilingual: list[Episode] = []
    for site_id in sorted({entry["site_id"] for entry in splits["test"]}):
        site = sites[site_id]
        days = [(d, *build(site, d)) for d in e2_dates(splits, site_id, seed)]
        offset = int(_rng(seed, "e2_phrasing", site_id).integers(2))
        for d, (day_date, day, tariff) in enumerate(days):
            templates = [t for t in intents.templates if applicable(t, site, tariff)]
            for j, template in enumerate(templates):
                goal = english_goal(template, site, tariff, (offset + d + j) % 2)
                english.append(Episode(episode_id(site, day_date, goal), site, day, tariff, goal))
        day_date, day, tariff = days[int(_rng(seed, "e2_multilingual", site_id).integers(len(days)))]
        for ml in intents.multilingual:
            if not applicable(intents.template(ml.template), site, tariff):
                continue
            for language in ("en", *LANGUAGES):
                goal = multilingual_goal(intents, ml, language, site, tariff)
                multilingual.append(Episode(episode_id(site, day_date, goal), site, day, tariff, goal))
    episodes = english + multilingual
    ids = [e.episode_id for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode ids in the hard set")
    return episodes


def pending_episodes(episodes: Sequence[Episode]) -> list[str]:
    return [e.episode_id for e in episodes if e.goal.text.strip() == PENDING]


# --- review listing ------------------------------------------------------------------------------


def describe(episodes: Sequence[Episode]) -> str:
    """Every resolved goal and gold spec, grouped by site and day, for review before any run."""
    lines: list[str] = []
    current = None
    for e in episodes:
        key = (e.site.site_id, e.day.date)
        if key != current:
            current = key
            periods = "; ".join(f"{name} {hours_to_windows(hours)}" for name, hours in sorted(e.tariff.periods.items())) or "none"
            lines.append(f"\n== {e.site.site_id} {e.day.date} (first-hour max SOC {first_hour_max_soc_pct(e.site):.1f}%; periods: {periods})")
        gold = json.dumps(e.goal.gold.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False)
        lines.append(f"  {e.goal.goal_id:<28} {e.goal.text}\n  {'':<28} gold {gold}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(prog="python -m voltproof.bench.intents_hard", description="E2 hard goals: resolved text and gold specs")
    parser.add_argument("--show", action="store_true", help="print every resolved goal and gold spec")
    args = parser.parse_args(argv)
    from voltproof.data.days import load_splits
    from voltproof.data.sites import SITES

    intents = load_hard_intents()
    episodes = build_hard_episodes(intents, load_splits(), SITES, allow_pending=True)
    if args.show:
        print(describe(episodes))
    pending = pending_episodes(episodes)
    print(f"\n{len(episodes)} episodes; {len(pending)} with {PENDING}; intents_hard.yaml sha256 {intents_sha256()[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
