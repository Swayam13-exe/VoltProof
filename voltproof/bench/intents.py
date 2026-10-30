"""Benchmark goals (SPEC §6.11): load `intents.yaml`, fill placeholders per site and day, build gold specs.

A template is site-agnostic. Its placeholders come from the site ({critical_load_name},
{critical_load_kw}) or from a named derivation over the site and the day's tariff (`params`), so
the same goal text and gold spec make sense for a 15 kWh battery and a 100 kWh one. Gold specs
are validated against each applicable site with `parse_spec`. The multilingual subset refuses to
load while any translation is still TRANSLATION_PENDING.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from voltproof.core.spec import format_errors, hours_to_windows, parse_spec
from voltproof.models import IntentSpec, Site, SiteType, Tariff

INTENTS_PATH = Path(__file__).with_name("intents.yaml")
PENDING = "TRANSLATION_PENDING"
LANGUAGES = ("es", "hi", "pt", "de")
BUILTIN_PLACEHOLDERS = ("critical_load_name", "critical_load_kw")
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_WHOLE_PLACEHOLDER_RE = re.compile(r"^\{(\w+)\}$")


class TranslationPendingError(ValueError):
    """The multilingual subset still has TRANSLATION_PENDING entries."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Derivation(_Model):
    derive: Literal["import_limit_frac_kw", "capacity_frac_kwh", "cover_hours_soc_pct", "tariff_period_windows"]
    frac: float | None = Field(default=None, gt=0, le=1)
    hours: float | None = Field(default=None, gt=0, le=24)
    period: str | None = None

    @model_validator(mode="after")
    def _arguments(self) -> Derivation:
        needed = {
            "import_limit_frac_kw": "frac",
            "capacity_frac_kwh": "frac",
            "cover_hours_soc_pct": "hours",
            "tariff_period_windows": "period",
        }[self.derive]
        if getattr(self, needed) is None:
            raise ValueError(f"{self.derive} needs `{needed}`")
        return self


class Template(_Model):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: str
    held_out: bool
    site_types: list[SiteType] = Field(min_length=1)
    requires: list[str] = Field(default_factory=list)
    params: dict[str, Derivation] = Field(default_factory=dict)
    phrasings: list[str] = Field(min_length=2, max_length=2)
    gold: dict[str, Any]

    @model_validator(mode="after")
    def _placeholders_known(self) -> Template:
        known = set(BUILTIN_PLACEHOLDERS) | set(self.params)
        for text in self.phrasings:
            unknown = set(_PLACEHOLDER_RE.findall(text)) - known
            if unknown:
                raise ValueError(f"template {self.id}: unknown placeholders {sorted(unknown)}")
        for requirement in self.requires:
            if requirement != "export_allowed" and not requirement.startswith("tariff_period:"):
                raise ValueError(f"template {self.id}: unknown requirement {requirement!r}")
        return self


class MultilingualGoal(_Model):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    template: str
    en: str
    es: str
    hi: str
    pt: str
    de: str

    def text(self, language: str) -> str:
        return getattr(self, language)

    def pending(self) -> list[str]:
        return [language for language in LANGUAGES if self.text(language).strip() == PENDING]


class IntentSet(_Model):
    version: int
    templates: list[Template]
    multilingual: list[MultilingualGoal] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> IntentSet:
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

    def template(self, template_id: str) -> Template:
        return next(t for t in self.templates if t.id == template_id)


@dataclass(frozen=True)
class GoalInstance:
    goal_id: str  # e.g. "storm_full_p0" or "ml_storm_full_de"
    template_id: str
    language: str
    text: str
    gold: IntentSpec
    held_out: bool


def load_intents(path: Path = INTENTS_PATH) -> IntentSet:
    return IntentSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def pending_translations(intents: IntentSet) -> list[str]:
    return [f"{goal.id}:{language}" for goal in intents.multilingual for language in goal.pending()]


def multilingual_goals(intents: IntentSet) -> list[MultilingualGoal]:
    """The multilingual subset; raises TranslationPendingError while any translation is missing."""
    pending = pending_translations(intents)
    if pending:
        raise TranslationPendingError(f"{len(pending)} multilingual goals still {PENDING}: {', '.join(pending)}")
    return list(intents.multilingual)


# --- filling a template for a site and day --------------------------------------------------


def applicable(template: Template, site: Site, tariff: Tariff) -> bool:
    if site.site_type not in template.site_types:
        return False
    for requirement in template.requires:
        if requirement == "export_allowed" and not site.grid.export_allowed:
            return False
        if requirement.startswith("tariff_period:") and not tariff.periods.get(requirement.split(":", 1)[1]):
            return False
    return True


def placeholder_values(template: Template, site: Site, tariff: Tariff) -> dict[str, Any]:
    values: dict[str, Any] = {"critical_load_name": site.critical_load_name, "critical_load_kw": site.critical_load_kw}
    for name, derivation in template.params.items():
        values[name] = derive(derivation, site, tariff)
    return values


def derive(derivation: Derivation, site: Site, tariff: Tariff) -> Any:
    battery = site.battery
    if derivation.derive == "import_limit_frac_kw":
        return max(1, math.floor(derivation.frac * site.grid.import_max_kw + 1e-9))
    if derivation.derive == "capacity_frac_kwh":
        return round(derivation.frac * battery.capacity_kwh, 1)
    if derivation.derive == "cover_hours_soc_pct":
        needed_kwh = derivation.hours * site.critical_load_kw / battery.eta_discharge
        return round(battery.soc_min_frac * 100.0 + 100.0 * needed_kwh / battery.capacity_kwh, 1)
    hours = tariff.periods.get(derivation.period or "")
    if not hours:
        raise ValueError(f"tariff {tariff.name!r} has no period {derivation.period!r}")
    return [{"start_hour": a, "end_hour": b} for a, b in hours_to_windows(hours)]


def format_value(value: Any) -> str:
    """Whole numbers without a decimal (100), else one decimal (60.5), or up to 3 decimals when one
    decimal would misstate a site value (0.15)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    if abs(value - round(value, 1)) < 1e-9:
        return f"{value:.1f}"
    return f"{round(value, 3):.3f}".rstrip("0")


def fill_text(text: str, values: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(f"no value for placeholder {{{name}}}")
        return format_value(values[name])

    return _PLACEHOLDER_RE.sub(replace, text)


def gold_spec(template: Template, site: Site, tariff: Tariff) -> IntentSpec:
    """The template's gold spec for this site and day, validated against the site."""
    data = _substitute(template.gold, placeholder_values(template, site, tariff))
    spec, errors = parse_spec(data, site)
    if spec is None:
        raise ValueError(f"gold spec of {template.id} is invalid for {site.site_id}:\n{format_errors(errors)}")
    return spec


def _substitute(node: Any, values: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        return {key: _substitute(value, values) for key, value in node.items()}
    if isinstance(node, list):
        return [_substitute(item, values) for item in node]
    if isinstance(node, str):
        match = _WHOLE_PLACEHOLDER_RE.match(node)
        if match is None:
            raise ValueError(f"gold values must be numbers, lists or one whole placeholder, got {node!r}")
        if match.group(1) not in values:
            raise ValueError(f"no value for placeholder {node}")
        return values[match.group(1)]
    return node


def goals_for(intents: IntentSet, site: Site, tariff: Tariff, split: Literal["dev", "test"]) -> list[GoalInstance]:
    """Goal instances for one site-day. dev: non-held-out templates, English phrasings only.
    test: every template, plus the multilingual subset (which must be fully translated)."""
    goals = []
    for template in intents.templates:
        if (split == "dev" and template.held_out) or not applicable(template, site, tariff):
            continue
        values = placeholder_values(template, site, tariff)
        gold = gold_spec(template, site, tariff)
        for k, phrasing in enumerate(template.phrasings):
            goals.append(GoalInstance(f"{template.id}_p{k}", template.id, "en", fill_text(phrasing, values), gold, template.held_out))
    if split == "test":
        for goal in multilingual_goals(intents):
            template = intents.template(goal.template)
            if not applicable(template, site, tariff):
                continue
            values = placeholder_values(template, site, tariff)
            gold = gold_spec(template, site, tariff)
            for language in ("en", *LANGUAGES):
                text = fill_text(goal.text(language), values)
                goals.append(GoalInstance(f"{goal.id}_{language}", template.id, language, text, gold, template.held_out))
    return goals
