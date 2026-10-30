"""intents.yaml: structure, placeholders, gold specs valid for every applicable site, splits, translations.

Sites used here: the two dev sites and a synthetic household site (never the held-out site).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from tests.fakes_data import DYNAMIC_TEMPLATE, make_site
from voltproof.bench.intents import (
    INTENTS_PATH,
    LANGUAGES,
    PENDING,
    TranslationPendingError,
    applicable,
    english_goal,
    fill_text,
    format_value,
    gold_spec,
    goals_for,
    load_intents,
    multilingual_goal,
    multilingual_goals,
    placeholder_values,
)
from voltproof.data.sites import SITES, dev_sites
from voltproof.data.tariff import build_tariff
from voltproof.models import Site, Tariff

DAY = date(2024, 6, 5)
SPEC_CATEGORIES = {
    "plain_cost",
    "evening_reserve",
    "announced_outage",
    "no_export",
    "full_by_time",
    "import_cap",
    "min_soc_all_day",
    "battery_peak_only",
    "cover_load_hours",
    "save_cycles",
}


def sites_with_tariffs() -> list[tuple[Site, Tariff]]:
    household = make_site("synthetic_household", site_type="household", tariff_template=DYNAMIC_TEMPLATE)
    sites = [*dev_sites(), household]
    return [(site, build_tariff(site.tariff_template, DAY, seed=2026, stream=site.site_id)) for site in sites]


def test_intents_load_with_the_spec_structure() -> None:
    intents = load_intents()
    assert len(intents.templates) == 12
    assert SPEC_CATEGORIES <= {t.category for t in intents.templates}
    assert sorted(t.id for t in intents.templates if t.held_out) == ["end_of_day_reserve", "outage_then_reserve"]
    assert all(len(t.phrasings) == 2 for t in intents.templates)
    assert len(intents.multilingual) == 6


def test_every_gold_spec_is_valid_for_every_applicable_site() -> None:
    intents = load_intents()
    pairs = sites_with_tariffs()
    for template in intents.templates:
        listed = [(site, tariff) for site, tariff in pairs if site.site_type in template.site_types]
        assert listed, f"{template.id}: no test site of an applicable type"
        for site, tariff in listed:
            assert applicable(template, site, tariff), f"{template.id} lists {site.site_type} but does not apply to {site.site_id}"
            gold_spec(template, site, tariff)  # raises if invalid for this site
            values = placeholder_values(template, site, tariff)
            for phrasing in template.phrasings:
                text = fill_text(phrasing, values)
                assert "{" not in text and "}" not in text


def test_requirements_exclude_sites_that_cannot_apply() -> None:
    intents = load_intents()
    clinic = SITES["nairobi_clinic"]
    tariff = build_tariff(clinic.tariff_template, DAY, seed=2026, stream=clinic.site_id)
    clinic_like_store = clinic.model_copy(update={"site_type": "commercial_cold_store"})
    assert not applicable(intents.template("no_export_midday"), clinic_like_store, tariff)  # export not allowed
    assert not applicable(intents.template("peak_only_discharge"), clinic_like_store, tariff)  # flat tariff: no peak


def test_derived_values_on_the_dev_sites() -> None:
    intents = load_intents()
    store, clinic = SITES["ahmedabad_cold_store"], SITES["nairobi_clinic"]

    def gold(template_id: str, site: Site):
        return gold_spec(intents.template(template_id), site, build_tariff(site.tariff_template, DAY, seed=2026, stream=site.site_id))

    assert gold("cover_critical_hours", store).soc_targets[0].min_soc_pct == 60.5
    assert gold("cover_critical_hours", clinic).soc_targets[0].min_soc_pct == 22.6
    assert gold("import_cap_afternoon", store).import_caps[0].max_kw == 30
    assert gold("import_cap_afternoon", clinic).import_caps[0].max_kw == 10
    assert gold("save_cycles", store).max_throughput_kwh == 100 and gold("save_cycles", clinic).max_throughput_kwh == 40
    peak = gold("peak_only_discharge", store).discharge_only_windows
    assert [(w.start_hour, w.end_hour) for w in peak] == [(6, 8), (17, 23)]


def test_placeholders_fill_with_the_number_format() -> None:
    assert [format_value(v) for v in (100.0, 30, 60.5, 0.8, 0.15, 8.0)] == ["100", "30", "60.5", "0.8", "0.15", "8"]
    store = SITES["ahmedabad_cold_store"]
    tariff = build_tariff(store.tariff_template, DAY, seed=2026, stream=store.site_id)
    template = load_intents().template("cover_critical_hours")
    text = fill_text(template.phrasings[0], placeholder_values(template, store, tariff))
    assert "run the cold room (8 kW) on its own for 6 hours" in text
    save = load_intents().template("save_cycles")
    assert "no more than 100 kWh" in fill_text(save.phrasings[0], placeholder_values(save, store, tariff))


def test_dev_goals_exclude_held_out_templates_and_test_goals_are_built_elsewhere() -> None:
    intents = load_intents()
    store = SITES["ahmedabad_cold_store"]
    tariff = build_tariff(store.tariff_template, DAY, seed=2026, stream=store.site_id)
    dev = goals_for(intents, store, tariff, "dev")
    assert dev and not any(g.held_out for g in dev) and {g.language for g in dev} == {"en"}
    assert len(dev) == 2 * sum(1 for t in intents.templates if not t.held_out)  # both phrasings
    with pytest.raises(ValueError, match="dev goals only"):
        goals_for(intents, store, tariff, "test")  # type: ignore[arg-type]
    held = english_goal(intents.template("outage_then_reserve"), store, tariff, 1)
    assert held.held_out and held.goal_id == "outage_then_reserve_p1" and held.gold.grid_outages


def test_multilingual_subset_loads_cleanly_with_the_same_gold_specs() -> None:
    intents = load_intents()
    goals = multilingual_goals(intents)  # raises if any TRANSLATION_PENDING remains
    raw = yaml.safe_load(INTENTS_PATH.read_text(encoding="utf-8"))
    assert all(PENDING not in str(value) for goal in raw["multilingual"] for value in goal.values())
    store = SITES["ahmedabad_cold_store"]
    tariff = build_tariff(store.tariff_template, DAY, seed=2026, stream=store.site_id)
    for goal in goals:
        gold = gold_spec(intents.template(goal.template), store, tariff)
        versions = [multilingual_goal(intents, goal, language, store, tariff) for language in ("en", *LANGUAGES)]
        assert [v.language for v in versions] == ["en", "es", "hi", "pt", "de"]
        assert all(v.gold == gold and v.template_id == goal.template for v in versions)
        assert all("{" not in v.text for v in versions)


def test_pending_translation_fails_loudly(tmp_path: Path) -> None:
    data = yaml.safe_load(INTENTS_PATH.read_text(encoding="utf-8"))
    data["multilingual"][1]["hi"] = PENDING
    data["multilingual"][4]["de"] = PENDING
    path = tmp_path / "intents.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    intents = load_intents(path)
    with pytest.raises(TranslationPendingError, match="2 multilingual goals still TRANSLATION_PENDING: ml_outage:hi, ml_import_cap:de"):
        multilingual_goals(intents)


def test_unknown_placeholders_are_rejected(tmp_path: Path) -> None:
    data = yaml.safe_load(INTENTS_PATH.read_text(encoding="utf-8"))
    data["templates"][0]["phrasings"][0] = "Keep the {mystery} low."
    path = tmp_path / "intents.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown placeholders"):
        load_intents(path)


def test_dev_episodes_never_use_held_out_sites_or_templates() -> None:
    from voltproof.bench.episode import dev_episodes

    episodes = dev_episodes(load_intents())
    held_out_sites = {s.site_id for s in SITES.values() if s.held_out}
    held_out_templates = {t.id for t in load_intents().templates if t.held_out}
    assert episodes and not any(e.site.site_id in held_out_sites for e in episodes)
    assert not any(e.goal.template_id in held_out_templates for e in episodes)
