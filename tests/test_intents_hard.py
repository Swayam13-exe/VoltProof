"""E2 hard goals: loading, derivations, requirements, gold specs per site type, the episode design.

Offline: synthetic sites (cold store, clinic, household) and the injected day builder of the runner
tests; no data file and no held-out site is touched.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

import pytest

from tests.fakes_data import TOU_TEMPLATE, make_site
from tests.test_runner import CLINIC, HOME, SITES, SPLITS, STORE, day_builder
from voltproof.bench import intents_hard as ih
from voltproof.bench.intents import LANGUAGES, PENDING, TranslationPendingError
from voltproof.data.sites import SITES as REAL_SITES
from voltproof.data.tariff import build_tariff

INTENTS = ih.load_hard_intents()


def filled(intents: ih.HardIntentSet) -> ih.HardIntentSet:
    goals = [g.model_copy(update={lang: f"{g.en} [{lang}]" for lang in LANGUAGES}) for g in intents.multilingual]
    return intents.model_copy(update={"multilingual": goals})


def tariff(site):
    return day_builder(site, date(2025, 1, 10))[1]


def test_the_set_has_ten_templates_with_two_phrasings_and_three_multilingual_goals() -> None:
    assert len(INTENTS.templates) == 10 and all(len(t.phrasings) == 2 for t in INTENTS.templates)
    assert [g.template for g in INTENTS.multilingual] == ["hard_negation", "hard_relative_time", "hard_tonight"]
    assert ih.pending_translations(INTENTS) == [f"{g.id}:{lang}" for g in INTENTS.multilingual for lang in LANGUAGES]
    assert all(g.en == INTENTS.template(g.template).phrasings[0] for g in INTENTS.multilingual)  # the source is phrasing 0


def test_every_gold_spec_is_valid_for_every_applicable_site_type() -> None:
    applicable = {}
    for site in (STORE, CLINIC, HOME):
        t = tariff(site)
        applicable[site.site_id] = [tpl.id for tpl in INTENTS.templates if ih.applicable(tpl, site, t)]
        for tpl in INTENTS.templates:
            if tpl.id in applicable[site.site_id]:
                assert ih.gold_spec(tpl, site, t) is not None
    assert "hard_compound3" not in applicable["syn_clinic"] and "hard_named_period" not in applicable["syn_clinic"]
    assert len(applicable["syn_clinic"]) == 8 and len(applicable["syn_store"]) == 10


def test_derivations_are_relative_to_the_site() -> None:
    t = tariff(STORE)
    kwh = ih.placeholder_values(INTENTS.template("hard_reserve_kwh"), STORE, t)["reserve_kwh"]
    assert kwh == round(0.4 * STORE.battery.capacity_kwh, 1)
    cap = ih.gold_spec(INTENTS.template("hard_cap_relative"), HOME, tariff(HOME)).import_caps[0].max_kw
    assert cap == round(0.5 * HOME.grid.import_max_kw, 1)
    odd = HOME.model_copy(update={"grid": HOME.grid.model_copy(update={"import_max_kw": 15.0})})
    assert ih.gold_spec(INTENTS.template("hard_cap_relative"), odd, tariff(odd)).import_caps[0].max_kw == 7.5  # no floor


def test_named_period_reserves_follow_the_day_tariff() -> None:
    site = make_site("syn_tou", tariff_template=TOU_TEMPLATE)
    t = build_tariff(TOU_TEMPLATE, date(2025, 1, 10), seed=2026, stream=site.site_id)
    spec = ih.gold_spec(INTENTS.template("hard_named_period"), site, t)
    windows = [(w.start_hour, w.end_hour) for w in spec.reserve_windows]
    assert windows == ih.hours_to_windows(t.periods["peak"]) and {w.min_soc_pct for w in spec.reserve_windows} == {40}


def test_the_impossible_goal_applies_only_where_the_first_hour_cannot_reach_it() -> None:
    template = INTENTS.template("hard_impossible")
    assert all(ih.applicable(template, s, tariff(s)) for s in (STORE, CLINIC, HOME))
    strong = STORE.model_copy(update={"battery": STORE.battery.model_copy(update={"p_charge_max_kw": STORE.battery.capacity_kwh, "eta_charge": 1.0})})
    assert ih.first_hour_max_soc_pct(strong) == pytest.approx(100.0) and not ih.applicable(template, strong, tariff(strong))
    for site in REAL_SITES.values():  # the reference sites cannot reach 100 % by the end of hour 0
        assert ih.first_hour_max_soc_pct(site) < 100.0


def test_phrasings_avoid_full_for_the_impossible_goal_and_the_distractor_names_hours() -> None:
    impossible = INTENTS.template("hard_impossible")
    assert all("100%" in p and "full" not in p.lower() for p in impossible.phrasings)
    distractor = INTENTS.template("hard_distractor")
    assert all(any(h in p for h in ("2 to 4 PM", "14:00 to 16:00")) for p in distractor.phrasings)
    assert ih.gold_spec(distractor, STORE, tariff(STORE)).grid_outages == []


def test_episode_design_counts_phrasings_and_multilingual_pairing() -> None:
    with pytest.raises(TranslationPendingError):
        ih.build_hard_episodes(INTENTS, SPLITS, SITES, day_builder)
    plan_only = ih.build_hard_episodes(INTENTS, SPLITS, SITES, day_builder, allow_pending=True)
    assert len(ih.pending_episodes(plan_only)) == 36
    episodes = ih.build_hard_episodes(filled(INTENTS), SPLITS, SITES, day_builder)
    english = [e for e in episodes if not e.goal.goal_id.startswith("ml_")]
    multilingual = [e for e in episodes if e.goal.goal_id.startswith("ml_")]
    assert len(english) == 2 * (10 + 8 + 10) and len(multilingual) == 3 * 5 * 3
    for site_id in SITES:  # each template uses both phrasings on each site
        for template in {e.goal.template_id for e in english if e.site.site_id == site_id}:
            used = {e.goal.goal_id[-1] for e in english if e.site.site_id == site_id and e.goal.template_id == template}
            assert used == {"0", "1"}
        ml_days = {e.day.date for e in multilingual if e.site.site_id == site_id}
        assert len(ml_days) == 1 and ml_days <= {e.day.date for e in english if e.site.site_id == site_id}
    assert Counter(e.goal.language for e in multilingual) == {lang: 9 for lang in ("en", *LANGUAGES)}
    assert ih.build_hard_episodes(filled(INTENTS), SPLITS, SITES, day_builder) == episodes  # deterministic


def test_hard_files_are_site_agnostic() -> None:
    text = ih.HARD_INTENTS_PATH.read_text(encoding="utf-8").lower()
    for site in REAL_SITES.values():
        assert site.site_id not in text and site.country.lower() not in text
    assert PENDING in ih.HARD_INTENTS_PATH.read_text(encoding="utf-8")


def test_unknown_derivation_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="pct"):
        ih.HardDerivation(derive="tariff_period_reserves", period="peak")
    with pytest.raises(ValueError, match="unreachable_first_hour"):
        ih.HardTemplate(id="hard_x", category="c", held_out=False, site_types=["household"], requires=["unreachable_first_hour"], phrasings=["a", "b"], gold={})
