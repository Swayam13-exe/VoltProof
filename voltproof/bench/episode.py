"""Benchmark episodes: one site, one day, one goal. Assembly, LLM-arm dispatch and evaluation.

Evaluation follows `scripts/run_baselines.run_case` exactly: the selected plan is checked for
physics on the forecast (`plan_physics_valid`), executed on actual data facing the gold spec's
outages with the plan's export limit and protected SOC, and scored with `episode_result` against
A0 (savings) and the oracle LP bound (gap). A fallback run executes A0 itself.

The test set (SPEC §6.11) is fixed by seed 2026:
- English: every test day x every template applicable to that site (held-out templates included) x
  one phrasing; phrasing = (site offset + day index + template index) % 2, so each template uses both
  phrasings on each site
- multilingual: the 6 goals x (en, es, hi, pt, de) on one seeded test day per site
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Mapping, Sequence

import zlib

import numpy as np

from voltproof.agents.llm_direct import run_a2
from voltproof.agents.llm_simulate import run_a3
from voltproof.agents.loop import ChatLLM
from voltproof.agents.tools import ArmRun, EpisodeContext
from voltproof.agents.voltproof import run_a4
from voltproof.baselines.optimizer_arms import a1_plan, a5_plan, a6_plan
from voltproof.baselines.rule_based import a0_setpoints
from voltproof.bench.intents import LANGUAGES, GoalInstance, IntentSet, applicable, english_goal, goals_for, multilingual_goal, multilingual_goals
from voltproof.core.executor import execute
from voltproof.core.metrics import episode_result, oracle_bound, spec_accuracy, trajectory_total_cost
from voltproof.core.verifier import verify
from voltproof.data import GLOBAL_SEED
from voltproof.data.days import build_day, day_tariff, dev_days, load_splits, split_dates
from voltproof.data.sites import SITES
from voltproof.data.weather import processed_csv_path, read_processed_csv
from voltproof.models import DayData, EpisodeResult, ExecutionResult, Plan, Site, Tariff, VerifyResult

LLM_ARMS: Mapping[str, Callable[..., ArmRun]] = {"A2": run_a2, "A3": run_a3, "A4": run_a4}
BASELINE_ARMS = ("A0", "A1", "A5", "A6")
DayBuilder = Callable[[Site, date], tuple[DayData, Tariff]]


@dataclass(frozen=True)
class Episode:
    episode_id: str
    site: Site
    day: DayData
    tariff: Tariff
    goal: GoalInstance

    def context(self) -> EpisodeContext:
        return EpisodeContext(self.episode_id, self.site, self.day, self.tariff, self.goal.text)


def processed_day_builder() -> DayBuilder:
    """DayData and tariff from the committed processed weather CSVs (one frame per site, loaded once)."""
    frames: dict[str, Any] = {}

    def build(site: Site, day_date: date) -> tuple[DayData, Tariff]:
        frame = frames.setdefault(site.site_id, read_processed_csv(processed_csv_path(site.site_id), site.timezone))
        return build_day(site, day_date, frame), day_tariff(site, day_date, frame)

    return build


def dev_episodes(intents: IntentSet, splits: Mapping[str, Any] | None = None, day_builder: DayBuilder | None = None) -> list[Episode]:
    """Every dev episode: dev days (never a held-out site) x non-held-out templates x 2 phrasings."""
    cases = dev_days(splits if splits is not None else load_splits(), list(SITES.values()))
    build = day_builder or processed_day_builder()
    episodes = []
    for site, day_date in cases:
        if site.held_out:
            raise ValueError(f"held-out site {site.site_id} in dev episodes")
        day, tariff = build(site, day_date)
        for goal in goals_for(intents, site, tariff, "dev"):
            if goal.held_out:
                raise ValueError(f"held-out template {goal.template_id} in dev episodes")
            episodes.append(Episode(episode_id(site, day_date, goal), site, day, tariff, goal))
    return episodes


def episode_id(site: Site, day_date: date, goal: GoalInstance) -> str:
    return f"{site.site_id}_{day_date.isoformat()}_{goal.goal_id}"


def _site_rng(seed: int, purpose: str, site_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(f"{purpose}:{site_id}".encode("utf-8"))])


def build_test_episodes(
    intents: IntentSet,
    splits: Mapping[str, Any],
    sites: Mapping[str, Site],
    day_builder: DayBuilder | None = None,
    seed: int = GLOBAL_SEED,
) -> list[Episode]:
    """The frozen test set (module docstring): English set, then the multilingual set, each in
    (site, date, template) order. Held-out sites and templates are included by design."""
    build = day_builder or processed_day_builder()
    site_ids = sorted({entry["site_id"] for entry in splits["test"]})
    days = {s: [(d, *build(sites[s], d)) for d in split_dates(splits, "test", s)] for s in site_ids}
    english: list[Episode] = []
    multilingual: list[Episode] = []
    goals = multilingual_goals(intents)  # refuses pending translations
    for site_id in site_ids:
        site = sites[site_id]
        offset = int(_site_rng(seed, "phrasing", site_id).integers(2))
        for d, (day_date, day, tariff) in enumerate(days[site_id]):
            templates = [t for t in intents.templates if applicable(t, site, tariff)]
            for j, template in enumerate(templates):
                goal = english_goal(template, site, tariff, (offset + d + j) % 2)
                english.append(Episode(episode_id(site, day_date, goal), site, day, tariff, goal))
        day_date, day, tariff = days[site_id][int(_site_rng(seed, "multilingual", site_id).integers(len(days[site_id])))]
        for ml in goals:
            if not applicable(intents.template(ml.template), site, tariff):
                continue
            for language in ("en", *LANGUAGES):
                goal = multilingual_goal(intents, ml, language, site, tariff)
                multilingual.append(Episode(episode_id(site, day_date, goal), site, day, tariff, goal))
    episodes = english + multilingual
    ids = [e.episode_id for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode ids in the test set")
    return episodes


def select_per_site(episodes: Sequence[Episode], n: int, seed: int = GLOBAL_SEED) -> list[Episode]:
    """n episodes per site in a seeded order: never a repeated template within a site, and different
    days while possible (the pilot's mixed sample)."""
    chosen: list[Episode] = []
    for site_id in sorted({e.site.site_id for e in episodes}):
        pool = [e for e in episodes if e.site.site_id == site_id]
        order = [pool[i] for i in _site_rng(seed, "pilot", site_id).permutation(len(pool))]
        picked: list[Episode] = []
        for distinct_days in (True, False):
            for episode in order:
                if len(picked) == n:
                    break
                if any(p.goal.template_id == episode.goal.template_id for p in picked):
                    continue
                if distinct_days and any(p.day.date == episode.day.date for p in picked):
                    continue
                picked.append(episode)
        chosen.extend(picked)
    return chosen


def select_episodes(episodes: Sequence[Episode], n: int, seed: int = GLOBAL_SEED) -> list[Episode]:
    """n episodes in a seeded order, alternating sites and never repeating a template while
    possible."""
    order = [episodes[i] for i in np.random.default_rng(seed).permutation(len(episodes))]
    chosen: list[Episode] = []
    for strict in (True, False):
        for episode in order:
            if len(chosen) == n:
                return chosen
            if any(c.episode_id == episode.episode_id for c in chosen):
                continue
            if strict and chosen:
                if episode.site.site_id == chosen[-1].site.site_id or any(c.goal.template_id == episode.goal.template_id for c in chosen):
                    continue
            chosen.append(episode)
    return chosen


def run_llm_arm(arm: str, llm: ChatLLM, episode: Episode, tags: dict[str, Any] | None = None) -> ArmRun:
    if arm not in LLM_ARMS:
        raise ValueError(f"unknown LLM arm {arm!r}; known: {', '.join(LLM_ARMS)}")
    return LLM_ARMS[arm](llm, episode.context(), tags=tags)


@dataclass(frozen=True)
class References:
    """A0's execution and costs and the oracle bound: the denominators every arm is scored against."""

    a0_execution: ExecutionResult
    a0_check: VerifyResult
    a0_money: float
    a0_total: float
    oracle_total: float
    a6: Plan


def references(episode: Episode) -> References:
    site, day, gold = episode.site, episode.day, episode.goal.gold
    a0_execution = execute(a0_setpoints(day, site, "actual"), day, site, gold.grid_outages)
    a0_cost = trajectory_total_cost(a0_execution, gold, day, site)
    a0_check = verify(a0_setpoints(day, site, "forecast"), day, site, gold, physics_only=True)
    a6 = a6_plan(day, site, gold)
    return References(a0_execution, a0_check, a0_cost.net_money, a0_cost.total, oracle_bound(a6), a6)


def _score(episode: Episode, arm: str, refs: References, execution: ExecutionResult, check: VerifyResult, solve_time_s: float | None = None) -> EpisodeResult:
    result = episode_result(
        arm=arm,
        spec_name=episode.goal.template_id,
        day=episode.day,
        site=episode.site,
        gold_spec=episode.goal.gold,
        execution=execution,
        plan_check=check,
        a0_money=refs.a0_money,
        a0_total=refs.a0_total,
        oracle_total=refs.oracle_total,
        solve_time_s=solve_time_s,
    )
    return result.model_copy(
        update={"episode_id": episode.episode_id, "language": episode.goal.language, "import_cap_hours": len(execution.import_cap_hours)}
    )


def baseline_results(episode: Episode) -> list[EpisodeResult]:
    """A0, A1, A5 and A6 for one episode, evaluated exactly as `scripts/run_baselines.run_case` does."""
    site, day, gold = episode.site, episode.day, episode.goal.gold
    refs = references(episode)
    rows = [_score(episode, "A0", refs, refs.a0_execution, refs.a0_check).model_copy(update={"plan_source": "baseline"})]
    for arm, plan in (("A1", a1_plan(day, site)), ("A5", a5_plan(day, site, gold)), ("A6", refs.a6)):
        check = verify(plan.battery_kw, day, site, gold, physics_only=True, export_limit_kw=plan.export_limit_kw)
        execution = execute(
            plan.battery_kw,
            day,
            site,
            gold.grid_outages,
            export_limit_kw=plan.export_limit_kw,
            plan_basis=plan.use,
            protected_soc_kwh=plan.protected_soc_kwh,
            import_limit_kw=plan.import_limit_kw,
        )
        rows.append(_score(episode, arm, refs, execution, check, plan.solve_time_s).model_copy(update={"plan_source": "baseline"}))
    return rows


def evaluate(episode: Episode, run: ArmRun, *, infra_retries: int | None = None) -> EpisodeResult:
    site, day, gold = episode.site, episode.day, episode.goal.gold
    refs = references(episode)
    a0_execution = refs.a0_execution
    if run.battery_kw is None:
        execution = a0_execution
        check = refs.a0_check
    else:
        check = verify(run.battery_kw, day, site, gold, physics_only=True, export_limit_kw=run.export_limit_kw)
        execution = execute(
            run.battery_kw,
            day,
            site,
            gold.grid_outages,
            export_limit_kw=run.export_limit_kw,
            plan_basis="forecast",
            protected_soc_kwh=run.protected_soc_kwh,
            import_limit_kw=run.import_limit_kw,
        )
    result = _score(episode, run.arm, refs, execution, check)
    precision, recall = spec_accuracy(run.spec, gold) if run.spec is not None else (None, None)
    totals = run.totals
    return result.model_copy(
        update={
            "spec_precision": precision,
            "spec_recall": recall,
            "tokens_in": totals.tokens_in,
            "tokens_out": totals.tokens_out,
            "tokens_cached": totals.tokens_cached,
            "cost_usd": totals.cost_usd,
            "latency_s": totals.latency_s,
            "tool_calls": run.tool_calls,
            "repair_turns": run.repair_turns,
            "fallback_used": run.fallback_used,
            "stop_reason": run.stop_reason,
            "plan_source": run.plan_source,
            "infra_retries": infra_retries,
            **run.metrics,
        }
    )
