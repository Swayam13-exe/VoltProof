"""Benchmark episodes: one site, one day, one goal. Assembly, LLM-arm dispatch and evaluation.

Evaluation follows `scripts/run_baselines.run_case` exactly: the selected plan is checked for
physics on the forecast (`plan_physics_valid`), executed on actual data facing the gold spec's
outages with the plan's export limit and protected SOC, and scored with `episode_result` against
A0 (savings) and the oracle LP bound (gap). A fallback run executes A0 itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from voltproof.agents.llm_direct import run_a2
from voltproof.agents.llm_simulate import run_a3
from voltproof.agents.loop import ChatLLM
from voltproof.agents.tools import ArmRun, EpisodeContext
from voltproof.agents.voltproof import run_a4
from voltproof.baselines.optimizer_arms import a6_plan
from voltproof.baselines.rule_based import a0_setpoints
from voltproof.bench.intents import GoalInstance, IntentSet, goals_for
from voltproof.core.executor import execute
from voltproof.core.metrics import episode_result, oracle_bound, spec_accuracy, trajectory_total_cost
from voltproof.core.verifier import verify
from voltproof.data import GLOBAL_SEED
from voltproof.data.days import build_day, day_tariff, dev_days, load_splits
from voltproof.data.sites import SITES
from voltproof.data.weather import processed_csv_path, read_processed_csv
from voltproof.models import DayData, EpisodeResult, Site, Tariff

LLM_ARMS: Mapping[str, Callable[..., ArmRun]] = {"A2": run_a2, "A3": run_a3, "A4": run_a4}


@dataclass(frozen=True)
class Episode:
    episode_id: str
    site: Site
    day: DayData
    tariff: Tariff
    goal: GoalInstance

    def context(self) -> EpisodeContext:
        return EpisodeContext(self.episode_id, self.site, self.day, self.tariff, self.goal.text)


def dev_episodes(intents: IntentSet, splits: Mapping[str, Any] | None = None) -> list[Episode]:
    """Every dev episode: dev days (never a held-out site) x non-held-out templates x 2 phrasings."""
    cases = dev_days(splits if splits is not None else load_splits(), list(SITES.values()))
    frames: dict[str, Any] = {}
    episodes = []
    for site, day_date in cases:
        if site.held_out:
            raise ValueError(f"held-out site {site.site_id} in dev episodes")
        frame = frames.setdefault(site.site_id, read_processed_csv(processed_csv_path(site.site_id), site.timezone))
        day, tariff = build_day(site, day_date, frame), day_tariff(site, day_date, frame)
        for goal in goals_for(intents, site, tariff, "dev"):
            if goal.held_out:
                raise ValueError(f"held-out template {goal.template_id} in dev episodes")
            episodes.append(Episode(episode_id(site, day_date, goal), site, day, tariff, goal))
    return episodes


def episode_id(site: Site, day_date: date, goal: GoalInstance) -> str:
    return f"{site.site_id}_{day_date.isoformat()}_{goal.goal_id}"


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


def evaluate(episode: Episode, run: ArmRun) -> EpisodeResult:
    site, day, gold = episode.site, episode.day, episode.goal.gold
    a0_execution = execute(a0_setpoints(day, site, "actual"), day, site, gold.grid_outages)
    a0_cost = trajectory_total_cost(a0_execution, gold, day, site)
    bound = oracle_bound(a6_plan(day, site, gold))
    if run.battery_kw is None:
        execution = a0_execution
        check = verify(a0_setpoints(day, site, "forecast"), day, site, gold, physics_only=True)
    else:
        check = verify(run.battery_kw, day, site, gold, physics_only=True, export_limit_kw=run.export_limit_kw)
        execution = execute(
            run.battery_kw,
            day,
            site,
            gold.grid_outages,
            run.export_limit_kw,
            plan_basis="forecast",
            protected_soc_kwh=run.protected_soc_kwh,
            import_limit_kw=run.import_limit_kw,
        )
    result = episode_result(
        arm=run.arm,
        spec_name=episode.goal.template_id,
        day=day,
        site=site,
        gold_spec=gold,
        execution=execution,
        plan_check=check,
        a0_money=a0_cost.net_money,
        a0_total=a0_cost.total,
        oracle_total=bound,
    )
    precision, recall = spec_accuracy(run.spec, gold) if run.spec is not None else (None, None)
    totals = run.totals
    return result.model_copy(
        update={
            "episode_id": episode.episode_id,
            "language": episode.goal.language,
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
            "import_cap_hours": len(execution.import_cap_hours),
            **run.metrics,
        }
    )
