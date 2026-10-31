"""E4 reasoning-effort sweep (SPEC §6.15): A4 at low and high effort on 60 fixed episodes.

Only the A4 agent's own turns change effort; every other call (the reviewer, purpose "review") stays
at medium, the frozen setting, so exactly one thing varies:
- `EffortLLM` routes purpose "agent" to a client built with the swept effort and everything else to
  the medium client (two `LLMClient`s, same cache, ledger and budget guard; run at concurrency 1)
- `CappedLLM` enforces the E4 cap ($5) on top of the global cap: before each call it sums the ledger
  lines of every E4 run (tags.run starting with "e4-", probes included) and refuses with
  `BudgetExceeded` when that plus the call's worst case would pass the cap

Episodes, fixed before any spend (seed 2026): 30 from the frozen main test set (10 per site: 2
multilingual in different languages, then English, every template distinct within a site) + 30 from
the E2 hard set (3 per hard template, at different sites where it applies; the English-set episodes
that duplicate a multilingual source request are left out). Medium results are the stored A4 rows of
the frozen test run and the E2 run, reused at $0.

PowerShell, from the repo root:
    python -m voltproof.bench.effort_sweep dry-run                     # cache keys of the first agent call
    python -m voltproof.bench.effort_sweep probe-summary --run runs\\e4-probe-low-...
    python -m voltproof.bench.effort_sweep project --low runs\\e4-probe-low-... --high runs\\e4-probe-high-...
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from voltproof.agents.loop import ChatLLM
from voltproof.agents.voltproof import A4Session
from voltproof.bench.episode import DayBuilder, Episode, build_test_episodes
from voltproof.bench.intents import IntentSet, load_intents
from voltproof.bench.intents_hard import HardIntentSet, build_hard_episodes, load_hard_intents
from voltproof.config import PROJECT_ROOT, Settings
from voltproof.data import GLOBAL_SEED
from voltproof.llm.client import BudgetExceeded, LLMClient, LLMResponse, cache_key, worst_case_cost_usd
from voltproof.llm.prompts import A4_SYSTEM, goal_message
from voltproof.models import Site

E4_CAP_USD = 5.0
E4_RUN_PREFIX = "e4-"
EFFORTS = ("low", "high")
MEDIUM = "medium"
AGENT_PURPOSE = "agent"
MAIN_PER_SITE = 10
MAIN_MULTILINGUAL_PER_SITE = 2
HARD_PER_TEMPLATE = 3
MAIN_RUN = PROJECT_ROOT / "runs" / "test-20261030-181315"
HARD_RUN = PROJECT_ROOT / "runs" / "e2-20261031-130234"
PILOT_RUN = PROJECT_ROOT / "runs" / "pilot-20261030-170121"


# --- routing and the cap ---------------------------------------------------------------------------


class EffortLLM:
    """A ChatLLM that sends the agent's own turns (purpose "agent", repairs included) to `agent` and
    every other call (the reviewer) to `other`."""

    def __init__(self, agent: ChatLLM, other: ChatLLM) -> None:
        self.agent = agent
        self.other = other

    def chat(self, messages: Sequence[Mapping[str, Any]], tools: Any = None, tool_choice: Any = None, response_format: Any = None, *, purpose: str, tags: Mapping[str, Any] | None = None) -> LLMResponse:
        client = self.agent if purpose == AGENT_PURPOSE else self.other
        return client.chat(messages, tools, tool_choice, response_format, purpose=purpose, tags=tags)


def effort_llm(settings: Settings, effort: str) -> EffortLLM:
    if effort not in EFFORTS:
        raise ValueError(f"E4 sweeps only {EFFORTS}, not {effort!r}")
    return EffortLLM(LLMClient(replace(settings, llm_reasoning_effort=effort)), LLMClient(replace(settings, llm_reasoning_effort=MEDIUM)))


def e4_spent_usd(ledger_path: Path, prefix: str = E4_RUN_PREFIX) -> float:
    """What every E4 run (probes included) has cost so far, from the ledger."""
    if not ledger_path.is_file():
        return 0.0
    total = 0.0
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if str((entry.get("tags") or {}).get("run", "")).startswith(prefix):
            total += float(entry.get("cost_usd") or 0.0)
    return total


class CappedLLM:
    """Refuses a call when E4's spend so far plus the call's worst case would pass the cap."""

    def __init__(self, inner: ChatLLM, settings: Settings, cap_usd: float = E4_CAP_USD) -> None:
        if cap_usd > E4_CAP_USD:
            raise ValueError(f"the E4 cap is ${E4_CAP_USD:.2f}; it can only be lowered")
        self.inner = inner
        self.settings = settings
        self.cap_usd = cap_usd
        self._lock = threading.Lock()

    def chat(self, messages: Sequence[Mapping[str, Any]], tools: Any = None, tool_choice: Any = None, response_format: Any = None, *, purpose: str, tags: Mapping[str, Any] | None = None) -> LLMResponse:
        with self._lock:
            spent = e4_spent_usd(self.settings.ledger_path)
            worst = worst_case_cost_usd(messages, tools, response_format, self.settings)
            if spent + worst > self.cap_usd:
                raise BudgetExceeded(f"E4 cap ${self.cap_usd:.2f} would be exceeded: E4 spent ${spent:.6f}, this call up to ${worst:.6f}")
        return self.inner.chat(messages, tools, tool_choice, response_format, purpose=purpose, tags=tags)


# --- the 60 episodes -----------------------------------------------------------------------------------


def _rng(seed: int, purpose: str, key: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(f"{purpose}:{key}".encode("utf-8"))])


def is_multilingual(episode: Episode) -> bool:
    return episode.goal.goal_id.startswith("ml_")


def select_main(episodes: Sequence[Episode], seed: int = GLOBAL_SEED) -> list[Episode]:
    """10 per site: 2 multilingual (different non-English languages and templates), then English with
    a template not yet picked at that site."""
    chosen: list[Episode] = []
    for site_id in sorted({e.site.site_id for e in episodes}):
        pool = [e for e in episodes if e.site.site_id == site_id]
        order = [pool[int(i)] for i in _rng(seed, "e4_main", site_id).permutation(len(pool))]
        picked: list[Episode] = []
        for episode in order:
            if sum(map(is_multilingual, picked)) == MAIN_MULTILINGUAL_PER_SITE:
                break
            if is_multilingual(episode) and episode.goal.language != "en" and all(
                p.goal.language != episode.goal.language and p.goal.template_id != episode.goal.template_id for p in picked
            ):
                picked.append(episode)
        for episode in order:
            if len(picked) == MAIN_PER_SITE:
                break
            if not is_multilingual(episode) and all(p.goal.template_id != episode.goal.template_id for p in picked):
                picked.append(episode)
        chosen.extend(picked)
    return chosen


def duplicate_requests(episodes: Sequence[Episode]) -> set[str]:
    """English-set episodes whose goal text, site and day equal a multilingual source episode's
    (they send identical requests; see the E2 report)."""
    sources = {(e.site.site_id, e.day.date, e.goal.text) for e in episodes if is_multilingual(e) and e.goal.language == "en"}
    return {e.episode_id for e in episodes if not is_multilingual(e) and (e.site.site_id, e.day.date, e.goal.text) in sources}


def select_hard(episodes: Sequence[Episode], seed: int = GLOBAL_SEED) -> list[Episode]:
    """3 per hard template, at different sites while the template allows; duplicates left out."""
    duplicates = duplicate_requests(episodes)
    pool = [e for e in episodes if e.episode_id not in duplicates]
    chosen: list[Episode] = []
    for template in dict.fromkeys(e.goal.template_id for e in pool):
        candidates = [e for e in pool if e.goal.template_id == template]
        order = [candidates[int(i)] for i in _rng(seed, "e4_hard", template).permutation(len(candidates))]
        picked: list[Episode] = []
        for distinct_sites in (True, False):
            for episode in order:
                if len(picked) == HARD_PER_TEMPLATE:
                    break
                if episode in picked or (distinct_sites and any(p.site.site_id == episode.site.site_id for p in picked)):
                    continue
                picked.append(episode)
        chosen.extend(picked)
    return chosen


def e4_episodes(
    intents: IntentSet | None,
    hard_intents: HardIntentSet | None,
    splits: Mapping[str, Any],
    sites: Mapping[str, Site],
    day_builder: DayBuilder | None = None,
    seed: int = GLOBAL_SEED,
) -> list[Episode]:
    """The E4 set: 30 main + 30 hard, in that order."""
    main = build_test_episodes(intents or load_intents(), splits, sites, day_builder, seed)
    hard = build_hard_episodes(hard_intents or load_hard_intents(), splits, sites, day_builder, seed)
    episodes = select_main(main, seed) + select_hard(hard, seed)
    ids = [e.episode_id for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode ids in the E4 set")
    return episodes


def subset(episode_id: str) -> str:
    return "hard" if "_hard_" in episode_id else "main"


# --- dry run: no cache hits at the swept efforts -----------------------------------------------------------


def first_call_key(episode: Episode, settings: Settings, effort: str) -> str:
    """The cache key of A4's first agent call, built exactly as run_a4 builds that request."""
    session = A4Session(None, episode.context(), {})  # type: ignore[arg-type]  # tools only; no call is made
    schemas = [tool.schema() for tool in session.tools()]
    messages = [{"role": "system", "content": A4_SYSTEM}, {"role": "user", "content": goal_message(episode.goal.text)}]
    return cache_key(
        model=settings.llm_model,
        messages=messages,
        tools=schemas,
        tool_choice="auto",
        response_format=None,
        reasoning_effort=effort,
        max_tokens=settings.llm_max_output_tokens,
    )


def dry_run(episodes: Sequence[Episode], settings: Settings) -> dict[str, int]:
    """Cached first agent calls per effort: expected 0 at low/high and all at medium (which shows the
    keys are computed as the frozen runs computed them)."""
    counts = {}
    for effort in (*EFFORTS, MEDIUM):
        keys = [first_call_key(e, settings, effort) for e in episodes]
        counts[effort] = sum((settings.cache_dir / k[:2] / f"{k}.json").is_file() for k in keys)
    return counts


# --- probes and projection ----------------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trace_stats(trace: Path) -> dict[str, Any]:
    """Agent and reviewer calls of one A4 trace: reasoning tokens, finish reasons, cache hits."""
    events = read_rows(trace)
    calls = [e for e in events if e["kind"] == "llm_call"]
    agent = [c for c in calls if c["name"] == AGENT_PURPOSE]
    return {
        "agent_calls": len(agent),
        "review_calls": len(calls) - len(agent),
        "agent_reasoning_tokens": sum(c["data"].get("tokens_reasoning") or 0 for c in agent),
        "reasoning_tokens": sum(c["data"].get("tokens_reasoning") or 0 for c in calls),
        "length_stops": sum(c["data"].get("finish_reason") == "length" for c in calls),
        "finish_reasons": sorted({str(c["data"].get("finish_reason")) for c in calls}),
        "cache_hits": sum(bool(c["data"].get("cache_hit")) for c in calls),
    }


def ledger_efforts(ledger_path: Path, run_id: str) -> dict[str, list[str]]:
    """The reasoning effort the client sent, per purpose, for one run (the routing proof)."""
    out: dict[str, set[str]] = {}
    for entry in read_rows(ledger_path):
        if (entry.get("tags") or {}).get("run") == run_id and entry.get("status") == "ok":
            out.setdefault(entry["purpose"], set()).add(str(entry.get("reasoning_effort")))
    return {purpose: sorted(efforts) for purpose, efforts in out.items()}


def probe_summary(run_dir: Path, ledger_path: Path) -> dict[str, Any]:
    rows = [r for r in read_rows(run_dir / "episodes.jsonl") if r["arm"] == "A4"]
    if len(rows) != 1:
        raise ValueError(f"{run_dir} should hold exactly one A4 result")
    row = rows[0]
    stats = trace_stats(run_dir / "traces" / f"A4__{row['episode_id']}.jsonl")
    accepted = not str(row.get("stop_reason") or "").startswith("llm_error") and not row.get("fallback_used")
    return {
        "run_id": run_dir.name,
        "episode_id": row["episode_id"],
        "accepted": accepted,
        "stop_reason": row.get("stop_reason"),
        "plan_source": row.get("plan_source"),
        "cost_usd": row.get("cost_usd") or 0.0,
        "latency_s": row.get("latency_s"),
        "gold_satisfied": row.get("gold_satisfied"),
        "efforts_sent": ledger_efforts(ledger_path, run_dir.name),
        **stats,
    }


def medium_rows(episode_ids: Sequence[str], main_run: Path = MAIN_RUN, hard_run: Path = HARD_RUN) -> dict[str, dict[str, Any]]:
    """The stored A4 (medium) rows of these episodes."""
    rows = {r["episode_id"]: r for run in (main_run, hard_run) for r in read_rows(run / "episodes.jsonl") if r["arm"] == "A4"}
    missing = [e for e in episode_ids if e not in rows]
    if missing:
        raise ValueError(f"no medium A4 result for {missing[:3]}")
    return {e: rows[e] for e in episode_ids}


def projection(probe: Mapping[str, Any], probe_medium_usd: float, medium_total_usd: float, n_episodes: int) -> dict[str, float]:
    """Two estimates for one effort over the E4 set; the larger is used."""
    per_episode = probe["cost_usd"] * n_episodes
    scaled = medium_total_usd * probe["cost_usd"] / probe_medium_usd if probe_medium_usd > 0 else float("inf")
    return {"probe_x_n": per_episode, "scaled_from_medium": scaled, "projected": max(per_episode, scaled)}


# --- the E4 report ---------------------------------------------------------------------------------------------

LEVELS = ("low", MEDIUM, "high")
SUBSETS = ("main", "hard", "all")
IMPOSSIBLE_TEMPLATE = "hard_impossible"


def effort_rows(plan_ids: Sequence[str], low_run: Path, high_run: Path, main_run: Path = MAIN_RUN, hard_run: Path = HARD_RUN) -> dict[str, dict[str, dict[str, Any]]]:
    """effort → episode_id → A4 row, with the trace statistics merged in (`_trace`)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    medium = medium_rows(plan_ids, main_run, hard_run)
    sources = {MEDIUM: medium}
    for effort, run in (("low", low_run), ("high", high_run)):
        rows = {r["episode_id"]: r for r in read_rows(run / "episodes.jsonl") if r["arm"] == "A4"}
        missing = [e for e in plan_ids if e not in rows]
        if missing:
            raise ValueError(f"{run.name} has no result for {len(missing)} plan episodes, e.g. {missing[:2]}")
        sources[effort] = {e: rows[e] for e in plan_ids}
    traces = {"low": low_run, "high": high_run}
    for effort in LEVELS:
        out[effort] = {}
        for eid, row in sources[effort].items():
            run = traces.get(effort) or (hard_run if subset(eid) == "hard" else main_run)
            out[effort][eid] = {**row, "_trace": trace_stats(run / "traces" / f"A4__{eid}.jsonl")}
    return out


def _exact(row: Mapping[str, Any]) -> bool:
    return row.get("spec_precision") == 1 and row.get("spec_recall") == 1


def _ids(plan_ids: Sequence[str], which: str) -> list[str]:
    return [e for e in plan_ids if which == "all" or subset(e) == which]


def effort_table(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], plan_ids: Sequence[str], usd: Any) -> list[dict[str, Any]]:
    """Per subset and effort: the SPEC §6.15 E4 measures. `usd(row)` is the list-price token cost."""
    out = []
    for which in SUBSETS:
        ids = _ids(plan_ids, which)
        for effort in LEVELS:
            mine = [rows[effort][e] for e in ids]
            n = len(mine)
            latency = [r["latency_s"] for r in mine if r.get("latency_s") is not None]
            rounds = [r.get("repair_rounds") or 0 for r in mine]
            out.append(
                {
                    "subset": which,
                    "effort": effort,
                    "episodes": n,
                    "goal satisfied": f"{100 * sum(bool(r['gold_satisfied']) for r in mine) / n:.1f}%",
                    "spec exact match": f"{100 * sum(map(_exact, mine)) / n:.1f}%",
                    "plan physics valid": f"{100 * sum(bool(r['plan_physics_valid']) for r in mine) / n:.1f}%",
                    "repair rounds mean (0/1/2)": f"{np.mean(rounds):.2f} ({rounds.count(0)}/{rounds.count(1)}/{rounds.count(2)})",
                    "escalations / fallbacks": f"{sum(bool(r.get('escalated')) for r in mine)} / {sum(bool(r.get('fallback_used')) for r in mine)}",
                    "ended at the turn cap": sum(r.get("stop_reason") == "max_turns" for r in mine),
                    "billed USD / episode": f"{np.mean([r.get('cost_usd') or 0.0 for r in mine]):.4f}",
                    "token USD / episode": f"{np.mean([usd(r) for r in mine]):.4f}",
                    "median latency s": f"{np.median(latency):.1f}" if latency else "-",
                    "p90 latency s": f"{np.percentile(latency, 90):.1f}" if latency else "-",
                    "agent reasoning tokens / episode": f"{np.mean([r['_trace']['agent_reasoning_tokens'] for r in mine]):.0f}",
                    "length stops": sum(r["_trace"]["length_stops"] for r in mine),
                    "cached calls": sum(r["_trace"]["cache_hits"] for r in mine),
                }
            )
    return out


def effort_bootstrap(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], plan_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Paired bootstrap 95% CIs (10,000, seed 2026) of low − medium and high − medium."""
    from voltproof.bench.report import bootstrap_mean_ci

    out = []
    for measure, value in (("goal satisfied", lambda r: float(bool(r["gold_satisfied"]))), ("spec exact match", lambda r: float(_exact(r)))):
        for effort in EFFORTS:
            row: dict[str, Any] = {"measure": measure, "comparison": f"{effort} - medium"}
            for which in SUBSETS:
                ids = _ids(plan_ids, which)
                diffs = [value(rows[effort][e]) - value(rows[MEDIUM][e]) for e in ids]
                mean, lo, hi = bootstrap_mean_ci(diffs)
                changed = sum(d != 0 for d in diffs)
                row[which] = f"{100 * mean:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}] ({changed} of {len(ids)} episodes differ)"
            out.append(row)
    return out


def impossible_rows(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], plan_ids: Sequence[str], all_targets: Mapping[str, tuple[list[Any], float]]) -> list[dict[str, Any]]:
    """hard_impossible: is the shortfall stated in the explanation (the E2 report's check: every
    shortfall of the gold spec's plan on the forecast quoted within 5 % or as a whole number)?"""
    from voltproof.agents.gate import extract_numbers, shortfall_stated

    targets = {e: t for e, t in all_targets.items() if e in plan_ids}
    out = []
    for effort in LEVELS:
        stated = 0
        mine = [rows[effort][e] for e in targets]
        for eid, (shortfalls, capacity) in targets.items():
            numbers = extract_numbers(rows[effort][eid].get("explanation") or "")
            stated += bool(shortfalls) and all(shortfall_stated(s, numbers, capacity) for s in shortfalls)
        out.append(
            {
                "effort": effort,
                "episodes": len(targets),
                "shortfall stated": f"{stated}/{len(targets)}",
                "spec exact match": f"{sum(map(_exact, mine))}/{len(mine)}",
                "no explanation (no accepted finalize)": sum(not r.get("explanation") for r in mine),
            }
        )
    return out


def effort_figure(rows: Mapping[str, Mapping[str, Mapping[str, Any]]], plan_ids: Sequence[str], usd: Any, path: Path, caption: str) -> Path:
    from voltproof.bench.report import INK, INK_2, SERIES, SURFACE, _figure, _save, bootstrap_mean_ci

    fig, ax = _figure(6.0, 3.6)
    for k, effort in enumerate(LEVELS):
        mine = [rows[effort][e] for e in plan_ids]
        mean, lo, hi = bootstrap_mean_ci([float(bool(r["gold_satisfied"])) for r in mine])
        cost = float(np.mean([usd(r) for r in mine]))
        color = SERIES[0] if effort == MEDIUM else SERIES[1]
        ax.errorbar([cost], [100 * mean], yerr=[[100 * (mean - lo)], [100 * (hi - mean)]], fmt="o", color=color, ecolor=INK, elinewidth=1.2, capsize=3, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        dx, dy = {"low": (-8, 6), MEDIUM: (8, -12)}.get(effort, (8, 6))  # low sits left of medium: label it on its left
        ax.annotate(f"{effort}{' (frozen)' if effort == MEDIUM else ''}: {100 * mean:.0f}%", (cost, 100 * mean), textcoords="offset points", xytext=(dx, dy), ha="left" if dx > 0 else "right", fontsize=8, color=INK)
    ax.set_xlabel("A4 LLM USD per episode (list-price tokens)", color=INK_2, fontsize=9)
    ax.set_ylabel("goal satisfied (%)", color=INK_2, fontsize=9)
    ax.set_title(f"E4: agent reasoning effort, {len(plan_ids)} episodes (95% bootstrap CI)", color=INK, fontsize=10, loc="left")
    ax.margins(x=0.35)
    return _save(fig, path, caption)


def e4_report(
    low_run: Path,
    high_run: Path,
    out_dir: Path,
    settings: Settings,
    *,
    main_run: Path = MAIN_RUN,
    hard_run: Path = HARD_RUN,
    shortfalls: Any = None,
) -> dict[str, Any]:
    """`shortfalls(hard_run)` gives the impossible-goal targets (tests inject it)."""
    from voltproof.bench.counterfactual import write_csv
    from voltproof.bench.report import Table, impossible_shortfalls, markdown
    from voltproof.bench.runner import token_usd

    plans = {json.loads((run / "episodes_plan.json").read_text(encoding="utf-8"))["plan_sha256"] for run in (low_run, high_run)}
    if len(plans) != 1:
        raise ValueError("the low and high runs were made from different plans")
    plan = json.loads((low_run / "episodes_plan.json").read_text(encoding="utf-8"))
    plan_ids = [e["episode_id"] for e in plan["episodes"]]
    rows = effort_rows(plan_ids, low_run, high_run, main_run, hard_run)

    def usd(row: Mapping[str, Any]) -> float:
        return token_usd(row, settings)

    target = out_dir / "e4"
    figure = effort_figure(rows, plan_ids, usd, target / "e4_effort.png", f"VoltProof Bench E4: A4 agent effort low / medium / high; reviewer at medium. Runs {low_run.name}, {high_run.name}; medium = frozen test run and E2 run.")
    spend = sum(r.get("cost_usd") or 0.0 for effort in EFFORTS for r in rows[effort].values())
    notes = (
        f"Plan: {len(plan_ids)} episodes ({sum(subset(e) == 'main' for e in plan_ids)} main, {sum(subset(e) == 'hard' for e in plan_ids)} hard), "
        f"plan sha256 `{plan['plan_sha256'][:12]}`. Only the A4 agent's reasoning effort varies (its own turns, repairs included); the "
        "reviewer stays at medium. Medium = the stored A4 rows of the frozen test run and the E2 run for the same episodes (not re-run). "
        f"E4 LLM cost of the two runs: ${spend:.2f} billed."
    )
    tables = [
        Table("e4_by_effort", "E4: A4 by reasoning effort and subset", effort_table(rows, plan_ids, usd), note=(
            "Billed USD is what the ledger charged; token USD prices every token at list price (runner.token_usd), because some medium rows "
            "were partly served from the LLM cache. Comparisons and the figure use token USD. Latency: medium was measured in other runs, at "
            "another time and concurrency (2, vs 1 here), so latency differences are indicative only. Agent reasoning tokens exclude the reviewer."
        )),
        Table("e4_bootstrap", "Paired bootstrap 95% CIs (10,000 resamples, seed 2026): each effort minus medium", effort_bootstrap(rows, plan_ids), note=(
            "One sample per setting: run-to-run LLM variation is not measured, so a difference of one or two episodes is reported as such, "
            "not as an effect."
        )),
        Table("e4_impossible", f"{IMPOSSIBLE_TEMPLATE}: shortfall stated in the explanation (the E2 report's check)", impossible_rows(rows, plan_ids, (shortfalls or impossible_shortfalls)(hard_run)), note=(
            "The check looks for the shortfall of the gold spec's plan on the forecast (A5). An explanation that states the shortfall of a "
            "differently read spec (e.g. a 100% reserve over [0, 24), which also constrains the starting charge) does not count; an episode "
            "that ended without an accepted finalize has no explanation."
        )),
    ]
    for table in tables:
        write_csv(target / "tables" / f"{table.name}.csv", table.rows)
    header = [
        "# VoltProof Bench E4: reasoning-effort sweep",
        "",
        notes,
        "",
        "Reported on its own, never pooled with any other run. Fixed in advance: the demo and every other result stay at medium, "
        "whatever E4 shows (it is the configuration that was benchmarked). Reasoning-effort sweep suggested by the Fireworks workshop at this hackathon.",
        "",
        "Limitation: \"actual\" weather is the same-day ICON analysis, not a measurement, so forecast error is understated.",
        "",
        f"![E4 figure]({figure.name})",
        "",
    ]
    (target / "REPORT.md").write_text("\n".join(header) + "\n" + "\n".join(markdown(t) for t in tables), encoding="utf-8")
    summary = {"plan_sha256": plan["plan_sha256"], "runs": [low_run.name, high_run.name], "billed_usd": spend, "tables": {t.name: t.rows for t in tables}}
    (target / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"summary": summary, "figure": figure, "tables": tables}


# --- CLI -------------------------------------------------------------------------------------------------------


def latest_plan(runs_dir: Path) -> Path:
    plans = sorted(p for p in runs_dir.glob("e4-plan-*") if (p / "episodes_plan.json").is_file())
    if not plans:
        raise SystemExit("no E4 plan: run python -m voltproof.bench.runner --split test --intents e4 --plan-only first")
    return plans[-1]


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    from voltproof.config import get_settings
    from voltproof.data.days import load_splits
    from voltproof.data.sites import SITES
    from voltproof.llm.client import read_ledger_totals

    parser = argparse.ArgumentParser(prog="python -m voltproof.bench.effort_sweep", description="E4 reasoning-effort sweep")
    parser.add_argument("command", choices=("dry-run", "probe-summary", "project", "report"))
    parser.add_argument("--run", type=Path, help="probe-summary: a probe run directory")
    parser.add_argument("--low", type=Path, help="project: the low-effort probe run; report: the low-effort E4 run")
    parser.add_argument("--high", type=Path, help="project: the high-effort probe run; report: the high-effort E4 run")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results", help="report: results directory (writes <out>/e4/)")
    args = parser.parse_args(argv)
    settings = get_settings()
    if args.command == "dry-run":
        episodes = e4_episodes(None, None, load_splits(), SITES)
        counts = dry_run(episodes, settings)
        print(f"E4 set: {len(episodes)} episodes; first agent call cached: " + ", ".join(f"{k} {v}/{len(episodes)}" for k, v in counts.items()))
        return 0
    if args.command == "probe-summary":
        print(json.dumps(probe_summary(args.run, settings.ledger_path), indent=1))
        return 0
    if args.command == "report":
        result = e4_report(args.low, args.high, args.out, settings)
        print(f"E4 report: {args.out / 'e4' / 'REPORT.md'}; figure {result['figure']}; billed ${result['summary']['billed_usd']:.2f}")
        return 0
    plan = json.loads((latest_plan(PROJECT_ROOT / "runs") / "episodes_plan.json").read_text(encoding="utf-8"))
    ids = [e["episode_id"] for e in plan["episodes"]]
    medium = medium_rows(ids)
    medium_total = sum(r.get("cost_usd") or 0.0 for r in medium.values())
    total = 0.0
    for effort, run in (("low", args.low), ("high", args.high)):
        probe = probe_summary(run, settings.ledger_path)
        pilot = {r["episode_id"]: r for r in read_rows(PILOT_RUN / "episodes.jsonl") if r["arm"] == "A4"}
        probe_medium = pilot[probe["episode_id"]].get("cost_usd") or 0.0
        estimate = projection(probe, probe_medium, medium_total, len(ids))
        total += estimate["projected"]
        print(
            f"{effort}: probe ${probe['cost_usd']:.4f} vs medium ${probe_medium:.4f} on {probe['episode_id']}; "
            f"x{len(ids)} = ${estimate['probe_x_n']:.2f}; scaled from medium (${medium_total:.2f} for the set) = ${estimate['scaled_from_medium']:.2f}; "
            f"projected ${estimate['projected']:.2f}"
        )
    spent = e4_spent_usd(settings.ledger_path)
    ledger = read_ledger_totals(settings.ledger_path).total_usd
    print(f"projected E4 runs ${total:.2f}; E4 spent so far ${spent:.4f}; E4 cap ${E4_CAP_USD:.2f} -> {'OK' if spent + total <= E4_CAP_USD else 'OVER: propose 20 + 20'}")
    print(f"global: ledger ${ledger:.2f} of ${settings.budget_usd_cap:.2f}; remaining ${settings.budget_usd_cap - ledger:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
