"""VoltProof Bench report (SPEC §6.11): tables, paired bootstrap CIs and figures from a finished run.

Reads a run directory (default: the frozen test run) and never calls the LLM or re-runs an arm; the
only computation besides aggregation is one oracle LP per episode, to recompute the oracle bound and
the day's p̂ (the tie tolerance). Aggregation rules (fixed after the test run, the same for every arm):
- per-episode percentages are never averaged (their base, A0's day cost, can be near zero)
- money: per site, pooled Σ(A0 − X)/Σ|A0|, the median per-episode difference in site currency and
  paired win/loss/tie counts against A0; never summed across currencies
- total cost with goals: per site, pooled gap Σ(X_total − bound)/Σ|A0_total|, the median gap, and
  A4's paired win/loss/tie counts against every arm
- goal satisfaction: rates with paired bootstrap 95% CIs (10,000 resamples, seed 2026) of A4 − X
- error decomposition: A4's pooled gap = translation (A4 − A5) + forecast (A5 − A6 executed) +
  remainder (A6 executed − bound)
Ties: |X − Y| ≤ 1e-6·max(1, p̂).

PowerShell, from the repo root:
    python -m voltproof.bench.report                 # report on the frozen test run
    python -m voltproof.bench.report --publish       # also copy the run into results/test-run/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from voltproof.bench.counterfactual import E1_CAPTION, FROZEN_RUN, RESULTS_DIR, run_episodes, write_csv
from voltproof.bench.runner import ALL_ARMS, code_fingerprint, read_jsonl
from voltproof.core.metrics import oracle_bound
from voltproof.core.rules import penalty_scale
from voltproof.agents.gate import extract_numbers, shortfall_stated
from voltproof.baselines.optimizer_arms import a5_plan, a6_plan
from voltproof.config import PROJECT_ROOT
from voltproof.data import GLOBAL_SEED
from voltproof.data.sites import SITES
from voltproof.llm.client import read_ledger_totals

BOOTSTRAP_RESAMPLES = 10_000
TIE_REL = 1e-6
BOUND_TOL = 1e-6
TRACES_ZIP_MAX_BYTES = 50 * 1024 * 1024
LLM_ARMS = ("A2", "A3", "A4")
GATE_CHECKS = ("verified_plan", "coverage", "grounded_explanation", "review")
ARM_NAMES = {
    "A0": "rule-based",
    "A1": "optimizer, cost only",
    "A2": "LLM writes schedule",
    "A3": "LLM + simulator",
    "A4": "VoltProof",
    "A5": "optimizer, gold spec",
    "A6": "oracle (actual data)",
}


@dataclass(frozen=True)
class EpisodeInfo:
    """What the aggregation needs beyond the rows: plan metadata, the day's p̂ and the oracle bound."""

    episode_id: str
    site_id: str
    template_id: str
    language: str
    multilingual: bool
    held_out_site: bool
    held_out_template: bool
    p_hat: float
    bound: float
    outage: bool = False  # the gold spec announces a grid outage


@dataclass
class Table:
    name: str
    title: str
    rows: list[dict[str, Any]]
    note: str | None = None


# --- loading -------------------------------------------------------------------------------------


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    plan = json.loads((run_dir / "episodes_plan.json").read_text(encoding="utf-8"))
    return config, plan, read_jsonl(run_dir / "episodes.jsonl")


def episode_infos(plan: Mapping[str, Any], p_hats: Mapping[str, float], bounds: Mapping[str, float]) -> dict[str, EpisodeInfo]:
    infos = {}
    for entry in plan["episodes"]:
        eid = entry["episode_id"]
        infos[eid] = EpisodeInfo(
            episode_id=eid,
            site_id=entry["site_id"],
            template_id=entry["template_id"],
            language=entry["language"],
            multilingual=not entry["goal_id"].startswith(f"{entry['template_id']}_p"),
            held_out_site=bool(entry["held_out_site"]),
            held_out_template=bool(entry["held_out_template"]),
            p_hat=p_hats[eid],
            bound=bounds[eid],
            outage=bool((entry.get("gold") or {}).get("grid_outages")),
        )
    return infos


def day_references(run_dir: Path, progress: bool = True) -> tuple[dict[str, float], dict[str, float]]:
    """p̂ and the oracle bound per episode, from the rebuilt days (one oracle LP each)."""
    from tqdm import tqdm

    p_hats, bounds = {}, {}
    for episode in tqdm(run_episodes(run_dir), desc="oracle bounds", unit="episode", disable=not progress):
        p_hats[episode.episode_id] = penalty_scale(episode.day.import_price_per_kwh)
        bounds[episode.episode_id] = oracle_bound(a6_plan(episode.day, episode.site, episode.goal.gold))
    return p_hats, bounds


def paired(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    """episode_id → arm → row."""
    out: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        out.setdefault(row["episode_id"], {})[row["arm"]] = row
    return out


def arms_in(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    present = {r["arm"] for r in rows}
    return [a for a in ALL_ARMS if a in present]


def select(infos: Mapping[str, EpisodeInfo], predicate: Callable[[EpisodeInfo], bool]) -> list[str]:
    return [eid for eid, info in infos.items() if predicate(info)]


# --- statistics ----------------------------------------------------------------------------------


def tie_tol(p_hat: float) -> float:
    return TIE_REL * max(1.0, p_hat)


def wlt(xs: Sequence[float], ys: Sequence[float], tols: Sequence[float]) -> str:
    """Wins/losses/ties of x against y where lower is better."""
    wins = sum(x < y - t for x, y, t in zip(xs, ys, tols))
    losses = sum(x > y + t for x, y, t in zip(xs, ys, tols))
    return f"{wins}/{losses}/{len(xs) - wins - losses}"


def bootstrap_mean_ci(values: Sequence[float], seed: int = GLOBAL_SEED, resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    """(mean, 2.5th, 97.5th percentile) of the bootstrap distribution of the mean; for paired data
    pass the per-episode differences."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(axis=1)
    return float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pct(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else 100.0 * numerator / denominator


# --- tables ----------------------------------------------------------------------------------------


def run_summary(rows: Sequence[Mapping[str, Any]], errors: Sequence[Mapping[str, Any]]) -> Table:
    infra = {}
    for e in errors:
        infra[e.get("arm")] = infra.get(e.get("arm"), 0) + 1
    out = []
    for arm in arms_in(rows):
        mine = [r for r in rows if r["arm"] == arm]
        sources: dict[str, int] = {}
        for r in mine:
            sources[r.get("plan_source") or "-"] = sources.get(r.get("plan_source") or "-", 0) + 1
        out.append(
            {
                "arm": arm,
                "name": ARM_NAMES.get(arm, ""),
                "results": len(mine),
                "plan_source": ", ".join(f"{k} {v}" for k, v in sorted(sources.items())),
                "fallbacks": sum(bool(r.get("fallback_used")) for r in mine),
                "escalations": sum(bool(r.get("escalated")) for r in mine),
                "infra_errors": infra.get(arm, 0),
                "infra_fallbacks": sum(bool(r.get("infra_retries")) for r in mine),
            }
        )
    return Table("run_summary", "Run summary", out)


def goal_rates(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], infos: Mapping[str, EpisodeInfo], arms: Sequence[str]) -> Table:
    groups: list[tuple[str, Callable[[EpisodeInfo], bool]]] = [("overall", lambda i: True)]
    for site in sorted({i.site_id for i in infos.values()}):
        held = any(i.held_out_site for i in infos.values() if i.site_id == site)
        groups.append((f"{site}{' (held out)' if held else ''}", lambda i, s=site: i.site_id == s))
    if any(i.held_out_template for i in infos.values()):
        groups.append(("held-out templates", lambda i: i.held_out_template))
    out = []
    for arm in arms:
        row: dict[str, Any] = {"arm": arm}
        for label, predicate in groups:
            ids = [e for e in select(infos, predicate) if arm in by_ep.get(e, {})]
            hits = sum(bool(by_ep[e][arm]["gold_satisfied"]) for e in ids)
            row[label] = f"{pct(hits, len(ids)):.1f}% ({hits}/{len(ids)})" if ids else "-"
        out.append(row)
    return Table("goal_satisfaction", "Goal satisfaction (gold spec met on the executed trajectory)", out)


def goal_bootstrap(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], infos: Mapping[str, EpisodeInfo], against: Sequence[str] = ("A0", "A1", "A2", "A3")) -> Table:
    groups: list[tuple[str, Callable[[EpisodeInfo], bool]]] = [("overall", lambda i: True)]
    groups += [(site, lambda i, s=site: i.site_id == s) for site in sorted({i.site_id for i in infos.values()})]
    out = []
    for other in against:
        row: dict[str, Any] = {"comparison": f"A4 - {other}"}
        for label, predicate in groups:
            ids = [e for e in select(infos, predicate) if {"A4", other} <= set(by_ep.get(e, {}))]
            diffs = [float(by_ep[e]["A4"]["gold_satisfied"]) - float(by_ep[e][other]["gold_satisfied"]) for e in ids]
            mean, lo, hi = bootstrap_mean_ci(diffs)
            row[label] = f"{100 * mean:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}]" if ids else "-"
        out.append(row)
    return Table(
        "goal_satisfaction_bootstrap",
        "A4 minus each arm in goal satisfaction, paired bootstrap 95% CI",
        out,
        note=f"{BOOTSTRAP_RESAMPLES:,} resamples of episodes (pairs kept), seed {GLOBAL_SEED}.",
    )


def money_table(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], infos: Mapping[str, EpisodeInfo], site: str, arms: Sequence[str]) -> Table:
    def stats(ids: list[str], arm: str) -> tuple[str, str, str]:
        ids = [e for e in ids if {"A0", arm} <= set(by_ep[e])]
        a0 = [by_ep[e]["A0"]["realized_cost"] for e in ids]
        x = [by_ep[e][arm]["realized_cost"] for e in ids]
        if not ids:
            return "-", "-", "-"
        pooled = pct(sum(a - b for a, b in zip(a0, x)), sum(abs(a) for a in a0))
        median = float(np.median([a - b for a, b in zip(a0, x)]))
        return (f"{pooled:+.2f}%" if pooled is not None else "n/a"), f"{median:+.2f}", wlt(x, a0, [tie_tol(infos[e].p_hat) for e in ids])

    all_ids = [e for e in select(infos, lambda i: i.site_id == site) if e in by_ep]
    cost_ids = [e for e in all_ids if infos[e].template_id == "cost_min"]
    out = []
    for arm in arms:
        pooled, median, record = stats(all_ids, arm)
        row = {"arm": arm, "pooled savings vs A0": pooled, "median (A0 - X) per day": median, "W/L/T vs A0": record}
        if cost_ids:  # the cost-only template (absent from the hard set)
            c_pooled, c_median, c_record = stats(cost_ids, arm)
            row.update({"cost_min: pooled": c_pooled, "cost_min: median": c_median, "cost_min: W/L/T": c_record})
        out.append(row)
    currency = _currency(by_ep, all_ids)
    return Table(
        f"money_{site}",
        f"Money (realized_cost, {currency}), {site}: {len(all_ids)} episodes, {len(cost_ids)} cost_min",
        out,
        note="Pooled = sum(A0 - X) / sum|A0|. Win = X cheaper than A0 by more than the tie tolerance.",
    )


def total_cost_table(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], infos: Mapping[str, EpisodeInfo], site: str, arms: Sequence[str]) -> Table:
    ids = [e for e in select(infos, lambda i: i.site_id == site) if e in by_ep]
    base = sum(abs(by_ep[e]["A0"]["realized_total_cost"]) for e in ids)
    out = []
    for arm in arms:
        mine = [e for e in ids if arm in by_ep[e]]
        gaps = [by_ep[e][arm]["realized_total_cost"] - infos[e].bound for e in mine]
        pooled = pct(sum(gaps), base)
        record = "-"
        if arm != "A4":
            both = [e for e in mine if "A4" in by_ep[e]]
            record = wlt([by_ep[e]["A4"]["realized_total_cost"] for e in both], [by_ep[e][arm]["realized_total_cost"] for e in both], [tie_tol(infos[e].p_hat) for e in both])
        out.append(
            {
                "arm": arm,
                "pooled gap to oracle bound": f"{pooled:+.2f}%" if pooled is not None else "n/a",
                "median gap per day": f"{float(np.median(gaps)):+.2f}" if gaps else "-",
                "A4 vs arm W/L/T": record,
            }
        )
    return Table(
        f"total_cost_{site}",
        f"Total cost with goals (realized_total_cost, {_currency(by_ep, ids)}), {site}: {len(ids)} episodes",
        out,
        note="Pooled gap = sum(X_total - oracle bound) / sum|A0_total|. W/L/T counts A4 wins (A4 lower).",
    )


def outage_table(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], infos: Mapping[str, EpisodeInfo], arms: Sequence[str], e1: Mapping[str, Any] | None) -> Table:
    ids = [e for e in select(infos, lambda i: i.outage) if e in by_ep]
    templates = ", ".join(dict.fromkeys(infos[e].template_id for e in ids))
    e1_arm = {r["arm"]: r for r in (e1 or {}).get("per_arm", [])}
    out = []
    for arm in arms:
        mine = [by_ep[e][arm] for e in ids if arm in by_ep[e]]
        valid = sum(bool(r["plan_physics_valid"]) for r in mine)
        row: dict[str, Any] = {
            "arm": arm,
            "episodes": len(mine),
            "plan physics valid": "n/a" if arm == "A6" else f"{pct(valid, len(mine)):.1f}%",
            "unserved critical kWh": f"{sum(r['unserved_critical_kwh'] for r in mine):.3f}",
            "executor overrides (hours / episodes)": f"{sum(r['overrides'] for r in mine)} / {sum(r['overrides'] > 0 for r in mine)}",
        }
        cf = e1_arm.get(arm)
        if cf is not None:
            row["E1: unserved critical kWh"] = f"{cf['cf_unserved_critical_kwh']:.3f}"
            row["E1: episodes with critical loss"] = cf["cf_episodes_any_critical_loss"]
            row["E1: unserved non-critical kWh"] = f"{cf['cf_unserved_noncritical_kwh']:.3f}"
        out.append(row)
    note = "Plan physics validity: did the plan as written avoid needing a grid that is down? A6 plans on actual data, so its forecast check is not applicable."
    if e1:
        note += " " + E1_CAPTION
    return Table("outage_safety", f"Outage safety ({templates or 'no outage goals'}: {len(ids)} episodes)", out, note=note)


def e1_site_table(e1: Mapping[str, Any]) -> Table:
    out = [
        {
            "site": r["site_id"],
            "arm": r["arm"],
            "episodes": r["episodes"],
            "critical kWh lost (with override)": f"{r['unserved_critical_kwh']:.3f}",
            "critical kWh lost (E1, no override)": f"{r['cf_unserved_critical_kwh']:.3f}",
            "episodes with critical loss (E1)": r["cf_episodes_any_critical_loss"],
            "non-critical kWh lost (E1)": f"{r['cf_unserved_noncritical_kwh']:.3f}",
            "gold satisfied (E1)": f"{r['cf_gold_satisfied_pct']:.1f}%",
        }
        for r in e1["per_site_arm"]
    ]
    return Table("e1_per_site", "E1 outage counterfactual per site", out, note=e1["caption"] + e1_finding(e1))


def e1_finding(e1: Mapping[str, Any]) -> str:
    """The reading of E1 for A4, stated only when the data support it (valid plans aside, which the
    outage table shows): A4 loses critical load without the override and the oracle loses none."""
    per_arm = {r["arm"]: r for r in e1.get("per_arm", [])}
    a4, a6 = per_arm.get("A4"), per_arm.get("A6")
    if not a4 or not a6 or a4["cf_episodes_any_critical_loss"] == 0 or a6["cf_episodes_any_critical_loss"] != 0:
        return ""
    return (
        " VoltProof's plans meet the announced outage on the forecast; the critical load lost in "
        f"{a4['cf_episodes_any_critical_loss']} episodes comes from solar falling short of the forecast (A6, with perfect "
        "information, loses none). The plan is verified, and the controller override remains as a second layer."
    )


def decomposition_table(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], infos: Mapping[str, EpisodeInfo]) -> Table:
    out = []
    for site in sorted({i.site_id for i in infos.values()}):
        ids = [e for e in select(infos, lambda i, s=site: i.site_id == s) if {"A0", "A4", "A5", "A6"} <= set(by_ep.get(e, {}))]
        base = sum(abs(by_ep[e]["A0"]["realized_total_cost"]) for e in ids)
        total = lambda arm: sum(by_ep[e][arm]["realized_total_cost"] for e in ids)  # noqa: E731
        bound = sum(infos[e].bound for e in ids)
        parts = {
            "translation (A4 - A5)": total("A4") - total("A5"),
            "forecast (A5 - A6 executed)": total("A5") - total("A6"),
            "remainder (A6 executed - bound)": total("A6") - bound,
        }
        gap = total("A4") - bound
        if abs(sum(parts.values()) - gap) > 1e-6 * max(1.0, abs(gap)):
            raise AssertionError(f"decomposition does not add up for {site}")
        out.append(
            {
                "site": site,
                "episodes": len(ids),
                "A4 pooled gap": f"{pct(gap, base):+.3f}%",
                **{k: f"{pct(v, base):+.3f}%" for k, v in parts.items()},
                "_values": {k: pct(v, base) for k, v in parts.items()},
            }
        )
    return Table(
        "error_decomposition",
        "Error decomposition of A4's pooled total-cost gap (share of sum|A0_total|)",
        out,
        note="translation + forecast + remainder = A4's pooled gap to the oracle bound, per site.",
    )


def spec_tables(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], infos: Mapping[str, EpisodeInfo]) -> list[Table]:
    def row(label: str, ids: list[str]) -> dict[str, Any]:
        rs = [by_ep[e]["A4"] for e in ids if "A4" in by_ep[e] and by_ep[e]["A4"].get("spec_precision") is not None]
        exact = sum(r["spec_precision"] == 1 and r["spec_recall"] == 1 for r in rs)
        return {
            "group": label,
            "episodes": len(rs),
            "exact match": f"{pct(exact, len(rs)):.1f}%" if rs else "-",
            "mean precision": f"{np.mean([r['spec_precision'] for r in rs]):.3f}" if rs else "-",
            "mean recall": f"{np.mean([r['spec_recall'] for r in rs]):.3f}" if rs else "-",
        }

    ids = [e for e in infos if e in by_ep]
    by_language = [row("English set", [e for e in ids if not infos[e].multilingual])]
    by_language += [row(f"multilingual: {lang}", [e for e in ids if infos[e].multilingual and infos[e].language == lang]) for lang in _languages(infos)]
    by_site = [row(site, [e for e in ids if infos[e].site_id == site]) for site in sorted({i.site_id for i in infos.values()})]
    by_template = [row(t, [e for e in ids if infos[e].template_id == t]) for t in _templates(infos)]
    return [
        Table("a4_spec_by_language", "A4 spec accuracy by language", by_language, note="Multilingual rows are the same goals, days and sites in each language (en = the English source sentence)."),
        Table("a4_spec_by_site", "A4 spec accuracy by site", by_site),
        Table("a4_spec_by_template", "A4 spec accuracy by template", by_template),
    ]


def repair_events(traces: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Every refused, non-final A4 finalize attempt (a repair round) and the checks that failed."""
    out = []
    for eid, events in sorted(traces.items()):
        for event in events:
            if event.get("kind") != "gate" or event.get("name") != "finalize":
                continue
            data = event.get("data", {})
            checks = [f["check"] for f in data.get("failed_checks", [])]
            if checks and not data.get("final"):
                out.append({"episode_id": eid, "attempt": data.get("attempt"), "checks": "+".join(dict.fromkeys(checks))})
    return out


def gate_tables(rows: Sequence[Mapping[str, Any]], repairs: Sequence[Mapping[str, Any]]) -> list[Table]:
    a4 = [r for r in rows if r["arm"] == "A4"]
    rounds: dict[int, int] = {}
    for r in a4:
        rounds[r.get("repair_rounds") or 0] = rounds.get(r.get("repair_rounds") or 0, 0) + 1
    by_check = [{"check": c, "repairs triggered (attempts failing it)": sum(c in rep["checks"].split("+") for rep in repairs)} for c in GATE_CHECKS]
    combos: dict[str, int] = {}
    for rep in repairs:
        combos[rep["checks"]] = combos.get(rep["checks"], 0) + 1
    verdicts = {
        "review approved": sum(r.get("review_approved") is True for r in a4),
        "review not approved": sum(r.get("review_approved") is False for r in a4),
        "review unavailable": sum(r.get("review_approved") is None for r in a4),
        "review disputed (accepted with flag)": sum(bool(r.get("review_disputed")) for r in a4),
        "explanation not grounded (accepted with flag)": sum(r.get("explanation_grounded") is False for r in a4),
        "escalated with plan": sum(r.get("escalation") == "with_plan" for r in a4),
        "escalated to fallback": sum(r.get("escalation") == "fallback" for r in a4),
    }
    return [
        Table("a4_repair_rounds", "A4 repair rounds per episode", [{"repair rounds": k, "episodes": v} for k, v in sorted(rounds.items())]),
        Table("a4_repair_triggers", f"Which gate check triggered each A4 repair ({len(repairs)} repairs)", by_check, note="An attempt can fail more than one check."),
        Table("a4_repair_combinations", "A4 repairs by combination of failed checks", [{"failed checks": k, "repairs": v} for k, v in sorted(combos.items(), key=lambda kv: -kv[1])]),
        Table("a4_gate_outcomes", "A4 gate outcomes and reviewer verdicts", [{"outcome": k, "episodes": v} for k, v in verdicts.items()]),
    ]


def cost_speed_table(rows: Sequence[Mapping[str, Any]]) -> Table:
    out = []
    for arm in arms_in(rows):
        if arm not in LLM_ARMS:
            continue
        mine = [r for r in rows if r["arm"] == arm]
        latency = [r["latency_s"] for r in mine if r.get("latency_s") is not None]
        usd = [r.get("cost_usd") or 0.0 for r in mine]
        out.append(
            {
                "arm": arm,
                "episodes": len(mine),
                "LLM USD / episode": f"{np.mean(usd):.4f}",
                "LLM USD total": f"{sum(usd):.2f}",
                "median latency s": f"{np.median(latency):.1f}" if latency else "-",
                "p90 latency s": f"{np.percentile(latency, 90):.1f}" if latency else "-",
                "tokens in / episode": f"{np.mean([r.get('tokens_in') or 0 for r in mine]):.0f}",
                "tokens out / episode": f"{np.mean([r.get('tokens_out') or 0 for r in mine]):.0f}",
            }
        )
    return Table("cost_speed", "LLM cost and speed per arm", out, note="Baselines make no LLM call. Latency is the LLM time per episode (agent + reviewer).")


def breakdown_tables(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], infos: Mapping[str, EpisodeInfo], arms: Sequence[str]) -> list[Table]:
    def rate(ids: list[str], arm: str) -> str:
        ids = [e for e in ids if arm in by_ep.get(e, {})]
        hits = sum(bool(by_ep[e][arm]["gold_satisfied"]) for e in ids)
        return f"{pct(hits, len(ids)):.0f}% ({hits}/{len(ids)})" if ids else "-"

    templates = [
        {"template": t, "held out": "yes" if any(i.held_out_template for i in infos.values() if i.template_id == t) else "", **{arm: rate(select(infos, lambda i, t=t: i.template_id == t), arm) for arm in arms}}
        for t in _templates(infos)
    ]
    languages = [
        {"language": lang, **{arm: rate(select(infos, lambda i, l=lang: i.multilingual and i.language == l), arm) for arm in arms}}
        for lang in _languages(infos)
    ]
    return [
        Table("goal_by_template", "Goal satisfaction by template", templates),
        Table("goal_by_language", "Goal satisfaction by language (multilingual set: same goals, days and sites)", languages),
    ]


def bound_check(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], infos: Mapping[str, EpisodeInfo]) -> dict[str, Any]:
    """The recomputed oracle bound against the one implied by A0's stored gap, and the oracle gate."""
    checked, skipped, mismatches, below = 0, 0, [], []
    for eid, info in infos.items():
        arms = by_ep.get(eid, {})
        a0 = arms.get("A0")
        if a0 is None:
            continue
        gap = a0.get("gap_to_oracle_pct")
        if gap is None:
            skipped += 1
        else:
            implied = a0["realized_total_cost"] - gap * abs(a0["realized_total_cost"]) / 100.0
            checked += 1
            if abs(implied - info.bound) > BOUND_TOL:
                mismatches.append({"episode_id": eid, "recomputed": info.bound, "implied": implied})
        for arm, row in arms.items():
            if row["realized_total_cost"] < info.bound - 1e-6 * max(1.0, info.p_hat):
                below.append({"episode_id": eid, "arm": arm})
    return {"checked": checked, "skipped_gap_none": skipped, "mismatches": mismatches, "below_bound": below, "passed": not mismatches and not below}


IMPOSSIBLE_TEMPLATE = "hard_impossible"


def impossible_shortfalls(run_dir: Path) -> dict[str, tuple[list[Any], float]]:
    """Per impossible-goal episode: the shortfalls of the gold spec's plan on the forecast (A5, what
    any arm can at best achieve) and the battery capacity."""
    out = {}
    for episode in run_episodes(run_dir):
        if episode.goal.template_id == IMPOSSIBLE_TEMPLATE:
            plan = a5_plan(episode.day, episode.site, episode.goal.gold)
            out[episode.episode_id] = (list(plan.shortfalls), episode.site.battery.capacity_kwh)
    return out


def shortfall_table(rows: Sequence[Mapping[str, Any]], infos: Mapping[str, EpisodeInfo], targets: Mapping[str, tuple[list[Any], float]]) -> Table:
    """E2's physically impossible goal: does each LLM arm's explanation state the shortfall (as A4's
    gate requires: quoted within 5 %, or rounded to the nearest whole unit)?"""
    out = []
    by_ep = paired(rows)
    for arm in LLM_ARMS:
        ids = [e for e in targets if arm in by_ep.get(e, {})]
        stated = 0
        for eid in ids:
            explanation = by_ep[eid][arm].get("explanation") or ""
            shortfalls, capacity = targets[eid]
            numbers = extract_numbers(explanation)
            stated += bool(shortfalls) and all(shortfall_stated(s, numbers, capacity) for s in shortfalls)
        row: dict[str, Any] = {"arm": arm, "episodes": len(ids), "shortfall stated": f"{stated}/{len(ids)}", "spec exact match": "-"}
        if arm == "A4":
            exact = sum(by_ep[e][arm].get("spec_precision") == 1 and by_ep[e][arm].get("spec_recall") == 1 for e in ids)
            row["spec exact match"] = f"{exact}/{len(ids)}"
        out.append(row)
    amounts = sorted({f"{infos[e].site_id}: {s.amount:.3f} {s.unit}" for e, (ss, _) in targets.items() for s in ss})
    return Table(
        "e2_impossible_shortfall",
        f"Physically impossible goal ({IMPOSSIBLE_TEMPLATE}): is the shortfall stated in the explanation?",
        out,
        note="Shortfall = the gold spec's plan on the forecast (A5), the least any arm can fall short by: " + "; ".join(amounts) + ".",
    )


def e2_notes(plan: Mapping[str, Any]) -> list[str]:
    """Disclosures for the E2 report header."""
    entries = plan["episodes"]
    ml_sources = {(e["site_id"], e["date"], e["goal"]) for e in entries if e["language"] == "en" and e["goal_id"].startswith("ml_")}
    duplicates = [e for e in entries if not e["goal_id"].startswith("ml_") and (e["site_id"], e["date"], e["goal"]) in ml_sources]
    return [
        "Reported separately from the main test run and never pooled with it. Same frozen code (plus the additive executor "
        "flag, default unchanged; the regression check passed before any spend), the same arms, metrics and aggregation rules.",
        "",
        f"Disclosure: {len(duplicates)} English-set episodes have the same goal text, site and day as a multilingual English "
        "source episode, so they sent identical requests (served from the LLM cache) and have identical results: "
        + ", ".join(e["episode_id"] for e in duplicates)
        + ".",
        "",
        "Multilingual goals: translations supplied by the author.",
        "",
    ]


def e3_report(e3_dir: Path, out_dir: Path) -> dict[str, Any]:
    """The E3 reviewer stress test: detection per error type in both conditions, side by side."""
    from voltproof.bench.reviewer_stress import CONTROL, ERROR_TYPES, summarize

    config = json.loads((e3_dir / "config.json").read_text(encoding="utf-8"))
    variants = read_jsonl(e3_dir / "variants.jsonl")
    results = read_jsonl(e3_dir / "results.jsonl")
    summary = summarize(variants, results)
    cell = {(r["condition"], r["error_type"]): r for r in summary}

    def show(r: Mapping[str, Any]) -> str:
        judged = r["variants"] - r["unavailable"]
        if not judged:
            return "-"
        return f"{100 * r['rate']:.0f}% [{100 * r['ci_low']:.0f}, {100 * r['ci_high']:.0f}] ({r['flagged']}/{judged})"

    rows = []
    for error_type in (CONTROL, *ERROR_TYPES):
        a, b = cell[("with_assumptions", error_type)], cell[("no_assumptions", error_type)]
        row: dict[str, Any] = {
            "error": "control (false alarm)" if error_type == CONTROL else error_type,
            "with assumptions: flagged": show(a),
            "no assumptions: flagged": show(b),
            "issue kind matches (with / no)": "-",
            "assumptions state original numbers: n, flagged": "-",
        }
        if error_type != CONTROL:
            row["issue kind matches (with / no)"] = f"{a['kind_match']}/{a['flagged']} / {b['kind_match']}/{b['flagged']}"
            yes, no = a.get("n_assumptions_state_original_true", 0), a.get("n_assumptions_state_original_false", 0)
            rate_yes, rate_no = a.get("rate_if_assumptions_state_original_true"), a.get("rate_if_assumptions_state_original_false")
            row["assumptions state original numbers: n, flagged"] = (
                f"yes {yes} ({'-' if rate_yes is None else f'{100 * rate_yes:.0f}%'}), no {no} ({'-' if rate_no is None else f'{100 * rate_no:.0f}%'})"
            )
        rows.append(row)
    unavailable = sum(r["unavailable"] for r in summary)
    same = sum(bool(v.get("same_input_as_with_assumptions")) for v in variants)
    same_eps = len({v["episode_id"] for v in variants if v.get("same_input_as_with_assumptions")})
    cost = sum(r.get("cost_usd") or 0.0 for r in results)
    notes = [
        f"{len(config['episodes'])} A4 episodes of the frozen run (spec exactly matching gold; cost_min excluded), {len(variants)} review calls, "
        f"{sum(bool(r.get('cache_hit')) for r in results)} served from the LLM cache, {unavailable} unusable verdicts, LLM cost ${cost:.2f}.",
        "Flagged = the reviewer did not approve. 95% Wilson intervals. Issue kind matches = the flagged verdict names the expected kind "
        "(wrong for shifted hours, wrong levels and wrong fields; missing; unneeded).",
        "Disclosure: the with-assumptions controls are byte-identical to the frozen run's review calls and replay its cached verdicts. "
        f"{same} no-assumptions variants in {same_eps} episodes are identical to their with-assumptions input (the run's reviewer saw no "
        "assumptions there), so they replay that verdict; they are kept and flagged.",
        "The reviewer prompt counts a requirement captured by the assumptions as not missing, so with assumptions present a dropped "
        "spec item the assumptions still describe is not necessarily an error to this reviewer.",
    ]
    table = Table("e3_detection", "E3: does the independent reviewer catch a wrong spec?", rows, note=" ".join(notes))
    target = out_dir / "e3"
    write_tables([table], target)
    merged = [{**{k: v for k, v in var.items() if k not in ("spec", "original_spec", "assumptions")}, **{k: v for k, v in res.items() if k not in ("raw",)}} for var in variants for res in results if res["variant_id"] == var["variant_id"]]
    write_csv(target / "tables" / "e3_variants.csv", [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in m.items()} for m in merged])
    for name in ("config.json", "variants.jsonl", "results.jsonl"):
        shutil.copyfile(e3_dir / name, target / name)
    figure = figure_e3(summary, out_dir / "figures" / "e3_detection.png", f"VoltProof Bench E3, run {e3_dir.name}. Reviewer = the frozen A4 reviewer (same prompt and schema).")
    (target / "REPORT.md").write_text(f"# VoltProof Bench E3, reviewer stress test: `{e3_dir.name}`\n\n" + markdown(table), encoding="utf-8")
    (target / "summary.json").write_text(json.dumps({"run_id": e3_dir.name, "summary": summary, "cost_usd": cost}, indent=1) + "\n", encoding="utf-8")
    return {"summary": summary, "figures": [str(figure.relative_to(out_dir))], "table": table}


def figure_e3(summary: Sequence[Mapping[str, Any]], path: Path, caption: str) -> Path:
    from voltproof.bench.reviewer_stress import CONDITIONS, CONTROL, ERROR_TYPES

    fig, ax = _figure(8.0, 3.8)
    types = [CONTROL, *ERROR_TYPES]
    cell = {(r["condition"], r["error_type"]): r for r in summary}
    x = np.arange(len(types))
    width = 0.38
    for k, condition in enumerate(CONDITIONS):
        rows = [cell[(condition, t)] for t in types]
        rates = [100 * (r["rate"] or 0.0) for r in rows]
        low = [max(0.0, 100 * ((r["rate"] or 0.0) - (r["ci_low"] or 0.0))) for r in rows]
        high = [max(0.0, 100 * ((r["ci_high"] or 0.0) - (r["rate"] or 0.0))) for r in rows]  # float noise at 0 % / 100 %
        xs = x + (k - 0.5) * width
        ax.bar(xs, rates, width=width - 0.03, color=SERIES[k], edgecolor=SURFACE, linewidth=2, label=condition.replace("_", " "))
        ax.errorbar(xs, rates, yerr=[low, high], fmt="none", ecolor=INK, elinewidth=1.0, capsize=2)
        for xi, r, v, h in zip(xs, rows, rates, high):
            ax.text(xi, v + h + 1.5, f"{r['flagged']}/{r['variants'] - r['unavailable']}", ha="center", fontsize=6, color=INK)
    ax.set_xticks(x, ["control\n(false alarm)", "shifted\nhours", "wrong\nlevel", "missing\nitem", "unneeded\nitem", "wrong\nfield"], fontsize=8)
    ax.set_ylim(0, 128)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("flagged by the reviewer (%)", color=INK_2, fontsize=9)
    ax.set_title("E3: reviewer detection of one injected spec error, 95% Wilson CI", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", ncol=2)
    return _save(fig, path, caption)


def _currency(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], ids: Sequence[str]) -> str:
    return next((r["currency"] for e in ids for r in by_ep[e].values()), "?")


def _templates(infos: Mapping[str, EpisodeInfo]) -> list[str]:
    return list(dict.fromkeys(i.template_id for i in infos.values()))


def _languages(infos: Mapping[str, EpisodeInfo]) -> list[str]:
    return list(dict.fromkeys(i.language for i in infos.values() if i.multilingual))


# --- assembly --------------------------------------------------------------------------------------


def build_tables(
    rows: Sequence[Mapping[str, Any]],
    infos: Mapping[str, EpisodeInfo],
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    errors: Sequence[Mapping[str, Any]] = (),
    e1: Mapping[str, Any] | None = None,
) -> list[Table]:
    by_ep = paired(rows)
    arms = arms_in(rows)
    sites = sorted({i.site_id for i in infos.values()})
    tables = [run_summary(rows, errors), goal_rates(by_ep, infos, arms)]
    if "A4" in arms:
        tables.append(goal_bootstrap(by_ep, infos, [a for a in ("A0", "A1", "A2", "A3") if a in arms]))
    tables += [money_table(by_ep, infos, site, arms) for site in sites]
    tables += [total_cost_table(by_ep, infos, site, arms) for site in sites]
    tables.append(outage_table(by_ep, infos, arms, e1))
    if e1:
        tables.append(e1_site_table(e1))
    tables.append(decomposition_table(by_ep, infos))
    tables += spec_tables(by_ep, infos)
    tables += gate_tables(rows, repair_events(traces))
    tables.append(cost_speed_table(rows))
    tables += breakdown_tables(by_ep, infos, arms)
    return tables


def markdown(table: Table) -> str:
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in table.rows]
    lines = [f"### {table.title}", ""]
    if rows:
        columns = list(rows[0])
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        lines += ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    if table.note:
        lines += ["", table.note]
    return "\n".join(lines) + "\n"


def write_tables(tables: Sequence[Table], out_dir: Path) -> None:
    for table in tables:
        write_csv(out_dir / "tables" / f"{table.name}.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in table.rows])


# --- figures -----------------------------------------------------------------------------------------

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # reference palette slots 1-3 (light)


def _axes(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width: float = 7.0, height: float = 3.6, ncols: int = 1) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)
    for ax in np.atleast_1d(axes):
        _axes(ax)
    return fig, axes


def _save(fig: Any, path: Path, caption: str) -> Path:
    lines = textwrap.wrap(caption, width=int(fig.get_figwidth() * 17))  # about 17 characters per inch at 7 pt
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02 + 0.035 * len(lines), 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def figure_goal_satisfaction(by_ep: Mapping[str, Mapping[str, Mapping[str, Any]]], arms: Sequence[str], path: Path, caption: str) -> Path:
    fig, ax = _figure()
    stats = [bootstrap_mean_ci([float(by_ep[e][a]["gold_satisfied"]) for e in by_ep if a in by_ep[e]]) for a in arms]
    x = np.arange(len(arms))
    colors = [SERIES[1] if a == "A4" else SERIES[0] for a in arms]
    ax.bar(x, [100 * s[0] for s in stats], width=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, [100 * s[0] for s in stats], yerr=[[100 * (s[0] - s[1]) for s in stats], [100 * (s[2] - s[0]) for s in stats]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    for xi, s in zip(x, stats):
        ax.text(xi, 100 * s[2] + 1.5, f"{100 * s[0]:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{a}\n" + "\n".join(textwrap.wrap(ARM_NAMES.get(a, ""), 12)) for a in arms], fontsize=7)
    ax.set_ylim(0, 108)
    ax.set_ylabel("goal satisfied (%)", color=INK_2, fontsize=9)
    ax.set_title("Goal satisfaction per arm, 95% bootstrap CI", color=INK, fontsize=10, loc="left")
    return _save(fig, path, caption)


def figure_outage(table: Table, path: Path, caption: str) -> Path:
    has_e1 = "E1: unserved critical kWh" in table.rows[0]
    fig, axes = _figure(9.0 if has_e1 else 6.0, 3.6, 2 if has_e1 else 1)
    left, right = (axes[0], axes[1]) if has_e1 else (axes, None)
    arms = [r["arm"] for r in table.rows]
    x = np.arange(len(arms))
    valid = [float(r["plan physics valid"].rstrip("%")) if r["plan physics valid"] != "n/a" else 0.0 for r in table.rows]
    left.bar(x, valid, width=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for xi, r, v in zip(x, table.rows, valid):
        left.text(xi, v + 2, r["plan physics valid"], ha="center", fontsize=7, color=INK)
    left.set_xticks(x, arms)
    left.set_ylim(0, 112)
    left.set_title("Outage plans physically valid (%)", color=INK, fontsize=10, loc="left")
    if right is not None:
        lost = [float(r["E1: unserved critical kWh"]) for r in table.rows]
        right.bar(x, lost, width=0.6, color=SERIES[1], edgecolor=SURFACE, linewidth=2)
        for xi, r, v in zip(x, table.rows, lost):
            right.text(xi, v, f"{v:.1f}\n({r['E1: episodes with critical loss']} ep)", ha="center", va="bottom", fontsize=7, color=INK)
        right.set_ylim(0, max(lost + [1.0]) * 1.25)
        right.set_xticks(x, arms)
        right.set_title("E1: critical kWh lost without the override", color=INK, fontsize=10, loc="left")
    return _save(fig, path, caption)


def figure_decomposition(table: Table, path: Path, caption: str) -> Path:
    fig, ax = _figure()
    sites = [r["site"] for r in table.rows]
    parts = list(table.rows[0]["_values"])
    width = 0.26
    x = np.arange(len(sites))
    for k, part in enumerate(parts):
        values = [r["_values"][part] for r in table.rows]
        ax.bar(x + (k - 1) * width, values, width=width - 0.02, color=SERIES[k], edgecolor=SURFACE, linewidth=2, label=part)
        for xi, v in zip(x + (k - 1) * width, values):
            ax.text(xi, max(v, 0.0), f"{v:+.2f}%", ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(x, sites, fontsize=8)
    ax.set_ylabel("% of sum|A0 total cost|", color=INK_2, fontsize=9)
    ax.set_title("Where A4's total-cost gap to the oracle comes from", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    return _save(fig, path, caption)


def figure_spec_language(table: Table, path: Path, caption: str) -> Path:
    fig, ax = _figure(6.0, 3.4)
    rows = [r for r in table.rows if r["exact match"] != "-"]
    x = np.arange(len(rows))
    values = [float(r["exact match"].rstrip("%")) for r in rows]
    ax.bar(x, values, width=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for xi, r, v in zip(x, rows, values):
        ax.text(xi, v + 1.5, f"{r['exact match']}\nn={r['episodes']}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [r["group"].replace("multilingual: ", "") for r in rows], fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("spec exact match (%)", color=INK_2, fontsize=9)
    ax.set_title("A4 spec accuracy by language", color=INK, fontsize=10, loc="left")
    return _save(fig, path, caption)


def figure_cost_vs_goal(rows: Sequence[Mapping[str, Any]], path: Path, caption: str) -> Path:
    fig, ax = _figure(6.0, 3.6)
    offsets = {"A4": (6, -12), "A5": (-8, -12), "A6": (6, 5)}  # the top cluster: keep labels apart
    for arm in arms_in(rows):
        mine = [r for r in rows if r["arm"] == arm]
        usd = float(np.mean([r.get("cost_usd") or 0.0 for r in mine]))
        rate = 100.0 * sum(bool(r["gold_satisfied"]) for r in mine) / len(mine)
        color = SERIES[1] if arm == "A4" else SERIES[0]
        ax.scatter([usd], [rate], s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        dx, dy = offsets.get(arm, (6, 4))
        ax.annotate(f"{arm} {ARM_NAMES.get(arm, '')}", (usd, rate), textcoords="offset points", xytext=(dx, dy), ha="left" if dx > 0 else "right", fontsize=7, color=INK)
    ax.set_xlim(-0.012, None)
    ax.set_xticks([t for t in ax.get_xticks() if t >= 0])
    ax.set_xlim(-0.012, None)
    ax.set_xlabel("LLM USD per episode", color=INK_2, fontsize=9)
    ax.set_ylabel("goal satisfied (%)", color=INK_2, fontsize=9)
    ax.set_title("Cost per episode vs goal satisfaction", color=INK, fontsize=10, loc="left")
    return _save(fig, path, caption)


def write_figures(tables: Sequence[Table], rows: Sequence[Mapping[str, Any]], fig_dir: Path, prefix: str, caption: str) -> list[Path]:
    named = {t.name: t for t in tables}
    by_ep = paired(rows)
    return [
        figure_goal_satisfaction(by_ep, arms_in(rows), fig_dir / f"{prefix}_goal_satisfaction.png", caption),
        figure_outage(named["outage_safety"], fig_dir / f"{prefix}_outage_safety.png", caption),
        figure_decomposition(named["error_decomposition"], fig_dir / f"{prefix}_error_decomposition.png", caption),
        figure_spec_language(named["a4_spec_by_language"], fig_dir / f"{prefix}_spec_accuracy_language.png", caption),
        figure_cost_vs_goal(rows, fig_dir / f"{prefix}_cost_vs_goal.png", caption),
    ]


# --- publishing the run ------------------------------------------------------------------------------

RUN_FILES = ("config.json", "episodes_plan.json", "episodes.jsonl")


def files_digest(run_dir: Path) -> str:
    """sha256 over every file in the run directory (names and bytes), to show it was not modified."""
    digest = hashlib.sha256()
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        digest.update(path.relative_to(run_dir).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def redact_paths(node: Any, root: Path) -> Any:
    """Absolute paths inside the repo become repo-relative (posix); other values are unchanged."""
    if isinstance(node, dict):
        return {key: redact_paths(value, root) for key, value in node.items()}
    if isinstance(node, list):
        return [redact_paths(item, root) for item in node]
    if isinstance(node, str) and Path(node).is_absolute():
        try:
            return Path(node).resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return node
    return node


def publish_run(run_dir: Path, out_dir: Path, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Copy the run's files into out_dir (config.json with repo-relative paths) and zip its traces
    (if under the size cap); read only on run_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        if name == "config.json":
            config = json.loads((run_dir / name).read_text(encoding="utf-8"))
            (out_dir / name).write_text(json.dumps(redact_paths(config, root), indent=1) + "\n", encoding="utf-8")
        else:
            shutil.copyfile(run_dir / name, out_dir / name)
    original_config_sha256 = hashlib.sha256((run_dir / "config.json").read_bytes()).hexdigest()
    zipped = out_dir / "traces.zip"
    with zipfile.ZipFile(zipped, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted((run_dir / "traces").glob("*.jsonl")):
            archive.write(path, f"traces/{path.name}")
    size = zipped.stat().st_size
    if size > TRACES_ZIP_MAX_BYTES:
        zipped.unlink()
    return {
        "traces_zip_bytes": size if size <= TRACES_ZIP_MAX_BYTES else None,
        "traces_zip_skipped": size > TRACES_ZIP_MAX_BYTES,
        "original_config_sha256": original_config_sha256,
    }


def run_readme(config: Mapping[str, Any], plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], context: Mapping[str, Any], kind: str = "test") -> str:
    regression = context.get("regression") or {}
    llm_usd = sum(r.get("cost_usd") or 0.0 for r in rows)
    intro = (
        "The frozen test run of VoltProof Bench. Every reported number comes from these files."
        if kind == "test"
        else "E2 (SPEC §6.15): the hard goal set, run on the frozen code plus the additive executor flag. Reported "
        f"separately from the main test run and never pooled with it. intents_hard.yaml sha256 `{config.get('intents_hard_sha256')}`."
    )
    lines = [
        f"# VoltProof Bench {'test run' if kind == 'test' else 'E2 run'} `{context['run_id']}`",
        "",
        intro,
        "",
        "| | |",
        "| --- | --- |",
        f"| run id | `{context['run_id']}` |",
        f"| code tag | `test-freeze` (commit `{context.get('commit', '?')}`) |",
        f"| run fingerprint | `{config['fingerprint']}` |",
        f"| code fingerprint at the run | `{config['code_fingerprint']}` |",
        f"| code fingerprint now | `{context['code_fingerprint_now']}` "
        + ("(same as at the run) |" if context["code_fingerprint_now"] == config["code_fingerprint"] else "(differs only by the additive `outage_override` flag, see below) |"),
        f"| plan sha256 | `{config['plan_sha256']}` |",
        f"| episodes | {plan['n_episodes']} x {len(arms_in(rows))} arms = {len(rows)} results |",
        f"| infrastructure errors | {context.get('infra_errors', 0)} |",
        f"| LLM cost of the run (ledger estimate) | ${llm_usd:.2f} |",
        f"| ledger total at report time | ${context.get('ledger_usd', float('nan')):.2f} of ${float(config['settings']['budget_usd_cap']):.0f} |",
        f"| sha256 of the run directory, before / after this report | `{context['digest_before']}` / `{context['digest_after']}` |",
        "",
        "Files: `config.json` (arguments, fingerprints, settings without the key; in this published copy, local absolute paths are "
        f"redacted to paths relative to the repo root; sha256 of the original `config.json`: `{context.get('original_config_sha256', '?')}`), "
        "`episodes_plan.json` (every episode with its goal "
        "and gold spec, written before the run), `episodes.jsonl` (one line per episode and arm), "
        + ("`traces.zip` (one trace per LLM arm and episode)." if context.get("traces_zip_bytes") else "traces not included (over 50 MB)."),
        "",
        "## Regression check after the freeze",
        "",
        (
            "One change was made to the frozen code after the run: an additive executor flag `outage_override` (default True, the frozen "
            "behaviour) used only by the E1 counterfactual. With it in place, all results were re-evaluated from their stored plans "
            if kind == "test"
            else "This run used the frozen code plus the additive executor flag `outage_override` (default True, the frozen behaviour; "
            "used only by E1). Before any spend, the regression check re-evaluated every result of the frozen test run from its stored plans "
        )
        + 
        "(baselines recomputed, A2/A3 schedules and declarations from the traces, A4 by re-solving its stored specs, whose tool "
        f"outputs must equal the traced ones exactly): {regression.get('rows_checked', '?')}/{regression.get('rows_stored', '?')} rows, "
        f"passed = {regression.get('passed')}, max |difference| in realized_cost {regression.get('max_abs_diff', {}).get('realized_cost')}, "
        f"realized_total_cost {regression.get('max_abs_diff', {}).get('realized_total_cost')} (tolerance 1e-9), gold_satisfied identical; "
        f"{regression.get('a4_replayed_tool_outputs_matching_trace', '?')} A4 tool outputs replayed identically.",
        "",
        "## Aggregation rules (fixed after the run; the per-episode definitions are unchanged)",
        "",
        "- Per-episode percentages (`savings_vs_rule_pct`, `gap_to_oracle_pct`) are never averaged: their base is the rule-based "
        "day cost, which is near zero or negative on sunny clinic and household days. The runner's end-of-run means are not reported.",
        "- Money (`realized_cost`), per site: pooled ratio sum(A0 - X)/sum|A0|, median per-episode difference in site currency per day, "
        "paired win/loss/tie counts against A0; the same for the cost-only template. Never summed across currencies.",
        "- Total cost with goals (`realized_total_cost`), per site: pooled gap sum(X_total - oracle bound)/sum|A0_total|, median "
        "per-episode gap, A4's paired win/loss/tie counts against every arm. The oracle bound is recomputed per episode and checked "
        "against the one implied by A0's stored gap (within 1e-6).",
        "- Ties: |X - Y| <= 1e-6 x max(1, the day's top import price).",
        f"- Goal satisfaction: rates per arm; paired bootstrap 95% CIs for A4 minus A0, A1, A2, A3 ({BOOTSTRAP_RESAMPLES:,} resamples, seed {GLOBAL_SEED}).",
        "- Outage safety: plan physics validity (not applicable to A6, which plans on actual data), unserved critical kWh, executor "
        "overrides, and the E1 counterfactual without the override.",
        "- Error decomposition per site: A4's pooled gap = translation (A4 - A5) + forecast (A5 - A6 executed) + remainder "
        "(A6 executed - oracle bound).",
        "",
        "Limitation: \"actual\" weather is the same-day ICON analysis, not a measurement, so forecast error is understated.",
        "",
    ]
    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------------------------------


def read_traces(run_dir: Path, arm: str = "A4") -> dict[str, list[dict[str, Any]]]:
    out = {}
    for path in sorted((run_dir / "traces").glob(f"{arm}__*.jsonl")):
        out[path.stem.split("__", 1)[1]] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


def report(
    run_dir: Path,
    out_dir: Path,
    *,
    publish: bool = False,
    progress: bool = True,
    references: Callable[[Path, bool], tuple[dict[str, float], dict[str, float]]] = day_references,
    kind: str = "test",
    shortfalls: Callable[[Path], dict[str, tuple[list[Any], float]]] | None = None,
) -> dict[str, Any]:
    """`references` gives p̂ and the oracle bound per episode (tests inject them). kind "test" is the
    main run (report in <out>/report, figures bench_*, published to <out>/test-run); kind "e2" is the
    hard-goal run, reported on its own (<out>/e2, figures e2_*, published to <out>/e2/run)."""
    if kind not in ("test", "e2"):
        raise ValueError(f"unknown report kind {kind!r}")
    digest_before = files_digest(run_dir)
    config, plan, rows = load_run(run_dir)
    errors = read_jsonl(run_dir / "errors.jsonl")
    p_hats, bounds = references(run_dir, progress)
    infos = episode_infos(plan, p_hats, bounds)
    e1_path = out_dir / "e1" / "e1_summary.json"
    e1 = json.loads(e1_path.read_text(encoding="utf-8")) if kind == "test" and e1_path.is_file() else None
    tables = build_tables(rows, infos, read_traces(run_dir), errors, e1)
    notes: list[str] = []
    if kind == "e2":
        targets = (shortfalls or impossible_shortfalls)(run_dir)
        tables.insert(3, shortfall_table(rows, infos, targets))
        notes = e2_notes(plan)
    checks = bound_check(paired(rows), infos)
    report_dir = out_dir / ("report" if kind == "test" else "e2")
    write_tables(tables, report_dir)
    write_csv(report_dir / "tables" / "a4_repairs.csv", repair_events(read_traces(run_dir)))
    caption = f"VoltProof Bench{' E2 (hard goals)' if kind == 'e2' else ''}, run {run_dir.name}. Same-day ICON analysis as 'actual' weather; sites and prices illustrative."
    figures = write_figures(tables, rows, out_dir / "figures", "bench" if kind == "test" else "e2", caption)
    regression_path = out_dir / "test-run" / "regression_check.json"
    regression = json.loads(regression_path.read_text(encoding="utf-8")) if regression_path.is_file() else None
    title = f"# VoltProof Bench report: `{run_dir.name}`" if kind == "test" else f"# VoltProof Bench E2, hard goals: `{run_dir.name}`"
    header = [
        title,
        "",
        *notes,
        f"Generated {datetime.now().isoformat(timespec='seconds')} by `python -m voltproof.bench.report`. "
        f"{plan['n_episodes']} episodes x {len(arms_in(rows))} arms. Oracle-bound check: {checks['checked']} episodes checked, "
        f"{checks['skipped_gap_none']} skipped (A0 gap None), {len(checks['mismatches'])} mismatches, "
        f"{len(checks['below_bound'])} results below the bound. Regression check: "
        + ("passed" if regression and regression.get("passed") else "NOT RUN OR FAILED")
        + ".",
        "",
        "Arms: " + "; ".join(f"{a} {n}" for a, n in ARM_NAMES.items()) + ".",
        "",
        "Limitation: \"actual\" weather is the same-day ICON analysis, not a measurement, so forecast error is understated.",
        "",
    ]
    (report_dir / "REPORT.md").write_text("\n".join(header) + "\n" + "\n".join(markdown(t) for t in tables), encoding="utf-8")
    summary = {
        "run_id": run_dir.name,
        "bound_check": checks,
        "regression_passed": bool(regression and regression.get("passed")),
        "figures": [str(p.relative_to(out_dir)) for p in figures],
        "tables": {t.name: [{k: v for k, v in r.items() if not k.startswith("_")} for r in t.rows] for t in tables},
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if publish:
        publish_dir = out_dir / ("test-run" if kind == "test" else "e2/run")
        published = publish_run(run_dir, publish_dir)
        context = {
            "run_id": run_dir.name,
            "code_fingerprint_now": code_fingerprint(),
            "digest_before": digest_before,
            "digest_after": files_digest(run_dir),
            "regression": regression,
            "infra_errors": len(errors),
            "ledger_usd": read_ledger_totals(Path(config["settings"]["ledger_path"])).total_usd,
            "commit": _commit(),
            **published,
        }
        (publish_dir / "README.md").write_text(run_readme(config, plan, rows, context, kind), encoding="utf-8")
    digest_after = files_digest(run_dir)
    if digest_after != digest_before:
        raise RuntimeError("the run directory changed while the report ran")
    summary["run_digest"] = digest_after
    return summary


def _commit() -> str:
    """The commit the test-freeze tag points to (read only)."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "test-freeze^{commit}"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "?"
    return result.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(prog="python -m voltproof.bench.report", description="VoltProof Bench report")
    parser.add_argument("--run", type=Path, default=FROZEN_RUN, help="finished run directory (read only)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--publish", action="store_true", help="copy the run files into <out>/test-run/ (or <out>/e2/run/) with a README")
    parser.add_argument("--kind", choices=("test", "e2"), default="test", help="e2: the hard-goal run (reported separately)")
    parser.add_argument("--e3", type=Path, help="an e3-* run directory: write the E3 reviewer stress report instead")
    args = parser.parse_args(argv)
    if args.e3:
        e3 = e3_report(args.e3, args.out)
        print(f"E3 report: {args.out / 'e3' / 'REPORT.md'}")
        for figure in e3["figures"]:
            print(f"figure: {figure}")
        return 0
    summary = report(args.run, args.out, publish=args.publish, kind=args.kind)
    print(f"report: {args.out / ('report' if args.kind == 'test' else 'e2') / 'REPORT.md'}")
    print(f"bound check passed: {summary['bound_check']['passed']}; regression passed: {summary['regression_passed']}")
    for figure in summary["figures"]:
        print(f"figure: {figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
