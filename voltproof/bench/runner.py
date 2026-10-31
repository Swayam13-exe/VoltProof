"""VoltProof Bench runner (SPEC §6.11): plan, pilot, sizing and resumable runs.

PowerShell, from the repo root:
    python -m voltproof.bench.runner --split test --plan-only     # free: write and summarize the test plan
    python -m voltproof.bench.runner --pilot                      # dev only: cost per episode, projection
    python -m voltproof.bench.runner --split test                 # the frozen test run (needs a pilot)
    python -m voltproof.bench.runner --split test --resume        # continue the newest test run

Units of work are jobs: (episode, A2|A3|A4) for the LLM arms and (episode, BASE) for the four
baseline rows (A0, A1, A5, A6). Jobs run in plan order with at most `--concurrency` in flight;
workers compute, and only the main thread writes files. A run directory holds config.json (with a
fingerprint of the frozen code and the plan), episodes_plan.json, episodes.jsonl (one line per
episode and arm), errors.jsonl (infrastructure-error attempts) and traces/.

Stops: a fatal LLM error saves what has finished and exits 3; BudgetExceeded exits 4 (that job is
not recorded, so it reruns on resume); a projection above the remaining budget minus the reserve,
or a fingerprint mismatch on resume, exits 5. Infrastructure errors (HTTP 5xx, timeouts, connection
errors after all retries) are retried on resume; after 2 failed resumes the A0 fallback is recorded
with infra_retries=2. Every other LLM outcome, including a rejected request, is a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from tqdm import tqdm

from voltproof.agents.loop import STOP_RUN_ERRORS, ChatLLM, to_json
from voltproof.agents.tools import ArmRun
from voltproof.bench.episode import (
    BASELINE_ARMS,
    LLM_ARMS,
    DayBuilder,
    Episode,
    baseline_results,
    build_test_episodes,
    dev_episodes,
    evaluate,
    run_llm_arm,
    select_per_site,
)
from voltproof.bench.intents import IntentSet, TranslationPendingError, load_intents
from voltproof.bench.effort_sweep import E4_CAP_USD, EFFORTS, CappedLLM, e4_episodes, effort_llm
from voltproof.bench.intents_hard import HardIntentSet, build_hard_episodes, intents_sha256, load_hard_intents
from voltproof.config import PROJECT_ROOT, ConfigError, Settings, get_settings
from voltproof.data import GLOBAL_SEED
from voltproof.data.days import load_splits, split_dates
from voltproof.data.sites import SITES
from voltproof.llm.client import LLMClient, Usage, estimate_cost_usd, read_ledger_totals
from voltproof.models import EpisodeResult, Site

EXIT_OK, EXIT_FATAL, EXIT_BUDGET, EXIT_REFUSED = 0, 3, 4, 5
ALL_ARMS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")
PILOT_ARMS = ("A2", "A3", "A4")
BASE = "BASE"
RESERVE_USD = 30.0
MAX_INFRA_RETRIES = 2
INFRA_STOP = "llm_error:LLMUnavailable"
PILOT_PER_SITE = 3
NOMINAL_TEST_EPISODES = 260  # SPEC §6.11; used only if no test plan has been written yet
RUNS_DIR = PROJECT_ROOT / "runs"
REGRESSION_PATH = PROJECT_ROOT / "results" / "test-run" / "regression_check.json"  # SPEC §6.15: E2/E3 spend only after it passed
FROZEN_PATHS = (
    "voltproof/models.py",
    "voltproof/llm/prompts.py",
    "voltproof/bench/intents.yaml",
    "voltproof/bench/intents.py",
    "voltproof/bench/episode.py",
    "voltproof/agents",
    "voltproof/core",
    "voltproof/baselines",
)

ArmRunner = Callable[[str, ChatLLM, Episode, dict[str, Any]], ArmRun]


# --- the plan ----------------------------------------------------------------------------------


def plan_entry(episode: Episode) -> dict[str, Any]:
    goal = episode.goal
    return {
        "episode_id": episode.episode_id,
        "site_id": episode.site.site_id,
        "date": episode.day.date.isoformat(),
        "template_id": goal.template_id,
        "goal_id": goal.goal_id,
        "language": goal.language,
        "held_out_template": goal.held_out,
        "held_out_site": episode.site.held_out,
        "goal": goal.text,
        "gold": goal.gold.model_dump(mode="json", exclude_defaults=True),
    }


def plan_counts(episodes: Sequence[Episode]) -> dict[str, dict[str, int]]:
    return {
        "site": dict(sorted(Counter(e.site.site_id for e in episodes).items())),
        "template": dict(sorted(Counter(e.goal.template_id for e in episodes).items())),
        "language": dict(sorted(Counter(e.goal.language for e in episodes).items())),
        "set": dict(sorted(Counter("english" if e.goal.goal_id.startswith(f"{e.goal.template_id}_p") else "multilingual" for e in episodes).items())),
    }


def plan_document(episodes: Sequence[Episode], split: str, seed: int) -> dict[str, Any]:
    entries = [plan_entry(e) for e in episodes]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"split": split, "seed": seed, "n_episodes": len(entries), "plan_sha256": digest, "counts": plan_counts(episodes), "episodes": entries}


def code_fingerprint(root: Path = PROJECT_ROOT) -> str:
    """sha256 over the frozen code (SPEC §6.11 test freeze); line endings normalized."""
    digest = hashlib.sha256()
    for path in _frozen_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _frozen_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative in FROZEN_PATHS:
        path = root / relative
        files.extend(sorted(p for p in path.rglob("*") if p.suffix in (".py", ".yaml")) if path.is_dir() else [path])
    return files


def run_fingerprint(plan: Mapping[str, Any], root: Path = PROJECT_ROOT) -> str:
    return hashlib.sha256(f"{code_fingerprint(root)}:{plan['plan_sha256']}".encode("utf-8")).hexdigest()


# --- jobs, results, resume ---------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    episode: Episode
    arm: str  # an LLM arm, or BASE for the baseline rows

    @property
    def key(self) -> tuple[str, str]:
        return (self.episode.episode_id, self.arm)


def make_jobs(episodes: Sequence[Episode], arms: Sequence[str]) -> list[Job]:
    """All arms of one episode before the next, so a stopped run leaves paired episodes."""
    jobs = []
    for episode in episodes:
        if any(arm in BASELINE_ARMS for arm in arms):
            jobs.append(Job(episode, BASE))
        jobs.extend(Job(episode, arm) for arm in arms if arm in LLM_ARMS)
    return jobs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every complete line; a partial or corrupt line (e.g. after a crash) is skipped."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def repair_tail(path: Path) -> None:
    """Make sure the next append starts on a new line, after a crash mid-line."""
    if path.is_file() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("a", encoding="utf-8") as out:
            out.write("\n")


def done_keys(run_dir: Path, base_arms: Sequence[str]) -> set[tuple[str, str]]:
    rows = read_jsonl(run_dir / "episodes.jsonl")
    keys = {(r["episode_id"], r["arm"]) for r in rows if "episode_id" in r and "arm" in r}
    episodes = {episode for episode, _ in keys}
    keys |= {(e, BASE) for e in episodes if base_arms and all((e, arm) in keys for arm in base_arms)}
    return keys


def infra_attempts(run_dir: Path) -> Counter[tuple[str, str]]:
    return Counter((r["episode_id"], r["arm"]) for r in read_jsonl(run_dir / "errors.jsonl"))


def write_line(path: Path, obj: Any) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        out.flush()


@dataclass
class Outcome:
    job: Job
    kind: str  # result | infra | budget
    results: list[EpisodeResult] = field(default_factory=list)
    run: ArmRun | None = None


def execute_job(job: Job, llm: ChatLLM, base_arms: Sequence[str], attempts: int, tags: dict[str, Any], run_arm: ArmRunner) -> Outcome:
    """Compute one job (no file writes: the main thread records outcomes)."""
    if job.arm == BASE:
        return Outcome(job, "result", [r for r in baseline_results(job.episode) if r.arm in base_arms])
    run = run_arm(job.arm, llm, job.episode, tags)
    if run.stop_reason == "budget_exceeded":
        return Outcome(job, "budget", run=run)
    if run.stop_reason == INFRA_STOP and attempts < MAX_INFRA_RETRIES:
        return Outcome(job, "infra", run=run)
    return Outcome(job, "result", [evaluate(job.episode, run, infra_retries=attempts or None)], run)


@dataclass
class RunStats:
    exit_code: int = EXIT_OK
    message: str | None = None
    recorded: int = 0
    infra_errors: int = 0
    cost_usd: float = 0.0
    results: list[EpisodeResult] = field(default_factory=list)


def run_jobs(
    jobs: Sequence[Job],
    llm: ChatLLM,
    run_dir: Path,
    *,
    run_id: str,
    base_arms: Sequence[str],
    concurrency: int = 2,
    run_arm: ArmRunner | None = None,
    progress: bool = True,
) -> RunStats:
    """Run jobs with at most `concurrency` in flight; record each finished job at once."""
    run_arm = run_arm or (lambda arm, client, episode, tags: run_llm_arm(arm, client, episode, tags))
    for name in ("episodes.jsonl", "errors.jsonl"):
        repair_tail(run_dir / name)
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    attempts = infra_attempts(run_dir)
    stats = RunStats()
    stop = threading.Event()
    queue = list(jobs)
    in_flight: dict[Future[Outcome], Job] = {}
    bar = tqdm(total=len(queue), desc=run_id, unit="job", disable=not progress)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        while queue or in_flight:
            while queue and not stop.is_set() and len(in_flight) < max(1, concurrency):
                job = queue.pop(0)
                tags = {"run": run_id, "episode": job.episode.episode_id}
                in_flight[pool.submit(execute_job, job, llm, base_arms, attempts[job.key], tags, run_arm)] = job
            if not in_flight:
                break
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                job = in_flight.pop(future)
                bar.update(1)
                try:
                    outcome = future.result()
                except STOP_RUN_ERRORS as exc:
                    if not stop.is_set():
                        status = f" (HTTP {exc.status_code})" if getattr(exc, "status_code", None) else ""
                        stats.exit_code, stats.message = EXIT_FATAL, f"STOPPED: {type(exc).__name__}{status}: {exc}"
                    stop.set()
                    continue
                _record(outcome, run_dir, attempts, stats)
                if outcome.kind == "budget" and stats.exit_code == EXIT_OK:
                    stats.exit_code, stats.message = EXIT_BUDGET, "STOPPED: BudgetExceeded (this job is not recorded; resume reruns it)"
                    stop.set()
                bar.set_postfix(cost=f"${stats.cost_usd:.3f}", infra=stats.infra_errors, refresh=False)
    bar.close()
    return stats


def _record(outcome: Outcome, run_dir: Path, attempts: Counter[tuple[str, str]], stats: RunStats) -> None:
    job, run = outcome.job, outcome.run
    if run is not None:
        stats.cost_usd += run.totals.cost_usd
        trace = run_dir / "traces" / f"{job.arm}__{job.episode.episode_id}.jsonl"
        trace.write_text("".join(json.dumps(e.model_dump(mode="json"), ensure_ascii=False) + "\n" for e in run.trace.events), encoding="utf-8")
    if outcome.kind == "infra":
        attempts[job.key] += 1
        stats.infra_errors += 1
        write_line(
            run_dir / "errors.jsonl",
            {"episode_id": job.episode.episode_id, "arm": job.arm, "attempt": attempts[job.key], "stop_reason": run.stop_reason if run else None, "ts": _now()},
        )
        return
    if outcome.kind == "budget":
        return
    for result in outcome.results:
        extra = {"explanation": run.explanation, "spec": run.spec.model_dump(mode="json") if run and run.spec else None} if run else {}
        write_line(run_dir / "episodes.jsonl", {**result.model_dump(mode="json"), **extra})
        stats.results.append(result)
        stats.recorded += 1


# --- pilot and projection ----------------------------------------------------------------------


def token_usd(result: Mapping[str, Any] | EpisodeResult, settings: Settings) -> float:
    """What the episode's tokens cost at list prices, even when the client cache made it free."""
    row = result.model_dump() if isinstance(result, EpisodeResult) else result
    usage = Usage(prompt=row.get("tokens_in") or 0, completion=row.get("tokens_out") or 0, cached=row.get("tokens_cached") or 0)
    return estimate_cost_usd(usage, settings)


def pilot_episodes(intents: IntentSet, splits: Mapping[str, Any], day_builder: DayBuilder | None = None, seed: int = GLOBAL_SEED) -> list[Episode]:
    """3 dev episodes per dev site, mixed templates; never a test day or a held-out site or template."""
    episodes = select_per_site(dev_episodes(intents, splits, day_builder), PILOT_PER_SITE, seed)
    dev_dates = {(e["site_id"], e["date"]) for e in splits["dev"]}
    for episode in episodes:
        if episode.site.held_out or episode.goal.held_out or (episode.site.site_id, episode.day.date.isoformat()) not in dev_dates:
            raise ValueError(f"pilot episode {episode.episode_id} is not a dev episode")
    return episodes


def summarize_arms(rows: Iterable[Mapping[str, Any] | EpisodeResult], settings: Settings, arms: Sequence[str] = PILOT_ARMS) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    dumped = [r.model_dump() if isinstance(r, EpisodeResult) else dict(r) for r in rows]
    for arm in arms:
        mine = [r for r in dumped if r.get("arm") == arm]
        if not mine:
            continue
        priced = [token_usd(r, settings) for r in mine]
        out[arm] = {
            "episodes": len(mine),
            "mean_tokens_in": sum(r.get("tokens_in") or 0 for r in mine) / len(mine),
            "mean_tokens_cached": sum(r.get("tokens_cached") or 0 for r in mine) / len(mine),
            "mean_tokens_out": sum(r.get("tokens_out") or 0 for r in mine) / len(mine),
            "mean_token_usd": sum(priced) / len(mine),
            "max_token_usd": max(priced),
            "mean_billed_usd": sum(r.get("cost_usd") or 0.0 for r in mine) / len(mine),
            "fallbacks": sum(1 for r in mine if r.get("fallback_used")),
            "escalations": sum(1 for r in mine if r.get("escalated")),
            "parse_failures": sum(1 for r in mine if r.get("stop_reason") == "parse_failure"),
            "infra_fallbacks": sum(1 for r in mine if r.get("infra_retries")),
        }
    return out


def projection(per_arm: Mapping[str, Mapping[str, float]], jobs_per_arm: Mapping[str, int], key: str = "mean_token_usd") -> float:
    missing = [arm for arm, n in jobs_per_arm.items() if n and arm not in per_arm]
    if missing:
        raise ValueError(f"the pilot has no cost estimate for arms {missing}")
    return sum(per_arm[arm][key] * n for arm, n in jobs_per_arm.items() if n)


def budget_check(projected: float, settings: Settings, reserve: float = RESERVE_USD) -> tuple[bool, str]:
    spent = read_ledger_totals(settings.ledger_path).total_usd
    remaining = settings.budget_usd_cap - spent
    allowed = remaining - reserve
    ok = projected <= allowed
    return ok, (
        f"projected ${projected:.2f}; spent ${spent:.4f} of ${settings.budget_usd_cap:.2f}; remaining ${remaining:.2f}; "
        f"allowed after the ${reserve:.0f} reserve ${allowed:.2f} -> {'OK' if ok else 'REFUSED'}"
    )


def latest_pilot_summary(runs_dir: Path) -> dict[str, Any] | None:
    paths = sorted(runs_dir.glob("pilot-*/pilot_summary.json"))
    return json.loads(paths[-1].read_text(encoding="utf-8")) if paths else None


def test_plan_size(runs_dir: Path) -> tuple[int, str]:
    """Episodes in the newest written test plan (no held-out data is loaded here), else SPEC's number."""
    paths = sorted(runs_dir.glob("test-*/episodes_plan.json"))
    if paths:
        return json.loads(paths[-1].read_text(encoding="utf-8"))["n_episodes"], str(paths[-1].parent.name)
    return NOMINAL_TEST_EPISODES, "SPEC nominal"


def jobs_per_arm(jobs: Sequence[Job]) -> dict[str, int]:
    counts = Counter(job.arm for job in jobs if job.arm in LLM_ARMS)
    return {arm: counts.get(arm, 0) for arm in LLM_ARMS}


# --- output --------------------------------------------------------------------------------------


def format_counts(plan: Mapping[str, Any]) -> str:
    lines = [f"{plan['split']} plan: {plan['n_episodes']} episodes (seed {plan['seed']}, sha256 {plan['plan_sha256'][:12]})"]
    for name, counts in plan["counts"].items():
        lines.append(f"  by {name}:")
        lines.extend(f"    {key:<28} {value:>4}" for key, value in counts.items())
    return "\n".join(lines)


def format_trace(run: ArmRun) -> str:
    lines = []
    for event in run.trace.events:
        data = event.data
        if event.kind == "llm_call":
            lines.append(
                f"[turn {event.turn}] LLM {event.name}: in {data.get('tokens_in')} (cached {data.get('tokens_cached')}) / "
                f"out {data.get('tokens_out')} (reasoning {data.get('tokens_reasoning')}), ${data.get('cost_usd', 0):.6f}, "
                f"finish={data.get('finish_reason')}, cache_hit={data.get('cache_hit')}"
            )
            if data.get("reasoning"):
                lines.append(f"    reasoning: {' '.join(str(data['reasoning']).split())}")
            if data.get("content"):
                lines.append(f"    content: {data['content']}")
        elif event.kind == "tool_call":
            lines.append(f"  -> {event.name} {to_json(data.get('args'))}")
        elif event.kind == "tool_result":
            lines.append(f"  <- {event.name} {to_json(data.get('result'))}")
        elif event.kind == "gate":
            lines.append(f"  GATE {event.name}: {event.summary}")
            for failed in data.get("failed_checks", []):
                lines.append(f"     x {failed['check']}: {failed['message']}")
        elif event.kind == "review":
            lines.append(f"  REVIEW approve={data.get('approved')} issues={to_json(data.get('issues'))}")
            lines.append(f"     raw: {data.get('raw')}")
        else:
            lines.append(f"  {event.kind.upper()} {event.name}: {event.summary or to_json(data)}")
    return "\n".join(lines)


def format_arm_table(rows: Sequence[Mapping[str, Any]], errors: Sequence[Mapping[str, Any]]) -> str:
    def mean(values: list[float | None]) -> str:
        present = [v for v in values if v is not None]
        return "   n/a" if not present else f"{sum(present) / len(present):+6.1f}"

    infra = Counter(e.get("arm") for e in errors)
    lines = ["arm   n   gold%  valid%   sav%   gap%  fallback  escal  infra_err  infra_fb   tok_in/ep  tok_out/ep   usd"]
    for arm in ALL_ARMS:
        mine = [r for r in rows if r.get("arm") == arm]
        if not mine:
            continue
        n = len(mine)
        lines.append(
            f"{arm:<4} {n:>3}  {100 * sum(bool(r['gold_satisfied']) for r in mine) / n:5.1f}  {100 * sum(bool(r['plan_physics_valid']) for r in mine) / n:6.1f}"
            f"  {mean([r.get('savings_vs_rule_pct') for r in mine])} {mean([r.get('gap_to_oracle_pct') for r in mine])}"
            f"  {sum(bool(r.get('fallback_used')) for r in mine):>8}  {sum(bool(r.get('escalated')) for r in mine):>5}"
            f"  {infra.get(arm, 0):>9}  {sum(bool(r.get('infra_retries')) for r in mine):>8}"
            f"  {sum(r.get('tokens_in') or 0 for r in mine) / n:>10.0f}  {sum(r.get('tokens_out') or 0 for r in mine) / n:>10.0f}"
            f"  ${sum(r.get('cost_usd') or 0.0 for r in mine):.4f}"
        )
    return "\n".join(lines)


def format_pilot(per_arm: Mapping[str, Mapping[str, float]], n_test: int, source: str, check: str, pessimistic: float, projected: float) -> str:
    lines = ["arm  episodes  tok_in  tok_cached  tok_out  token_usd/ep  billed_usd/ep  max_usd/ep  fallbacks  escal  parse_fail  infra_fb"]
    for arm, s in per_arm.items():
        lines.append(
            f"{arm:<4} {s['episodes']:>8.0f}  {s['mean_tokens_in']:>6.0f}  {s['mean_tokens_cached']:>10.0f}  {s['mean_tokens_out']:>7.0f}"
            f"  {s['mean_token_usd']:>12.4f}  {s['mean_billed_usd']:>13.4f}  {s['max_token_usd']:>10.4f}  {s['fallbacks']:>9.0f}"
            f"  {s['escalations']:>5.0f}  {s['parse_failures']:>10.0f}  {s['infra_fallbacks']:>8.0f}"
        )
    lines.append(f"\nTest set: {n_test} episodes x {', '.join(per_arm)} ({source}).")
    for arm, s in per_arm.items():
        lines.append(f"  {arm}: {s['mean_token_usd']:.4f} x {n_test} = ${s['mean_token_usd'] * n_test:.2f}   (pessimistic, max x n: ${s['max_token_usd'] * n_test:.2f})")
    lines.append(f"Projected test cost: ${projected:.2f} (pessimistic ${pessimistic:.2f})")
    lines.append(f"Budget: {check}")
    return "\n".join(lines)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- CLI ---------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m voltproof.bench.runner", description="VoltProof Bench runner")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--arms", help=f"comma-separated, default all ({','.join(ALL_ARMS)}); the pilot uses {','.join(PILOT_ARMS)}")
    parser.add_argument("--sites", help="comma-separated site ids")
    parser.add_argument("--limit", type=int, help="the first N episodes in plan order")
    parser.add_argument("--episodes", type=Path, help="JSON list of episode ids, or an episodes_plan.json")
    parser.add_argument("--pilot", action="store_true", help="dev-only cost pilot and test-cost projection")
    parser.add_argument("--plan-only", action="store_true", help="write and summarize the plan; no LLM call")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--force", action="store_true", help="skip the projection refusal")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--intents", choices=("main", "hard", "e4"), default="main", help="hard: the E2 goal set; e4: the E4 effort-sweep set (SPEC §6.15); test split only")
    parser.add_argument("--agent-effort", choices=EFFORTS, help="E4: the A4 agent's reasoning effort (the reviewer stays medium); A4 only, concurrency 1, E4 cap")
    parser.add_argument("--cost-cap", type=float, help=f"E4: lower the E4 cap (at most ${E4_CAP_USD:.0f}, the default)")
    args = parser.parse_args(argv)
    if args.intents in ("hard", "e4") and (args.pilot or args.split != "test"):
        parser.error(f"--intents {args.intents} runs on the test split only (no pilot)")
    if args.cost_cap is not None and not args.agent_effort:
        parser.error("--cost-cap applies to E4 runs (--agent-effort) only")
    if args.cost_cap is not None and not 0 < args.cost_cap <= E4_CAP_USD:
        parser.error(f"--cost-cap must be in (0, {E4_CAP_USD:g}]")
    if args.pilot:
        args.split = "dev"
    default = PILOT_ARMS if args.pilot else ALL_ARMS
    args.arms = [a.strip() for a in (args.arms.split(",") if args.arms else default) if a.strip()]
    unknown = [a for a in args.arms if a not in ALL_ARMS]
    if unknown:
        parser.error(f"unknown arms {unknown}")
    if args.agent_effort:
        if args.arms != ["A4"]:
            parser.error("--agent-effort sweeps the A4 agent only: use --arms A4")
        args.concurrency = 1  # two clients (swept effort, medium reviewer) share the ledger: one call at a time
    if args.intents == "e4" and not args.plan_only and not args.agent_effort:
        parser.error("--intents e4 runs need --agent-effort (medium results are reused from the frozen runs, not re-run)")
    return args


def select(episodes: list[Episode], args: argparse.Namespace, sites: Mapping[str, Site]) -> list[Episode]:
    if args.sites:
        wanted = [s.strip() for s in args.sites.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in sites]
        if unknown:
            raise SystemExit(f"unknown sites {unknown}")
        if args.split == "dev" and any(sites[s].held_out for s in wanted):
            raise SystemExit("held-out sites are not part of the dev split")
        episodes = [e for e in episodes if e.site.site_id in wanted]
    if args.episodes:
        data = json.loads(args.episodes.read_text(encoding="utf-8"))
        ids = [e["episode_id"] for e in data["episodes"]] if isinstance(data, dict) else list(data)
        known = {e.episode_id for e in episodes}
        missing = [i for i in ids if i not in known]
        if missing:
            raise SystemExit(f"unknown episode ids: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        episodes = [e for e in episodes if e.episode_id in set(ids)]
    return episodes[: args.limit] if args.limit is not None else episodes


def regression_refusal(path: Path) -> str | None:
    """Why an E2/E3 run may not spend (SPEC §6.15), or None: the regression check must have passed on
    the code as it is now."""
    if not path.is_file():
        return f"no regression check at {path}; run python -m voltproof.bench.counterfactual regression first"
    check = json.loads(path.read_text(encoding="utf-8"))
    if not check.get("passed"):
        return f"the regression check at {path} did not pass"
    if check.get("code_fingerprint") != code_fingerprint():
        return "the frozen code changed since the regression check passed; run it again"
    return None


def run_directory(args: argparse.Namespace, runs_dir: Path) -> Path:
    prefix = "pilot" if args.pilot else ("e2" if args.intents == "hard" else args.split)
    if args.intents == "e4":
        prefix = f"e4-{args.agent_effort}" if args.agent_effort else "e4-plan"
    elif args.agent_effort:
        prefix = f"e4-probe-{args.agent_effort}"
    if args.resume:
        if args.run_id:
            path = runs_dir / args.run_id
        else:
            candidates = sorted(p for p in runs_dir.glob(f"{prefix}-*") if _resumable(p))
            if not candidates:
                raise SystemExit(f"nothing to resume: no {prefix}-* run in {runs_dir}")
            path = candidates[-1]
        if not (path / "config.json").is_file():
            raise SystemExit(f"nothing to resume at {path}")
        return path
    if args.run_id:
        return runs_dir / args.run_id
    base = f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path, n = runs_dir / base, 1
    while path.exists():  # two runs started in the same second
        n += 1
        path = runs_dir / f"{base}-{n}"
    return path


def _resumable(path: Path) -> bool:
    config = path / "config.json"
    return config.is_file() and not json.loads(config.read_text(encoding="utf-8")).get("plan_only")


def main(
    argv: Sequence[str] | None = None,
    *,
    llm: ChatLLM | None = None,
    settings: Settings | None = None,
    intents: IntentSet | None = None,
    splits: Mapping[str, Any] | None = None,
    sites: Mapping[str, Site] | None = None,
    day_builder: DayBuilder | None = None,
    runs_dir: Path | None = None,
    run_arm: ArmRunner | None = None,
    progress: bool = True,
    hard_intents: HardIntentSet | None = None,
    regression_path: Path | None = None,
) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    args = parse_args(argv)
    intents = intents or load_intents()
    splits = splits if splits is not None else load_splits()
    sites = sites or SITES
    runs_dir = runs_dir or RUNS_DIR

    if args.intents == "hard":
        try:
            episodes = build_hard_episodes(hard_intents or load_hard_intents(), splits, sites, day_builder, args.seed, allow_pending=args.plan_only)
        except TranslationPendingError as exc:
            print(f"REFUSED: {exc}")
            return EXIT_REFUSED
        if not args.plan_only:
            refusal = regression_refusal(regression_path or REGRESSION_PATH)
            if refusal:
                print(f"REFUSED: {refusal}")
                return EXIT_REFUSED
    elif args.intents == "e4":
        episodes = e4_episodes(intents, hard_intents, splits, sites, day_builder, args.seed)
        if not args.plan_only:
            refusal = regression_refusal(regression_path or REGRESSION_PATH)
            if refusal:
                print(f"REFUSED: {refusal}")
                return EXIT_REFUSED
    elif args.pilot:
        episodes = pilot_episodes(intents, splits, day_builder, args.seed)
    elif args.split == "dev":
        episodes = dev_episodes(intents, splits, day_builder)
    else:
        episodes = build_test_episodes(intents, splits, sites, day_builder, args.seed)
    episodes = select(episodes, args, sites)
    split_label = {"hard": "test-hard", "e4": "test-e4"}.get(args.intents, args.split)
    plan = plan_document(episodes, "pilot" if args.pilot else split_label, args.seed)
    fingerprint = run_fingerprint(plan)

    run_dir = run_directory(args, runs_dir)
    if args.resume:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("fingerprint") != fingerprint:
            print(f"REFUSED: {run_dir.name} was made with different frozen code or a different plan; a rerun must be a new run.")
            return EXIT_REFUSED
    elif run_dir.exists():
        print(f"REFUSED: {run_dir} already exists (use --resume, or another --run-id)")
        return EXIT_REFUSED
    print(format_counts(plan))
    if args.plan_only:
        _create_run(run_dir, args, fingerprint, None, plan)
        print(f"plan written to {run_dir / 'episodes_plan.json'} (no LLM call)")
        return EXIT_OK

    try:
        settings = settings or get_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    base_arms = [a for a in args.arms if a in BASELINE_ARMS]
    jobs = make_jobs(episodes, args.arms)
    finished = done_keys(run_dir, base_arms) if args.resume else set()
    pending = [job for job in jobs if job.key not in finished]
    print(f"\n{len(jobs)} jobs ({len(pending)} pending) for arms {', '.join(args.arms)}")

    if args.split == "test" and not args.pilot:
        if _check_projection(pending, settings, runs_dir, args.force):
            return EXIT_REFUSED  # nothing was created or run
    if not args.resume:
        _create_run(run_dir, args, fingerprint, settings, plan)

    client = llm or LLMClient(settings)
    if args.agent_effort:
        client = CappedLLM(llm or effort_llm(settings, args.agent_effort), settings, args.cost_cap or E4_CAP_USD)
        print(f"E4: A4 agent at reasoning_effort={args.agent_effort}, reviewer at medium; concurrency 1; E4 cap ${args.cost_cap or E4_CAP_USD:.2f}")
    stats = run_jobs(pending, client, run_dir, run_id=run_dir.name, base_arms=base_arms, concurrency=args.concurrency, run_arm=run_arm, progress=progress)
    if stats.message:
        print(f"\n{stats.message}. Partial results saved in {run_dir}")
    rows = read_jsonl(run_dir / "episodes.jsonl")
    print("\n" + format_arm_table(rows, read_jsonl(run_dir / "errors.jsonl")))
    if args.pilot:
        _finish_pilot(rows, settings, runs_dir, run_dir)
    spent = read_ledger_totals(settings.ledger_path).total_usd
    print(f"\nthis session: {stats.recorded} results, {stats.infra_errors} infrastructure errors, LLM ${stats.cost_usd:.4f}; ledger ${spent:.4f} of ${settings.budget_usd_cap:.2f}")
    print(f"results: {run_dir}")
    return stats.exit_code


def _check_projection(pending: Sequence[Job], settings: Settings, runs_dir: Path, force: bool) -> bool:
    """True when the run must be refused (SPEC §6.11)."""
    summary = latest_pilot_summary(runs_dir)
    if summary is None:
        if force:
            print("WARNING: no pilot summary; --force given, starting without a projection")
            return False
        print("REFUSED: no pilot summary found; run --pilot first (or pass --force)")
        return True
    try:
        projected = projection(summary["arms"], jobs_per_arm(pending))
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return not force
    ok, message = budget_check(projected, settings)
    print(f"Projection from {summary['run_id']}: {message}")
    if not ok and not force:
        print("REFUSED: the projection exceeds the remaining budget minus the reserve (pass --force to override)")
        return True
    return False


def _finish_pilot(rows: Sequence[Mapping[str, Any]], settings: Settings, runs_dir: Path, run_dir: Path) -> None:
    per_arm = summarize_arms(rows, settings)
    n_test, source = test_plan_size(runs_dir)
    counts = {arm: n_test for arm in per_arm}
    projected = projection(per_arm, counts)
    pessimistic = projection(per_arm, counts, "max_token_usd")
    _, check = budget_check(projected, settings)
    print("\nPILOT (dev split only)\n" + format_pilot(per_arm, n_test, source, check, pessimistic, projected))
    summary = {"run_id": run_dir.name, "created": _now(), "arms": per_arm, "test_episodes": n_test, "test_plan_source": source, "projected_usd": projected, "pessimistic_usd": pessimistic}
    (run_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


def _create_run(run_dir: Path, args: argparse.Namespace, fingerprint: str, settings: Settings | None, plan: Mapping[str, Any]) -> None:
    """Create the run directory with its plan (written before anything runs) and config."""
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "episodes_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    safe = None
    if settings is not None:  # never the key
        safe = {f.name: str(getattr(settings, f.name)) for f in fields(settings) if f.name != "baseten_api_key"}
    config = {
        "created": _now(),
        "plan_only": bool(args.plan_only),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "fingerprint": fingerprint,
        "code_fingerprint": code_fingerprint(),
        "plan_sha256": plan["plan_sha256"],
        "intents_hard_sha256": intents_sha256() if getattr(args, "intents", "main") in ("hard", "e4") else None,
        "agent_reasoning_effort": getattr(args, "agent_effort", None),
        "reviewer_reasoning_effort": "medium" if getattr(args, "agent_effort", None) else None,
        "e4_cap_usd": (args.cost_cap or E4_CAP_USD) if getattr(args, "agent_effort", None) else None,
        "settings": safe,
        "ledger_usd_at_start": read_ledger_totals(settings.ledger_path).total_usd if settings else None,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
