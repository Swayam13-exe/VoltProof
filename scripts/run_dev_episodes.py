"""Run N dev episodes per LLM arm (A2, A3, A4) end to end, for the P3 gate (SPEC §8).

PowerShell, from the repo root:
    python scripts\\run_dev_episodes.py --arms A4 --n 1 --dry-run      # free: lists episodes, no LLM call
    python scripts\\run_dev_episodes.py --arms A4 --n 1 --show-trace   # paid: one A4 episode, full trace
    python scripts\\run_dev_episodes.py --arms A2,A3,A4 --n 3

DEV days only (held-out sites and held-out templates are refused). All arms get the same
episodes. Each episode is executed on actual data with P2's executor and scored with P2's
metrics against A0 and the oracle. Results and traces go to runs/dev-<timestamp>/.

Stops immediately on a fatal LLM error (HTTP 401/402/403, a 429 that outlasts the retries, a
missing key): prints the error class and exits 3, without further calls. BudgetExceeded stops
cleanly with exit 4.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

from voltproof.agents.loop import STOP_RUN_ERRORS, ChatLLM, to_json  # noqa: E402
from voltproof.baselines.optimizer_arms import a5_plan  # noqa: E402
from voltproof.bench.episode import LLM_ARMS, Episode, dev_episodes, evaluate, run_llm_arm, select_episodes  # noqa: E402
from voltproof.bench.intents import load_intents  # noqa: E402
from voltproof.bench.runner import format_trace, write_line  # noqa: E402
from voltproof.config import PROJECT_ROOT, ConfigError, Settings, get_settings  # noqa: E402
from voltproof.data import GLOBAL_SEED  # noqa: E402
from voltproof.llm.client import LLMClient, read_ledger_totals  # noqa: E402
from voltproof.models import EpisodeResult  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "runs"
EXIT_FATAL, EXIT_BUDGET = 3, 4


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", default="A4", help="comma-separated LLM arms: A2,A3,A4")
    parser.add_argument("--n", type=int, default=1, help="episodes per arm")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--episode", action="append", default=[], help="run this episode id (repeatable) instead of the seeded pick")
    parser.add_argument("--dry-run", action="store_true", help="list the episodes and gold feasibility; no LLM call")
    parser.add_argument("--show-trace", action="store_true", help="print every episode's full trace")
    args = parser.parse_args(argv)
    args.arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = [arm for arm in args.arms if arm not in LLM_ARMS]
    if unknown:
        parser.error(f"unknown arms {unknown}; choose from {', '.join(LLM_ARMS)}")
    return args


def choose(episodes: list[Episode], args: argparse.Namespace) -> list[Episode]:
    if args.episode:
        by_id = {e.episode_id: e for e in episodes}
        missing = [i for i in args.episode if i not in by_id]
        if missing:
            raise SystemExit(f"unknown episode ids: {missing}")
        return [by_id[i] for i in args.episode]
    return select_episodes(episodes, args.n, args.seed)


def print_dry_run(selected: list[Episode], episodes: list[Episode], arms: list[str]) -> None:
    print(f"{len(episodes)} dev episodes available; selected {len(selected)} for arms {', '.join(arms)}:")
    for e in selected:
        print(f"\n  {e.episode_id}\n    goal: {e.goal.text}\n    gold: {to_json(e.goal.gold.model_dump(mode='json', exclude_defaults=True))}")
    print("\nGold feasibility on the forecast (A5 = optimizer with the gold spec), every dev site-day x template:")
    seen = set()
    for e in episodes:
        key = (e.site.site_id, e.day.date, e.goal.template_id)
        if key in seen:
            continue
        seen.add(key)
        plan = a5_plan(e.day, e.site, e.goal.gold)
        status = "ok" if not plan.shortfalls else "; ".join(s.message for s in plan.shortfalls)
        print(f"  {e.site.site_id:<21} {e.day.date}  {e.goal.template_id:<21} {status}")


def row(result: EpisodeResult) -> str:
    def pct(value: float | None) -> str:
        return "   n/a" if value is None else f"{value:+6.1f}"

    return (
        f"{result.arm:<3} {result.episode_id:<58} {result.plan_source or '-':<20} {'yes' if result.gold_satisfied else 'NO ':<4}"
        f"{pct(result.savings_vs_rule_pct)} {pct(result.gap_to_oracle_pct)}  {'yes' if result.plan_physics_valid else 'NO ':<4}"
        f"{result.repair_turns or 0:>3} {result.tokens_in or 0:>7} {result.tokens_out or 0:>6}  ${result.cost_usd or 0:.4f}"
    )


def main(argv: Sequence[str] | None = None, *, llm: ChatLLM | None = None, settings: Settings | None = None) -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    args = parse_args(argv)
    episodes = dev_episodes(load_intents())
    selected = choose(episodes, args)
    if args.dry_run:
        print_dry_run(selected, episodes, args.arms)
        return 0
    try:
        settings = settings or get_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    llm = llm or LLMClient(settings)

    run_dir = RUNS_DIR / f"dev-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"arms": args.arms, "seed": args.seed, "episodes": [e.episode_id for e in selected]}, indent=2), encoding="utf-8"
    )
    spent_before = read_ledger_totals(settings.ledger_path).total_usd
    results: list[EpisodeResult] = []
    stop_code = 0
    jobs = [(arm, episode) for arm in args.arms for episode in selected]
    for arm, episode in tqdm(jobs, desc="episodes", unit="ep"):
        try:
            run = run_llm_arm(arm, llm, episode, tags={"script": "run_dev_episodes"})
        except STOP_RUN_ERRORS as exc:
            status = f" (HTTP {exc.status_code})" if getattr(exc, "status_code", None) else ""
            print(f"\nSTOPPED: {type(exc).__name__}{status}: {exc}. No further LLM calls were made. Partial results: {run_dir}")
            stop_code = EXIT_FATAL
            break
        result = evaluate(episode, run)
        results.append(result)
        write_line(run_dir / "episodes.jsonl", {**result.model_dump(mode="json"), "explanation": run.explanation, "spec": run.spec.model_dump(mode="json") if run.spec else None})
        trace_path = run_dir / "traces" / f"{arm}__{episode.episode_id}.jsonl"
        for event in run.trace.events:
            write_line(trace_path, event.model_dump(mode="json"))
        if args.show_trace:
            tqdm.write(f"\n===== {arm} {episode.episode_id}\nGOAL: {episode.goal.text}\nGOLD: {to_json(episode.goal.gold.model_dump(mode='json', exclude_defaults=True))}")
            tqdm.write(format_trace(run))
            tqdm.write(f"EXPLANATION: {run.explanation}")
            tqdm.write(f"SELECTED SPEC: {to_json(run.spec.model_dump(mode='json', exclude_defaults=True)) if run.spec else None}")
            tqdm.write(f"RESULT: {to_json(result.model_dump(mode='json'))}")
        if run.stop_reason == "budget_exceeded":
            print(f"\nSTOPPED: BudgetExceeded. Partial results: {run_dir}")
            stop_code = EXIT_BUDGET
            break

    print("\narm episode                                                    source               gold  sav%   gap%  valid rep  tok_in tok_out  cost")
    for result in results:
        print(row(result))
    spent_after = read_ledger_totals(settings.ledger_path).total_usd
    total = sum(r.cost_usd or 0.0 for r in results)
    print(f"\n{len(results)} episodes, LLM cost ${total:.4f}; ledger ${spent_before:.4f} -> ${spent_after:.4f} of ${settings.budget_usd_cap:.2f}")
    print(f"results: {run_dir}")
    return stop_code


if __name__ == "__main__":
    sys.exit(main())
