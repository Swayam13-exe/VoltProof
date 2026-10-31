# VoltProof Bench E2 run `e2-20261031-130234`

E2 (SPEC §6.15): the hard goal set, run on the frozen code plus the additive executor flag. Reported separately from the main test run and never pooled with it. intents_hard.yaml sha256 `f09813e0cf3758d92dd3a19bb3ebe5a170e05c151e992930636ab70a43583602`.

| | |
| --- | --- |
| run id | `e2-20261031-130234` |
| code tag | `test-freeze` (commit `2dacaa5`) |
| run fingerprint | `2a21a541c78f316d06b98dc340c791b2a4b68f268ec8eb89452c5c72c62365a3` |
| code fingerprint at the run | `20abe8704fc57c9aa9132bf53e7da21c2023985f3a00a3ccc5c8813f6e48f55a` |
| code fingerprint now | `20abe8704fc57c9aa9132bf53e7da21c2023985f3a00a3ccc5c8813f6e48f55a` (same as at the run) |
| plan sha256 | `64e83b05b71cd941df5a6ce2ad03e562484c0a7e50180dd79dfd4dec9b6fcf87` |
| episodes | 101 x 7 arms = 707 results |
| infrastructure errors | 0 |
| LLM cost of the run (ledger estimate) | $8.34 |
| ledger total at report time | $32.04 of $85 |
| sha256 of the run directory, before / after this report | `e30757fad9210fcdce97fad4743378175945725d046cb2d8399dadcb44fdb9be` / `e30757fad9210fcdce97fad4743378175945725d046cb2d8399dadcb44fdb9be` |

Files: `config.json` (arguments, fingerprints, settings without the key; in this published copy, local absolute paths are redacted to paths relative to the repo root; sha256 of the original `config.json`: `c68afcc927fdb1147c49553fbe430c9b1d45ba7f5d8b46fa6d78a1ad703fc4e6`), `episodes_plan.json` (every episode with its goal and gold spec, written before the run), `episodes.jsonl` (one line per episode and arm), `traces.zip` (one trace per LLM arm and episode).

## Regression check after the freeze

This run used the frozen code plus the additive executor flag `outage_override` (default True, the frozen behaviour; used only by E1). Before any spend, the regression check re-evaluated every result of the frozen test run from its stored plans (baselines recomputed, A2/A3 schedules and declarations from the traces, A4 by re-solving its stored specs, whose tool outputs must equal the traced ones exactly): 1820/1820 rows, passed = True, max |difference| in realized_cost 0.0, realized_total_cost 0.0 (tolerance 1e-9), gold_satisfied identical; 524 A4 tool outputs replayed identically.

## Aggregation rules (fixed after the run; the per-episode definitions are unchanged)

- Per-episode percentages (`savings_vs_rule_pct`, `gap_to_oracle_pct`) are never averaged: their base is the rule-based day cost, which is near zero or negative on sunny clinic and household days. The runner's end-of-run means are not reported.
- Money (`realized_cost`), per site: pooled ratio sum(A0 - X)/sum|A0|, median per-episode difference in site currency per day, paired win/loss/tie counts against A0; the same for the cost-only template. Never summed across currencies.
- Total cost with goals (`realized_total_cost`), per site: pooled gap sum(X_total - oracle bound)/sum|A0_total|, median per-episode gap, A4's paired win/loss/tie counts against every arm. The oracle bound is recomputed per episode and checked against the one implied by A0's stored gap (within 1e-6).
- Ties: |X - Y| <= 1e-6 x max(1, the day's top import price).
- Goal satisfaction: rates per arm; paired bootstrap 95% CIs for A4 minus A0, A1, A2, A3 (10,000 resamples, seed 2026).
- Outage safety: plan physics validity (not applicable to A6, which plans on actual data), unserved critical kWh, executor overrides, and the E1 counterfactual without the override.
- Error decomposition per site: A4's pooled gap = translation (A4 - A5) + forecast (A5 - A6 executed) + remainder (A6 executed - oracle bound).

Limitation: "actual" weather is the same-day ICON analysis, not a measurement, so forecast error is understated.
