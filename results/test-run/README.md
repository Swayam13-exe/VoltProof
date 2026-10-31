# VoltProof Bench test run `test-20261030-181315`

The frozen test run of VoltProof Bench. Every reported number comes from these files.

| | |
| --- | --- |
| run id | `test-20261030-181315` |
| code tag | `test-freeze` (commit `2dacaa5`) |
| run fingerprint | `cb2f74a5a7ec2f5981cba2a6d8613173cd51133215840ff059e909c5dd4bfda9` |
| code fingerprint at the run | `79958a98d7cc9d2291a3d57390fef973a9e76a9ed141acddc58a8a66efd5f5eb` |
| code fingerprint now | `20abe8704fc57c9aa9132bf53e7da21c2023985f3a00a3ccc5c8813f6e48f55a` (differs only by the additive `outage_override` flag, see below) |
| plan sha256 | `2ac37bf2cb4d225d1e90b6f7a59517ada69eda66602c3b83876a9599c35413bc` |
| episodes | 260 x 7 arms = 1820 results |
| infrastructure errors | 0 |
| LLM cost of the run (ledger estimate) | $21.50 |
| ledger total at report time | $32.04 of $85 |
| sha256 of the run directory, before / after this report | `a13e0d472207fa471079be5d7432a079391a4e0b4e14807f9c303e61c28ea971` / `a13e0d472207fa471079be5d7432a079391a4e0b4e14807f9c303e61c28ea971` |

Files: `config.json` (arguments, fingerprints, settings without the key; in this published copy, local absolute paths are redacted to paths relative to the repo root; sha256 of the original `config.json`: `0da5d6b427edc471b70ac1de32bc9d5f45d0088a6d17d8cb9e61d0ba4a218bf0`), `episodes_plan.json` (every episode with its goal and gold spec, written before the run), `episodes.jsonl` (one line per episode and arm), `traces.zip` (one trace per LLM arm and episode).

## Regression check after the freeze

One change was made to the frozen code after the run: an additive executor flag `outage_override` (default True, the frozen behaviour) used only by the E1 counterfactual. With it in place, all results were re-evaluated from their stored plans (baselines recomputed, A2/A3 schedules and declarations from the traces, A4 by re-solving its stored specs, whose tool outputs must equal the traced ones exactly): 1820/1820 rows, passed = True, max |difference| in realized_cost 0.0, realized_total_cost 0.0 (tolerance 1e-9), gold_satisfied identical; 524 A4 tool outputs replayed identically.

## Aggregation rules (fixed after the run; the per-episode definitions are unchanged)

- Per-episode percentages (`savings_vs_rule_pct`, `gap_to_oracle_pct`) are never averaged: their base is the rule-based day cost, which is near zero or negative on sunny clinic and household days. The runner's end-of-run means are not reported.
- Money (`realized_cost`), per site: pooled ratio sum(A0 - X)/sum|A0|, median per-episode difference in site currency per day, paired win/loss/tie counts against A0; the same for the cost-only template. Never summed across currencies.
- Total cost with goals (`realized_total_cost`), per site: pooled gap sum(X_total - oracle bound)/sum|A0_total|, median per-episode gap, A4's paired win/loss/tie counts against every arm. The oracle bound is recomputed per episode and checked against the one implied by A0's stored gap (within 1e-6).
- Ties: |X - Y| <= 1e-6 x max(1, the day's top import price).
- Goal satisfaction: rates per arm; paired bootstrap 95% CIs for A4 minus A0, A1, A2, A3 (10,000 resamples, seed 2026).
- Outage safety: plan physics validity (not applicable to A6, which plans on actual data), unserved critical kWh, executor overrides, and the E1 counterfactual without the override.
- Error decomposition per site: A4's pooled gap = translation (A4 - A5) + forecast (A5 - A6 executed) + remainder (A6 executed - oracle bound).

Limitation: "actual" weather is the same-day ICON analysis, not a measurement, so forecast error is understated.
