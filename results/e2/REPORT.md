# VoltProof Bench E2, hard goals: `e2-20261031-130234`

Reported separately from the main test run and never pooled with it. Same frozen code (plus the additive executor flag, default unchanged; the regression check passed before any spend), the same arms, metrics and aggregation rules.

Disclosure: 4 English-set episodes have the same goal text, site and day as a multilingual English source episode, so they sent identical requests (served from the LLM cache) and have identical results: ahmedabad_cold_store_2025-01-12_hard_negation_p0, ahmedabad_cold_store_2025-01-12_hard_tonight_p0, munich_home_2025-08-30_hard_relative_time_p0, nairobi_clinic_2025-03-28_hard_relative_time_p0.

Multilingual goals: translations written by AI and not reviewed by native speakers (Hindi checked by the author).

Generated 2026-10-31T14:06:33 by `python -m voltproof.bench.report`. 101 episodes x 7 arms. Oracle-bound check: 101 episodes checked, 0 skipped (A0 gap None), 0 mismatches, 0 results below the bound. Regression check: passed.

Arms: A0 rule-based; A1 optimizer, cost only; A2 LLM writes schedule; A3 LLM + simulator; A4 VoltProof; A5 optimizer, gold spec; A6 oracle (actual data).

Limitation: "actual" weather is the same-day ICON analysis, not a measurement, so forecast error is understated.

### Run summary

| arm | name | results | plan_source | fallbacks | escalations | infra_errors | infra_fallbacks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | rule-based | 101 | baseline 101 | 0 | 0 | 0 | 0 |
| A1 | optimizer, cost only | 101 | baseline 101 | 0 | 0 | 0 | 0 |
| A2 | LLM writes schedule | 101 | fallback 1, finalized 100 | 1 | 0 | 0 | 0 |
| A3 | LLM + simulator | 101 | finalized 100, log_physics_valid 1 | 0 | 0 | 0 | 0 |
| A4 | VoltProof | 101 | finalized 101 | 0 | 0 | 0 | 0 |
| A5 | optimizer, gold spec | 101 | baseline 101 | 0 | 0 | 0 | 0 |
| A6 | oracle (actual data) | 101 | baseline 101 | 0 | 0 | 0 | 0 |

### Goal satisfaction (gold spec met on the executed trajectory)

| arm | overall | ahmedabad_cold_store | munich_home (held out) | nairobi_clinic |
| --- | --- | --- | --- | --- |
| A0 | 27.7% (28/101) | 5.7% (2/35) | 57.1% (20/35) | 19.4% (6/31) |
| A1 | 35.6% (36/101) | 22.9% (8/35) | 40.0% (14/35) | 45.2% (14/31) |
| A2 | 88.1% (89/101) | 82.9% (29/35) | 94.3% (33/35) | 87.1% (27/31) |
| A3 | 80.2% (81/101) | 80.0% (28/35) | 85.7% (30/35) | 74.2% (23/31) |
| A4 | 94.1% (95/101) | 94.3% (33/35) | 94.3% (33/35) | 93.5% (29/31) |
| A5 | 94.1% (95/101) | 94.3% (33/35) | 94.3% (33/35) | 93.5% (29/31) |
| A6 | 94.1% (95/101) | 94.3% (33/35) | 94.3% (33/35) | 93.5% (29/31) |

### A4 minus each arm in goal satisfaction, paired bootstrap 95% CI

| comparison | overall | ahmedabad_cold_store | munich_home | nairobi_clinic |
| --- | --- | --- | --- | --- |
| A4 - A0 | +66.3 pts [+57.4, +75.2] | +88.6 pts [+77.1, +97.1] | +37.1 pts [+22.9, +54.3] | +74.2 pts [+58.1, +87.1] |
| A4 - A1 | +58.4 pts [+48.5, +68.3] | +71.4 pts [+57.1, +85.7] | +54.3 pts [+37.1, +71.4] | +48.4 pts [+29.0, +64.5] |
| A4 - A2 | +5.9 pts [+2.0, +10.9] | +11.4 pts [+2.9, +22.9] | +0.0 pts [+0.0, +0.0] | +6.5 pts [+0.0, +16.1] |
| A4 - A3 | +13.9 pts [+7.9, +20.8] | +14.3 pts [+2.9, +25.7] | +8.6 pts [+0.0, +20.0] | +19.4 pts [+6.5, +35.5] |

10,000 resamples of episodes (pairs kept), seed 2026.

### Physically impossible goal (hard_impossible): is the shortfall stated in the explanation?

| arm | episodes | shortfall stated | spec exact match |
| --- | --- | --- | --- |
| A2 | 6 | 0/6 | - |
| A3 | 6 | 5/6 | - |
| A4 | 6 | 6/6 | 4/6 |

Shortfall = the gold spec's plan on the forecast (A5), the least any arm can fall short by: ahmedabad_cold_store: 3.450 kWh; ahmedabad_cold_store: 7.565 kWh; munich_home: 2.750 kWh; nairobi_clinic: 5.750 kWh.

### Money (realized_cost, INR), ahmedabad_cold_store: 35 episodes, 0 cost_min

| arm | pooled savings vs A0 | median (A0 - X) per day | W/L/T vs A0 |
| --- | --- | --- | --- |
| A0 | +0.00% | +0.00 | 0/0/35 |
| A1 | +4.28% | +72.96 | 35/0/0 |
| A2 | +2.33% | +54.21 | 30/4/1 |
| A3 | +1.04% | +35.52 | 23/12/0 |
| A4 | +3.33% | +64.76 | 33/2/0 |
| A5 | +3.33% | +64.76 | 33/2/0 |
| A6 | +3.47% | +65.27 | 33/2/0 |

Pooled = sum(A0 - X) / sum|A0|. Win = X cheaper than A0 by more than the tie tolerance.

### Money (realized_cost, EUR), munich_home: 35 episodes, 0 cost_min

| arm | pooled savings vs A0 | median (A0 - X) per day | W/L/T vs A0 |
| --- | --- | --- | --- |
| A0 | +0.00% | +0.00 | 0/0/35 |
| A1 | +0.32% | -0.01 | 11/24/0 |
| A2 | -65.69% | -0.30 | 0/35/0 |
| A3 | -138.09% | -1.12 | 0/35/0 |
| A4 | -24.21% | -0.01 | 5/30/0 |
| A5 | -24.21% | -0.01 | 5/30/0 |
| A6 | -17.11% | +0.05 | 21/14/0 |

Pooled = sum(A0 - X) / sum|A0|. Win = X cheaper than A0 by more than the tie tolerance.

### Money (realized_cost, KES), nairobi_clinic: 31 episodes, 0 cost_min

| arm | pooled savings vs A0 | median (A0 - X) per day | W/L/T vs A0 |
| --- | --- | --- | --- |
| A0 | +0.00% | +0.00 | 0/0/31 |
| A1 | +4.62% | +34.20 | 31/0/0 |
| A2 | -8.39% | -28.07 | 7/24/0 |
| A3 | -8.24% | -11.59 | 13/18/0 |
| A4 | -3.78% | +11.31 | 20/11/0 |
| A5 | -3.78% | +11.31 | 20/11/0 |
| A6 | -3.67% | +11.31 | 20/11/0 |

Pooled = sum(A0 - X) / sum|A0|. Win = X cheaper than A0 by more than the tie tolerance.

### Total cost with goals (realized_total_cost, INR), ahmedabad_cold_store: 35 episodes

| arm | pooled gap to oracle bound | median gap per day | A4 vs arm W/L/T |
| --- | --- | --- | --- |
| A0 | +83.07% | +7973.48 | 35/0/0 |
| A1 | +54.93% | +3924.95 | 27/8/0 |
| A2 | +3.03% | +14.16 | 31/4/0 |
| A3 | +0.45% | +30.92 | 34/1/0 |
| A4 | +0.03% | +0.91 | - |
| A5 | +0.03% | +0.91 | 0/0/35 |
| A6 | +0.00% | +0.00 | 0/35/0 |

Pooled gap = sum(X_total - oracle bound) / sum|A0_total|. W/L/T counts A4 wins (A4 lower).

### Total cost with goals (realized_total_cost, EUR), munich_home: 35 episodes

| arm | pooled gap to oracle bound | median gap per day | A4 vs arm W/L/T |
| --- | --- | --- | --- |
| A0 | +90.51% | +0.05 | 20/15/0 |
| A1 | +117.31% | +2.89 | 23/0/12 |
| A2 | +2.54% | +0.13 | 33/2/0 |
| A3 | +7.25% | +1.10 | 33/2/0 |
| A4 | +0.36% | +0.06 | - |
| A5 | +0.36% | +0.06 | 0/0/35 |
| A6 | +0.00% | -0.00 | 0/35/0 |

Pooled gap = sum(X_total - oracle bound) / sum|A0_total|. W/L/T counts A4 wins (A4 lower).

### Total cost with goals (realized_total_cost, KES), nairobi_clinic: 31 episodes

| arm | pooled gap to oracle bound | median gap per day | A4 vs arm W/L/T |
| --- | --- | --- | --- |
| A0 | +89.35% | +6180.32 | 31/0/0 |
| A1 | +31.93% | +2037.11 | 17/0/14 |
| A2 | +0.40% | +19.35 | 30/1/0 |
| A3 | +0.66% | +18.95 | 24/2/5 |
| A4 | +0.02% | +0.00 | - |
| A5 | +0.02% | +0.00 | 0/0/31 |
| A6 | -0.00% | +0.00 | 0/9/22 |

Pooled gap = sum(X_total - oracle bound) / sum|A0_total|. W/L/T counts A4 wins (A4 lower).

### Outage safety (hard_compound3: 4 episodes)

| arm | episodes | plan physics valid | unserved critical kWh | executor overrides (hours / episodes) |
| --- | --- | --- | --- | --- |
| A0 | 4 | 75.0% | 0.000 | 0 / 0 |
| A1 | 4 | 50.0% | 0.000 | 12 / 4 |
| A2 | 4 | 0.0% | 0.000 | 12 / 4 |
| A3 | 4 | 75.0% | 0.000 | 11 / 4 |
| A4 | 4 | 100.0% | 0.000 | 12 / 4 |
| A5 | 4 | 100.0% | 0.000 | 12 / 4 |
| A6 | 4 | n/a | 0.000 | 4 / 2 |

Plan physics validity: did the plan as written avoid needing a grid that is down? A6 plans on actual data, so its forecast check is not applicable.

### Error decomposition of A4's pooled total-cost gap (share of sum|A0_total|)

| site | episodes | A4 pooled gap | translation (A4 - A5) | forecast (A5 - A6 executed) | remainder (A6 executed - bound) |
| --- | --- | --- | --- | --- | --- |
| ahmedabad_cold_store | 35 | +0.034% | +0.000% | +0.034% | +0.000% |
| munich_home | 35 | +0.363% | +0.000% | +0.363% | -0.000% |
| nairobi_clinic | 31 | +0.017% | +0.000% | +0.017% | -0.000% |

translation + forecast + remainder = A4's pooled gap to the oracle bound, per site.

### A4 spec accuracy by language

| group | episodes | exact match | mean precision | mean recall |
| --- | --- | --- | --- | --- |
| English set | 56 | 96.4% | 0.964 | 0.964 |
| multilingual: en | 9 | 100.0% | 1.000 | 1.000 |
| multilingual: es | 9 | 100.0% | 1.000 | 1.000 |
| multilingual: hi | 9 | 100.0% | 1.000 | 1.000 |
| multilingual: pt | 9 | 100.0% | 1.000 | 1.000 |
| multilingual: de | 9 | 100.0% | 1.000 | 1.000 |

Multilingual rows are the same goals, days and sites in each language (en = the English source sentence).

### A4 spec accuracy by site

| group | episodes | exact match | mean precision | mean recall |
| --- | --- | --- | --- | --- |
| ahmedabad_cold_store | 35 | 97.1% | 0.971 | 0.971 |
| munich_home | 35 | 97.1% | 0.971 | 0.971 |
| nairobi_clinic | 31 | 100.0% | 1.000 | 1.000 |

### A4 spec accuracy by template

| group | episodes | exact match | mean precision | mean recall |
| --- | --- | --- | --- | --- |
| hard_reserve_kwh | 6 | 100.0% | 1.000 | 1.000 |
| hard_cap_relative | 6 | 100.0% | 1.000 | 1.000 |
| hard_negation | 21 | 100.0% | 1.000 | 1.000 |
| hard_relative_time | 21 | 100.0% | 1.000 | 1.000 |
| hard_tonight | 21 | 100.0% | 1.000 | 1.000 |
| hard_noon_midnight | 6 | 100.0% | 1.000 | 1.000 |
| hard_distractor | 6 | 100.0% | 1.000 | 1.000 |
| hard_compound3 | 4 | 100.0% | 1.000 | 1.000 |
| hard_impossible | 6 | 66.7% | 0.667 | 0.667 |
| hard_named_period | 4 | 100.0% | 1.000 | 1.000 |

### A4 repair rounds per episode

| repair rounds | episodes |
| --- | --- |
| 0 | 73 |
| 1 | 26 |
| 2 | 2 |

### Which gate check triggered each A4 repair (30 repairs)

| check | repairs triggered (attempts failing it) |
| --- | --- |
| verified_plan | 0 |
| coverage | 0 |
| grounded_explanation | 30 |
| review | 0 |

An attempt can fail more than one check.

### A4 repairs by combination of failed checks

| failed checks | repairs |
| --- | --- |
| grounded_explanation | 30 |

### A4 gate outcomes and reviewer verdicts

| outcome | episodes |
| --- | --- |
| review approved | 101 |
| review not approved | 0 |
| review unavailable | 0 |
| review disputed (accepted with flag) | 0 |
| explanation not grounded (accepted with flag) | 0 |
| escalated with plan | 0 |
| escalated to fallback | 0 |

### LLM cost and speed per arm

| arm | episodes | LLM USD / episode | LLM USD total | median latency s | p90 latency s | tokens in / episode | tokens out / episode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A2 | 101 | 0.0210 | 2.12 | 17.7 | 24.3 | 1459 | 5034 |
| A3 | 101 | 0.0481 | 4.86 | 38.7 | 58.2 | 32606 | 9414 |
| A4 | 101 | 0.0135 | 1.36 | 10.7 | 14.3 | 15824 | 2441 |

Baselines make no LLM call. Latency is the LLM time per episode (agent + reviewer).

### Goal satisfaction by template

| template | held out | A0 | A1 | A2 | A3 | A4 | A5 | A6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_reserve_kwh |  | 50% (3/6) | 67% (4/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) |
| hard_cap_relative |  | 100% (6/6) | 67% (4/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) |
| hard_negation |  | 0% (0/21) | 29% (6/21) | 100% (21/21) | 100% (21/21) | 100% (21/21) | 100% (21/21) | 100% (21/21) |
| hard_relative_time |  | 38% (8/21) | 38% (8/21) | 81% (17/21) | 76% (16/21) | 100% (21/21) | 100% (21/21) | 100% (21/21) |
| hard_tonight |  | 38% (8/21) | 38% (8/21) | 90% (19/21) | 81% (17/21) | 100% (21/21) | 100% (21/21) | 100% (21/21) |
| hard_noon_midnight |  | 0% (0/6) | 0% (0/6) | 100% (6/6) | 33% (2/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) |
| hard_distractor |  | 50% (3/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) | 100% (6/6) |
| hard_compound3 |  | 0% (0/4) | 0% (0/4) | 100% (4/4) | 100% (4/4) | 100% (4/4) | 100% (4/4) | 100% (4/4) |
| hard_impossible |  | 0% (0/6) | 0% (0/6) | 0% (0/6) | 0% (0/6) | 0% (0/6) | 0% (0/6) | 0% (0/6) |
| hard_named_period |  | 0% (0/4) | 0% (0/4) | 100% (4/4) | 75% (3/4) | 100% (4/4) | 100% (4/4) | 100% (4/4) |

### Goal satisfaction by language (multilingual set: same goals, days and sites)

| language | A0 | A1 | A2 | A3 | A4 | A5 | A6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| en | 22% (2/9) | 33% (3/9) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (9/9) | 100% (9/9) |
| es | 22% (2/9) | 33% (3/9) | 100% (9/9) | 67% (6/9) | 100% (9/9) | 100% (9/9) | 100% (9/9) |
| hi | 22% (2/9) | 33% (3/9) | 67% (6/9) | 89% (8/9) | 100% (9/9) | 100% (9/9) | 100% (9/9) |
| pt | 22% (2/9) | 33% (3/9) | 89% (8/9) | 89% (8/9) | 100% (9/9) | 100% (9/9) | 100% (9/9) |
| de | 22% (2/9) | 33% (3/9) | 89% (8/9) | 89% (8/9) | 100% (9/9) | 100% (9/9) | 100% (9/9) |
