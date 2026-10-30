# VoltProof Bench report: `test-20261030-181315`

Generated 2026-10-30T23:29:10 by `python -m voltproof.bench.report`. 260 episodes x 7 arms. Oracle-bound check: 260 episodes checked, 0 skipped (A0 gap None), 0 mismatches, 0 results below the bound. Regression check: passed.

Arms: A0 rule-based; A1 optimizer, cost only; A2 LLM writes schedule; A3 LLM + simulator; A4 VoltProof; A5 optimizer, gold spec; A6 oracle (actual data).

Limitation: "actual" weather is the same-day ICON analysis, not a measurement, so forecast error is understated.

### Run summary

| arm | name | results | plan_source | fallbacks | escalations | infra_errors | infra_fallbacks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | rule-based | 260 | baseline 260 | 0 | 0 | 0 | 0 |
| A1 | optimizer, cost only | 260 | baseline 260 | 0 | 0 | 0 | 0 |
| A2 | LLM writes schedule | 260 | fallback 4, finalized 256 | 4 | 0 | 0 | 0 |
| A3 | LLM + simulator | 260 | finalized 257, log_physics_valid 3 | 0 | 0 | 0 | 0 |
| A4 | VoltProof | 260 | finalized 260 | 0 | 0 | 0 | 0 |
| A5 | optimizer, gold spec | 260 | baseline 260 | 0 | 0 | 0 | 0 |
| A6 | oracle (actual data) | 260 | baseline 260 | 0 | 0 | 0 | 0 |

### Goal satisfaction (gold spec met on the executed trajectory)

| arm | overall | ahmedabad_cold_store | munich_home (held out) | nairobi_clinic | held-out templates |
| --- | --- | --- | --- | --- | --- |
| A0 | 53.5% (139/260) | 44.4% (40/90) | 58.9% (53/90) | 57.5% (46/80) | 40.0% (12/30) |
| A1 | 56.9% (148/260) | 28.9% (26/90) | 58.9% (53/90) | 86.2% (69/80) | 53.3% (16/30) |
| A2 | 90.4% (235/260) | 85.6% (77/90) | 93.3% (84/90) | 92.5% (74/80) | 76.7% (23/30) |
| A3 | 93.1% (242/260) | 92.2% (83/90) | 93.3% (84/90) | 93.8% (75/80) | 76.7% (23/30) |
| A4 | 98.8% (257/260) | 97.8% (88/90) | 100.0% (90/90) | 98.8% (79/80) | 90.0% (27/30) |
| A5 | 98.8% (257/260) | 97.8% (88/90) | 100.0% (90/90) | 98.8% (79/80) | 90.0% (27/30) |
| A6 | 99.6% (259/260) | 98.9% (89/90) | 100.0% (90/90) | 100.0% (80/80) | 96.7% (29/30) |

### A4 minus each arm in goal satisfaction, paired bootstrap 95% CI

| comparison | overall | ahmedabad_cold_store | munich_home | nairobi_clinic |
| --- | --- | --- | --- | --- |
| A4 - A0 | +45.4 pts [+39.2, +51.5] | +53.3 pts [+43.3, +63.3] | +41.1 pts [+31.1, +51.1] | +41.2 pts [+31.2, +52.5] |
| A4 - A1 | +41.9 pts [+36.2, +48.1] | +68.9 pts [+58.9, +77.8] | +41.1 pts [+31.1, +51.1] | +12.5 pts [+6.2, +20.0] |
| A4 - A2 | +8.5 pts [+5.0, +11.9] | +12.2 pts [+5.6, +20.0] | +6.7 pts [+2.2, +12.2] | +6.2 pts [+1.2, +12.5] |
| A4 - A3 | +5.8 pts [+2.7, +8.8] | +5.6 pts [+0.0, +11.1] | +6.7 pts [+2.2, +12.2] | +5.0 pts [+1.2, +10.0] |

10,000 resamples of episodes (pairs kept), seed 2026.

### Money (realized_cost, INR), ahmedabad_cold_store: 90 episodes, 5 cost_min

| arm | pooled savings vs A0 | median (A0 - X) per day | W/L/T vs A0 | cost_min: pooled | cost_min: median | cost_min: W/L/T |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | +0.00% | +0.00 | 0/0/90 | +0.00% | +0.00 | 0/0/5 |
| A1 | +1.42% | +69.89 | 79/11/0 | +2.75% | +72.96 | 5/0/0 |
| A2 | -0.04% | +49.36 | 71/18/1 | +1.43% | +52.32 | 4/1/0 |
| A3 | -0.09% | +42.60 | 73/17/0 | +2.27% | +52.57 | 5/0/0 |
| A4 | +0.62% | +61.43 | 78/12/0 | +2.75% | +72.96 | 5/0/0 |
| A5 | +0.62% | +61.43 | 78/12/0 | +2.75% | +72.96 | 5/0/0 |
| A6 | +0.92% | +70.64 | 78/12/0 | +3.20% | +76.05 | 5/0/0 |

Pooled = sum(A0 - X) / sum|A0|. Win = X cheaper than A0 by more than the tie tolerance.

### Money (realized_cost, EUR), munich_home: 90 episodes, 5 cost_min

| arm | pooled savings vs A0 | median (A0 - X) per day | W/L/T vs A0 | cost_min: pooled | cost_min: median | cost_min: W/L/T |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | +0.00% | +0.00 | 0/0/90 | +0.00% | +0.00 | 0/0/5 |
| A1 | -3.99% | +0.00 | 47/43/0 | -11.97% | -0.05 | 1/4/0 |
| A2 | -62.99% | -0.25 | 3/85/2 | -23.30% | -0.09 | 0/4/1 |
| A3 | -149.21% | -1.05 | 2/88/0 | -104.06% | -1.41 | 0/5/0 |
| A4 | -10.06% | -0.03 | 33/57/0 | -11.97% | -0.05 | 1/4/0 |
| A5 | -10.06% | -0.03 | 33/57/0 | -11.97% | -0.05 | 1/4/0 |
| A6 | +9.49% | +0.05 | 71/19/0 | +8.18% | +0.05 | 5/0/0 |

Pooled = sum(A0 - X) / sum|A0|. Win = X cheaper than A0 by more than the tie tolerance.

### Money (realized_cost, KES), nairobi_clinic: 80 episodes, 5 cost_min

| arm | pooled savings vs A0 | median (A0 - X) per day | W/L/T vs A0 | cost_min: pooled | cost_min: median | cost_min: W/L/T |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | +0.00% | +0.00 | 0/0/80 | +0.00% | +0.00 | 0/0/5 |
| A1 | +7.33% | +36.45 | 79/1/0 | +8.10% | +34.20 | 5/0/0 |
| A2 | -3.12% | -6.48 | 27/52/1 | -0.80% | -0.57 | 2/3/0 |
| A3 | -5.23% | -5.62 | 33/47/0 | -0.78% | -0.02 | 2/3/0 |
| A4 | +5.02% | +34.20 | 72/8/0 | +8.10% | +34.20 | 5/0/0 |
| A5 | +5.02% | +34.20 | 72/8/0 | +8.10% | +34.20 | 5/0/0 |
| A6 | +5.10% | +34.20 | 72/8/0 | +8.12% | +34.20 | 5/0/0 |

Pooled = sum(A0 - X) / sum|A0|. Win = X cheaper than A0 by more than the tie tolerance.

### Total cost with goals (realized_total_cost, INR), ahmedabad_cold_store: 90 episodes

| arm | pooled gap to oracle bound | median gap per day | A4 vs arm W/L/T |
| --- | --- | --- | --- |
| A0 | +78.95% | +5548.62 | 90/0/0 |
| A1 | +46.13% | +3976.21 | 74/2/14 |
| A2 | +1.44% | +14.44 | 70/20/0 |
| A3 | +4.96% | +26.34 | 84/6/0 |
| A4 | +0.08% | +5.09 | - |
| A5 | +0.08% | +5.09 | 0/2/88 |
| A6 | +0.00% | +0.00 | 0/84/6 |

Pooled gap = sum(X_total - oracle bound) / sum|A0_total|. W/L/T counts A4 wins (A4 lower).

### Total cost with goals (realized_total_cost, EUR), munich_home: 90 episodes

| arm | pooled gap to oracle bound | median gap per day | A4 vs arm W/L/T |
| --- | --- | --- | --- |
| A0 | +95.67% | +0.34 | 63/27/0 |
| A1 | +84.56% | +0.34 | 44/4/42 |
| A2 | +12.24% | +0.39 | 87/3/0 |
| A3 | +16.26% | +1.06 | 86/4/0 |
| A4 | +1.53% | +0.06 | - |
| A5 | +1.53% | +0.06 | 0/0/90 |
| A6 | +0.00% | +0.00 | 0/90/0 |

Pooled gap = sum(X_total - oracle bound) / sum|A0_total|. W/L/T counts A4 wins (A4 lower).

### Total cost with goals (realized_total_cost, KES), nairobi_clinic: 80 episodes

| arm | pooled gap to oracle bound | median gap per day | A4 vs arm W/L/T |
| --- | --- | --- | --- |
| A0 | +87.31% | +41.34 | 80/0/0 |
| A1 | +26.84% | +0.00 | 11/1/68 |
| A2 | +9.16% | +42.43 | 80/0/0 |
| A3 | +4.35% | +41.94 | 74/1/5 |
| A4 | +0.14% | +0.00 | - |
| A5 | +0.14% | +0.00 | 0/0/80 |
| A6 | +0.00% | +0.00 | 0/20/60 |

Pooled gap = sum(X_total - oracle bound) / sum|A0_total|. W/L/T counts A4 wins (A4 lower).

### Outage safety (outage_critical, outage_then_reserve: 45 episodes)

| arm | episodes | plan physics valid | unserved critical kWh | executor overrides (hours / episodes) | E1: unserved critical kWh | E1: episodes with critical loss | E1: unserved non-critical kWh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 45 | 68.9% | 0.147 | 41 / 22 | 19.036 | 9 | 560.775 |
| A1 | 45 | 46.7% | 0.000 | 130 / 44 | 414.686 | 16 | 682.681 |
| A2 | 45 | 17.8% | 0.000 | 121 / 43 | 12.825 | 4 | 255.170 |
| A3 | 45 | 57.8% | 0.000 | 125 / 44 | 8.764 | 5 | 310.548 |
| A4 | 45 | 100.0% | 0.000 | 136 / 44 | 9.455 | 5 | 60.333 |
| A5 | 45 | 100.0% | 0.000 | 136 / 44 | 9.455 | 5 | 60.333 |
| A6 | 45 | n/a | 0.000 | 5 / 2 | 0.000 | 0 | 0.000 |

Plan physics validity: did the plan as written avoid needing a grid that is down? A6 plans on actual data, so its forecast check is not applicable. E1 counterfactual: only the executor's deficit-covering outage override is removed; in outage hours the battery follows its clipped setpoint and still stores surplus PV within its limits (the no-wasted-solar rule every arm gets outside outages), and any deficit goes unserved, non-critical load first. Planned charging above the PV surplus during an outage is a deficit, as in the verifier. This is a conservative estimate of what the plans alone would have lost.

### E1 outage counterfactual per site

| site | arm | episodes | critical kWh lost (with override) | critical kWh lost (E1, no override) | episodes with critical loss (E1) | non-critical kWh lost (E1) | gold satisfied (E1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ahmedabad_cold_store | A0 | 15 | 0.147 | 19.036 | 9 | 548.270 | 13.3% |
| ahmedabad_cold_store | A1 | 15 | 0.000 | 406.728 | 11 | 642.020 | 0.0% |
| ahmedabad_cold_store | A2 | 15 | 0.000 | 0.000 | 0 | 212.340 | 100.0% |
| ahmedabad_cold_store | A3 | 15 | 0.000 | 1.163 | 2 | 275.209 | 86.7% |
| ahmedabad_cold_store | A4 | 15 | 0.000 | 0.000 | 0 | 22.031 | 93.3% |
| ahmedabad_cold_store | A5 | 15 | 0.000 | 0.000 | 0 | 22.031 | 93.3% |
| ahmedabad_cold_store | A6 | 15 | 0.000 | 0.000 | 0 | 0.000 | 93.3% |
| munich_home | A0 | 15 | 0.000 | 0.000 | 0 | 0.000 | 93.3% |
| munich_home | A1 | 15 | 0.000 | 0.397 | 3 | 1.360 | 80.0% |
| munich_home | A2 | 15 | 0.000 | 0.014 | 1 | 1.075 | 93.3% |
| munich_home | A3 | 15 | 0.000 | 0.048 | 1 | 0.218 | 93.3% |
| munich_home | A4 | 15 | 0.000 | 1.893 | 3 | 2.120 | 80.0% |
| munich_home | A5 | 15 | 0.000 | 1.893 | 3 | 2.120 | 80.0% |
| munich_home | A6 | 15 | 0.000 | 0.000 | 0 | 0.000 | 100.0% |
| nairobi_clinic | A0 | 15 | 0.000 | 0.000 | 0 | 12.505 | 80.0% |
| nairobi_clinic | A1 | 15 | 0.000 | 7.562 | 2 | 39.301 | 86.7% |
| nairobi_clinic | A2 | 15 | 0.000 | 12.811 | 3 | 41.755 | 80.0% |
| nairobi_clinic | A3 | 15 | 0.000 | 7.552 | 2 | 35.121 | 86.7% |
| nairobi_clinic | A4 | 15 | 0.000 | 7.562 | 2 | 36.182 | 86.7% |
| nairobi_clinic | A5 | 15 | 0.000 | 7.562 | 2 | 36.182 | 86.7% |
| nairobi_clinic | A6 | 15 | 0.000 | 0.000 | 0 | 0.000 | 100.0% |

E1 counterfactual: only the executor's deficit-covering outage override is removed; in outage hours the battery follows its clipped setpoint and still stores surplus PV within its limits (the no-wasted-solar rule every arm gets outside outages), and any deficit goes unserved, non-critical load first. Planned charging above the PV surplus during an outage is a deficit, as in the verifier. This is a conservative estimate of what the plans alone would have lost.

### Error decomposition of A4's pooled total-cost gap (share of sum|A0_total|)

| site | episodes | A4 pooled gap | translation (A4 - A5) | forecast (A5 - A6 executed) | remainder (A6 executed - bound) |
| --- | --- | --- | --- | --- | --- |
| ahmedabad_cold_store | 90 | +0.075% | +0.000% | +0.075% | +0.000% |
| munich_home | 90 | +1.530% | +0.000% | +1.530% | +0.000% |
| nairobi_clinic | 80 | +0.143% | +0.000% | +0.143% | +0.000% |

translation + forecast + remainder = A4's pooled gap to the oracle bound, per site.

### A4 spec accuracy by language

| group | episodes | exact match | mean precision | mean recall |
| --- | --- | --- | --- | --- |
| English set | 170 | 99.4% | 0.997 | 1.000 |
| multilingual: en | 18 | 100.0% | 1.000 | 1.000 |
| multilingual: es | 18 | 100.0% | 1.000 | 1.000 |
| multilingual: hi | 18 | 100.0% | 1.000 | 1.000 |
| multilingual: pt | 18 | 100.0% | 1.000 | 1.000 |
| multilingual: de | 18 | 100.0% | 1.000 | 1.000 |

Multilingual rows are the same goals, days and sites in each language (en = the English source sentence).

### A4 spec accuracy by site

| group | episodes | exact match | mean precision | mean recall |
| --- | --- | --- | --- | --- |
| ahmedabad_cold_store | 90 | 100.0% | 1.000 | 1.000 |
| munich_home | 90 | 100.0% | 1.000 | 1.000 |
| nairobi_clinic | 80 | 98.8% | 0.994 | 1.000 |

### A4 spec accuracy by template

| group | episodes | exact match | mean precision | mean recall |
| --- | --- | --- | --- | --- |
| cost_min | 15 | 100.0% | 1.000 | 1.000 |
| evening_reserve | 30 | 100.0% | 1.000 | 1.000 |
| outage_critical | 30 | 100.0% | 1.000 | 1.000 |
| no_export_midday | 10 | 100.0% | 1.000 | 1.000 |
| storm_full | 30 | 100.0% | 1.000 | 1.000 |
| import_cap_afternoon | 30 | 100.0% | 1.000 | 1.000 |
| never_below | 30 | 100.0% | 1.000 | 1.000 |
| peak_only_discharge | 10 | 100.0% | 1.000 | 1.000 |
| cover_critical_hours | 15 | 93.3% | 0.967 | 1.000 |
| save_cycles | 30 | 100.0% | 1.000 | 1.000 |
| end_of_day_reserve | 15 | 100.0% | 1.000 | 1.000 |
| outage_then_reserve | 15 | 100.0% | 1.000 | 1.000 |

### A4 repair rounds per episode

| repair rounds | episodes |
| --- | --- |
| 0 | 194 |
| 1 | 63 |
| 2 | 3 |

### Which gate check triggered each A4 repair (69 repairs)

| check | repairs triggered (attempts failing it) |
| --- | --- |
| verified_plan | 0 |
| coverage | 0 |
| grounded_explanation | 68 |
| review | 1 |

An attempt can fail more than one check.

### A4 repairs by combination of failed checks

| failed checks | repairs |
| --- | --- |
| grounded_explanation | 68 |
| review | 1 |

### A4 gate outcomes and reviewer verdicts

| outcome | episodes |
| --- | --- |
| review approved | 260 |
| review not approved | 0 |
| review unavailable | 0 |
| review disputed (accepted with flag) | 0 |
| explanation not grounded (accepted with flag) | 0 |
| escalated with plan | 0 |
| escalated to fallback | 0 |

### LLM cost and speed per arm

| arm | episodes | LLM USD / episode | LLM USD total | median latency s | p90 latency s | tokens in / episode | tokens out / episode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A2 | 260 | 0.0198 | 5.15 | 16.1 | 22.0 | 1460 | 4627 |
| A3 | 260 | 0.0515 | 13.39 | 39.3 | 60.4 | 33336 | 9993 |
| A4 | 260 | 0.0114 | 2.97 | 8.5 | 13.3 | 15161 | 2024 |

Baselines make no LLM call. Latency is the LLM time per episode (agent + reviewer).

### Goal satisfaction by template

| template | held out | A0 | A1 | A2 | A3 | A4 | A5 | A6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cost_min |  | 100% (15/15) | 100% (15/15) | 100% (15/15) | 100% (15/15) | 100% (15/15) | 100% (15/15) | 100% (15/15) |
| evening_reserve |  | 37% (11/30) | 63% (19/30) | 87% (26/30) | 93% (28/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| outage_critical |  | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| no_export_midday |  | 60% (6/10) | 40% (4/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) |
| storm_full |  | 13% (4/30) | 10% (3/30) | 80% (24/30) | 77% (23/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| import_cap_afternoon |  | 100% (30/30) | 67% (20/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| never_below |  | 10% (3/30) | 40% (12/30) | 83% (25/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| peak_only_discharge |  | 0% (0/10) | 50% (5/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) | 100% (10/10) |
| cover_critical_hours |  | 53% (8/15) | 80% (12/15) | 93% (14/15) | 100% (15/15) | 100% (15/15) | 100% (15/15) | 100% (15/15) |
| save_cycles |  | 67% (20/30) | 40% (12/30) | 93% (28/30) | 93% (28/30) | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| end_of_day_reserve | yes | 40% (6/15) | 47% (7/15) | 67% (10/15) | 80% (12/15) | 100% (15/15) | 100% (15/15) | 100% (15/15) |
| outage_then_reserve | yes | 40% (6/15) | 60% (9/15) | 87% (13/15) | 73% (11/15) | 80% (12/15) | 80% (12/15) | 93% (14/15) |

### Goal satisfaction by language (multilingual set: same goals, days and sites)

| language | A0 | A1 | A2 | A3 | A4 | A5 | A6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| en | 50% (9/18) | 50% (9/18) | 89% (16/18) | 94% (17/18) | 100% (18/18) | 100% (18/18) | 100% (18/18) |
| es | 50% (9/18) | 50% (9/18) | 94% (17/18) | 100% (18/18) | 100% (18/18) | 100% (18/18) | 100% (18/18) |
| hi | 50% (9/18) | 50% (9/18) | 94% (17/18) | 100% (18/18) | 100% (18/18) | 100% (18/18) | 100% (18/18) |
| pt | 50% (9/18) | 50% (9/18) | 100% (18/18) | 94% (17/18) | 100% (18/18) | 100% (18/18) | 100% (18/18) |
| de | 50% (9/18) | 50% (9/18) | 89% (16/18) | 94% (17/18) | 100% (18/18) | 100% (18/18) | 100% (18/18) |
