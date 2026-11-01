# Why VoltProof sometimes costs more than the rule-based battery on outage days

**Short answer:** on a few grid-outage days, VoltProof keeps the whole site running through the outage. The rule-based battery keeps only the critical load on and switches the rest off. Realized cost doesn't count load that was switched off, so the rule-based battery looks cheaper for serving less. When both serve the same load, VoltProof is cheaper.

Everything below was computed offline from the frozen code and the committed results (`results/test-run`, `results/e2/run`), with no LLM calls. No frozen file or result was changed.

## The measure
`realized_cost` is grid purchases minus export income, plus battery wear, plus the value of the change in stored battery energy. That value is priced at the day's mean import price × discharge efficiency, the same for both sides (SPEC §6.7). It doesn't include non-critical ("other") load that was left unserved. Only `realized_total_cost` counts that load, at 100 × the day's top import price per kWh. `savings_vs_rule_pct` and the page's cost line both use `realized_cost`.

During a grid outage the executor applies the same safety rule to every arm (SPEC §6.6). The battery covers the critical load first, and covers other load only with energy the rest of the outage's critical load won't need. The rest is shed. VoltProof's optimizer prices unserved other load at 100 × the top import price per kWh, far above the ~₹9.20/kWh it costs to serve it from the battery. So VoltProof's plans charge enough beforehand to serve the whole site.

## Worked example: Ahmedabad cold store, 2025-12-17, grid off 14:00–17:00
Goal: keep the cold room running and the bill low. Spec: `{"grid_outages":[{"start_hour":14,"end_hour":17}]}`. Both sides meet every goal and serve 100% of the critical load.

| | VoltProof | rule-based |
|---|---|---|
| Realized cost | ₹2,419.51 | ₹2,097.22 |
| Other load switched off, 14:00–17:00 | **0 kWh** | **41.12 kWh** (4.91 + 15.96 + 20.25) |
| Battery at 14:00 | 99.5 kWh | 20.0 kWh (its 20% floor) |
| Battery at midnight | 10.0 kWh | 10.0 kWh (same energy credit, ₹314.13 each) |

Where the ₹322.29 comes from (VoltProof minus rule-based):

| | difference | what happens |
|---|---|---|
| 12:00–13:00 | +₹475.97 | VoltProof pre-charges 74.37 kWh at ₹6.40, the cheapest price of the day |
| 00:00–05:00 | +₹228.00 | the rule-based battery discharges overnight at ₹8.00 |
| 06:00–07:00 | −₹192.67 | VoltProof discharges at the ₹9.60 peak |
| 17:00–18:00 | −₹330.51 | VoltProof discharges at the ₹9.60 peak |
| battery wear | +₹141.49 | throughput 179.49 vs 38.00 kWh |
| **total** | **+₹322.28** | |

**Equal service:** the rule-based battery's 41.12 kWh of shed load was removed from the day's load (forecast and actual), and the same spec was re-solved with the frozen optimizer. VoltProof then costs ₹2,027.74 against ₹2,097.22: **₹69.48 cheaper (3.3%)**. The rule-based battery's cost on the reduced load is unchanged at ₹2,097.22.

## How often this happens in the benchmark
**Main test run (260 episodes):**
- VoltProof (A4) and the rule-based battery (A0) both meet every gold goal in 139 episodes. A4 costs more in 36 of them.
- **10 of those 36 have this cause.** All are `outage_critical` goals where A0 sheds other load and A4 sheds none:

  | episode | A4 costs more by | other load A0 switched off |
  |---|---|---|
  | Ahmedabad 2025-08-30 (p1) | ₹402.64 | 49.31 kWh |
  | Ahmedabad 2025-11-01 (p0) | ₹326.80 | 40.32 kWh |
  | Ahmedabad 2025-12-17 (p1 and 5 translations: en, es, hi, pt, de) | ₹322.29 each | 41.12 kWh |
  | Ahmedabad 2025-06-08 (p0) | ₹39.45 | 10.64 kWh |
  | Nairobi 2025-03-28 (p1) | KES 49.41 | 2.99 kWh |

- **The other 26 are all Munich, €0.01–0.64 each.** These are small forecast-error effects on low-bill days, not this cause.

**The same 10 episodes for the optimizer-only arms:**
- **A5**, the optimizer given the hand-written gold spec, shows exactly the same 10 episodes with identical costs (A4 − A5 = ₹0.00 on each). A perfect translation of the goal gives the same result, so this comes from the optimizer and the cost measure, not from the AI.
- **A1**, the cost-only optimizer that doesn't know about the outage, also shows these 10. Its gaps are smaller on the four Ahmedabad days (for example ₹252.16 on 2025-12-17), because it doesn't pre-charge for the outage and sheds some other load itself (about 9.1 kWh on 2025-12-17).

**E2 (hard goals, 101 episodes):** A4 costs more in 15 of the 28 episodes where both meet every goal. All 15 are Munich 2025-08-30 at €0.01. **None** have this cause.

**Method:** for episodes where both sides meet every gold goal, `realized_total_cost − realized_cost` is the unserved-load penalty plus a negligible curtailment term. Dividing the difference between the two sides by 100 × the day's top import price gives the extra kWh A0 shed. The 2025-12-17 figure was confirmed by re-running the executor (41.12 kWh).

## What the page shows
The same-day comparison now reports, for each side:
- `non_critical_shed_kwh`: the whole day
- `non_critical_shed_outage_kwh`: the outage hours, when the spec has outages

When VoltProof costs more and the rule-based battery switched off at least 0.1 kWh more other load, the cost line is shown in neutral ink. It reads: "Costs more because it kept the whole site running; the rule-based battery switched off 41.12 kWh of other load during the outage." Realized cost and every benchmark metric are unchanged.
