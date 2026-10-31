# VoltProof Bench E3, reviewer stress test: `e3-20261031-001625`

### E3: does the independent reviewer catch a wrong spec?

| error | with assumptions: flagged | no assumptions: flagged | issue kind matches (with / no) | assumptions state original numbers: n, flagged |
| --- | --- | --- | --- | --- |
| control (false alarm) | 0% [0, 9] (0/40) | 0% [0, 9] (0/40) | - | - |
| shift_hours | 100% [88, 100] (29/29) | 100% [88, 100] (29/29) | 29/29 / 29/29 | yes 28 (100%), no 1 (100%) |
| wrong_level | 100% [89, 100] (30/30) | 87% [70, 95] (26/30) | 30/30 / 26/26 | yes 20 (100%), no 10 (100%) |
| missing_item | 82% [68, 91] (33/40) | 100% [91, 100] (40/40) | 33/33 / 40/40 | yes 33 (79%), no 7 (100%) |
| unneeded_item | 100% [91, 100] (40/40) | 100% [91, 100] (40/40) | 40/40 / 40/40 | yes 0 (-), no 0 (-) |
| wrong_field | 97% [86, 100] (35/36) | 100% [90, 100] (36/36) | 29/35 / 29/36 | yes 15 (100%), no 3 (100%) |

40 A4 episodes of the frozen run (spec exactly matching gold; cost_min excluded), 430 review calls, 71 served from the LLM cache, 0 unusable verdicts, LLM cost $1.38. Flagged = the reviewer did not approve. 95% Wilson intervals. Issue kind matches = the flagged verdict names the expected kind (wrong for shifted hours, wrong levels and wrong fields; missing; unneeded). Disclosure: the with-assumptions controls are byte-identical to the frozen run's review calls and replay its cached verdicts. 29 no-assumptions variants in 6 episodes are identical to their with-assumptions input (the run's reviewer saw no assumptions there), so they replay that verdict; they are kept and flagged. The reviewer prompt counts a requirement captured by the assumptions as not missing, so with assumptions present a dropped spec item the assumptions still describe is not necessarily an error to this reviewer.
