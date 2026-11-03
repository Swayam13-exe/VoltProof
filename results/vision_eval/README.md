# Bill reader: evaluation on ten harder bills

One pass on 2026-11-02 18:08 (UTC), model `meta-models/muse-spark-open-1026`, reasoning effort `medium`. No prompt tuning and no re-run: these are the first and only results. Bills: `data/vision_eval/` (drawn by `python scripts/make_vision_samples.py --hard`); script: `scripts/vision_eval.py`; raw answers: `results.json`.

Each bill went through the app's own path for an upload: `sanitize_image` → `read_bill` (one uncached call) → `check_bill` against a site in the bill's `site_currency`.

## Counts

| outcome | bills |
|---|---|
| read correctly | 7 |
| correctly refused | 3 |
| wrong | 0 |
| not run (error or budget stop) | 0 |

Expected: 7 to read, 3 to refuse.

Cost: $0.0435 by the ledger (cap $0.15); per-call costs in `results.json`.

## Per bill

| bill | tests | expect | outcome | details | cost | latency |
|---|---|---|---|---|---|---|
| h01_unreadable_blur.jpg | unreadable: heavy blur | refuse | correctly refused | model said unreadable | $0.0033 | 6.39 s |
| h02_hindi_labels.png | Hindi labels | read | read correctly |  | $0.0043 | 7.45 s |
| h03_half_hour.png | half-hour slot boundaries | refuse | correctly refused | model said unreadable | $0.0056 | 7.61 s |
| h04_wrong_currency.png | USD bill for an INR site | refuse | correctly refused | validator: This bill is in USD, but the site uses INR. Pick the INR site or another bill. | $0.0047 | 6.67 s |
| h05_rotated_photo.jpg | sideways phone photo | read | read correctly |  | $0.0054 | 7.48 s |
| h06_flat.png | flat tariff, rate in cents | read | read correctly |  | $0.0034 | 8.11 s |
| h07_three_period.png | three rates, one split in two | read | read correctly |  | $0.0044 | 6.14 s |
| h08_export_rate.png | export credit to read | read | read correctly |  | $0.0034 | 6.45 s |
| h09_many_fees.png | many fee lines around the rates | read | read correctly |  | $0.0043 | 6.98 s |
| h10_low_contrast.jpg | pale print, photographed | read | read correctly |  | $0.0047 | 10.2 s |

## Failures

None.

Warnings ("check this rate") are shown to the user but do not block; they are listed per bill in `results.json`.

## Notes (added after the run, by hand; the numbers above are unchanged)

- **h03 (half-hour boundaries) was refused for the wrong reason.** The bill is clearly legible, but the model answered `{"unreadable": true}` instead of reading the 22:30 / 06:30 / 17:30 boundaries. The outcome is still a refusal (no tariff is accepted), but the user sees "The bill could not be read clearly" rather than the specific "half-hour slots are not supported yet" message. The half-hour validator itself is covered offline in `tests/test_bill_reader.py`.
- h04 (USD bill on the INR site) was read faithfully as USD and blocked by the currency check; the plausibility check also flagged all three rates.
- Fixed charges: read correctly on all 8 bills that were read (the unreadable and half-hour bills returned none). The fixed charge is shown read-only and never used.
- All ten replies were strict JSON.
