# Clinical forecasting with partial data — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Predict ICU stays of seven or more days from first-day lab sequences, observation flags and demographics. The larger model trains with a block of lab inputs and flags removed in one-third of each shuffled batch. The same missing-block condition is used during stress evaluation.

[Research connection](https://bme.gatech.edu/bio/may-dongmei-wang)

[Dataset source](https://physionet.org/content/challenge-2012/1.0.0/)

Input shape is 18 × 24. The experiment uses 4,000 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.71413 | 0.58187 | 0.76306 | 0.20191 | 2402 | 19970 | 1.5 | 5.5 |
| compact_stress | 0.67283 | 0.56972 | 0.76016 | 0.21939 | — | — | — | — |
| heavy | 0.71413 | 0.57153 | 0.75657 | 0.20257 | 2402 | 1.2594e+06 | 1 | 5 |
| heavy_stress | 0.71467 | 0.57433 | 0.75405 | 0.20192 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 4 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
