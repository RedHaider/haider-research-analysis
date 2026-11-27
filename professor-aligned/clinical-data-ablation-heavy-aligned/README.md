# Clinical forecasting with partial data — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://bme.gatech.edu/bio/may-dongmei-wang)

Input shape is 16 × 24. The experiment uses 2,250 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.79054 | 0.8222 | 0.70671 | 0.14739 | 1350 | 19810 | 2 | 6 |
| compact_stress | 0.70463 | 0.74078 | 0.55913 | 0.24657 | — | — | — | — |
| heavy | 0.79344 | 0.8387 | 0.72178 | 0.15587 | 1350 | 1.259e+06 | 1.5 | 5.5 |
| heavy_stress | 0.71332 | 0.79103 | 0.62941 | 0.1886 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 4` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
