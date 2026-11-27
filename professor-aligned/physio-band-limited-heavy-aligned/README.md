# Sensor classification under bandwidth reduction — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://ceps.unh.edu/person/md-shaad-mahmud)

Input shape is 8 × 128. The experiment uses 1,200 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 1 | 1 | 1 | 4.1639e-08 | 700 | 19170 | 8 | 8 |
| compact_stress | 0.74821 | 1 | 1 | 0.2478 | — | — | — | — |
| heavy | 1 | 1 | 1 | 3.3494e-11 | 700 | 1.2574e+06 | 2.5 | 6.5 |
| heavy_stress | 0.75357 | 1 | 1 | 0.15558 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 10` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
