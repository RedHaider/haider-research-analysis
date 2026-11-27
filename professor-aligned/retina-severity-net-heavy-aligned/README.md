# Retinal severity classification — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://www.hrs.sc.edu/study/colleges_schools/engineering_and_computing/faculty-staff/jianjunhu.php)

Input shape is 3 × 32 × 32. The experiment uses 1,400 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 1 | 1 | 1 | 1.0592e-07 | 840 | 31874 | 7 | 8 |
| heavy | 1 | 1 | 1 | 6.0265e-09 | 840 | 1.5253e+06 | 4 | 8 |

Run `python ../../training-logs/batch79/train.py 16` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
