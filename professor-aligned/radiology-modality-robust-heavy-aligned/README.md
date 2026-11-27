# Robust imageâ€“text matching with degraded images — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://www.uvm.edu/cems/cs/profile/safwan-wshah)

Input shape is 2048. The experiment uses 3,520 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.98672 | 0.99951 | 0.99859 | 0.0084671 | 2240 | 1.3347e+05 | 3 | 7 |
| compact_stress | 0.9875 | 0.9993 | 0.99805 | 0.0091879 | — | — | — | — |
| fixed_reference | 0.27344 | 0.99993 | 0.99981 | 0.24118 | — | — | — | — |
| heavy | 0.98516 | 0.99896 | 0.99701 | 0.011628 | 2240 | 2.2346e+06 | 3.5 | 7.5 |
| heavy_stress | 0.98516 | 0.99802 | 0.99476 | 0.013102 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 12` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
