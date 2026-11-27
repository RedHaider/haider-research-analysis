# Thin-vessel segmentation with residual context — Professor-aligned simulation

Add a high-frequency image channel to thin-vessel segmentation. Train both architectures using foreground-weighted BCE and soft Dice. Use the fixed DRIVE test partition.

For this run the source observations and labels are simulated.
Input shape is 2 × 64 × 64. There are 160 observations before splitting.

[Research connection](https://www.uwyo.edu/eecs/faculty-staff/faculty/Suresh-Muknahallipatna/index.html)

## Mean held-out metrics

| Arm | dice | boundary_dice | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 0.62058 | 0.36467 | 44864 | 2.4337 | 8 | 8 |
| heavy | 0.8701 | 0.76598 | 1.5494e+06 | 21.745 | 12 | 12 |

Run `python ../../training-logs/batch100/train.py 15` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
