# Thin-vessel segmentation with residual context — Public-data companion

Add a high-frequency image channel to thin-vessel segmentation. Train both architectures using foreground-weighted BCE and soft Dice. Use the fixed DRIVE test partition.

This run uses the public source linked below.
Input shape is 2 × 64 × 64. There are 40 observations before splitting.

[Research connection](https://www.uwyo.edu/eecs/faculty-staff/faculty/Suresh-Muknahallipatna/index.html)

[Dataset source](https://drive.grand-challenge.org/)

## Mean held-out metrics

| Arm | dice | boundary_dice | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 0.17668 | 0.17511 | 44864 | 0.61775 | 8 | 8 |
| heavy | 0.23946 | 0.12987 | 1.5494e+06 | 3.026 | 12 | 12 |

Run `python ../../training-logs/batch100/train.py 15 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
