# Heng Huang — professor-aligned simulation

Compare a two-block temporal CNN with an eight-convolution temporal CNN on hourly physiology and observation indicators.

Simulated 24-hour trajectories with eight measured channels, intermittent observations and an outcome-dependent trend.

Train on local CPU with scaled inputs, AdamW, and validation-selected checkpoints across three seeds. The compact baseline has a budget of seven epochs and the medium model has a budget of nine. The stored split file holds indices and training-only scaling statistics. Accuracy is reported as a fraction and glucose MAE is in mg/dL.

[Research reference](https://www.cs.umd.edu/article/2026/10/umd-computer-scientist-heng-huang-receives-funding-develop-wearable-ai-vitals)

## Mean held-out metrics

| Model | auroc | average_precision | accuracy | parameters | selected_epoch |
|---|---:|---:|---:|---:|---:|
| compact | 0.61241 | 0.38889 | 0.71349 | 1562 | 4.3333 |
| medium | 0.61498 | 0.38491 | 0.70569 | 88706 | 1.3333 |

Run `python train.py` with the versions in `requirements.txt`. For public-data runs, set `PORTFOLIO_DATA` to your local datasets directory.

See `results.json` for individual seeds and `checkpoints/` for saved weights and split statistics.
