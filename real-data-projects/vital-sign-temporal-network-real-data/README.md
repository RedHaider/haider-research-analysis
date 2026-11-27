# Heng Huang — public-data companion

Compare a two-block temporal CNN with an eight-convolution temporal CNN on hourly physiology and observation indicators.

PhysioNet Challenge 2012 Set A, 4,000 stays. Hourly aggregation of early vital signs and observation indicators through the first 24 hours. The target is in-hospital mortality. This retains the temporal risk-estimation idea using ICU measurements and a different endpoint from wearable readmission.

Train on local CPU with scaled inputs, AdamW, and validation-selected checkpoints across three seeds. The compact baseline has a budget of seven epochs and the medium model has a budget of nine. The stored split file holds indices and training-only scaling statistics. Accuracy is reported as a fraction and glucose MAE is in mg/dL.

[Research reference](https://www.cs.umd.edu/article/2026/10/umd-computer-scientist-heng-huang-receives-funding-develop-wearable-ai-vitals)

[Dataset source](https://physionet.org/content/challenge-2012/1.0.0/)

## Mean held-out metrics

| Model | auroc | average_precision | accuracy | parameters | selected_epoch |
|---|---:|---:|---:|---:|---:|
| compact | 0.68178 | 0.24452 | 0.85906 | 1562 | 5.6667 |
| medium | 0.63556 | 0.20255 | 0.85978 | 88706 | 2 |

Run `python train.py` with the versions in `requirements.txt`. For public-data runs, set `PORTFOLIO_DATA` to your local datasets directory.

See `results.json` for individual seeds and `checkpoints/` for saved weights and split statistics.
