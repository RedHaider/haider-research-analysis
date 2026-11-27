# Breast imaging classification from image and texture cues — Professor-aligned simulation

Combine image intensity with high-frequency residual texture. Train compact and residual CNNs to classify breast ultrasound images. This is a public ultrasound analogue of computer-aided breast image classification.

For this run the source observations and labels are simulated.
Input shape is 2 × 64 × 64. There are 950 observations before splitting.

[Research connection](https://midas.umich.edu/directory/lubomir-hadjiyski/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 1 | 1 | 1 | 3.7273e-06 | 31618 | 3.8359 | 8 | 8 |
| heavy | 1 | 1 | 1 | 2.1789e-09 | 1.5247e+06 | 45.608 | 5 | 9 |

Run `python ../../training-logs/batch100/train.py 2` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
