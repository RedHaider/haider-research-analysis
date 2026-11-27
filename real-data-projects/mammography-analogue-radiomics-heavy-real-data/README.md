# Breast imaging classification from image and texture cues — Public-data companion

Combine image intensity with high-frequency residual texture. Train compact and residual CNNs to classify breast ultrasound images. This is a public ultrasound analogue of computer-aided breast image classification.

This run uses the public source linked below.
Input shape is 2 × 64 × 64. There are 702 observations before splitting.

[Research connection](https://midas.umich.edu/directory/lubomir-hadjiyski/)

[Dataset source](https://medmnist.com/)

## Mean held-out metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | elapsed_seconds | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.80769 | 0.83114 | 0.90934 | 0.14514 | 31618 | 2.9872 | 6 | 8 |
| heavy | 0.51282 | 0.8254 | 0.91457 | 0.23648 | 1.5247e+06 | 24.521 | 2.5 | 6.5 |

Run `python ../../training-logs/batch100/train.py 2 --real` from this folder. Set `PORTFOLIO_DATA` if the datasets are stored outside the shared data folder.

See results.json for seed-level metrics and checkpoints for weights and split statistics.
