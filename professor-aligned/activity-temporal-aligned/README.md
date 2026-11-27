# Human activity recognition — Professor-aligned simulation

Use 64 time samples per window from three accelerometer channels and three gyroscope channels to classify six activities. Participants are disjoint across splits.

This benchmark uses synthetic records or images designed for a controlled experiment.

[Research connection](https://eng.ua.edu/eng-directory/dr-jiaqi-jackey-gong/)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | parameters | epoch |
|---|---:|---:|---:|
| compact | 0.56781 | 2118 | 6 |
| residual | 0.93382 | 37614 | 6 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 5` from this folder using the dependencies listed in `requirements.txt`.
