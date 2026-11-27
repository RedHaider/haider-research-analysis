# Transition detection from wearable heart rate — Professor-aligned simulation

Predict whether the next 30-second epoch crosses between sleep and wake, using ten past wearable heart-rate epochs and a missing-measurement indicator. The simulated transition label comes from consecutive thresholded synthetic heart-rate samples.

This benchmark uses synthetic records or images designed for a controlled experiment.

[Research connection](https://www.nature.com/articles/s41598-019-44004-w)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | auroc | brier | parameters | epoch |
|---|---:|---:|---:|---:|---:|
| compact | 0.65152 | 0.69741 | 0.21784 | 1846 | 4.6667 |
| residual | 0.63531 | 0.69529 | 0.22516 | 37010 | 1.3333 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 4` from this folder using the dependencies listed in `requirements.txt`.
