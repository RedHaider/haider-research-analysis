# Segmentation of thin vessels — Professor-aligned simulation

Segment thin structures using a shallow encoder–decoder and a residual variant with a larger receptive field. Use a foreground-weighted BCE and soft Dice objective. The test set contains unseen images.

This benchmark uses synthetic records or images designed for a controlled experiment.

[Research connection](https://www.sci.utah.edu/people/tolga.html)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | dice | parameters | epoch |
|---|---:|---:|---:|
| compact | 0.92919 | 5845 | 10 |
| residual | 0.97933 | 1.0992e+05 | 8.6667 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 8` from this folder using the dependencies listed in `requirements.txt`.
