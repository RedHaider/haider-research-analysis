# Convolutional histology tissue classification — Professor-aligned simulation

Classify nine histology tissue types with compact and residual convolutional classifiers using spatial pooling. Real-data experiments use the official partition and a bounded per-partition sample.

This benchmark uses synthetic records or images designed for a controlled experiment.

[Research connection](https://pacaid.cc.hawaii.edu/cores/medai/)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | parameters | epoch |
|---|---:|---:|---:|
| compact | 0.2926 | 3789 | 6 |
| residual | 0.90032 | 1.0033e+05 | 5.3333 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 6` from this folder using the dependencies listed in `requirements.txt`.
