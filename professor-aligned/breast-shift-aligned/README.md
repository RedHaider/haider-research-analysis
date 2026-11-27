# Breast imaging under acquisition shift — Professor-aligned simulation

Classify breast-image labels, then repeat evaluation with a fixed synthetic blur, gain change and additive noise. Each architecture sees the same clean training data.

This benchmark uses synthetic records or images designed for a controlled experiment.

[Research connection](https://www.sci.pitt.edu/people/shandong-wu)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | auroc | brier | parameters | epoch |
|---|---:|---:|---:|---:|---:|
| compact | 0.99464 | 0.99994 | 0.021391 | 3286 | 5.3333 |
| compact_stress | 0.49839 | 0.99478 | 0.50152 | — | — |
| residual | 1 | 1 | 3.3352e-08 | 99218 | 4.6667 |
| residual_stress | 0.49839 | 1 | 0.50159 | — | — |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 3` from this folder using the dependencies listed in `requirements.txt`.
