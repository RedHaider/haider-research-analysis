# Nocturnal low-glucose risk — Professor-aligned simulation

Binary classification of an observed overnight low-glucose reading using the preceding six readings, observation intervals and time of day. The forecast window is midnight–06:00. Nights require at least one observed outcome-period reading. The estimand is an observed low in an intermittently sampled diary.

This benchmark uses synthetic records or images designed for a controlled experiment.

[Research connection](https://apps.ohsu.edu/research/tech-portal/technology/view/4282492)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | auroc | brier | parameters | epoch |
|---|---:|---:|---:|---:|---:|
| compact | 0.74427 | 0.81101 | 0.17638 | 1882 | 3.3333 |
| residual | 0.72252 | 0.80745 | 0.1846 | 37082 | 4.3333 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 0` from this folder using the dependencies listed in `requirements.txt`.
