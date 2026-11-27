# Wearable classification with missing sensors — Professor-aligned simulation

Classify six activities from accelerometer and gyroscope channels, then set gyroscope readings to zero at test time. During residual-model training, mask gyroscope readings for one third of each shuffled batch. The compact baseline trains on intact inputs. Compare both models on intact and missing-stream data. This comparison changes model depth and training augmentation together.

This benchmark uses synthetic records or images designed for a controlled experiment.

[Research connection](https://engineering.tamu.edu/cse/profiles/mortazavi-bobak.html)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | parameters | epoch |
|---|---:|---:|---:|
| compact | 0.61356 | 2118 | 6 |
| compact_stress | 0.15359 | — | — |
| residual | 0.82516 | 37614 | 5 |
| residual_stress | 0.87663 | — | — |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 7` from this folder using the dependencies listed in `requirements.txt`.
