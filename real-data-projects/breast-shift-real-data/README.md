# Breast imaging under acquisition shift — Public-data companion

Classify breast-image labels, then repeat evaluation with a fixed synthetic blur, gain change and additive noise. Each architecture sees the same clean training data.

Use BreastMNIST ultrasound rather than mammograms. The corruption is artificial and does not establish scanner-level or hospital-level robustness.

[Research connection](https://www.sci.pitt.edu/people/shandong-wu)

[Dataset source](https://medmnist.com/)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | auroc | brier | parameters | epoch |
|---|---:|---:|---:|---:|---:|
| compact | 0.73077 | 0.75487 | 0.17349 | 3286 | 6 |
| compact_stress | 0.73077 | 0.69472 | 0.19196 | — | — |
| residual | 0.80983 | 0.80416 | 0.15022 | 99218 | 4.3333 |
| residual_stress | 0.67308 | 0.67043 | 0.21493 | — | — |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 3 --real` from this folder using the dependencies listed in `requirements.txt`.
