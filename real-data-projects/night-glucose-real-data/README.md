# Nocturnal low-glucose risk — Public-data companion

Binary classification of an observed overnight low-glucose reading using the preceding six readings, observation intervals and time of day. The forecast window is midnight–06:00. Nights require at least one observed outcome-period reading. The estimand is an observed low in an intermittently sampled diary.

Use the public UCI Diabetes records with glucose dated by patient. The dataset comprises intermittent log entries, with no continuous glucose traces.

[Research connection](https://apps.ohsu.edu/research/tech-portal/technology/view/4282492)

[Dataset source](https://archive.ics.uci.edu/dataset/34/diabetes)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | accuracy | auroc | brier | parameters | epoch |
|---|---:|---:|---:|---:|---:|
| compact | 0.36643 | 0.50803 | 0.25355 | 1882 | 3 |
| residual | 0.64368 | 0.51791 | 0.24284 | 37082 | 2.6667 |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 0 --real` from this folder using the dependencies listed in `requirements.txt`.
