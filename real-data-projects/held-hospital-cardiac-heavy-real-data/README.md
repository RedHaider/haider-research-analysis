# Cardiac risk on a held-out hospital — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Hold one site out for testing. Fill nonfinite inputs with zero and append missingness indicators. Classification is presence of recorded heart disease.

[Research connection](https://medicine.osu.edu/departments/biomedical-informatics/our-people)

[Dataset source](https://archive.ics.uci.edu/dataset/45/heart+disease)

Input shape is 26. The experiment uses 920 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.78061 | 0.87003 | 0.8193 | 0.15623 | 488 | 4066 | 7.5 | 8 |
| heavy | 0.66837 | 0.80846 | 0.69661 | 0.22023 | 488 | 6.8173e+05 | 2.5 | 6.5 |

Run `python ../../training-logs/batch79/train.py 1 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
