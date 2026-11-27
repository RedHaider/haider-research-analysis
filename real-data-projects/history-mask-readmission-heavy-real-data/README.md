# Readmission prediction with missing prior utilization — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Train on intact utilization features and score both intact data and cases with three history fields zeroed. The larger model sees zeroed history fields in one-third of each shuffled batch. Prediction target is 30-day readmission.

[Research connection](https://cancer.osu.edu/find-a-researcher/search-researcher-directory/ping-zhang-100003433)

[Dataset source](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

Input shape is 13. The experiment uses 18,365 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.8859 | 0.6132 | 0.19146 | 0.098625 | 3600 | 3234 | 3 | 7 |
| compact_stress | 0.88602 | 0.54006 | 0.12575 | 0.10224 | — | — | — | — |
| heavy | 0.88602 | 0.62216 | 0.19976 | 0.098555 | 3600 | 6.7175e+05 | 2 | 6 |
| heavy_stress | 0.88602 | 0.56103 | 0.13538 | 0.10081 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 0 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
