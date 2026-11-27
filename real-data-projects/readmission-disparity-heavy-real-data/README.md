# Age-stratified hospital readmission ranking — Public-data companion

Compare neural readmission rankers, and report AUROC, average precision and Brier score for the full test set and by age below or at least 70. Hold out people. The public-data endpoint is observed readmission within 30 days for diabetes encounters.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://cph.osu.edu/people/fareed5)

[Data source](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

## Mean test metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.8859 | 0.6132 | 0.19146 | 0.098625 | 3234 | 3 | 7 |
| compact_age_70_plus | 0.87318 | 0.57886 | 0.19215 | 0.10929 | — | — | — |
| compact_under_70 | 0.89682 | 0.63891 | 0.2068 | 0.089476 | — | — | — |
| heavy | 0.88602 | 0.62936 | 0.20162 | 0.098214 | 6.7175e+05 | 2 | 6 |
| heavy_age_70_plus | 0.87318 | 0.59722 | 0.1968 | 0.10894 | — | — | — |
| heavy_under_70 | 0.89704 | 0.6492 | 0.21171 | 0.089009 | — | — | — |

Run `python ../../training-logs/batch60/train.py 2 --real` using the pinned dependencies. Public data are read from the project datasets directory.
