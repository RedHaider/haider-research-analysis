# Group reweighting in readmission prediction — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Compare an unweighted compact model with a wider model trained using inverse frequency weights for combinations of age group and outcome. Report discrimination and Brier score separately by age. Architecture and training weights both change.

[Research connection](https://sail.uark.edu/)

[Dataset source](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

Input shape is 13. The experiment uses 18,365 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.8859 | 0.6132 | 0.19146 | 0.098625 | 3600 | 3234 | 3 | 7 |
| compact_age_70_plus | 0.87318 | 0.57886 | 0.19215 | 0.10929 | — | — | — | — |
| compact_under_70 | 0.89682 | 0.63891 | 0.2068 | 0.089476 | — | — | — | — |
| heavy | 0.81637 | 0.61447 | 0.18597 | 0.17932 | 3600 | 6.7175e+05 | 3.5 | 7.5 |
| heavy_age_70_plus | 0.82357 | 0.59361 | 0.18951 | 0.17654 | — | — | — | — |
| heavy_under_70 | 0.81013 | 0.63838 | 0.192 | 0.18173 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 6 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
