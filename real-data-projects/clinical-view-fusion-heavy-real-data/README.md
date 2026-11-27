# Fusion of demographics and early laboratory sequences — Public-data companion

Combine temporal laboratory values and measurement indicators with age and recorded sex. Predict a stay of at least seven days from the first 24 hours of public ICU records. Report held-out stays and standard binary prediction metrics.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://bime.uw.edu/faculty/chen-liang/)

[Data source](https://physionet.org/content/challenge-2012/1.0.0/)

## Mean test metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.71413 | 0.58187 | 0.76306 | 0.20191 | 19970 | 1.5 | 5.5 |
| heavy | 0.70761 | 0.56677 | 0.75328 | 0.20474 | 1.2594e+06 | 1 | 5 |

Run `python ../../training-logs/batch60/train.py 8 --real` using the pinned dependencies. Public data are read from the project datasets directory.
