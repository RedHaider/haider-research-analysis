# Early hospital stay-length classification — Public-data companion

Test neural prediction of stays of seven or more days. The public-data model uses admission-type and source codes, prior utilization, age and a gender indicator, with patient-disjoint splits.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://www.cis.udel.edu/people/faculty/rahmat-beheshti/)

[Data source](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

## Mean test metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.7873 | 0.56516 | 0.25982 | 0.16577 | 2850 | 7 | 8 |
| heavy | 0.78801 | 0.56602 | 0.2614 | 0.16613 | 6.6714e+05 | 7.5 | 10 |

Run `python ../../training-logs/batch60/train.py 4 --real` using the pinned dependencies. Public data are read from the project datasets directory.
