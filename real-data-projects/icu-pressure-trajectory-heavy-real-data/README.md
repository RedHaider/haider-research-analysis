# Forecasting observed hypotension from early ICU vitals — Public-data companion

Use 24 temporal bins before a fixed landmark. The public-data study uses observations before ICU hour six and predicts an observed systolic reading below 90 mmHg in hours 6–12, conditional on recorded normal prior blood pressure and available follow-up. The endpoint is a recorded low reading.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://profiles.ucsd.edu/shamim.nemati)

[Data source](https://physionet.org/content/challenge-2012/1.0.0/)

## Mean test metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.83417 | 0.66931 | 0.28523 | 0.1339 | 19810 | 2 | 6 |
| heavy | 0.83333 | 0.6696 | 0.26978 | 0.13557 | 1.259e+06 | 1 | 5 |

Run `python ../../training-logs/batch60/train.py 3 --real` using the pinned dependencies. Public data are read from the project datasets directory.
