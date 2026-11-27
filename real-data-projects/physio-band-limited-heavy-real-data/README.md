# Sensor classification under bandwidth reduction — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Classify left versus right motor imagery using EEG time series. Hold out people and repeat evaluation after additional temporal smoothing. Smoothing approximates reduced sensor bandwidth.

[Research connection](https://ceps.unh.edu/person/md-shaad-mahmud)

[Dataset source](https://physionet.org/content/eegmmidb/1.0.0/)

Input shape is 8 × 128. The experiment uses 480 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.5 | 0.48458 | 0.50651 | 0.25057 | 270 | 19170 | 2 | 6 |
| compact_stress | 0.5 | 0.4693 | 0.48515 | 0.25044 | — | — | — | — |
| heavy | 0.5 | 0.49055 | 0.48727 | 0.25033 | 270 | 1.2574e+06 | 1.5 | 5.5 |
| heavy_stress | 0.5 | 0.49236 | 0.49954 | 0.25024 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 10 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
