# Sample-size scaling of health records models — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Compare compact and larger readmission classifiers under different training-data budgets. Sample at most 650 training records for the compact arm and at most 3,600 for the wider arm. Patients are held out. This design changes architecture and sample size together.

[Research connection](https://eecs.wsu.edu/~assefaw/home.html)

[Dataset source](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

Input shape is 13. The experiment uses 6,200 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.89173 | 0.57563 | 0.14253 | 0.096277 | 650 | 3234 | 3.5 | 7 |
| heavy | 0.89173 | 0.6158 | 0.18979 | 0.094178 | 3600 | 6.7175e+05 | 3 | 7 |

Run `python ../../training-logs/batch79/train.py 14 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
