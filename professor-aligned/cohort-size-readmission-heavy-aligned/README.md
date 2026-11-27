# Sample-size scaling of health records models — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://eecs.wsu.edu/~assefaw/home.html)

Input shape is 15. The experiment uses 2,600 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.73579 | 0.78687 | 0.68481 | 0.17998 | 650 | 3362 | 6 | 8 |
| heavy | 0.73829 | 0.79487 | 0.68523 | 0.1766 | 1560 | 6.7328e+05 | 2 | 6 |

Run `python ../../training-logs/batch79/train.py 14` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
