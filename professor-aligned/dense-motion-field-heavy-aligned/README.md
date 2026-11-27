# Dense displacement estimation from image pairs — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://www.ece.lsu.edu/xinli/)

Input shape is 2 × 40 × 40. The experiment uses 1,000 observations before splitting.

## Held-out results

| Arm | endpoint_error_px | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 9.4329 | 600 | 45136 | 7.5 | 8 |
| heavy | 9.269 | 600 | 1.5501e+06 | 10.5 | 12 |
| fixed_reference | 2.8047 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 7` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
