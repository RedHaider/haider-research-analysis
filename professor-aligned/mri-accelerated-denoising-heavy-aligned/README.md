# MRI restoration after simulated low-SNR acquisition — Professor-aligned simulation

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Use generated data to test the main method in a controlled setting. See the paired public study for the dataset-specific protocol and limitations.

[Research connection](https://www.boisestate.edu/coen-ece/directory/bangerter/)

Input shape is 1 × 64 × 64. The experiment uses 370 observations before splitting.

## Held-out results

| Arm | psnr | mae | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 21.331 | 0.066887 | 221 | 44608 | 8 | 8 |
| fixed_reference | 19.312 | 0.086423 | — | — | — | — |
| heavy | 25.284 | 0.042605 | 221 | 1.5488e+06 | 12 | 12 |

Run `python ../../training-logs/batch79/train.py 18` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
