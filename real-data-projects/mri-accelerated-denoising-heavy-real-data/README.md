# MRI restoration after simulated low-SNR acquisition — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Restore MRI image crops after synthetic Rician-like magnitude noise. Compare both trained architectures with the corrupted input. The data are 2D images with file-level deduplication and no patient identifiers.

[Research connection](https://www.boisestate.edu/coen-ece/directory/bangerter/)

[Dataset source](https://huggingface.co/datasets/miladfa7/Brain-MRI-Images-for-Brain-Tumor-Detection)

Input shape is 1 × 64 × 64. The experiment uses 228 observations before splitting.

## Held-out results

| Arm | psnr | mae | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|
| compact | 17.758 | 0.0871 | 136 | 44608 | 8 | 8 |
| fixed_reference | 18.277 | 0.099833 | — | — | — | — |
| heavy | 20.703 | 0.06095 | 136 | 1.5488e+06 | 12 | 12 |

Run `python ../../training-logs/batch79/train.py 18 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
