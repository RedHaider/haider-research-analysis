# Robust imageâ€“text matching with degraded images — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Train a pair verifier on frozen radiology image and caption embeddings. Train the larger verifier with image-embedding noise in one-third of each batch. Test both models with the same intensity of added noise. Training blocks and the test gallery are separated.

[Research connection](https://www.uvm.edu/cems/cs/profile/safwan-wshah)

[Dataset source](https://huggingface.co/datasets/unsloth/Radiology_mini)

Input shape is 2048. The experiment uses 1,120 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.76406 | 0.73992 | 0.4511 | 0.16596 | 600 | 1.3347e+05 | 6 | 8 |
| compact_stress | 0.75313 | 0.72594 | 0.45805 | 0.16801 | — | — | — | — |
| fixed_reference | 0.25 | 0.70203 | 0.51523 | 0.34216 | — | — | — | — |
| heavy | 0.75625 | 0.72336 | 0.46245 | 0.16938 | 600 | 2.2346e+06 | 4.5 | 8.5 |
| heavy_stress | 0.7625 | 0.72339 | 0.46825 | 0.16978 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 12 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
