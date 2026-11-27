# Dense displacement estimation from image pairs — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Train two dense image-to-image networks to predict two components of a displacement field. Bound the predicted displacement to eight pixels per axis. Apply smooth synthetic deformation to source images and keep derivatives of each original in the same split. The real-data study uses real endoscopy backgrounds and artificial motion.

[Research connection](https://www.ece.lsu.edu/xinli/)

[Dataset source](https://datasets.simula.no/kvasir-seg/)

Input shape is 2 × 40 × 40. The experiment uses 1,000 observations before splitting.

## Held-out results

| Arm | endpoint_error_px | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|
| compact | 9.397 | 600 | 45136 | 7.5 | 8 |
| heavy | 8.7577 | 600 | 1.5501e+06 | 11.5 | 12 |
| fixed_reference | 2.8047 | — | — | — | — |

Run `python ../../training-logs/batch79/train.py 7 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
