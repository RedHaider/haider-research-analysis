# Retinal severity classification — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Classify RetinaMNIST grades 2–4 versus grades 0–1. Evaluate the official test partition as a related retinal disease classification task.

[Research connection](https://www.hrs.sc.edu/study/colleges_schools/engineering_and_computing/faculty-staff/jianjunhu.php)

[Dataset source](https://medmnist.com/)

Input shape is 3 × 32 × 32. The experiment uses 1,480 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.73125 | 0.78718 | 0.69346 | 0.18728 | 842 | 31874 | 7 | 8 |
| heavy | 0.71 | 0.78033 | 0.68405 | 0.18945 | 842 | 1.5253e+06 | 6 | 10 |

Run `python ../../training-logs/batch79/train.py 16 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
