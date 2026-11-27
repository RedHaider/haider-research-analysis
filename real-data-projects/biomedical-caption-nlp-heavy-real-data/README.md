# Radiology caption negation detection — Public-data companion

Use a compact baseline and a wider network trained from scratch with AdamW. Select checkpoints by validation loss.

Detect lexicon-defined negation cues in radiology figure captions after removing the source cue words. Tokenized word n-grams are hashed to 1,024 features. The labels are weak labels, and the task concerns captions rather than clinical notes.

[Research connection](https://bime.uw.edu/faculty/meliha-yetisgen/)

[Dataset source](https://huggingface.co/datasets/unsloth/Radiology_mini)

Input shape is 1024. The experiment uses 280 observations before splitting.

## Held-out results

| Arm | accuracy | auroc | average_precision | brier | training_records | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.9125 | 0.54403 | 0.11623 | 0.086969 | 156 | 67938 | 7 | 8 |
| heavy | 0.9125 | 0.34638 | 0.078585 | 0.08598 | 156 | 1.4482e+06 | 2.5 | 6.5 |

Run `python ../../training-logs/batch79/train.py 2 --real` from this folder. Install the pinned requirements and put the public datasets in the shared datasets folder.

See `results.json` for per-seed metrics and `checkpoints/` for trained weights, validation curves and split statistics.
