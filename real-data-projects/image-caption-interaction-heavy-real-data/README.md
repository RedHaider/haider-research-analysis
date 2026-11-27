# Multimodal radiology pair verification — Public-data companion

Train compact and wide multilayer perceptrons on frozen image and text embeddings, absolute differences and elementwise products. Select three high-similarity negative captions within the same ten-pair block. Keep blocks disjoint between training and validation and use a separate test gallery. Compare with cosine similarity.

Compact and larger networks train with AdamW, with the best checkpoint selected by validation loss. Raw metrics and split sizes are stored per seed.

[Research reference](https://midas.umich.edu/directory/chuan-zhou/)

[Data source](https://huggingface.co/datasets/unsloth/Radiology_mini)

## Mean test metrics

| Arm | accuracy | auroc | average_precision | brier | parameters | best_epoch | epochs_ran |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact | 0.76406 | 0.73992 | 0.4511 | 0.16596 | 1.3347e+05 | 6 | 8 |
| fixed_reference | 0.25 | 0.70203 | 0.51523 | 0.34216 | — | — | — |
| heavy | 0.77031 | 0.74305 | 0.4937 | 0.16198 | 2.2346e+06 | 5 | 9 |

Run `python ../../training-logs/batch60/train.py 1 --real` using the pinned dependencies. Public data are read from the project datasets directory.
