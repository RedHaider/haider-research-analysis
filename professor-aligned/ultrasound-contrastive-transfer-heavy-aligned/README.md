# Professor-aligned simulation

Test whether contrastive initialization improves training on 64×64 ultrasound images. Train a compact supervised CNN, a larger supervised CNN, and the same larger CNN after six epochs of label-free paired-view contrastive pretraining.
Pretraining uses only training images. The two larger models have the same architecture and each receives ten supervised epochs. Select checkpoints using the separate validation set. Evaluate original images and a fixed, artificial speckle-like perturbation.

This study adapts ultrasound representation learning to static image classification. The public task uses BreastMNIST breast ultrasound and separates malignant from normal or benign images. The simulation uses echo phantoms with known lesion shapes.

Training, validation and test sizes are 550, 100 and 150.

[Research context](https://arxiv.org/abs/2310.10689)

## Evaluation

| Model | AUROC | Perturbed AUROC | Parameters |
|---|---:|---:|---:|
| compact | 1.0000 | 1.0000 | 31,362 |
| heavy_supervised | 1.0000 | 1.0000 | 1,524,042 |
| heavy_contrastive | 1.0000 | 1.0000 | 1,524,042 |

Run `python ../../training-logs/batch80/train.py` from this folder.
See results.json for per-seed metrics and checkpoints for trained weights, normalization statistics and training curves.
