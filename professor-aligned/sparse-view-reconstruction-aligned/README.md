# Sparse-view CT reconstruction — Professor-aligned simulation

Apply a nine-angle parallel-beam projection and a discretized filtered backprojection. Compare reconstructions against the original slice. Evaluate compact and residual image-to-image CNNs, with the uncorrected backprojection as a reference. Measurements are simulated.

This benchmark uses synthetic records or images designed for a controlled experiment.

[Research connection](https://eecs.utk.edu/people/jens-gregor/)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | psnr_db | mae | parameters | epoch |
|---|---:|---:|---:|---:|
| compact | 16.678 | 0.11831 | 5845 | 6 |
| residual | 16.872 | 0.11586 | 1.0992e+05 | 6 |
| input_baseline | 14.784 | 0.14791 | — | — |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 2` from this folder using the dependencies listed in `requirements.txt`.
