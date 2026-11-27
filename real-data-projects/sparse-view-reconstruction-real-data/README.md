# Sparse-view CT reconstruction — Public-data companion

Apply a nine-angle parallel-beam projection and a discretized filtered backprojection. Compare reconstructions against the original slice. Evaluate compact and residual image-to-image CNNs, with the uncorrected backprojection as a reference. Measurements are simulated.

Use abdominal CT slices from OrganAMNIST as the reference signal. Simulate sparse projection measurements from the downsampled slices. Evaluation separates slices. Patient identifiers are unavailable in this input.

[Research connection](https://eecs.utk.edu/people/jens-gregor/)

[Dataset source](https://medmnist.com/)

Models train locally with AdamW for up to six epochs, with ten epochs for vessel segmentation. Select the checkpoint by validation loss. Report means across three seeded repeats.

## Results

| Model | psnr_db | mae | parameters | epoch |
|---|---:|---:|---:|---:|
| compact | 15.212 | 0.13151 | 5845 | 6 |
| residual | 14.635 | 0.13811 | 1.0992e+05 | 5 |
| input_baseline | 13.811 | 0.16582 | — | — |

The saved results include split sizes, per-seed metrics and model parameters. The checkpoints directory holds weights and training-only normalization statistics.

Run `python ../../training-logs/batch50/train.py 2 --real` from this folder using the dependencies listed in `requirements.txt`.
