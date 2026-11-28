# Three-dimensional deformable registration

[An Unsupervised Learning Model for Deformable Medical Image Registration](https://arxiv.org/abs/1802.02604), co-authored by Guha Balakrishnan, motivates predicting a dense deformation with a CNN and training from image similarity and displacement smoothness. The separate [probabilistic diffeomorphic model](https://www.sciencedirect.com/science/article/pii/S1361841519300635) motivates integrating a velocity field. This independent implementation compares an identity warp, a direct displacement U-Net, and scaling-and-squaring integration. Both learned arms train without segmentation supervision on the same source pairs.

The aligned benchmark uses 900 generated 3D phantoms. The real-data companion uses the labeled T1-weighted crops in [Medical Segmentation Decathlon Task 04](https://medicaldecathlon.com/). Resampling maps volumes to a common normalized grid and does not preserve metric physical distances. It uses disjoint source volumes across training, validation and test pairs. The archive labels crops with volume IDs, so this evaluation does not establish patient-level disjointness. Held-out Dice is computed on the provided subregion masks. Local normalized cross-correlation and the fraction of nonpositive Jacobians measure alignment and fold behavior.

    python training-logs/full-scale/registration/train.py
    python training-logs/full-scale/registration/train.py --real

Install PyTorch, NumPy and SciPy, and install nibabel for the MRI archive. The real run reads datasets/msd-hippocampus.tar. Both runs save split and pair IDs, validation records and the best model states.
