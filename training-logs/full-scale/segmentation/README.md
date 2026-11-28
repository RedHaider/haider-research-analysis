# Loss-conditioned segmentation and contour probabilities

Polina Golland co-authored [Hypernet-Ensemble Learning of Segmentation Probability](https://arxiv.org/abs/2112.06693), which varies Tversky's false-positive and false-negative penalties and marginalizes predictions to estimate contour probabilities. This independent implementation uses feature-wise scale and bias in a residual U-Net bottleneck. A fixed-Tversky version provides the reference. Both arms also use a fixed auxiliary binary cross-entropy term, the same split, and matched augmentation and optimization. The conditioned arm averages predictions at five loss settings. Its output includes foreground area at each setting.

* The professor-aligned experiment uses 1,800 generated lesion phantoms. The contour threshold is randomized independently of image appearance to model annotation variation.
* The real-data experiment uses all [BUS-BRA](https://zenodo.org/records/8231412) image-mask pairs that pass the input assertions. It splits by the supplied Case identifier. The benchmark evaluates each scan against its one available reference mask. It measures agreement and probabilistic error near that reference contour. It has no inter-rater validation.

Run from the research project root with PyTorch, NumPy, SciPy, Pillow, and scikit-learn installed.

    python training-logs/full-scale/segmentation/train.py --epochs 12 --threads 2
    python training-logs/full-scale/segmentation/train.py --real --epochs 12 --threads 2

The real run reads datasets/busbra.zip. The expected MD5 reported by Zenodo is 1f8b2be6476d58fc97bfb5e5a1ea9bab. The trainer stores per-epoch validation Dice, exact splits, optimizer state, best weights, and held-out Dice and Brier scores. Test comparison uses a paired bootstrap over cases. Resume an interrupted run with the same arguments. Use a distinct seed to collect an independent optimization run with the same patient split.
