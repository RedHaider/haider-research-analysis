# Adapting a frozen classifier to sensor drift

[Adapting in the Dark](https://arxiv.org/abs/2604.15609), co-authored by Jihun Hamm, motivates a trainable local model around a frozen source predictor. This independent variant trains a bounded residual logit correction. It compares a frozen classifier, an unguarded entropy loss and an anchored update using confident predictions and consistency. The local experiment uses the frozen source model's returned scores as its interface.

The aligned project generates 30,000 records across ten increasingly shifted batches. The companion uses 13,910 measurements from the [UCI Gas Sensor Array Drift dataset](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset). It trains on batches 1–3 and evaluates online on batches 5–10. It preserves the batch order and permutes records within each batch using a fixed seed. Every prediction is scored before learning from the corresponding batch. Deployment labels are used for evaluation only.

    python training-logs/full-scale/stream-adaptation/train.py
    python training-logs/full-scale/stream-adaptation/train.py --real

Install NumPy, PyTorch, scikit-learn, joblib and threadpoolctl. The companion expects datasets/gas-sensor-drift.zip. Reports include accuracy, macro-F1, log loss and update counts. The trainer saves frozen classifiers, adapters and batch-level prediction arrays.
