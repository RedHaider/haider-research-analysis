from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


def expected_calibration_error(probability, target, bins=10):
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=int)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        keep = (probability >= low) & (probability <= high if index == bins - 1 else probability < high)
        if keep.any():
            value += keep.mean() * abs(target[keep].mean() - probability[keep].mean())
    return float(value)


def binary_metrics(probability, target):
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=int)
    prediction = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "ece": expected_calibration_error(probability, target),
    }

