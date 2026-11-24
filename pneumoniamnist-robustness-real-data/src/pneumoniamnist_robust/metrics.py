from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def expected_calibration_error(probability, target, bins: int = 15) -> float:
    probability = np.asarray(probability, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=int).reshape(-1)
    confidence = np.maximum(probability, 1.0 - probability)
    prediction = (probability >= 0.5).astype(int)
    correctness = (prediction == target).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        keep = (confidence >= low) & (confidence <= high if index == bins - 1 else confidence < high)
        if keep.any():
            ece += keep.mean() * abs(correctness[keep].mean() - confidence[keep].mean())
    return float(ece)


def binary_metrics(probability, target) -> dict[str, float]:
    probability = np.asarray(probability, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=int).reshape(-1)
    prediction = (probability >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "f1": float(f1_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "ece": expected_calibration_error(probability, target),
    }

