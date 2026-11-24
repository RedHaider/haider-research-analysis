from __future__ import annotations

import numpy as np


def retrieval_metrics(similarity):
    similarity = np.asarray(similarity)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Similarity matrix must be square")
    order = np.argsort(-similarity, axis=1)
    targets = np.arange(len(similarity))[:, None]
    ranks = np.argmax(order == targets, axis=1) + 1
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }

