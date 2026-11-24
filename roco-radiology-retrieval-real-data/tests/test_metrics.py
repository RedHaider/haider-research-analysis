import numpy as np
import pytest

from roco_retrieval.metrics import retrieval_metrics


def test_identity_similarity_is_perfect():
    result = retrieval_metrics(np.eye(8))
    assert result["recall_at_1"] == 1.0
    assert result["mrr"] == 1.0


def test_metric_ranges():
    result = retrieval_metrics(np.arange(25).reshape(5, 5))
    assert all(0 <= result[key] <= 1 for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"))


def test_non_square_input_is_rejected():
    with pytest.raises(ValueError):
        retrieval_metrics(np.zeros((3, 4)))

