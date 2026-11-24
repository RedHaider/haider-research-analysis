import torch
from kvasir_robust.metrics import segmentation_metrics


def test_perfect_nonempty_mask_scores_one():
    target = torch.tensor([[[[1., 0.], [1., 0.]]]])
    logits = torch.where(target.bool(), torch.tensor(20.), torch.tensor(-20.))
    metrics = segmentation_metrics(logits, target)
    assert metrics["dice"] > .999999
    assert metrics["iou"] > .999999


def test_all_background_prediction_is_not_perfect_for_nonempty_mask():
    target = torch.tensor([[[[1., 0.], [1., 0.]]]])
    logits = torch.full_like(target, -20.)
    metrics = segmentation_metrics(logits, target)
    assert metrics["dice"] < 1e-5
    assert metrics["recall"] < 1e-5

