from __future__ import annotations

import numpy as np
import torch


def soft_dice_loss(logits, target, eps=1e-6):
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * intersection + eps) / (denominator + eps)).mean()


def segmentation_statistics(logits, target, threshold=.5):
    probability = logits.sigmoid(); prediction = probability >= threshold; truth = target >= .5
    return {
        "tp": int((prediction & truth).sum()),
        "fp": int((prediction & ~truth).sum()),
        "fn": int((~prediction & truth).sum()),
        "confidence": torch.maximum(probability, 1-probability).detach().cpu().numpy().ravel(),
        "correctness": (prediction == truth).detach().cpu().numpy().ravel().astype(float),
    }


def metrics_from_statistics(tp, fp, fn, confidence, correctness):
    eps = 1e-6
    dice = (2*tp+eps)/(2*tp+fp+fn+eps); iou=(tp+eps)/(tp+fp+fn+eps)
    precision=(tp+eps)/(tp+fp+eps); recall=(tp+eps)/(tp+fn+eps)
    bins=np.linspace(.5,1,11); ece=0.0
    for low,high in zip(bins[:-1],bins[1:]):
        keep=(confidence>=low)&(confidence<high if high < 1 else confidence<=high)
        if keep.any(): ece += keep.mean()*abs(correctness[keep].mean()-confidence[keep].mean())
    return {"dice":float(dice),"iou":float(iou),"precision":float(precision),
            "recall":float(recall),"ece":float(ece)}


def segmentation_metrics(logits, target, threshold=.5):
    stats = segmentation_statistics(logits, target, threshold)
    return metrics_from_statistics(**stats)

