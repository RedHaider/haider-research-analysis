from __future__ import annotations

import numpy as np
import torch


def soft_dice_loss(logits, target, eps=1e-6):
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * intersection + eps) / (denominator + eps)).mean()


def segmentation_metrics(logits, target, threshold=.5):
    probability = logits.sigmoid(); prediction = probability >= threshold; truth = target >= .5
    axes = (1, 2, 3); tp = (prediction & truth).sum(axes).float(); fp = (prediction & ~truth).sum(axes).float()
    fn = (~prediction & truth).sum(axes).float(); eps = 1e-6
    dice = (2*tp+eps)/(2*tp+fp+fn+eps); iou=(tp+eps)/(tp+fp+fn+eps)
    precision=(tp+eps)/(tp+fp+eps); recall=(tp+eps)/(tp+fn+eps)
    confidence = torch.maximum(probability, 1-probability).detach().cpu().numpy().ravel()
    correctness = (prediction == truth).detach().cpu().numpy().ravel().astype(float)
    bins=np.linspace(.5,1,11); ece=0.0
    for low,high in zip(bins[:-1],bins[1:]):
        keep=(confidence>=low)&(confidence<high)
        if keep.any(): ece += keep.mean()*abs(correctness[keep].mean()-confidence[keep].mean())
    return {"dice":float(dice.mean()),"iou":float(iou.mean()),"precision":float(precision.mean()),
            "recall":float(recall.mean()),"ece":float(ece)}
