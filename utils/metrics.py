"""
Metrics for evaluating the tissue segmentation model.

`balanced_accuracy` mirrors the metric the target platform actually reports (per the
interview notes): it's the average of per-class recall, which prevents the
dominant classes (dermis, adipose, background) from masking poor performance
on the rare-but-critical tumor classes (BCC/SCC).
"""
from typing import Dict, List

import torch


@torch.no_grad()
def confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    preds, targets: any shape, integer class ids (e.g. flattened pixel predictions).
    Returns an (num_classes, num_classes) matrix where rows = true class,
    columns = predicted class.
    """
    preds = preds.flatten()
    targets = targets.flatten()
    idx = targets * num_classes + preds
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


@torch.no_grad()
def balanced_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> float:
    """Mean per-class recall (a.k.a. balanced accuracy), pixel-level."""
    cm = confusion_matrix(preds, targets, num_classes).float()
    per_class_total = cm.sum(dim=1)
    per_class_correct = cm.diag()
    # Avoid div-by-zero for classes absent from this batch — exclude them
    # from the mean rather than counting them as 0 recall.
    present = per_class_total > 0
    recall = torch.zeros(num_classes)
    recall[present] = per_class_correct[present] / per_class_total[present]
    return recall[present].mean().item() if present.any() else 0.0


@torch.no_grad()
def per_class_dice(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6
) -> torch.Tensor:
    """Soft-free (hard, argmax-based) Dice score per class, for reporting."""
    dice = torch.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        dice[c] = (2 * intersection + eps) / (denom + eps)
    return dice


def format_class_metrics(values: torch.Tensor, class_names: List[str]) -> str:
    return ", ".join(f"{name}={v:.3f}" for name, v in zip(class_names, values.tolist()))
