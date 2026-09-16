"""
Loss function for the imbalanced 7-class segmentation problem.

Combines:
  - Class-weighted cross-entropy (weights derived from inverse pixel
    frequency in the training set, so rare classes like BCC/SCC contribute
    proportionally more to the gradient).
  - Soft Dice loss (computed on softmax probabilities), which is known to
    help segmentation models directly optimize for region overlap rather
    than just per-pixel classification, and tends to be more robust to
    severe class imbalance than cross-entropy alone.

This combination is a common, well-tested default for imbalanced medical
image segmentation (rather than picking one over the other).
"""

import torch
import torch.nn.functional as F
from torch import nn


def soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6
) -> torch.Tensor:
    """logits: (B, C, H, W); targets: (B, H, W) long."""
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    denom = torch.sum(probs + targets_onehot, dims)
    dice_per_class = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice_per_class.mean()


class WeightedCEDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        dice_weight: float = 0.5,
        use_dice: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight if use_dice else 0.0
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        if self.dice_weight > 0:
            d_loss = soft_dice_loss(logits, targets, self.num_classes)
            return (1 - self.dice_weight) * ce_loss + self.dice_weight * d_loss
        return ce_loss


@torch.no_grad()
def compute_inverse_frequency_weights(
    dataset, num_classes: int, sample_batches: int = 8
) -> torch.Tensor:
    """
    Estimate class weights from observed pixel frequency across a handful of
    sampled items (not the whole dataset - this is meant to be a cheap
    approximation, same as you'd do with a real, much larger dataset where
    scanning every pixel of every mosaic up front would be too slow).
    """
    counts = torch.zeros(num_classes)
    n = min(sample_batches, len(dataset))
    for i in range(n):
        _, mask = dataset[i]
        for c in range(num_classes):
            counts[c] += (mask == c).sum().item()
    counts = torch.clamp(counts, min=1.0)
    freq = counts / counts.sum()
    weights = 1.0 / freq
    weights = weights / weights.sum() * num_classes  # normalize to mean ~1
    return weights
