import torch

from model.losses import (
    WeightedCEDiceLoss,
    compute_inverse_frequency_weights,
    soft_dice_loss,
)


def _one_hot_logits(targets: torch.Tensor, num_classes: int, confidence: float = 20.0):
    """Build logits that strongly predict `targets` (near-perfect classifier)."""
    b, h, w = targets.shape
    logits = torch.full((b, num_classes, h, w), -confidence)
    logits.scatter_(1, targets.unsqueeze(1), confidence)
    return logits


def test_soft_dice_loss_near_zero_for_perfect_prediction():
    targets = torch.randint(0, 4, (2, 8, 8))
    logits = _one_hot_logits(targets, num_classes=4)
    loss = soft_dice_loss(logits, targets, num_classes=4)
    assert loss.item() < 0.05


def test_soft_dice_loss_near_one_for_disjoint_prediction():
    # num_classes=2 so both classes are actually present in targets/wrong;
    # with more unused classes, epsilon smoothing gives those a trivial
    # dice of 1 and dilutes the mean, so keep this to exactly the classes
    # in play.
    targets = torch.zeros(2, 8, 8, dtype=torch.long)
    wrong = torch.ones(2, 8, 8, dtype=torch.long)
    logits = _one_hot_logits(wrong, num_classes=2)
    loss = soft_dice_loss(logits, targets, num_classes=2)
    assert loss.item() > 0.9


def test_weighted_ce_dice_loss_is_scalar_and_differentiable():
    targets = torch.randint(0, 4, (2, 8, 8))
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    criterion = WeightedCEDiceLoss(num_classes=4, use_dice=True)
    loss = criterion(logits, targets)
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_use_dice_false_matches_plain_weighted_ce():
    targets = torch.randint(0, 4, (2, 8, 8))
    logits = torch.randn(2, 4, 8, 8)
    weights = torch.tensor([1.0, 2.0, 0.5, 1.0])

    criterion = WeightedCEDiceLoss(num_classes=4, class_weights=weights, use_dice=False)
    loss = criterion(logits, targets)

    expected = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
    assert torch.isclose(loss, expected)


def test_compute_inverse_frequency_weights_favors_rare_class():
    class _FixedDataset:
        def __init__(self, masks):
            self.masks = masks

        def __len__(self):
            return len(self.masks)

        def __getitem__(self, idx):
            return None, self.masks[idx]

    # class 0 dominates every sample, class 1 is rare -> class 1 should get a
    # higher inverse-frequency weight.
    mask = torch.zeros(4, 4, dtype=torch.long)
    mask[0, 0] = 1
    dataset = _FixedDataset([mask, mask, mask])

    weights = compute_inverse_frequency_weights(dataset, num_classes=2, sample_batches=3)
    assert weights[1] > weights[0]
