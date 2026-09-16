"""
Overfit test: mathematical sanity, not just "does it run".

A model+loss+optimizer wired correctly should be able to drive the loss on a
tiny FIXED batch down sharply within a few hundred steps. If this fails while
tests/smoke passes, the bug is almost certainly a frozen parameter, a loss
reading the wrong tensor, gradients not flowing, or an optimizer built on the
wrong parameter set -- code that runs cleanly but never actually learns.
"""
import pytest
import torch

from data.synthetic_dataset import SyntheticTissueDataset
from model.losses import WeightedCEDiceLoss
from model.unet import TissueSegUNet

# Production uses lr=1e-4, tuned for many epochs over a large dataset. Here we
# want fast convergence on 2 fixed samples, so a higher LR and more steps
# than production are intentional and specific to this test.
_STEPS = 250
_LR = 5e-3
_MAX_LOSS_RATIO = 0.25  # final loss must be <= 25% of the initial loss


@pytest.mark.overfit
def test_model_overfits_a_fixed_tiny_batch():
    torch.manual_seed(0)

    dataset = SyntheticTissueDataset(num_samples=2, patch_size=32, seed=1)
    images = torch.stack([dataset[i][0] for i in range(2)])
    masks = torch.stack([dataset[i][1] for i in range(2)])

    model = TissueSegUNet(in_channels=3, num_classes=7, base_features=16, depth=2)
    criterion = WeightedCEDiceLoss(num_classes=7, use_dice=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LR)

    model.train()
    losses = []
    for step in range(_STEPS):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        assert torch.isfinite(loss), f"loss diverged to non-finite at step {step}: {loss.item()}"
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    initial_loss, final_loss = losses[0], losses[-1]
    assert final_loss <= _MAX_LOSS_RATIO * initial_loss, (
        f"model failed to overfit a fixed 2-sample batch: "
        f"initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f} "
        f"(needed <= {_MAX_LOSS_RATIO * initial_loss:.4f})"
    )
