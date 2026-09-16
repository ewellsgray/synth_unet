"""
Smoke tests: does the training machinery run at all, for a handful of steps,
with no exceptions and no NaN/Inf losses?

Deliberately NOT going through train() (see tests/e2e for that — it exercises
the real entry point including checkpoint/log file I/O). This wires the
pieces together directly and skips all file I/O, so it stays fast enough to
run on every commit.
"""
import pytest
import torch
from torch.utils.data import DataLoader

from data.synthetic_dataset import SyntheticTissueDataset
from model.losses import WeightedCEDiceLoss
from model.unet import TissueSegUNet


@pytest.mark.smoke
def test_a_few_training_steps_run_without_exception():
    dataset = SyntheticTissueDataset(num_samples=8, patch_size=32, seed=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    model = TissueSegUNet(in_channels=3, num_classes=7, base_features=8, depth=2)
    criterion = WeightedCEDiceLoss(num_classes=7, use_dice=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    steps_run = 0
    for images, masks in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)

        assert torch.isfinite(loss), f"non-finite loss at step {steps_run}: {loss.item()}"

        loss.backward()
        optimizer.step()

        steps_run += 1
        if steps_run >= 4:
            break

    assert steps_run == 4

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)
