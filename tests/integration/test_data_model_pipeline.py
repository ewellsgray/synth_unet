import torch
from torch.utils.data import DataLoader

from data.synthetic_dataset import SyntheticTissueDataset
from model.losses import WeightedCEDiceLoss
from model.unet import TissueSegUNet


def test_dataset_dataloader_model_loss_end_to_end():
    # patch_size=32 is the smallest clean size that stays above the
    # synthetic mask generator's minimum blob-radius floor (>=29px).
    dataset = SyntheticTissueDataset(num_samples=4, patch_size=32, seed=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = TissueSegUNet(in_channels=3, num_classes=7, base_features=4, depth=2)
    criterion = WeightedCEDiceLoss(num_classes=7, use_dice=True)

    images, masks = next(iter(loader))
    assert images.shape == (2, 3, 32, 32)
    assert masks.shape == (2, 32, 32)

    logits = model(images)
    assert logits.shape == (2, 7, 32, 32)

    loss = criterion(logits, masks)
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)
