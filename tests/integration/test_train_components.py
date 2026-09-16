import torch

from data.transforms import JointAugment
from model.unet import TissueSegUNet
from train import build_dataloaders, run_validation


def test_build_dataloaders_matches_config_sizes(tiny_cfg):
    train_loader, val_loader, train_ds_for_weights = build_dataloaders(tiny_cfg, JointAugment())

    assert len(train_loader.dataset) == tiny_cfg.synthetic_train_size
    assert len(val_loader.dataset) == tiny_cfg.synthetic_val_size
    assert len(train_ds_for_weights) == tiny_cfg.synthetic_train_size

    images, masks = next(iter(train_loader))
    assert images.shape == (tiny_cfg.batch_size, tiny_cfg.in_channels,
                             tiny_cfg.patch_size, tiny_cfg.patch_size)
    assert masks.shape == (tiny_cfg.batch_size, tiny_cfg.patch_size, tiny_cfg.patch_size)


def test_run_validation_returns_valid_metrics(tiny_cfg):
    _, val_loader, _ = build_dataloaders(tiny_cfg, JointAugment())
    model = TissueSegUNet(
        in_channels=tiny_cfg.in_channels,
        num_classes=tiny_cfg.num_classes,
        base_features=4,
        depth=2,
    )
    device = torch.device("cpu")

    bal_acc, dice = run_validation(model, val_loader, device, tiny_cfg.num_classes)

    assert isinstance(bal_acc, float)
    assert 0.0 <= bal_acc <= 1.0
    assert dice.shape == (tiny_cfg.num_classes,)
    assert torch.all(dice >= 0.0) and torch.all(dice <= 1.0)
