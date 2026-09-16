import os

import pytest
import torch

from data.synthetic_dataset import make_synthetic_case
from inference import sliding_window_inference
from model.unet import TissueSegUNet
from train import train
from utils.checkpoint import load_checkpoint


@pytest.mark.slow
def test_train_checkpoint_inference_pipeline(tiny_cfg):
    train(tiny_cfg)

    best_ckpt = os.path.join(tiny_cfg.checkpoint_dir, "best.pt")
    assert os.path.exists(best_ckpt)

    loaded_model = TissueSegUNet(
        in_channels=tiny_cfg.in_channels,
        num_classes=tiny_cfg.num_classes,
        base_features=tiny_cfg.base_features,
        depth=tiny_cfg.depth,
    )
    load_checkpoint(best_ckpt, loaded_model, map_location="cpu")

    image_np, _ = make_synthetic_case(size=tiny_cfg.full_image_size, seed=999)
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

    prob_maps = sliding_window_inference(
        loaded_model, image, patch_size=tiny_cfg.patch_size,
        num_classes=tiny_cfg.num_classes, device=torch.device("cpu"),
    )

    assert prob_maps.shape == (tiny_cfg.num_classes, tiny_cfg.full_image_size, tiny_cfg.full_image_size)
    assert torch.isfinite(prob_maps).all()
    assert prob_maps.min() >= 0.0 and prob_maps.max() <= 1.0 + 1e-4
