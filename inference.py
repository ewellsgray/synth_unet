"""
Sliding-window inference: run the patch-trained model over a full-size image
and reassemble 7 full-resolution grayscale probability maps.

This matters because the model is trained on small patches (patch_size, e.g.
256x256) but real mosaics are much larger, so at inference time we tile the
full image, run each tile through the model, and stitch the per-class
probability maps back together — averaging overlapping regions to avoid
seam artifacts at patch boundaries.

Run with:
    python inference.py
(uses a freshly-initialized model + a synthetic full-size mosaic by default,
purely to demonstrate the tiling/stitching logic end to end.)
"""
import numpy as np
import torch
import torch.nn.functional as F

from config import TrainConfig, TISSUE_CLASSES
from data.synthetic_dataset import make_synthetic_case
from model.unet import TissueSegUNet
from utils.device import get_device


@torch.no_grad()
def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,          # (C, H, W) float in [0,1]
    patch_size: int,
    num_classes: int,
    device: torch.device,
    overlap: float = 0.25,
) -> torch.Tensor:
    """Returns per-class probability maps of shape (num_classes, H, W)."""
    model.eval()
    c, h, w = image.shape
    stride = max(1, int(patch_size * (1 - overlap)))

    prob_sum = torch.zeros(num_classes, h, w)
    weight_sum = torch.zeros(1, h, w)

    ys = list(range(0, max(h - patch_size, 0) + 1, stride))
    xs = list(range(0, max(w - patch_size, 0) + 1, stride))
    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)

    for y in ys:
        for x in xs:
            patch = image[:, y:y + patch_size, x:x + patch_size].unsqueeze(0).to(device)
            logits = model(patch)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()  # (num_classes, ph, pw)
            prob_sum[:, y:y + patch_size, x:x + patch_size] += probs
            weight_sum[:, y:y + patch_size, x:x + patch_size] += 1.0

    weight_sum = torch.clamp(weight_sum, min=1e-6)
    return prob_sum / weight_sum


def probability_maps_to_uint8(prob_maps: torch.Tensor) -> np.ndarray:
    """(num_classes, H, W) in [0,1] -> (num_classes, H, W) uint8 grayscale images."""
    return (prob_maps.clamp(0, 1).numpy() * 255).astype(np.uint8)


if __name__ == "__main__":
    cfg = TrainConfig()
    device = get_device(cfg.device_preference)

    model = TissueSegUNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_features=cfg.base_features,
        depth=cfg.depth,
    ).to(device)
    # NOTE: untrained weights here purely to exercise the tiling/stitching
    # pipeline. In real use, load a trained checkpoint first, e.g.:
    #   from utils.checkpoint import load_checkpoint
    #   load_checkpoint("checkpoints/best.pt", model)

    image_np, _ = make_synthetic_case(cfg.full_image_size, seed=123)
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

    prob_maps = sliding_window_inference(
        model, image, patch_size=cfg.patch_size, num_classes=cfg.num_classes, device=device
    )
    maps_uint8 = probability_maps_to_uint8(prob_maps)

    print(f"Full image size: {tuple(image.shape)}")
    print(f"Stitched probability maps shape: {tuple(prob_maps.shape)}")
    for i, name in enumerate(TISSUE_CLASSES):
        print(f"  class {i} ({name}): mean prob={prob_maps[i].mean():.3f}, "
              f"max prob={prob_maps[i].max():.3f}")
