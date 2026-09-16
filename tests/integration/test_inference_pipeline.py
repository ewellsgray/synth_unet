import numpy as np
import torch

from data.synthetic_dataset import make_synthetic_case
from inference import probability_maps_to_uint8, sliding_window_inference
from model.unet import TissueSegUNet


def test_sliding_window_inference_shape_and_probability_sum():
    image_np, _ = make_synthetic_case(size=48, seed=1)
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

    model = TissueSegUNet(in_channels=3, num_classes=7, base_features=4, depth=2)
    device = torch.device("cpu")

    prob_maps = sliding_window_inference(
        model, image, patch_size=16, num_classes=7, device=device, overlap=0.25
    )

    assert prob_maps.shape == (7, 48, 48)
    # overlap-averaged softmax outputs should still sum to ~1 across classes
    # at each pixel (weighted average of simplex points is itself in the simplex).
    per_pixel_sum = prob_maps.sum(dim=0)
    assert torch.allclose(per_pixel_sum, torch.ones_like(per_pixel_sum), atol=1e-4)


def test_probability_maps_to_uint8_range():
    prob_maps = torch.rand(7, 8, 8)
    out = probability_maps_to_uint8(prob_maps)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    assert out.shape == (7, 8, 8)
