import pytest
import torch

from model.unet import TissueSegUNet, count_parameters


@pytest.mark.parametrize("patch_size", [16, 32])
def test_forward_output_shape(patch_size):
    model = TissueSegUNet(in_channels=3, num_classes=7, base_features=4, depth=2)
    x = torch.randn(2, 3, patch_size, patch_size)
    out = model(x)
    assert out.shape == (2, 7, patch_size, patch_size)


def test_forward_handles_non_power_of_two_size():
    # depth=2 downsamples by 4x; 33 is not evenly divisible, exercising the
    # padding logic in Up.forward.
    model = TissueSegUNet(in_channels=3, num_classes=7, base_features=4, depth=2)
    x = torch.randn(1, 3, 33, 33)
    out = model(x)
    assert out.shape == (1, 7, 33, 33)


def test_count_parameters_positive_and_matches_manual_sum():
    model = TissueSegUNet(in_channels=3, num_classes=7, base_features=4, depth=1)
    n = count_parameters(model)
    manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n > 0
    assert n == manual


def test_different_num_classes_changes_output_channels():
    model = TissueSegUNet(in_channels=3, num_classes=3, base_features=4, depth=1)
    x = torch.randn(1, 3, 16, 16)
    out = model(x)
    assert out.shape == (1, 3, 16, 16)
