import random

import torch

from data.transforms import JointAugment


def _sample_image_mask():
    image = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4) / 32.0
    mask = torch.arange(4 * 4, dtype=torch.int64).reshape(4, 4)
    return image, mask


def test_no_op_when_all_probabilities_zero():
    image, mask = _sample_image_mask()
    aug = JointAugment(flip_prob=0.0, rotate_prob=0.0, color_jitter_prob=0.0)
    out_image, out_mask = aug(image, mask)
    assert torch.equal(out_image, image)
    assert torch.equal(out_mask, mask)


def test_both_flips_applied_in_lockstep():
    # flip_prob=1.0 makes both the horizontal and vertical flip checks fire.
    image, mask = _sample_image_mask()
    aug = JointAugment(flip_prob=1.0, rotate_prob=0.0, color_jitter_prob=0.0)
    out_image, out_mask = aug(image, mask)
    expected_image = torch.flip(torch.flip(image, dims=[2]), dims=[1])
    expected_mask = torch.flip(torch.flip(mask, dims=[1]), dims=[0])
    assert torch.equal(out_image, expected_image)
    assert torch.equal(out_mask, expected_mask)


def test_rotate_applied_in_lockstep():
    image, mask = _sample_image_mask()
    aug = JointAugment(flip_prob=0.0, rotate_prob=1.0, color_jitter_prob=0.0)
    random.seed(0)
    k = random.choice([1, 2, 3])
    random.seed(0)
    out_image, out_mask = aug(image, mask)
    expected_image = torch.rot90(image, k=k, dims=[1, 2])
    expected_mask = torch.rot90(mask, k=k, dims=[0, 1])
    assert torch.equal(out_image, expected_image)
    assert torch.equal(out_mask, expected_mask)


def test_color_jitter_only_affects_image_not_mask():
    image, mask = _sample_image_mask()
    aug = JointAugment(flip_prob=0.0, rotate_prob=0.0, color_jitter_prob=1.0,
                        brightness=0.15, contrast=0.15)
    out_image, out_mask = aug(image, mask)
    assert torch.equal(out_mask, mask)
    assert out_image.shape == image.shape
    assert out_image.min() >= 0.0 and out_image.max() <= 1.0


def test_output_shapes_preserved():
    image, mask = _sample_image_mask()
    aug = JointAugment()
    random.seed(42)
    out_image, out_mask = aug(image, mask)
    assert out_image.shape == image.shape
    assert out_mask.shape == mask.shape
