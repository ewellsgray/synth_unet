import numpy as np
import torch

from data.synthetic_dataset import (
    SyntheticTissueDataset,
    colorize_mask,
    make_synthetic_case,
)


def test_make_synthetic_case_shapes_and_dtypes():
    image, mask = make_synthetic_case(size=32, seed=1)
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8
    assert mask.shape == (32, 32)
    assert mask.dtype == np.int64


def test_make_synthetic_case_mask_values_in_range():
    _, mask = make_synthetic_case(size=32, seed=1)
    assert mask.min() >= 0
    assert mask.max() <= 6


def test_make_synthetic_case_deterministic_given_same_seed():
    image_a, mask_a = make_synthetic_case(size=32, seed=7)
    image_b, mask_b = make_synthetic_case(size=32, seed=7)
    assert np.array_equal(image_a, image_b)
    assert np.array_equal(mask_a, mask_b)


def test_make_synthetic_case_differs_across_seeds():
    _, mask_a = make_synthetic_case(size=32, seed=1)
    _, mask_b = make_synthetic_case(size=32, seed=2)
    assert not np.array_equal(mask_a, mask_b)


def test_colorize_mask_shape_and_dtype():
    _, mask = make_synthetic_case(size=32, seed=1)
    colored = colorize_mask(mask)
    assert colored.shape == (32, 32, 3)
    assert colored.dtype == np.uint8


def test_colorize_mask_is_flat_per_class():
    mask = np.zeros((4, 4), dtype=np.int64)  # all EPIDERMIS (class 0)
    colored = colorize_mask(mask)
    # every pixel should be exactly the same fixed base color for class 0
    unique_colors = np.unique(colored.reshape(-1, 3), axis=0)
    assert unique_colors.shape[0] == 1


def test_synthetic_dataset_len():
    ds = SyntheticTissueDataset(num_samples=5, patch_size=16, seed=0)
    assert len(ds) == 5


def test_synthetic_dataset_getitem_shapes_and_ranges():
    # patch_size must be large enough for _make_mask's blob-radius calc
    # (int(0.035 * size) >= 1, i.e. size >= 29) or numpy's rng.integers
    # raises ValueError("high <= 0") — 32 is the smallest clean size above
    # that floor.
    ds = SyntheticTissueDataset(num_samples=3, patch_size=32, seed=0)
    image_t, mask_t = ds[0]
    assert isinstance(image_t, torch.Tensor)
    assert isinstance(mask_t, torch.Tensor)
    assert image_t.shape == (3, 32, 32)
    assert image_t.dtype == torch.float32
    assert image_t.min() >= 0.0 and image_t.max() <= 1.0
    assert mask_t.shape == (32, 32)
    assert mask_t.dtype == torch.int64


def test_synthetic_dataset_items_are_deterministic():
    ds_a = SyntheticTissueDataset(num_samples=3, patch_size=32, seed=0)
    ds_b = SyntheticTissueDataset(num_samples=3, patch_size=32, seed=0)
    image_a, mask_a = ds_a[1]
    image_b, mask_b = ds_b[1]
    assert torch.equal(image_a, image_b)
    assert torch.equal(mask_a, mask_b)


def test_synthetic_dataset_different_indices_differ():
    ds = SyntheticTissueDataset(num_samples=3, patch_size=32, seed=0)
    _, mask_0 = ds[0]
    _, mask_1 = ds[1]
    assert not torch.equal(mask_0, mask_1)
