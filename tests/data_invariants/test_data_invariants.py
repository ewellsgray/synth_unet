"""
Data invariants: pure checks on the synthetic data pipeline's output ranges,
shapes, and dtypes -- no model involved. Guards against a data-generation or
augmentation regression silently corrupting what the model trains on.
"""
import random

import pytest
import torch

from config import TISSUE_CLASSES
from data.synthetic_dataset import SyntheticTissueDataset, make_synthetic_case
from data.transforms import JointAugment

NUM_CLASSES = len(TISSUE_CLASSES)


@pytest.mark.data_invariant
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("size", [32, 48])
def test_make_synthetic_case_invariants(seed, size):
    image, mask = make_synthetic_case(size=size, seed=seed)

    assert image.shape == (size, size, 3)
    assert mask.shape == (size, size)
    assert image.dtype.name == "uint8"
    assert int(mask.min()) >= 0
    assert int(mask.max()) < NUM_CLASSES


@pytest.mark.data_invariant
def test_synthetic_dataset_getitem_invariants():
    ds = SyntheticTissueDataset(num_samples=6, patch_size=32, seed=3)
    for i in range(len(ds)):
        image_t, mask_t = ds[i]

        assert image_t.dtype == torch.float32
        assert torch.isfinite(image_t).all()
        assert image_t.min() >= 0.0 and image_t.max() <= 1.0

        assert mask_t.dtype == torch.int64
        assert torch.isfinite(mask_t.float()).all()
        assert int(mask_t.min()) >= 0
        assert int(mask_t.max()) < NUM_CLASSES


@pytest.mark.data_invariant
def test_joint_augment_preserves_invariants_across_random_states():
    ds = SyntheticTissueDataset(num_samples=1, patch_size=32, seed=4)
    image, mask = ds[0]
    aug = JointAugment()

    for trial_seed in range(20):
        random.seed(trial_seed)
        out_image, out_mask = aug(image, mask)

        assert out_image.shape == image.shape
        assert out_mask.shape == mask.shape

        assert torch.isfinite(out_image).all()
        assert out_image.min() >= 0.0 and out_image.max() <= 1.0

        assert torch.isfinite(out_mask.float()).all()
        assert int(out_mask.min()) >= 0
        assert int(out_mask.max()) < NUM_CLASSES
