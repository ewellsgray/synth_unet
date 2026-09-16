import os

from PIL import Image

from scripts.export_synthetic_samples import export_samples


def test_export_samples_writes_expected_files(tmp_path):
    out_dir = str(tmp_path / "samples")
    export_samples(count=3, out_dir=out_dir, patch_size=32, seed=1)

    for subdir in ("images", "masks", "masks_color"):
        d = os.path.join(out_dir, subdir)
        assert os.path.isdir(d)
        files = sorted(os.listdir(d))
        assert files == [f"sample_{i:04d}.png" for i in range(3)]


def test_exported_image_and_mask_have_expected_shape_and_mode(tmp_path):
    out_dir = str(tmp_path / "samples")
    export_samples(count=1, out_dir=out_dir, patch_size=32, seed=1)

    image = Image.open(os.path.join(out_dir, "images", "sample_0000.png"))
    assert image.size == (32, 32)
    assert image.mode == "RGB"

    mask = Image.open(os.path.join(out_dir, "masks", "sample_0000.png"))
    assert mask.size == (32, 32)
    assert mask.mode == "L"

    mask_color = Image.open(os.path.join(out_dir, "masks_color", "sample_0000.png"))
    assert mask_color.size == (32, 32)
    assert mask_color.mode == "RGB"


def test_export_samples_matches_dataset_seed_derivation(tmp_path):
    import numpy as np

    from data.synthetic_dataset import make_synthetic_case

    out_dir = str(tmp_path / "samples")
    export_samples(count=1, out_dir=out_dir, patch_size=32, seed=5)

    expected_image, _ = make_synthetic_case(32, seed=5 * 100_003 + 0)
    actual_image = np.array(Image.open(os.path.join(out_dir, "images", "sample_0000.png")))
    assert np.array_equal(expected_image, actual_image)
