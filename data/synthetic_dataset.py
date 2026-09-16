"""
Synthetic data generator standing in for real digitally-stained confocal mosaics.

We don't have access to real platform data, so this module procedurally
generates plausible *layered skin* images: a rough epidermis band on top, a
dermis layer below it, adipose tissue at the bottom, scattered appendageal
structures (hair follicles/sebaceous glands), random BCC/SCC tumor nest blobs,
and a background/void border — matching the 7-class scheme from config.py.

This exists purely so the training loop below has something real to run on
end-to-end. Swap this module out for `data/real_dataset.py` once real
OME-TIFF/OME-Zarr mosaics and pathologist labels are available (see that
file's docstring for the intended interface).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is an optional soft dependency here
    _HAS_SCIPY = False

from config import TISSUE_CLASSES

# Class indices, mirroring config.TISSUE_CLASSES order.
EPIDERMIS, DERMIS, BCC, SCC, ADIPOSE, APPENDAGE, BACKGROUND = range(7)

# Approximate per-class base RGB "digital stain" color (mimicking the H&E-like
# palette the platform produces: pink/eosinophilic stroma, purple/hematoxylin
# nuclei-dense tumor nests, pale adipose, etc.)
_BASE_COLOR = {
    EPIDERMIS: (219, 150, 180),    # pink-purple squamous layer
    DERMIS: (231, 190, 200),       # pale pink collagen stroma
    BCC: (120, 70, 130),           # dense basophilic (purple) tumor nests
    SCC: (150, 90, 110),           # atypical keratinocyte nests, darker pink/purple
    ADIPOSE: (245, 235, 225),      # pale, near-white fat
    APPENDAGE: (170, 110, 150),    # follicle/gland, purple-ish but distinct texture
    BACKGROUND: (10, 10, 10),      # glass-free void / imaging artifact — near black
}


def _smooth_noise(h: int, w: int, rng: np.random.Generator, scale: float = 8.0) -> np.ndarray:
    """Cheap low-frequency noise field in [0, 1] for per-class texture variation."""
    low = rng.random((max(2, h // int(scale)), max(2, w // int(scale))))
    if _HAS_SCIPY:
        # Upsample via zoom-like repeat, then blur for smoothness.
        reps_h = h // low.shape[0] + 1
        reps_w = w // low.shape[1] + 1
        big = np.kron(low, np.ones((reps_h, reps_w)))[:h, :w]
        big = gaussian_filter(big, sigma=scale / 2)
    else:
        reps_h = h // low.shape[0] + 1
        reps_w = w // low.shape[1] + 1
        big = np.kron(low, np.ones((reps_h, reps_w)))[:h, :w]
    big -= big.min()
    denom = big.max() - big.min() if big.max() > big.min() else 1.0
    return big / denom


def _make_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Procedurally build a plausible layered-skin label mask (H, W) of class ids."""
    mask = np.full((h, w), DERMIS, dtype=np.int64)

    # Border "void" region — imaging artifact / glass-free boundary.
    border = max(4, int(0.02 * min(h, w)))
    mask[:border, :] = BACKGROUND
    mask[-border:, :] = BACKGROUND
    mask[:, :border] = BACKGROUND
    mask[:, -border:] = BACKGROUND

    # Wavy epidermis band along the top ~15-25% of the tissue region.
    epi_frac = rng.uniform(0.15, 0.25)
    epi_height = int(epi_frac * h)
    wave = (np.sin(np.linspace(0, rng.uniform(2, 5) * np.pi, w)) * h * 0.02).astype(int)
    for x in range(w):
        top = border
        bottom = min(h - border, border + epi_height + wave[x])
        mask[top:bottom, x] = EPIDERMIS

    # Adipose band along the bottom ~20-30%.
    adi_frac = rng.uniform(0.20, 0.30)
    adi_height = int(adi_frac * h)
    wave2 = (np.sin(np.linspace(0, rng.uniform(2, 5) * np.pi, w) + 1.0) * h * 0.02).astype(int)
    for x in range(w):
        bottom = h - border
        top = max(border, h - border - adi_height + wave2[x])
        mask[top:bottom, x] = ADIPOSE

    # Scattered appendageal structures (small circular blobs) in the dermis.
    n_append = rng.integers(2, 6)
    for _ in range(n_append):
        cy = rng.integers(int(0.3 * h), int(0.75 * h))
        cx = rng.integers(border, w - border)
        r = rng.integers(int(0.015 * min(h, w)), int(0.035 * min(h, w)))
        yy, xx = np.ogrid[:h, :w]
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        mask[blob & (mask == DERMIS)] = APPENDAGE

    # Random tumor nests (BCC and/or SCC) — irregular blobs, sometimes near the
    # epidermis (invasive pattern), sometimes deeper.
    n_tumor = rng.integers(1, 4)
    for _ in range(n_tumor):
        tumor_class = BCC if rng.random() < 0.6 else SCC
        cy = rng.integers(int(0.15 * h), int(0.85 * h))
        cx = rng.integers(border, w - border)
        r = rng.integers(int(0.03 * min(h, w)), int(0.08 * min(h, w)))
        yy, xx = np.ogrid[:h, :w]
        # Irregular blob: circle perturbed by low-freq noise threshold.
        noise = _smooth_noise(h, w, rng, scale=6.0)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        blob = (dist <= r * (0.7 + 0.6 * noise))
        eligible = blob & (mask != BACKGROUND)
        mask[eligible] = tumor_class

    return mask


def _colorize(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn a label mask into an RGB image mimicking the digital H&E stain."""
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    texture = _smooth_noise(h, w, rng, scale=4.0)
    for cls_id, color in _BASE_COLOR.items():
        region = mask == cls_id
        if not region.any():
            continue
        color_arr = np.array(color, dtype=np.float32)
        variation = (texture[region][:, None] - 0.5) * 30.0  # +/- texture shading
        noise = rng.normal(0, 8.0, size=(region.sum(), 3))
        img[region] = np.clip(color_arr[None, :] + variation + noise, 0, 255)
    return img.astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Flat (no texture/noise) palette rendering of a label mask, for human-readable
    visualization of exported samples. image: (H, W, 3) uint8."""
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in _BASE_COLOR.items():
        img[mask == cls_id] = color
    return img


def make_synthetic_case(
    size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one (image, mask) pair. image: (H, W, 3) uint8, mask: (H, W) int64."""
    rng = np.random.default_rng(seed)
    mask = _make_mask(size, size, rng)
    image = _colorize(mask, rng)
    return image, mask


class SyntheticTissueDataset(Dataset):
    """
    Generates synthetic (image, mask) patches on the fly.

    Each item simulates one training patch already cropped from a larger
    mosaic (see data/real_dataset.py for how real patches would be extracted
    from an actual large confocal mosaic via sliding-window tiling).
    """

    def __init__(self, num_samples: int, patch_size: int, seed: int = 0):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        image, mask = make_synthetic_case(self.patch_size, seed=self.seed * 100_003 + idx)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (3, H, W) in [0,1]
        mask_t = torch.from_numpy(mask).long()  # (H, W)
        return image_t, mask_t


if __name__ == "__main__":
    # Quick visual sanity check — run with `python -m data.synthetic_dataset`
    img, msk = make_synthetic_case(256, seed=1)
    print("Image shape:", img.shape, "dtype:", img.dtype)
    print("Mask shape:", msk.shape, "unique classes present:",
          sorted(set(msk.flatten().tolist())))
    print("Classes:", TISSUE_CLASSES)
