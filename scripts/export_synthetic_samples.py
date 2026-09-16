"""
Export synthetic tissue samples to disk as viewable PNG files.

The synthetic dataset (data/synthetic_dataset.py) normally generates
(image, mask) pairs on the fly, in memory, and never writes them anywhere.
This script exists purely to let you inspect what the generator produces.

Run with:
    python scripts/export_synthetic_samples.py --count 20
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TrainConfig
from data.synthetic_dataset import make_synthetic_case, colorize_mask


def export_samples(count: int, out_dir: str, patch_size: int, seed: int):
    images_dir = os.path.join(out_dir, "images")
    masks_dir = os.path.join(out_dir, "masks")
    masks_color_dir = os.path.join(out_dir, "masks_color")
    for d in (images_dir, masks_dir, masks_color_dir):
        os.makedirs(d, exist_ok=True)

    for idx in range(count):
        # Same seed derivation as SyntheticTissueDataset.__getitem__, so
        # sample `idx` here matches what training would sample at that index.
        image, mask = make_synthetic_case(patch_size, seed=seed * 100_003 + idx)
        mask_color = colorize_mask(mask)

        name = f"sample_{idx:04d}.png"
        Image.fromarray(image).save(os.path.join(images_dir, name))
        Image.fromarray(mask.astype(np.uint8), mode="L").save(os.path.join(masks_dir, name))
        Image.fromarray(mask_color).save(os.path.join(masks_color_dir, name))

    print(f"[done] wrote {count} samples to {out_dir}/ (images/, masks/, masks_color/)")


def main():
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20, help="number of samples to export")
    parser.add_argument("--out", type=str, default="data/synthetic_samples", help="output directory")
    parser.add_argument("--patch-size", type=int, default=cfg.patch_size, help="side length of each sample")
    parser.add_argument("--seed", type=int, default=cfg.seed, help="base seed")
    args = parser.parse_args()

    export_samples(args.count, args.out, args.patch_size, args.seed)


if __name__ == "__main__":
    main()
