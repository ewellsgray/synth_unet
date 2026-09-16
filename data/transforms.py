"""
Lightweight augmentations applied jointly to (image, mask) pairs.

Kept dependency-free (pure torch) since a mask has to be transformed exactly
in lockstep with its image (no interpolation on the mask - it's categorical
labels, not continuous pixel values).
"""
import random
from typing import Tuple

import torch


class JointAugment:
    """Random flips/rotations (safe for masks) + light color jitter (image only)."""

    def __init__(self, flip_prob: float = 0.5, rotate_prob: float = 0.5,
                 color_jitter_prob: float = 0.5, brightness: float = 0.15,
                 contrast: float = 0.15):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # image: (C, H, W) float in [0,1], mask: (H, W) long

        if random.random() < self.flip_prob:
            image = torch.flip(image, dims=[2])  # horizontal
            mask = torch.flip(mask, dims=[1])

        if random.random() < self.flip_prob:
            image = torch.flip(image, dims=[1])  # vertical
            mask = torch.flip(mask, dims=[0])

        if random.random() < self.rotate_prob:
            k = random.choice([1, 2, 3])  # 90/180/270 degrees
            image = torch.rot90(image, k=k, dims=[1, 2])
            mask = torch.rot90(mask, k=k, dims=[0, 1])

        if random.random() < self.color_jitter_prob:
            b = 1.0 + random.uniform(-self.brightness, self.brightness)
            c = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = torch.clamp((image - mean) * c + mean * b, 0.0, 1.0)

        return image, mask
