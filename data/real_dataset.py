"""
STUB — intended interface for loading real confocal mosaics.

This is not wired up to any real files; it documents the shape of the pipeline
you'd build once real data is available, based on what we know about the
platform:

  - Raw captures and Digitally Stained Confocal Mosaics (DSCMs) are large,
    likely 3D volumetric, sub-micron-resolution images — almost certainly
    too large to load whole onto a GPU, hence PATCH-BASED extraction below.
  - Modern multi-dimensional microscopy formats (OME-TIFF / OME-Zarr) are a
    much better fit here than flat 2D whole-slide formats like DICOM/SVS,
    since they natively support multi-channel, multi-z-plane volumes.
  - Given the "lab-in-a-box" edge architecture and the PHI-sensitivity of
    patient tissue images, raw data plausibly lives on-prem/edge storage
    first, with a curated, de-identified, patch-indexed dataset (this class's
    real job) built on top for training — see the metadata DB note below.

Swap `SyntheticTissueDataset` for `RealTissuePatchDataset` in train.py once
the paths below point at real data and `_read_region` is implemented against
whatever format/storage backend is actually in use (e.g. `tifffile`, `zarr`,
or a vendor SDK).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class PatchIndexEntry:
    """
    One row of a "patch index" — the metadata layer that turns a handful of
    huge whole-mosaic files into many trainable patches without duplicating
    image data on disk.

    In practice this index would be built once (offline) by scanning each
    mosaic + its pathologist-reviewed label map and recording patch
    coordinates, then persisted as a small table (SQLite/Parquet/CSV) —
    much cheaper to version and query than the images themselves.
    """
    case_id: str
    mosaic_path: str          # path or URI to the OME-TIFF/OME-Zarr file
    mask_path: str            # path to the corresponding pathologist-reviewed label map
    x: int                    # top-left x of the patch within the mosaic
    y: int                    # top-left y of the patch within the mosaic
    z: int = 0                # z-plane index, for volumetric data (0 for a single 2D plane)
    model_confidence: float = None  # confidence score from the current model, if this
                                     # patch came through the confidence-based review queue
    reviewed_by_pathologist: bool = False


class RealTissuePatchDataset(Dataset):
    """
    Loads fixed-size patches from a pre-built patch index (see PatchIndexEntry)
    rather than whole mosaics, so training patches can be shuffled/batched
    normally while the actual pixel data for a huge mosaic is only ever
    partially read into memory.

    TODO when wiring this up for real:
      1. Implement `_load_index` to read the real patch-index table.
      2. Implement `_read_region` to read just the requested (x, y, patch_size)
         window from `mosaic_path` — e.g. via `tifffile.TiffFile(...).asarray(
         key=..., ...)` for OME-TIFF, or `zarr.open(...)[y:y+p, x:x+p]` for
         OME-Zarr — instead of loading the full mosaic into memory.
      3. Decide how model_confidence / reviewed_by_pathologist feed into
         sampling — e.g. oversampling rare tumor classes, or weighting loss
         per-patch by review status.
    """

    def __init__(self, index_path: str, patch_size: int, transform=None):
        self.index_path = Path(index_path)
        self.patch_size = patch_size
        self.transform = transform
        self.entries: List[PatchIndexEntry] = self._load_index()

    def _load_index(self) -> List[PatchIndexEntry]:
        raise NotImplementedError(
            "Point this at the real patch index (SQLite/Parquet/CSV) once it exists. "
            "This stub intentionally has no fake data — see synthetic_dataset.py "
            "for something you can actually train against today."
        )

    def _read_region(self, entry: PatchIndexEntry) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Implement windowed reads against the real mosaic format "
            "(tifffile/zarr/vendor SDK) here."
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image, mask = self._read_region(entry)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask
