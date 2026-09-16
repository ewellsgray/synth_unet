# Synthetic Tissue Segmentation — Training Simulation (synth_unet)

A self-contained PyTorch project simulating GPU training of a U-Net that
predicts **7 per-pixel tissue probability maps** (epidermis, dermis, BCC,
SCC, adipose, appendageal structures, background/void) from digitally-stained
confocal mosaic images — modeled on a Mohs-margin-mapping use case discussed
in the interview prep notes.

There's no real dataset available, so a **synthetic data generator**
(`data/synthetic_dataset.py`) procedurally creates plausible layered-skin
images + label masks, so the whole pipeline actually runs end to end today.
`data/real_dataset.py` documents the interface you'd implement against real
patch-indexed OME-TIFF/OME-Zarr data later — nothing else needs to change.

## Project layout

```
config.py              All hyperparameters / settings in one place
train.py                Main training loop (this is what you run)
inference.py             Sliding-window inference + probability-map stitching
model/
  unet.py                Configurable U-Net architecture
  losses.py               Class-weighted CE + soft Dice loss, weight estimation
data/
  synthetic_dataset.py    Procedural synthetic image/mask generator (runs today)
  real_dataset.py          STUB documenting the real-data pipeline interface
  transforms.py            Joint image+mask augmentations
utils/
  device.py                CUDA -> Intel XPU -> CPU device selection
  metrics.py                Balanced accuracy + per-class Dice
  checkpoint.py             Save/load model+optimizer state
  logging.py                 Run logging: console+file log, hardware info, metrics CSV
checkpoints/               Where trained models get saved
logs/                       Per-run logs (train.log + metrics.csv), one timestamped dir each
```

## Running it

```bash
pip install -r requirements.txt
python train.py          # trains for config.epochs, prints per-epoch metrics
python inference.py       # demonstrates patch-tiling inference on a full mosaic
python -m model.unet      # quick shape/param-count sanity check
python -m data.synthetic_dataset   # inspect one generated (image, mask) pair
```

Training auto-detects hardware: NVIDIA GPU (CUDA) -> Intel GPU (XPU, if
`intel_extension_for_pytorch`/`torch.xpu` is available) -> CPU fallback. On
CPU, reduce `synthetic_train_size`/`epochs`/`patch_size` in `config.py` for a
quick smoke test — the full config is sized for actual GPU hardware.

## Testing

```bash
pip install -e ".[test]"   # installs pytest/pytest-cov on top of requirements.txt
pytest                      # unit + integration tests (fast, CPU, seconds)
pytest -m slow              # end-to-end tests: real tiny train() runs + checkpoint/inference
pytest -m ""                 # everything, including slow
pytest --cov=. --cov-report=term-missing -m ""   # with coverage
```

```
tests/
  unit/             pure functions/classes in isolation (config, dataset generator,
                     model shapes, losses, metrics, device selection, checkpoint I/O, logging)
  integration/       multiple components wired together (dataset -> loader -> model -> loss,
                     build_dataloaders/run_validation, sliding-window inference, checkpoint
                     save/load roundtrip, the export_synthetic_samples CLI script)
  e2e/               a real (tiny) train() run end to end, and train -> checkpoint -> inference
  smoke/             a few real training steps with no file I/O -- catches crashes/NaNs fast
  overfit/           trains on a fixed tiny batch -- catches a model that runs but never learns
  data_invariants/   dataset/augmentation output range/shape/dtype checks, no model involved
```

All tests run on CPU regardless of local hardware (`tiny_cfg` fixture in
`tests/conftest.py` forces `device_preference=["cpu"]` and shrinks
`patch_size`/dataset sizes/epochs) so they're fast and reproducible in CI
without a GPU. An autouse fixture snapshots/restores `torch`/`random` RNG
state around every test, since `train()` and `JointAugment` both mutate
global RNG state.

### Pre-commit gate

Regular tests verify the code *runs*; they don't catch a model that trains
cleanly but never actually learns (a frozen layer, a loss wired to the wrong
tensor, gradients not flowing). The `smoke`/`overfit`/`data_invariant` tiers
above exist specifically for that, and are wired into a git pre-commit hook
alongside lint/type checks so they run automatically on every commit:

```bash
pip install -e ".[dev]"   # ruff, mypy, pre-commit, on top of the test deps
pre-commit install         # one-time: registers the git hook
```

After that, `git commit` runs, in order: `ruff check .` → `mypy` →
`pytest tests/smoke` (~5s) → `pytest tests/overfit` (~30s) →
`pytest tests/data_invariants` (~2s) — and blocks the commit if any stage
fails. Run the whole gate on demand without committing:

```bash
pre-commit run --all-files
```

`smoke` and `data_invariant` are cheap enough to also run as part of plain
`pytest` (they're not excluded by `addopts`); `overfit` stays opt-in
(`pytest -m overfit`) alongside `slow`, since letting it into the everyday
default run would make routine `pytest` calls ~10x slower for a check that's
really a pre-commit-gate concern, not a "type `pytest` and wait a second"
one.

## Design choices worth being able to explain in an interview

- **Mixed precision (AMP)** is on by default (`config.use_amp`) — real
  speedup lever for multi-hour training runs, and directly relevant to
  "why does training take hours even with a GPU."
- **Class-weighted + Dice loss**: the 7 classes are nowhere near balanced
  (background/dermis/adipose dominate; BCC/SCC are rare but are the whole
  point of the model), so plain cross-entropy alone would under-train on
  the classes that matter most. Weights are auto-derived from observed
  pixel frequency by default.
- **Balanced accuracy** as the reported validation metric, matching what
  the real system tracks — computed as mean per-class recall from a
  confusion matrix, not overall pixel accuracy (which would look great
  while ignoring the tumor classes).
- **Patch-based training + sliding-window inference**: mosaics are assumed
  too large to fit on GPU whole, so the model only ever sees fixed-size
  patches, and `inference.py` shows how those get tiled back into a
  full-resolution result with overlap-averaging to avoid seam artifacts.
- **Checkpointing + resumability**: `checkpoints/last.pt` and `best.pt` are
  saved automatically — a direct answer to "how do we get this out of a
  notebook" (reproducible runs, resumable training, a real artifact to
  deploy rather than an in-memory model object).
