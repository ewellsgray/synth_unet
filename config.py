"""
Central configuration for the 7-class tissue segmentation U-Net (synth_unet).

Editing this file is the primary way to change how a training run behaves —
nothing else in the project should need to be touched for routine experiments.
"""
from dataclasses import dataclass, field

# The 7 mutually-exclusive per-pixel classes the model predicts a probability
# map for, based on the tissue/lesion categories relevant to Mohs margin mapping.
TISSUE_CLASSES: list[str] = [
    "epidermis",
    "dermis_normal_stroma",
    "bcc_tumor_nests",
    "scc_keratinocyte_atypia",
    "adipose_tissue",
    "appendageal_structures",
    "background_void",
]

# Rough relative prevalence used to seed synthetic data generation and to derive
# default class weights. In real margin-mapping data, background/normal stroma
# dominates and tumor classes (the ones you actually care about) are rare —
# this is the class-imbalance problem "balanced accuracy" is meant to address.
DEFAULT_CLASS_PREVALENCE: list[float] = [0.12, 0.30, 0.04, 0.03, 0.30, 0.06, 0.15]


@dataclass
class TrainConfig:
    # --- Data / model shape ---
    num_classes: int = len(TISSUE_CLASSES)
    in_channels: int = 3          # digitally-stained image behaves like an RGB (H&E-mimic) image
    patch_size: int = 256         # side length of the square patch fed to the network
    full_image_size: int = 1024   # simulated size of a full mosaic before patching (see inference.py)

    # --- Optimization ---
    batch_size: int = 8
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_accum_steps: int = 1     # >1 lets you simulate a larger effective batch on limited GPU memory

    # --- U-Net architecture ---
    base_features: int = 32       # channel count after the first conv block
    depth: int = 4                # number of down/up-sampling stages

    # --- Hardware / performance ---
    device_preference: list[str] = field(default_factory=lambda: ["cuda", "xpu", "cpu"])
    use_amp: bool = True          # mixed-precision training (FP16/BF16 autocast)
    num_workers: int = 4

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1     # epochs between checkpoint saves
    resume_from: str | None = None

    # --- Logging ---
    log_dir: str = "logs"         # each run gets its own logs/<timestamp>/ subdirectory

    # --- Data split / reproducibility ---
    val_split: float = 0.15
    seed: int = 42

    # --- Class imbalance handling ---
    # If None, weights are auto-derived at runtime from the training set's
    # observed pixel frequency (inverse-frequency weighting) — see train.py.
    class_weights: list[float] | None = None
    use_dice_loss: bool = True    # combine weighted CE with soft Dice, common for imbalanced segmentation

    # --- Synthetic dataset (used until a real data pipeline is wired in) ---
    synthetic_train_size: int = 1000
    synthetic_val_size: int = 200
