"""
Training entry point for the 7-class tissue segmentation U-Net.

Run with:
    python train.py

What this simulates, end to end:
  - GPU-accelerated training of a U-Net that outputs 7 per-pixel probability
    maps (softmax over the class channel), one per tissue/margin category.
  - Mixed-precision (AMP) training, the way you'd actually speed up multi-hour
    training runs on real hardware.
  - Class-weighted + Dice loss to handle the severe imbalance between common
    classes (dermis, adipose, background) and rare-but-critical ones (BCC/SCC).
  - "Balanced accuracy" validation reporting, matching the metric the target
    platform actually tracks.
  - Checkpointing, so a run can be resumed rather than living only in a
    notebook kernel's memory.

Swap `SyntheticTissueDataset` for `RealTissuePatchDataset` (data/real_dataset.py)
once real patch-indexed data exists — nothing else in this file needs to change.
"""
import os
import time

import torch
from torch.utils.data import DataLoader

from config import TISSUE_CLASSES, TrainConfig
from data.synthetic_dataset import SyntheticTissueDataset
from data.transforms import JointAugment
from model.losses import WeightedCEDiceLoss, compute_inverse_frequency_weights
from model.unet import TissueSegUNet, count_parameters
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import amp_is_supported, get_device
from utils.logging import (
    MetricsRecorder,
    format_duration,
    gpu_memory_mb,
    log_config,
    log_environment,
    reset_peak_memory_stats,
    setup_run_logger,
)
from utils.metrics import balanced_accuracy, format_class_metrics, per_class_dice


def build_dataloaders(cfg: TrainConfig, augment: JointAugment):
    train_ds: torch.utils.data.Dataset = SyntheticTissueDataset(
        num_samples=cfg.synthetic_train_size, patch_size=cfg.patch_size, seed=cfg.seed
    )
    val_ds: torch.utils.data.Dataset = SyntheticTissueDataset(
        num_samples=cfg.synthetic_val_size, patch_size=cfg.patch_size, seed=cfg.seed + 1
    )

    class _AugmentedWrapper(torch.utils.data.Dataset):
        def __init__(self, base, transform=None):
            self.base = base
            self.transform = transform

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            image, mask = self.base[idx]
            if self.transform is not None:
                image, mask = self.transform(image, mask)
            return image, mask

    train_ds = _AugmentedWrapper(train_ds, transform=augment)
    # No augmentation on validation — we want a stable measure of real performance.
    val_ds = _AugmentedWrapper(val_ds, transform=None)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_ds


def run_validation(model, val_loader, device, num_classes):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
    preds_cat = torch.cat([p.flatten() for p in all_preds])
    targets_cat = torch.cat([t.flatten() for t in all_targets])
    bal_acc = balanced_accuracy(preds_cat, targets_cat, num_classes)
    dice = per_class_dice(preds_cat, targets_cat, num_classes)
    return bal_acc, dice


def train(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)

    logger, run_dir = setup_run_logger(cfg.log_dir)
    logger.info(f"[setup] run directory: {run_dir}")
    log_config(logger, cfg)

    device = get_device(cfg.device_preference)
    use_amp = cfg.use_amp and amp_is_supported(device)
    log_environment(logger, device)
    logger.info(f"[setup] device={device}  mixed_precision={use_amp}")

    augment = JointAugment()
    train_loader, val_loader, train_ds_for_weights = build_dataloaders(cfg, augment)
    logger.info(
        f"[setup] train_samples={len(train_loader.dataset)}  val_samples={len(val_loader.dataset)}  "
        f"batch_size={cfg.batch_size}  steps_per_epoch={len(train_loader)}"
    )

    model = TissueSegUNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_features=cfg.base_features,
        depth=cfg.depth,
    ).to(device)
    logger.info(f"[setup] model parameters: {count_parameters(model):,}")

    if cfg.class_weights is not None:
        class_weights = torch.tensor(cfg.class_weights, dtype=torch.float32)
    else:
        # base dataset (before the augment wrapper) for weight estimation
        class_weights = compute_inverse_frequency_weights(
            train_ds_for_weights.base, cfg.num_classes, sample_batches=8
        )
    logger.info(f"[setup] class weights: {format_class_metrics(class_weights, TISSUE_CLASSES)}")
    class_weights = class_weights.to(device)

    criterion = WeightedCEDiceLoss(
        num_classes=cfg.num_classes, class_weights=class_weights, use_dice=cfg.use_dice_loss
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # GradScaler only meaningfully applies to CUDA; other backends run AMP
    # without loss scaling here for simplicity.
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    start_epoch = 0
    best_metric = 0.0
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        payload = load_checkpoint(cfg.resume_from, model, optimizer, map_location=str(device))
        start_epoch = payload["epoch"] + 1
        best_metric = payload["best_metric"]
        logger.info(f"[resume] continuing from epoch {start_epoch}, best_metric={best_metric:.4f}")

    metrics_recorder = MetricsRecorder(run_dir, TISSUE_CLASSES)
    run_start = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad()
        reset_peak_memory_stats(device)

        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            autocast_ctx = torch.autocast(device_type=device.type, enabled=use_amp)
            with autocast_ctx:
                logits = model(images)
                loss = criterion(logits, masks) / cfg.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if (step + 1) % cfg.grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % cfg.grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * cfg.grad_accum_steps

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        bal_acc, dice = run_validation(model, val_loader, device, cfg.num_classes)
        epoch_time = time.time() - epoch_start
        mem_mb = gpu_memory_mb(device)
        lr = optimizer.param_groups[0]["lr"]

        mem_str = f"  peak_mem={mem_mb:.0f}MB" if mem_mb is not None else ""
        logger.info(
            f"[epoch {epoch + 1}/{cfg.epochs}] "
            f"train_loss={avg_loss:.4f}  val_balanced_accuracy={bal_acc:.4f}  "
            f"lr={lr:.2e}  time={format_duration(epoch_time)}{mem_str}"
        )
        logger.info(f"           per-class dice: {format_class_metrics(dice, TISSUE_CLASSES)}")
        metrics_recorder.log_epoch(
            epoch=epoch + 1, train_loss=avg_loss, val_balanced_accuracy=bal_acc,
            lr=lr, epoch_time_sec=epoch_time, dice=dice, gpu_mem_mb=mem_mb,
        )

        is_best = bal_acc > best_metric
        best_metric = max(bal_acc, best_metric)

        if (epoch + 1) % cfg.checkpoint_every == 0 or is_best:
            ckpt_path = os.path.join(cfg.checkpoint_dir, "last.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_metric)
            logger.info(f"           checkpoint saved -> {ckpt_path}")
            if is_best:
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                save_checkpoint(best_path, model, optimizer, epoch, best_metric)
                logger.info(f"           new best model saved -> {best_path}")

    total_time = time.time() - run_start
    epochs_run = cfg.epochs - start_epoch
    avg_epoch_time = total_time / epochs_run if epochs_run else 0.0
    logger.info(
        f"[done] best validation balanced accuracy: {best_metric:.4f}  "
        f"total_time={format_duration(total_time)}  avg_epoch_time={format_duration(avg_epoch_time)}"
    )
    logger.info(f"[done] logs written to {run_dir}")
    return model


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
