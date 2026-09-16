"""
Run logging: console + persisted log file, hardware/environment info,
and a per-epoch metrics CSV — the record you'd actually want after a
multi-hour training run, not just scrollback from a terminal.

Each call to `setup_run_logger` creates a timestamped directory under
`cfg.log_dir` (e.g. `logs/20260916-083000/`) containing:
  train.log     full text log (everything also printed to the console)
  metrics.csv   one row per epoch: loss, accuracy, per-class dice, timing
"""
import csv
import dataclasses
import logging
import os
import platform
import sys
import time

import torch


def setup_run_logger(log_dir: str, name: str = "train") -> "tuple[logging.Logger, str]":
    """Create a timestamped run directory and a logger that writes to both
    the console and `<run_dir>/train.log`."""
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(log_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger(f"{name}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Guard against duplicate handlers if this is ever called twice with the
    # same logger name (e.g. interactive re-runs in the same process).
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(run_dir, "train.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger, run_dir


def log_environment(logger: logging.Logger, device: torch.device) -> None:
    """Log everything needed to reproduce/interpret a run later: what
    hardware it actually trained on, and which software versions."""
    logger.info(
        f"[env] python={platform.python_version()}  torch={torch.__version__}  "
        f"platform={platform.platform()}"
    )

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_mem_gb = props.total_memory / (1024 ** 3)
        logger.info(
            f"[hardware] training on GPU (CUDA): {torch.cuda.get_device_name(idx)}  "
            f"total_memory={total_mem_gb:.1f}GB  compute_capability={props.major}.{props.minor}  "
            f"cuda_version={torch.version.cuda}  gpu_count={torch.cuda.device_count()}"
        )
    elif device.type == "xpu":
        try:
            name = torch.xpu.get_device_name(device)
        except Exception:
            name = "unknown Intel XPU device"
        logger.info(f"[hardware] training on GPU (XPU): {name}")
    else:
        logger.info(
            f"[hardware] training on CPU: {platform.processor() or platform.machine()}  "
            f"logical_cores={os.cpu_count()}  "
            "(no GPU detected/available — expect training to be substantially slower)"
        )


def log_config(logger: logging.Logger, cfg) -> None:
    logger.info("[config] " + ", ".join(f"{k}={v}" for k, v in dataclasses.asdict(cfg).items()))


def format_duration(seconds: float) -> str:
    """Human-readable elapsed time, e.g. 5423.1 -> '1h 30m 23s'."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def gpu_memory_mb(device: torch.device) -> float | None:
    """Peak allocated GPU memory for this device since the last reset, in MB.
    Returns None on backends without a comparable stat (e.g. CPU)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "max_memory_allocated"):
        try:
            return torch.xpu.max_memory_allocated(device) / (1024 ** 2)
        except Exception:
            return None
    return None


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "reset_peak_memory_stats"):
        try:
            torch.xpu.reset_peak_memory_stats(device)
        except Exception:
            pass


class MetricsRecorder:
    """Appends one CSV row per epoch to `<run_dir>/metrics.csv`, so a run's
    learning curves can be plotted/compared later without re-parsing logs."""

    def __init__(self, run_dir: str, class_names: list[str]):
        self.class_names = class_names
        self.path = os.path.join(run_dir, "metrics.csv")
        self._fieldnames = (
            ["epoch", "train_loss", "val_balanced_accuracy", "lr", "epoch_time_sec", "gpu_mem_mb"]
            + [f"dice_{name}" for name in class_names]
        )
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self._fieldnames)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_balanced_accuracy: float,
        lr: float,
        epoch_time_sec: float,
        dice: torch.Tensor,
        gpu_mem_mb: float | None,
    ) -> None:
        row = [epoch, train_loss, val_balanced_accuracy, lr, epoch_time_sec, gpu_mem_mb or ""]
        row += [v for v in dice.tolist()]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
