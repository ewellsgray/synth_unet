import csv
import glob
import os

import pytest
import torch

from model.unet import TissueSegUNet
from train import train


@pytest.mark.slow
def test_train_runs_and_produces_checkpoints_and_logs(tiny_cfg):
    model = train(tiny_cfg)

    assert isinstance(model, TissueSegUNet)

    last_ckpt = os.path.join(tiny_cfg.checkpoint_dir, "last.pt")
    assert os.path.exists(last_ckpt)

    # a single-epoch run's validation is always >= the initial best_metric of
    # 0.0, so "best.pt" is always written on epoch 1 too.
    best_ckpt = os.path.join(tiny_cfg.checkpoint_dir, "best.pt")
    assert os.path.exists(best_ckpt)

    run_dirs = glob.glob(os.path.join(tiny_cfg.log_dir, "*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    log_path = os.path.join(run_dir, "train.log")
    assert os.path.exists(log_path)
    with open(log_path) as f:
        assert "[done]" in f.read()

    metrics_path = os.path.join(run_dir, "metrics.csv")
    assert os.path.exists(metrics_path)
    with open(metrics_path, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1 + tiny_cfg.epochs  # header + one row per epoch


@pytest.mark.slow
def test_train_produces_finite_loss(tiny_cfg):
    # A second, independent tiny run confirming training doesn't diverge to
    # NaN/inf on the very first step (a common early-integration bug when
    # model shape or loss wiring is subtly wrong).
    model = train(tiny_cfg)
    for p in model.parameters():
        assert torch.isfinite(p).all()
