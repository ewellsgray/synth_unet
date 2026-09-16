import logging
import random

import pytest
import torch

from config import TrainConfig


@pytest.fixture(autouse=True)
def rng_isolation():
    """Snapshot/restore torch and random RNG state so torch.manual_seed(...)
    calls (train.py) or unseeded random.* calls (JointAugment) in one test
    never affect another test's outcome."""
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    yield
    torch.set_rng_state(torch_state)
    random.setstate(random_state)


# FUTURE WORK: tests/smoke, tests/overfit, and tests/data_invariants each
# build their own tiny dataset/model inline (patch_size=32, small
# base_features/depth) instead of using this fixture, since they intentionally
# skip TrainConfig/file I/O to stay fast. That leaves the "tiny" parameters
# duplicated across three files -- consider factoring a shared
# `tiny_model_and_data` fixture here if they drift out of sync.
@pytest.fixture
def tiny_cfg(tmp_path):
    """A TrainConfig sized for fast, deterministic CPU test runs."""
    return TrainConfig(
        patch_size=32,
        full_image_size=64,
        batch_size=2,
        epochs=1,
        num_workers=0,
        device_preference=["cpu"],
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        synthetic_train_size=6,
        synthetic_val_size=4,
    )


@pytest.fixture
def cleanup_logging_handlers():
    """Close/remove handlers on any logger created during the test so file
    handles (opened by utils.logging.setup_run_logger inside tmp_path) are
    released promptly instead of leaking across the test session."""
    created_loggers = []

    def _register(logger: logging.Logger):
        created_loggers.append(logger)
        return logger

    yield _register

    for logger in created_loggers:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
