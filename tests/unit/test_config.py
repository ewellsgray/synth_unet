import pytest

from config import DEFAULT_CLASS_PREVALENCE, TISSUE_CLASSES, TrainConfig


def test_tissue_classes_count():
    assert len(TISSUE_CLASSES) == 7
    assert len(set(TISSUE_CLASSES)) == 7  # no duplicates


def test_default_class_prevalence_matches_classes():
    assert len(DEFAULT_CLASS_PREVALENCE) == len(TISSUE_CLASSES)
    assert sum(DEFAULT_CLASS_PREVALENCE) == pytest.approx(1.0, abs=1e-6)


def test_train_config_defaults_construct():
    cfg = TrainConfig()
    assert cfg.num_classes == len(TISSUE_CLASSES)
    assert cfg.in_channels == 3
    assert cfg.device_preference == ["cuda", "xpu", "cpu"]
    assert cfg.class_weights is None


def test_train_config_overrides():
    cfg = TrainConfig(patch_size=32, epochs=1, batch_size=2)
    assert cfg.patch_size == 32
    assert cfg.epochs == 1
    assert cfg.batch_size == 2
    # unrelated defaults remain untouched
    assert cfg.num_classes == len(TISSUE_CLASSES)
