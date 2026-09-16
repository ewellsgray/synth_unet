import pytest
import torch

from utils.metrics import (
    balanced_accuracy,
    confusion_matrix,
    format_class_metrics,
    per_class_dice,
)


def test_confusion_matrix_known_values():
    # true=0 pred=0, true=0 pred=1, true=1 pred=1, true=1 pred=1
    preds = torch.tensor([0, 1, 1, 1])
    targets = torch.tensor([0, 0, 1, 1])
    cm = confusion_matrix(preds, targets, num_classes=2)
    expected = torch.tensor([[1, 1], [0, 2]])
    assert torch.equal(cm, expected)


def test_balanced_accuracy_perfect_prediction():
    preds = torch.tensor([0, 1, 2, 0, 1, 2])
    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    assert balanced_accuracy(preds, targets, num_classes=3) == 1.0


def test_balanced_accuracy_matches_manual_mean_recall():
    preds = torch.tensor([0, 1, 1, 1])
    targets = torch.tensor([0, 0, 1, 1])
    # class 0: 1/2 correct recall=0.5; class 1: 2/2 correct recall=1.0
    expected = (0.5 + 1.0) / 2
    assert balanced_accuracy(preds, targets, num_classes=2) == expected


def test_balanced_accuracy_excludes_absent_classes():
    # class 2 never appears in targets -> excluded from the mean, not
    # counted as 0 recall.
    preds = torch.tensor([0, 1])
    targets = torch.tensor([0, 1])
    assert balanced_accuracy(preds, targets, num_classes=3) == 1.0


def test_balanced_accuracy_empty_all_classes_absent_returns_zero():
    preds = torch.tensor([], dtype=torch.long)
    targets = torch.tensor([], dtype=torch.long)
    assert balanced_accuracy(preds, targets, num_classes=3) == 0.0


def test_per_class_dice_perfect_and_absent():
    preds = torch.tensor([0, 0, 1, 1])
    targets = torch.tensor([0, 0, 1, 1])
    dice = per_class_dice(preds, targets, num_classes=3)
    assert dice[0].item() == pytest.approx(1.0, abs=1e-4)
    assert dice[1].item() == pytest.approx(1.0, abs=1e-4)
    assert dice[2].item() == pytest.approx(1.0, abs=1e-4)  # absent from both -> trivial 1.0 via eps


def test_per_class_dice_no_overlap():
    preds = torch.tensor([0, 0])
    targets = torch.tensor([1, 1])
    dice = per_class_dice(preds, targets, num_classes=2)
    assert dice[0].item() == pytest.approx(0.0, abs=1e-4)
    assert dice[1].item() == pytest.approx(0.0, abs=1e-4)


def test_format_class_metrics():
    values = torch.tensor([0.5, 1.0])
    out = format_class_metrics(values, ["a", "b"])
    assert out == "a=0.500, b=1.000"
