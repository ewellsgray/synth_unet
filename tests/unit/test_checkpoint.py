import os

import torch

from utils.checkpoint import load_checkpoint, save_checkpoint


def _tiny_model_and_optimizer():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def test_save_checkpoint_creates_parent_dir_and_file(tmp_path):
    model, optimizer = _tiny_model_and_optimizer()
    path = str(tmp_path / "nested" / "ckpt.pt")
    save_checkpoint(path, model, optimizer, epoch=3, best_metric=0.75)
    assert os.path.exists(path)


def test_save_load_roundtrip_restores_weights(tmp_path):
    model_a, optimizer_a = _tiny_model_and_optimizer()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model_a, optimizer_a, epoch=5, best_metric=0.42)

    model_b, optimizer_b = _tiny_model_and_optimizer()
    payload = load_checkpoint(path, model_b, optimizer_b)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)
    assert payload["epoch"] == 5
    assert payload["best_metric"] == 0.42


def test_load_checkpoint_without_optimizer(tmp_path):
    model_a, optimizer_a = _tiny_model_and_optimizer()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model_a, optimizer_a, epoch=1, best_metric=0.1)

    model_b, _ = _tiny_model_and_optimizer()
    payload = load_checkpoint(path, model_b)  # optimizer=None
    assert payload["epoch"] == 1
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_save_checkpoint_stores_extra_payload(tmp_path):
    model, optimizer = _tiny_model_and_optimizer()
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, model, optimizer, epoch=0, best_metric=0.0, extra={"note": "hello"})
    payload = load_checkpoint(path, model, optimizer)
    assert payload["extra"] == {"note": "hello"}
