import torch

from model.unet import TissueSegUNet
from utils.checkpoint import load_checkpoint, save_checkpoint


def _build_model_and_optimizer():
    model = TissueSegUNet(in_channels=3, num_classes=4, base_features=4, depth=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_checkpoint_resume_restores_model_and_run_state(tmp_path):
    model_a, optimizer_a = _build_model_and_optimizer()

    # simulate a step of training so weights/optimizer state are non-trivial
    x = torch.randn(1, 3, 8, 8)
    target = torch.randint(0, 4, (1, 8, 8))
    loss = torch.nn.functional.cross_entropy(model_a(x), target)
    loss.backward()
    optimizer_a.step()

    ckpt_path = str(tmp_path / "checkpoints" / "last.pt")
    save_checkpoint(ckpt_path, model_a, optimizer_a, epoch=2, best_metric=0.55)

    # a fresh model/optimizer with different random init
    model_b, optimizer_b = _build_model_and_optimizer()
    assert any(
        not torch.equal(p_a, p_b)
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters())
    )

    payload = load_checkpoint(ckpt_path, model_b, optimizer_b, map_location="cpu")

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)
    assert payload["epoch"] == 2
    assert payload["best_metric"] == 0.55
