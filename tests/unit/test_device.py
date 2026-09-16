import torch

from utils.device import amp_is_supported, get_device


def test_get_device_picks_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = get_device(("cuda", "cpu"))
    assert device.type == "cuda"


def test_get_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = get_device(("cuda", "cpu"))
    assert device.type == "cpu"


def test_get_device_skips_missing_xpu_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "xpu", None, raising=False)
    device = get_device(("cuda", "xpu", "cpu"))
    assert device.type == "cpu"


def test_get_device_defaults_to_cpu_with_empty_preference():
    assert get_device(()).type == "cpu"


def test_amp_is_supported_true_for_cuda_and_xpu():
    assert amp_is_supported(torch.device("cuda")) is True
    assert amp_is_supported(torch.device("xpu")) is True


def test_amp_is_supported_false_for_cpu():
    assert amp_is_supported(torch.device("cpu")) is False
