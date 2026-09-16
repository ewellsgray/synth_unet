"""
Device selection that tries NVIDIA CUDA first, then Intel XPU (oneAPI /
Data Center GPU Max, via the `intel_extension_for_pytorch` package or
PyTorch's built-in `torch.xpu` backend on newer versions), then falls back
to CPU.

This directly reflects the real migration question the target platform is facing:
most segmentation code is written and tested against `.cuda()` calls, and
moving to an Intel GPU is not a transparent swap — it needs an explicit
backend check like this one, and any CUDA-only ops/libraries in the
pipeline need auditing.
"""
import torch


def get_device(preference=("cuda", "xpu", "cpu")) -> torch.device:
    for backend in preference:
        if backend == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if backend == "xpu" and getattr(torch, "xpu", None) is not None:
            try:
                if torch.xpu.is_available():
                    return torch.device("xpu")
            except Exception:
                pass  # xpu backend not built into this torch install
        if backend == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def amp_is_supported(device: torch.device) -> bool:
    """Mixed precision autocast is meaningful on cuda/xpu; on cpu it's a no-op
    (bf16 autocast on CPU exists but rarely helps without dedicated support)."""
    return device.type in ("cuda", "xpu")
