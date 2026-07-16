from __future__ import annotations


def select_device(requested: str = "auto") -> str:
    if requested == "rocm":
        requested = "cuda"
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype_for_device(device: str):
    import torch

    if device == "cpu":
        return torch.float32
    return torch.float16
