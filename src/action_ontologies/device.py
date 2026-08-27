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


def needs_eager_attention(device: str, *, training: bool = False) -> bool:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return False
    # HIP exposes AMD accelerators through torch.cuda, but its reported
    # "capability" is a gfx architecture tuple rather than NVIDIA compute
    # capability. ROCm SDPA backward has produced illegal memory accesses on
    # gfx1030; eager attention is slower but stable for LoRA training.
    if torch.version.hip is not None:
        return training
    # Pre-Volta GPUs (compute capability < 7, e.g. Pascal) fall back to the
    # "math" SDPA backend, which runs softmax in fp16 without upcasting and
    # overflows to NaN/garbage tokens. Eager attention upcasts softmax to
    # fp32 internally, which is numerically stable at a modest speed cost.
    return min(
        (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())),
        default=(7, 0),
    )[0] < 7
