from types import SimpleNamespace

from action_ontologies.device import needs_eager_attention, select_device


def test_rocm_alias_uses_torch_cuda_backend_name():
    assert select_device("rocm") == "cuda"


def test_rocm_uses_eager_attention(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "version", SimpleNamespace(hip="7.2"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert needs_eager_attention("cuda") is False
    assert needs_eager_attention("cuda", training=True) is True
