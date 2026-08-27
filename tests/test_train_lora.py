import importlib.util
from pathlib import Path

import torch


def _module():
    path = Path(__file__).parents[1] / "scripts" / "train_lora.py"
    spec = importlib.util.spec_from_file_location("train_lora", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mask_prompt_tokens_handles_right_padding():
    ids = torch.tensor([[10, 11, 12, 13, 0], [20, 21, 22, 23, 24]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    labels = _module().mask_prompt_tokens(ids, mask, [2, 3])
    assert labels.tolist() == [[-100, -100, 12, 13, -100], [-100, -100, -100, 23, 24]]


def test_mask_prompt_tokens_handles_left_padding():
    ids = torch.tensor([[0, 10, 11, 12, 13]])
    mask = torch.tensor([[0, 1, 1, 1, 1]])
    labels = _module().mask_prompt_tokens(ids, mask, [2])
    assert labels.tolist() == [[-100, -100, -100, 12, 13]]
