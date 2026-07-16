from action_ontologies.device import select_device


def test_rocm_alias_uses_torch_cuda_backend_name():
    assert select_device("rocm") == "cuda"
