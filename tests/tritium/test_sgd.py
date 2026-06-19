import pytest
import torch
from _utils import DTYPES, assert_close, cuda_required

import tritium


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 128, 4096])
def test_sgd_step_matches_reference(dtype: torch.dtype, n: int) -> None:
    param = torch.randn(n, device="cuda", dtype=dtype)
    grad = torch.randn(n, device="cuda", dtype=dtype)
    lr = 1e-2

    ref = (param.float() - lr * grad.float()).to(dtype)
    tritium.sgd_step(param, grad, lr)
    assert_close(param, ref)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_sgd_step_weight_decay_matches_reference(dtype: torch.dtype) -> None:
    param = torch.randn(4096, device="cuda", dtype=dtype)
    grad = torch.randn_like(param)
    lr = 1e-2
    weight_decay = 0.1

    ref = (param.float() - lr * (grad.float() + weight_decay * param.float())).to(dtype)
    tritium.sgd_step(param, grad, lr, weight_decay=weight_decay)
    assert_close(param, ref)


@cuda_required
def test_sgd_step_updates_in_place() -> None:
    param = torch.randn(1024, device="cuda", dtype=torch.float32)
    grad = torch.randn(1024, device="cuda", dtype=torch.float32)
    before = param.clone()
    tritium.sgd_step(param, grad, 0.1)
    assert not torch.equal(before, param)


def test_sgd_step_rejects_cpu_tensors() -> None:
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.sgd_step(torch.randn(4), torch.randn(4), 1e-2)
