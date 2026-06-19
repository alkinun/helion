import pytest
import torch
import torch.nn.functional as F
from _utils import DTYPES, assert_close, cuda_required

import tritium


def _gelu_reference(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 16, 1024, 8192])
def test_gelu_forward(dtype: torch.dtype, n: int) -> None:
    x = torch.randn(n, device="cuda", dtype=dtype)
    out = tritium.gelu(x)
    assert_close(out, _gelu_reference(x))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_backward(dtype: torch.dtype) -> None:
    x = torch.randn(1024, device="cuda", dtype=dtype)
    dy = torch.randn(1024, device="cuda", dtype=dtype)
    dx = tritium.gelu_backward(dy, x)

    x_ref = x.clone().requires_grad_(True)
    _gelu_reference(x_ref).backward(dy)
    assert_close(dx, x_ref.grad)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_autograd(dtype: torch.dtype) -> None:
    x = torch.randn(1024, device="cuda", dtype=dtype, requires_grad=True)
    out = tritium.gelu(x)
    out.backward(torch.randn_like(out))
    assert x.grad is not None
    assert x.grad.shape == x.shape


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 16, 1024])
def test_add_gelu_forward(dtype: torch.dtype, n: int) -> None:
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)
    out = tritium.add_gelu(x, y)
    assert_close(out, _gelu_reference(x + y))


def test_gelu_rejects_cpu_tensors() -> None:
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.gelu(torch.randn(4))
