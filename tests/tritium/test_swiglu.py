import pytest
import torch
import torch.nn.functional as F
from _utils import DTYPES, assert_close, cuda_required

import tritium


def _swiglu_reference(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return F.silu(x) * gate


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 16, 1024, 8192])
def test_swiglu_forward(dtype: torch.dtype, n: int) -> None:
    x = torch.randn(n, device="cuda", dtype=dtype)
    gate = torch.randn(n, device="cuda", dtype=dtype)
    out = tritium.swiglu(x, gate)
    assert_close(out, _swiglu_reference(x, gate))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_backward(dtype: torch.dtype) -> None:
    x = torch.randn(1024, device="cuda", dtype=dtype)
    gate = torch.randn(1024, device="cuda", dtype=dtype)
    dy = torch.randn(1024, device="cuda", dtype=dtype)

    dx, dgate = tritium.swiglu_backward(dy, x, gate)

    x_ref = x.clone().requires_grad_(True)
    gate_ref = gate.clone().requires_grad_(True)
    _swiglu_reference(x_ref, gate_ref).backward(dy)
    assert_close(dx, x_ref.grad)
    assert_close(dgate, gate_ref.grad)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_autograd(dtype: torch.dtype) -> None:
    x = torch.randn(1024, device="cuda", dtype=dtype, requires_grad=True)
    gate = torch.randn(1024, device="cuda", dtype=dtype, requires_grad=True)
    out = tritium.swiglu(x, gate)
    out.backward(torch.randn_like(out))
    assert x.grad is not None
    assert gate.grad is not None


def test_swiglu_rejects_cpu_tensors() -> None:
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.swiglu(torch.randn(4), torch.randn(4))
