import pytest
import torch
from _utils import DTYPES, assert_close, cuda_required

import tritium


def _relu_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 16, 1024, 8192])
def test_relu_forward(dtype: torch.dtype, n: int) -> None:
    x = torch.randn(n, device="cuda", dtype=dtype)
    out = tritium.relu(x)
    assert_close(out, _relu_reference(x))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_relu_autograd(dtype: torch.dtype) -> None:
    x = torch.randn(1024, device="cuda", dtype=dtype, requires_grad=True)
    out = tritium.relu(x)
    out.backward(torch.ones_like(out))
    assert_close(x.grad, (x > 0).to(dtype))


def test_relu_rejects_cpu_tensors() -> None:
    x = torch.randn(4)
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.relu(x)
