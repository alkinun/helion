import pytest
import torch
from _utils import DTYPES, assert_close, cuda_required

import tritium


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 16, 1024, 8192])
def test_add_relu_forward(dtype: torch.dtype, n: int) -> None:
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)
    out = tritium.add_relu(x, y)
    assert_close(out, torch.relu(x + y))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_add_relu_autograd(dtype: torch.dtype) -> None:
    x = torch.randn(1024, device="cuda", dtype=dtype, requires_grad=True)
    y = torch.randn(1024, device="cuda", dtype=dtype, requires_grad=True)
    out = tritium.add_relu(x, y)
    out.backward(torch.ones_like(out))

    mask = (x + y > 0).to(dtype)
    assert_close(x.grad, mask)
    assert_close(y.grad, mask)


def test_add_relu_rejects_cpu_tensors() -> None:
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.add_relu(torch.randn(4), torch.randn(4))
