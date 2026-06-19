import pytest
import torch
from _utils import DTYPES, assert_close, cuda_required

import tritium


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 16, 1024, 4096])
def test_add_forward(dtype: torch.dtype, n: int) -> None:
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)

    out = tritium.add(x, y)
    assert_close(out, x + y)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_add_autograd(dtype: torch.dtype) -> None:
    x = torch.randn(1024, device="cuda", dtype=dtype, requires_grad=True)
    y = torch.randn(1024, device="cuda", dtype=dtype, requires_grad=True)

    out = tritium.add(x, y)
    out.backward(torch.ones_like(out))

    assert_close(x.grad, torch.ones(1024, device="cuda", dtype=dtype))
    assert_close(y.grad, torch.ones(1024, device="cuda", dtype=dtype))


def test_add_rejects_cpu_tensors() -> None:
    x = torch.randn(4)
    y = torch.randn(4)
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.add(x, y)
