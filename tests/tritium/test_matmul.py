import pytest
import torch
from _utils import DTYPES, assert_close, cuda_required

import tritium


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(16, 16, 16), (64, 64, 64), (128, 256, 32), (33, 17, 9)],
)
def test_matmul_forward(dtype: torch.dtype, m: int, n: int, k: int) -> None:
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    out = tritium.matmul(a, b)
    assert_close(out, torch.matmul(a.float(), b.float()).to(dtype))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_accepts_non_contiguous_second_input(dtype: torch.dtype) -> None:
    m, n, k = 64, 64, 64
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype)
    b = weight.T

    assert not b.is_contiguous()
    out = tritium.matmul(a, b)
    assert_close(out, torch.matmul(a.float(), b.float()).to(dtype))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("n", "k"), [(16, 16), (96, 64), (1024, 384), (65, 33)])
def test_matmul_vec_forward(dtype: torch.dtype, n: int, k: int) -> None:
    a = torch.randn(k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)

    out = tritium.matmul_vec(a, b)
    assert_close(out, torch.matmul(a.float(), b.float()).to(dtype))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_vec_accepts_non_contiguous_second_input(dtype: torch.dtype) -> None:
    n, k = 96, 64
    a = torch.randn(k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype)
    b = weight.T

    assert not b.is_contiguous()
    out = tritium.matmul_vec(a, b)
    assert_close(out, torch.matmul(a.float(), b.float()).to(dtype))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_vec_autograd(dtype: torch.dtype) -> None:
    a = torch.randn(64, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(64, 32, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(32, device="cuda", dtype=dtype)

    out = tritium.matmul_vec(a, b)
    out.backward(grad)

    a_ref = a.detach().float().clone().requires_grad_(True)
    b_ref = b.detach().float().clone().requires_grad_(True)
    torch.matmul(a_ref, b_ref).backward(grad.float())

    assert_close(a.grad, a_ref.grad.to(dtype))
    assert_close(b.grad, b_ref.grad.to(dtype))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("m", "n", "k"), [(64, 64, 64), (128, 32, 17)])
def test_matmul_autograd(dtype: torch.dtype, m: int, n: int, k: int) -> None:
    a = torch.randn(m, k, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(k, n, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(m, n, device="cuda", dtype=dtype)

    out = tritium.matmul(a, b)
    out.backward(grad)

    a_ref = a.detach().float().clone().requires_grad_(True)
    b_ref = b.detach().float().clone().requires_grad_(True)
    torch.matmul(a_ref, b_ref).backward(grad.float())

    assert_close(a.grad, a_ref.grad.to(dtype))
    assert_close(b.grad, b_ref.grad.to(dtype))


@cuda_required
def test_matmul_forward_skips_autograd_under_no_grad() -> None:
    a = torch.randn(16, 16, device="cuda", requires_grad=True)
    b = torch.randn(16, 16, device="cuda", requires_grad=True)
    with torch.no_grad():
        out = tritium.matmul(a, b)
    assert not out.requires_grad


def test_matmul_rejects_cpu_tensors() -> None:
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.matmul(torch.randn(4, 4), torch.randn(4, 4))


@cuda_required
def test_matmul_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        tritium.matmul(
            torch.randn(4, 8, device="cuda"), torch.randn(4, 8, device="cuda")
        )


@cuda_required
def test_matmul_rejects_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="Dtype mismatch"):
        tritium.matmul(
            torch.randn(4, 8, device="cuda", dtype=torch.float16),
            torch.randn(8, 4, device="cuda", dtype=torch.float32),
        )


@cuda_required
def test_matmul_vec_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        tritium.matmul_vec(
            torch.randn(8, device="cuda"), torch.randn(4, 8, device="cuda")
        )
