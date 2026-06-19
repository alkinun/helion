import pytest
import torch
from _utils import cuda_required

import tritium

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
HIDDEN_SIZES = [1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 8192, 11008]
N_ROWS = [1, 16, 128, 1024, 4096]


def _rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.float()
    weight_float = weight.float()
    rstd = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_float * rstd * weight_float).to(x.dtype)


def _rmsnorm_backward_reference(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dy_float = dy.float()
    x_float = x.float()
    weight_float = weight.float()
    hidden_size = x.shape[-1]
    rstd = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    dot = (dy_float * weight_float * x_float).sum(dim=-1, keepdim=True)
    dx = dy_float * weight_float * rstd - x_float * dot * rstd.pow(3) / hidden_size
    dweight = (dy_float * x_float * rstd).sum(dim=tuple(range(x.dim() - 1)))
    return dx.to(x.dtype), dweight.to(weight.dtype)


def _residual_rmsnorm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = (x.float() + residual.float()).to(x.dtype)
    out = _rmsnorm_reference(residual_out, weight, eps)
    return out, residual_out


def _tolerances(
    dtype: torch.dtype,
    *,
    accumulated: bool = False,
) -> tuple[float, float]:
    if dtype == torch.float32:
        if accumulated:
            return 5e-5, 5e-5
        return 2e-5, 2e-5
    if dtype == torch.float16:
        return 2e-2, 2e-2
    return 3e-2, 3e-2


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    accumulated: bool = False,
) -> None:
    rtol, atol = _tolerances(actual.dtype, accumulated=accumulated)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_rmsnorm_forward_numerical_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    out = tritium.rmsnorm(x, weight)
    ref = _rmsnorm_reference(x, weight, eps=1e-6)

    _assert_close(out, ref)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_rmsnorm_backward_dx_and_dweight_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)

    dx, dweight = tritium.rmsnorm_backward(dy, x, weight)
    ref_dx, ref_dweight = _rmsnorm_backward_reference(dy, x, weight, eps=1e-6)

    _assert_close(dx, ref_dx)
    _assert_close(dweight, ref_dweight, accumulated=True)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", [1024, 4096, 11008])
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_rmsnorm_autograd_backward_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)

    out = tritium.rmsnorm(x, weight)
    out.backward(dy)
    ref_dx, ref_dweight = _rmsnorm_backward_reference(
        dy,
        x.detach(),
        weight.detach(),
        eps=1e-6,
    )

    assert x.grad is not None
    assert weight.grad is not None
    _assert_close(x.grad, ref_dx)
    _assert_close(weight.grad, ref_dweight, accumulated=True)


@cuda_required
def test_rmsnorm_autograd_accepts_non_contiguous_grad_output() -> None:
    hidden_size = 1024
    x = torch.randn(
        16,
        hidden_size,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.randn(
        hidden_size,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    dy = torch.randn(hidden_size, 16, device="cuda", dtype=torch.float32).T

    assert not dy.is_contiguous()

    out = tritium.rmsnorm(x, weight)
    out.backward(dy)
    ref_dx, ref_dweight = _rmsnorm_backward_reference(
        dy,
        x.detach(),
        weight.detach(),
        eps=1e-6,
    )

    assert x.grad is not None
    assert weight.grad is not None
    _assert_close(x.grad, ref_dx)
    _assert_close(weight.grad, ref_dweight, accumulated=True)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", [1024, 4096, 11008])
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_residual_rmsnorm_forward_numerical_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)

    out, residual_out = tritium.residual_rmsnorm(x, residual, weight)
    ref_out, ref_residual_out = _residual_rmsnorm_reference(
        x,
        residual,
        weight,
        eps=1e-6,
    )

    _assert_close(out, ref_out)
    _assert_close(residual_out, ref_residual_out)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", [1024, 4096, 11008])
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_residual_rmsnorm_autograd_backward_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)
    dresidual_out = torch.randn_like(x)

    out, residual_out = tritium.residual_rmsnorm(x, residual, weight)
    torch.autograd.backward((out, residual_out), (dy, dresidual_out))

    residual_out_ref = (x.detach().float() + residual.detach().float()).to(dtype)
    ref_dz, ref_dweight = _rmsnorm_backward_reference(
        dy,
        residual_out_ref,
        weight.detach(),
        eps=1e-6,
    )
    ref_dz = ref_dz + dresidual_out

    assert x.grad is not None
    assert residual.grad is not None
    assert weight.grad is not None
    _assert_close(x.grad, ref_dz)
    _assert_close(residual.grad, ref_dz)
    _assert_close(weight.grad, ref_dweight, accumulated=True)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_rows", N_ROWS)
def test_rmsnorm_all_requested_row_counts(dtype: torch.dtype, n_rows: int) -> None:
    hidden_size = 1024
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)

    out = tritium.rmsnorm(x, weight)
    dx, dweight = tritium.rmsnorm_backward(dy, x, weight)
    ref = _rmsnorm_reference(x, weight, eps=1e-6)
    ref_dx, ref_dweight = _rmsnorm_backward_reference(dy, x, weight, eps=1e-6)

    _assert_close(out, ref)
    _assert_close(dx, ref_dx)
    _assert_close(dweight, ref_dweight, accumulated=True)


@cuda_required
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_rmsnorm_large_dweight_error(hidden_size: int) -> None:
    n_rows = 4096
    dtype = torch.float16
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)

    _, dweight = tritium.rmsnorm_backward(dy, x, weight)
    _, ref_dweight = _rmsnorm_backward_reference(dy, x, weight, eps=1e-6)

    _assert_close(dweight, ref_dweight, accumulated=True)


@cuda_required
@pytest.mark.parametrize("n_rows", [1024, 4096, 4097])
def test_rmsnorm_large_batch_tokens(n_rows: int) -> None:
    hidden_size = 1024
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    dy = torch.randn_like(x)

    out = tritium.rmsnorm(x, weight)
    dx, dweight = tritium.rmsnorm_backward(dy, x, weight)

    assert out.shape == x.shape
    assert dx.shape == x.shape
    assert dweight.shape == weight.shape


def test_rmsnorm_rejects_cpu_tensors() -> None:
    x = torch.randn(2, 1024)
    weight = torch.randn(1024)

    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.rmsnorm(x, weight)


@cuda_required
def test_rmsnorm_rejects_non_contiguous_input() -> None:
    x = torch.randn(1024, 16, device="cuda").T
    weight = torch.randn(1024, device="cuda")

    with pytest.raises(ValueError, match="x must be contiguous"):
        tritium.rmsnorm(x, weight)


@cuda_required
def test_rmsnorm_accepts_non_whitelisted_hidden_size() -> None:
    hidden_size = 2560
    x = torch.randn(2, hidden_size, device="cuda")
    weight = torch.randn(hidden_size, device="cuda")

    out = tritium.rmsnorm(x, weight)
    ref = _rmsnorm_reference(x, weight, eps=1e-6)

    _assert_close(out, ref)


@cuda_required
def test_rmsnorm_rejects_too_large_hidden_size() -> None:
    hidden_size = 65537
    x = torch.randn(2, hidden_size, device="cuda")
    weight = torch.randn(hidden_size, device="cuda")

    with pytest.raises(ValueError, match="Unsupported hidden size"):
        tritium.rmsnorm(x, weight)
