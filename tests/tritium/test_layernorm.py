import pytest
import torch
from _utils import cuda_required

import tritium

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
HIDDEN_SIZES = [1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 8192, 11008]
N_ROWS = [1, 16, 128, 1024, 4096]


def _layernorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.float()
    mean = x_float.mean(dim=-1, keepdim=True)
    var = x_float.var(dim=-1, keepdim=True, correction=0)
    x_hat = (x_float - mean) * torch.rsqrt(var + eps)
    return (x_hat * weight.float() + bias.float()).to(x.dtype)


def _layernorm_backward_reference(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dy_float = dy.float()
    x_float = x.float()
    weight_float = weight.float()
    mean = x_float.mean(dim=-1, keepdim=True)
    var = x_float.var(dim=-1, keepdim=True, correction=0)
    rstd = torch.rsqrt(var + eps)
    x_hat = (x_float - mean) * rstd

    g = dy_float * weight_float
    c1 = g.mean(dim=-1, keepdim=True)
    c2 = (g * x_hat).mean(dim=-1, keepdim=True)
    dx = rstd * (g - c1 - x_hat * c2)
    reduce_dims = tuple(range(x.dim() - 1))
    dweight = (dy_float * x_hat).sum(dim=reduce_dims)
    dbias = dy_float.sum(dim=reduce_dims)
    return dx.to(x.dtype), dweight.to(weight.dtype), dbias.to(weight.dtype)


def _tolerances(
    dtype: torch.dtype,
    *,
    accumulated: bool = False,
) -> tuple[float, float]:
    if dtype == torch.float32:
        if accumulated:
            return 2e-4, 2e-4
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
def test_layernorm_forward_numerical_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    bias = torch.randn(hidden_size, device="cuda", dtype=dtype)

    out = tritium.layernorm(x, weight, bias)
    ref = _layernorm_reference(x, weight, bias, eps=1e-5)

    _assert_close(out, ref)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_layernorm_backward_dx_dweight_dbias_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)

    dx, dweight, dbias = tritium.layernorm_backward(dy, x, weight)
    ref_dx, ref_dweight, ref_dbias = _layernorm_backward_reference(
        dy, x, weight, eps=1e-5
    )

    _assert_close(dx, ref_dx)
    _assert_close(dweight, ref_dweight, accumulated=True)
    _assert_close(dbias, ref_dbias, accumulated=True)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", [1024, 4096, 11008])
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_layernorm_autograd_backward_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)

    out = tritium.layernorm(x, weight, bias)
    out.backward(dy)
    ref_dx, ref_dweight, ref_dbias = _layernorm_backward_reference(
        dy, x.detach(), weight.detach(), eps=1e-5
    )

    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None
    _assert_close(x.grad, ref_dx)
    _assert_close(weight.grad, ref_dweight, accumulated=True)
    _assert_close(bias.grad, ref_dbias, accumulated=True)


@cuda_required
def test_layernorm_autograd_accepts_non_contiguous_grad_output() -> None:
    hidden_size = 1024
    x = torch.randn(
        16, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(
        hidden_size, device="cuda", dtype=torch.float32, requires_grad=True
    )
    bias = torch.randn(
        hidden_size, device="cuda", dtype=torch.float32, requires_grad=True
    )
    dy = torch.randn(hidden_size, 16, device="cuda", dtype=torch.float32).T

    assert not dy.is_contiguous()

    out = tritium.layernorm(x, weight, bias)
    out.backward(dy)
    ref_dx, ref_dweight, ref_dbias = _layernorm_backward_reference(
        dy, x.detach(), weight.detach(), eps=1e-5
    )

    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None
    _assert_close(x.grad, ref_dx)
    _assert_close(weight.grad, ref_dweight, accumulated=True)
    _assert_close(bias.grad, ref_dbias, accumulated=True)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_rows", N_ROWS)
def test_layernorm_all_requested_row_counts(dtype: torch.dtype, n_rows: int) -> None:
    hidden_size = 1024
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    bias = torch.randn(hidden_size, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)

    out = tritium.layernorm(x, weight, bias)
    dx, dweight, dbias = tritium.layernorm_backward(dy, x, weight)
    ref = _layernorm_reference(x, weight, bias, eps=1e-5)
    ref_dx, ref_dweight, ref_dbias = _layernorm_backward_reference(
        dy, x, weight, eps=1e-5
    )

    _assert_close(out, ref)
    _assert_close(dx, ref_dx)
    _assert_close(dweight, ref_dweight, accumulated=True)
    _assert_close(dbias, ref_dbias, accumulated=True)


@cuda_required
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_layernorm_large_grad_error(hidden_size: int) -> None:
    n_rows = 4096
    dtype = torch.float16
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)

    _, dweight, dbias = tritium.layernorm_backward(dy, x, weight)
    _, ref_dweight, ref_dbias = _layernorm_backward_reference(dy, x, weight, eps=1e-5)

    _assert_close(dweight, ref_dweight, accumulated=True)
    _assert_close(dbias, ref_dbias, accumulated=True)


@cuda_required
@pytest.mark.parametrize("n_rows", [1024, 4096, 4097])
def test_layernorm_large_batch_tokens(n_rows: int) -> None:
    hidden_size = 1024
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    bias = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    dy = torch.randn_like(x)

    out = tritium.layernorm(x, weight, bias)
    dx, dweight, dbias = tritium.layernorm_backward(dy, x, weight)

    assert out.shape == x.shape
    assert dx.shape == x.shape
    assert dweight.shape == weight.shape
    assert dbias.shape == bias.shape


@cuda_required
def test_layernorm_matches_torch_functional() -> None:
    hidden_size = 1024
    x = torch.randn(64, hidden_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(hidden_size, device="cuda", dtype=torch.float32)

    out = tritium.layernorm(x, weight, bias)
    ref = torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias, eps=1e-5)
    _assert_close(out, ref)


@cuda_required
def test_layernorm_only_needs_dx() -> None:
    hidden_size = 1024
    x = torch.randn(
        128, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True
    )
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    dy = torch.randn_like(x)

    out = tritium.layernorm(x, weight, bias)
    out.backward(dy)
    ref_dx, _, _ = _layernorm_backward_reference(dy, x.detach(), weight, eps=1e-5)

    assert x.grad is not None
    _assert_close(x.grad, ref_dx)


def test_layernorm_rejects_cpu_tensors() -> None:
    x = torch.randn(2, 1024)
    weight = torch.randn(1024)
    bias = torch.randn(1024)

    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.layernorm(x, weight, bias)


@cuda_required
def test_layernorm_rejects_non_contiguous_input() -> None:
    x = torch.randn(1024, 16, device="cuda").T
    weight = torch.randn(1024, device="cuda")
    bias = torch.randn(1024, device="cuda")

    with pytest.raises(ValueError, match="x must be contiguous"):
        tritium.layernorm(x, weight, bias)


@cuda_required
def test_layernorm_rejects_param_shape_mismatch() -> None:
    hidden_size = 1024
    x = torch.randn(2, hidden_size, device="cuda")
    weight = torch.randn(hidden_size - 1, device="cuda")
    bias = torch.randn(hidden_size, device="cuda")

    with pytest.raises(ValueError, match="Shape mismatch"):
        tritium.layernorm(x, weight, bias)


@cuda_required
def test_layernorm_accepts_non_whitelisted_hidden_size() -> None:
    hidden_size = 2560
    x = torch.randn(2, hidden_size, device="cuda")
    weight = torch.randn(hidden_size, device="cuda")
    bias = torch.randn(hidden_size, device="cuda")

    out = tritium.layernorm(x, weight, bias)
    ref = _layernorm_reference(x, weight, bias, eps=1e-5)

    _assert_close(out, ref)


@cuda_required
def test_layernorm_rejects_too_large_hidden_size() -> None:
    hidden_size = 65537
    x = torch.randn(2, hidden_size, device="cuda")
    weight = torch.randn(hidden_size, device="cuda")
    bias = torch.randn(hidden_size, device="cuda")

    with pytest.raises(ValueError, match="Unsupported hidden size"):
        tritium.layernorm(x, weight, bias)
