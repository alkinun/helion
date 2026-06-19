import pytest
import torch
from _utils import cuda_required

import tritium

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
HIDDEN_SIZES = [1, 2, 16, 128, 1024, 1536, 2048, 4096, 8192, 11008]
N_ROWS = [1, 16, 128, 1024, 4096]


def _softmax_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def _softmax_backward_reference(
    dy: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    dy_float = dy.float()
    out_float = out.float()
    c = (dy_float * out_float).sum(dim=-1, keepdim=True)
    return (out_float * (dy_float - c)).to(out.dtype)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    return 2e-2, 2e-2


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    rtol, atol = _tolerances(actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_softmax_forward_numerical_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    out = tritium.softmax(x)
    ref = _softmax_reference(x)
    _assert_close(out, ref)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", [16, 1024, 4096, 11008])
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_softmax_backward_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)
    out = tritium.softmax(x)
    dx = tritium.softmax_backward(dy, out)

    ref_dx = _softmax_backward_reference(dy, out)
    _assert_close(dx, ref_dx)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden_size", [1024, 4096, 11008])
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_softmax_autograd_backward_error(
    dtype: torch.dtype,
    hidden_size: int,
    n_rows: int,
) -> None:
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)

    out = tritium.softmax(x)
    out.backward(dy)
    ref_dx = _softmax_backward_reference(dy, out.detach())

    assert x.grad is not None
    _assert_close(x.grad, ref_dx)


@cuda_required
def test_softmax_autograd_accepts_non_contiguous_grad_output() -> None:
    hidden_size = 1024
    x = torch.randn(
        16, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True
    )
    dy = torch.randn(hidden_size, 16, device="cuda", dtype=torch.float32).T

    assert not dy.is_contiguous()

    out = tritium.softmax(x)
    out.backward(dy)
    ref_dx = _softmax_backward_reference(dy, out.detach())

    assert x.grad is not None
    _assert_close(x.grad, ref_dx)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_rows", N_ROWS)
def test_softmax_all_requested_row_counts(dtype: torch.dtype, n_rows: int) -> None:
    hidden_size = 1024
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)

    out = tritium.softmax(x)
    dx = tritium.softmax_backward(dy, out)
    _assert_close(out, _softmax_reference(x))
    _assert_close(dx, _softmax_backward_reference(dy, out))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_rows_sum_to_one_and_nonneg(dtype: torch.dtype) -> None:
    x = torch.randn(128, 1024, device="cuda", dtype=dtype)
    out = tritium.softmax(x)
    row_sums = out.float().sum(dim=-1)
    torch.testing.assert_close(
        row_sums, torch.ones_like(row_sums), rtol=1e-3, atol=1e-3
    )
    assert bool((out.float() >= 0).all())


@cuda_required
def test_softmax_matches_torch_functional() -> None:
    hidden_size = 1024
    x = torch.randn(64, hidden_size, device="cuda", dtype=torch.float32)
    out = tritium.softmax(x)
    ref = torch.softmax(x, dim=-1)
    _assert_close(out, ref)


@cuda_required
@pytest.mark.parametrize("n_rows", [1024, 4096, 4097])
def test_softmax_large_batch_tokens(n_rows: int) -> None:
    hidden_size = 1024
    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    dy = torch.randn_like(x)

    out = tritium.softmax(x)
    dx = tritium.softmax_backward(dy, out)

    assert out.shape == x.shape
    assert dx.shape == x.shape


@cuda_required
def test_softmax_numerical_stability_large_values() -> None:
    x = torch.full((16, 1024), 1000.0, device="cuda", dtype=torch.float32)
    x += torch.randn_like(x)
    out = tritium.softmax(x)
    assert bool(torch.isfinite(out).all())
    row_sums = out.sum(dim=-1)
    torch.testing.assert_close(
        row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-4
    )


def test_softmax_rejects_cpu_tensors() -> None:
    x = torch.randn(2, 1024)
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.softmax(x)


@cuda_required
def test_softmax_rejects_non_contiguous_input() -> None:
    x = torch.randn(1024, 16, device="cuda").T
    with pytest.raises(ValueError, match="x must be contiguous"):
        tritium.softmax(x)


@cuda_required
def test_softmax_accepts_non_whitelisted_hidden_size() -> None:
    hidden_size = 2560
    x = torch.randn(2, hidden_size, device="cuda")
    out = tritium.softmax(x)
    _assert_close(out, _softmax_reference(x))


@cuda_required
def test_softmax_rejects_too_large_hidden_size() -> None:
    hidden_size = 65537
    x = torch.randn(2, hidden_size, device="cuda")
    with pytest.raises(ValueError, match="Unsupported hidden size"):
        tritium.softmax(x)
