import pytest
import torch
from _utils import DTYPES, assert_close, cuda_required

import tritium

P_VALUES = [0.1, 0.5, 0.9]


def _scale(p: float) -> float:
    return 1.0 / (1.0 - p)


def _kept_mask(out: torch.Tensor) -> torch.Tensor:
    return out != 0


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", P_VALUES)
def test_dropout_forward_value_correctness(dtype: torch.dtype, p: float) -> None:
    n = 8192
    x = torch.randn(n, device="cuda", dtype=dtype)
    out = tritium.dropout(x, p, seed=0)

    kept = _kept_mask(out)
    assert_close(out[kept], (x[kept].float() * _scale(p)).to(dtype))

    keep_rate = kept.to(torch.float32).mean()
    torch.testing.assert_close(
        keep_rate, torch.tensor(1.0 - p, device="cuda"), rtol=0.05, atol=0.02
    )


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", P_VALUES)
def test_dropout_preserves_mean(dtype: torch.dtype, p: float) -> None:
    n = 1 << 18
    x = torch.randn(n, device="cuda", dtype=dtype)
    out = tritium.dropout(x, p, seed=1)
    torch.testing.assert_close(
        out.float().mean(),
        x.float().mean(),
        rtol=0.02,
        atol=0.02,
    )


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 16, 1024, 8193])
def test_dropout_forward_shapes_and_mask(dtype: torch.dtype, n: int) -> None:
    x = torch.randn(n, device="cuda", dtype=dtype)
    out = tritium.dropout(x, 0.5, seed=42)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    kept = _kept_mask(out)
    assert_close(out[kept], (x[kept].float() * _scale(0.5)).to(dtype))
    dropped = ~kept
    assert bool((out[dropped] == 0).all())


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("p", P_VALUES)
def test_dropout_backward(dtype: torch.dtype, p: float) -> None:
    n = 8192
    x = torch.randn(n, device="cuda", dtype=dtype)
    dy = torch.randn(n, device="cuda", dtype=dtype)
    seed = 7

    out = tritium.dropout(x, p, seed=seed)
    dx = tritium.dropout_backward(dy, p, seed)

    kept = _kept_mask(out)
    expected = torch.where(kept, dy.float() * _scale(p), torch.zeros(())).to(dy.dtype)
    assert_close(dx, expected)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_dropout_autograd(dtype: torch.dtype) -> None:
    n = 4096
    x = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(n, device="cuda", dtype=dtype)
    out = tritium.dropout(x, 0.5, seed=99)
    out.backward(dy)

    assert x.grad is not None
    assert x.grad.shape == x.shape
    kept = _kept_mask(out.detach())
    expected = torch.where(kept, dy.float() * _scale(0.5), torch.zeros(())).to(dy.dtype)
    assert_close(x.grad, expected)


@cuda_required
def test_dropout_autograd_drops_have_zero_grad() -> None:
    x = torch.randn(8192, device="cuda", dtype=torch.float32, requires_grad=True)
    out = tritium.dropout(x, 0.5, seed=3)
    out.sum().backward()
    assert x.grad is not None
    scale = _scale(0.5)
    kept = out.detach() != 0
    assert_close(x.grad, torch.where(kept, torch.full((), scale), torch.zeros(())))


@cuda_required
def test_dropout_deterministic_with_seed() -> None:
    x = torch.randn(2048, device="cuda", dtype=torch.float32)
    a = tritium.dropout(x, 0.3, seed=123)
    b = tritium.dropout(x, 0.3, seed=123)
    assert bool(torch.equal(a, b))


@cuda_required
def test_dropout_different_seeds_differ() -> None:
    x = torch.randn(8192, device="cuda", dtype=torch.float32)
    a = tritium.dropout(x, 0.5, seed=1)
    b = tritium.dropout(x, 0.5, seed=2)
    assert not bool(torch.equal(a, b))


@cuda_required
def test_dropout_p_zero_is_identity() -> None:
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    out = tritium.dropout(x, 0.0)
    assert out is x


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_dropout_empty_tensor(dtype: torch.dtype) -> None:
    x = torch.randn(0, device="cuda", dtype=dtype)
    out = tritium.dropout(x, 0.5, seed=0)
    assert out.shape == (0,)


@cuda_required
def test_dropout_global_seed_reproducible() -> None:
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    torch.manual_seed(0)
    a = tritium.dropout(x, 0.5)
    torch.manual_seed(0)
    b = tritium.dropout(x, 0.5)
    assert bool(torch.equal(a, b))


@cuda_required
def test_dropout_accepts_non_contiguous_input() -> None:
    x = torch.randn(1024, 16, device="cuda", dtype=torch.float32).T
    assert not x.is_contiguous()
    out = tritium.dropout(x, 0.5, seed=4)
    assert out.shape == x.shape
    assert out.is_contiguous()
    kept = _kept_mask(out)
    assert_close(out[kept], x.contiguous()[kept] * _scale(0.5))


def test_dropout_rejects_cpu_tensors() -> None:
    x = torch.randn(8)
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.dropout(x, 0.5, seed=0)


@pytest.mark.parametrize("bad_p", [-0.1, 1.0, 1.5])
def test_dropout_rejects_invalid_p(bad_p: float) -> None:
    x = torch.randn(16, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="0 <= p < 1"):
        tritium.dropout(x, bad_p, seed=0)
