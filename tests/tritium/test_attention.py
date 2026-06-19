import pytest
import torch
import torch.nn.functional as F
from _utils import DTYPES, assert_close, cuda_required

import tritium


def _attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    n_heads = q.shape[1]
    n_kv = k.shape[1]
    if n_kv != n_heads:
        group = n_heads // n_kv
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [16, 32, 64])
@pytest.mark.parametrize(("batch", "n_heads", "seq"), [(1, 4, 64), (2, 8, 128)])
def test_attention_forward(
    dtype: torch.dtype,
    head_dim: int,
    batch: int,
    n_heads: int,
    seq: int,
) -> None:
    q = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=dtype)

    out = tritium.attention(q, k, v, is_causal=True)
    ref = _attention_reference(q, k, v, is_causal=True)
    assert_close(out, ref)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_attention_grouped_query(dtype: torch.dtype) -> None:
    batch, n_heads, n_kv, seq, head_dim = 2, 8, 2, 128, 64
    q = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, n_kv, seq, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, n_kv, seq, head_dim, device="cuda", dtype=dtype)

    out = tritium.attention(q, k, v, is_causal=True)
    ref = _attention_reference(q, k, v, is_causal=True)
    assert_close(out, ref)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_attention_non_causal(dtype: torch.dtype) -> None:
    batch, n_heads, seq, head_dim = 1, 4, 64, 32
    q = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=dtype)

    out = tritium.attention(q, k, v, is_causal=False)
    ref = _attention_reference(q, k, v, is_causal=False)
    assert_close(out, ref)


@cuda_required
def test_attention_kv_cache_single_query() -> None:
    batch, n_heads, kv_len, head_dim = 1, 4, 64, 32
    k = torch.randn(batch, n_heads, kv_len, head_dim, device="cuda")
    v = torch.randn(batch, n_heads, kv_len, head_dim, device="cuda")
    q = k[:, :, -1:, :].contiguous()

    out = tritium.attention(q, k, v, is_causal=False)
    ref = _attention_reference(q, k, v, is_causal=False)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_attention_autograd(dtype: torch.dtype) -> None:
    batch, n_heads, seq, head_dim = 1, 4, 64, 32
    q = torch.randn(
        batch, n_heads, seq, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch, n_heads, seq, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch, n_heads, seq, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )
    grad = torch.randn(batch, n_heads, seq, head_dim, device="cuda", dtype=dtype)

    out = tritium.attention(q, k, v, is_causal=True)
    out.backward(grad)

    q_ref = q.detach().float().clone().requires_grad_(True)
    k_ref = k.detach().float().clone().requires_grad_(True)
    v_ref = v.detach().float().clone().requires_grad_(True)
    _attention_reference(q_ref, k_ref, v_ref, is_causal=True).backward(grad.float())

    assert_close(q.grad, q_ref.grad.to(dtype))
    assert_close(k.grad, k_ref.grad.to(dtype))
    assert_close(v.grad, v_ref.grad.to(dtype))


def test_attention_rejects_cpu_tensors() -> None:
    q = torch.randn(1, 2, 4, 16)
    k = torch.randn(1, 2, 4, 16)
    v = torch.randn(1, 2, 4, 16)
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.attention(q, k, v)


@cuda_required
def test_attention_rejects_non_power_of_two_head_dim() -> None:
    q = torch.randn(1, 2, 4, 24, device="cuda")
    k = torch.randn(1, 2, 4, 24, device="cuda")
    v = torch.randn(1, 2, 4, 24, device="cuda")
    with pytest.raises(ValueError, match="power of 2"):
        tritium.attention(q, k, v)


@cuda_required
def test_attention_rejects_undivisible_heads() -> None:
    q = torch.randn(1, 8, 4, 16, device="cuda")
    k = torch.randn(1, 3, 4, 16, device="cuda")
    v = torch.randn(1, 3, 4, 16, device="cuda")
    with pytest.raises(ValueError, match="divisible"):
        tritium.attention(q, k, v)
