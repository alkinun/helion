import pytest
import torch
from _utils import DTYPES, assert_close, cuda_required

import tritium

HEAD_DIMS = [32, 64, 128]
N_TOKENS = [1, 16, 128]


def _rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_tokens = q.shape[0]
    half = q.shape[-1] // 2
    if position_ids is None:
        pos = torch.arange(n_tokens, device=q.device)
    else:
        pos = position_ids
    idx = pos.to(torch.long)
    c = cos[idx]
    s = sin[idx]
    if q.dim() == 3:
        c = c.unsqueeze(1)
        s = s.unsqueeze(1)

    def rot(t: torch.Tensor) -> torch.Tensor:
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat((t1 * c - t2 * s, t2 * c + t1 * s), dim=-1)

    return rot(q), rot(k)


def _make_cos_sin(
    max_seq_len: int, head_dim: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().contiguous(), freqs.sin().contiguous()


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("n_tokens", N_TOKENS)
def test_rope_forward_2d(dtype: torch.dtype, head_dim: int, n_tokens: int) -> None:
    q = torch.randn(n_tokens, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(n_tokens, head_dim, device="cuda", dtype=dtype)
    cos, sin = _make_cos_sin(n_tokens, head_dim, "cuda")

    q_out, k_out = tritium.rope(q, k, cos, sin)
    ref_q, ref_k = _rope_reference(q, k, cos.float(), sin.float())
    assert_close(q_out, ref_q.to(dtype))
    assert_close(k_out, ref_k.to(dtype))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_forward_multi_head(dtype: torch.dtype) -> None:
    n_tokens, n_heads, head_dim = 64, 8, 64
    q = torch.randn(n_tokens, n_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(n_tokens, n_heads, head_dim, device="cuda", dtype=dtype)
    cos, sin = _make_cos_sin(n_tokens, head_dim, "cuda")

    q_out, k_out = tritium.rope(q, k, cos, sin)
    ref_q, ref_k = _rope_reference(q, k, cos.float(), sin.float())
    assert_close(q_out, ref_q.to(dtype))
    assert_close(k_out, ref_k.to(dtype))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_uses_position_ids(dtype: torch.dtype) -> None:
    n_tokens, head_dim = 16, 64
    q = torch.randn(n_tokens, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(n_tokens, head_dim, device="cuda", dtype=dtype)
    cos, sin = _make_cos_sin(512, head_dim, "cuda")
    pos = torch.tensor(
        [10, 20, 30, 0, 5, 7, 200, 3, 99, 1, 2, 8, 4, 6, 9, 11], device="cuda"
    )

    q_out, k_out = tritium.rope(q, k, cos, sin, position_ids=pos)
    ref_q, ref_k = _rope_reference(q, k, cos.float(), sin.float(), position_ids=pos)
    assert_close(q_out, ref_q.to(dtype))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_in_place_matches_out_of_place(dtype: torch.dtype) -> None:
    n_tokens, head_dim = 32, 64
    q = torch.randn(n_tokens, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(n_tokens, head_dim, device="cuda", dtype=dtype)
    cos, sin = _make_cos_sin(n_tokens, head_dim, "cuda")

    q_ref, k_ref = tritium.rope(q, k, cos, sin)
    q_mut = q.clone()
    k_mut = k.clone()
    tritium.rope_(q_mut, k_mut, cos, sin)
    assert_close(q_mut, q_ref)
    assert_close(k_mut, k_ref)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_autograd(dtype: torch.dtype) -> None:
    n_tokens, head_dim = 16, 64
    q = torch.randn(n_tokens, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(n_tokens, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    cos, sin = _make_cos_sin(n_tokens, head_dim, "cuda")

    out = tritium.rope(q, k, cos, sin)
    out[0].sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert q.grad.shape == q.shape


def test_rope_rejects_cpu_tensors() -> None:
    q = torch.randn(4, 16)
    k = torch.randn(4, 16)
    cos = torch.randn(4, 8)
    sin = torch.randn(4, 8)
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.rope(q, k, cos, sin)


@cuda_required
def test_rope_rejects_non_contiguous() -> None:
    head_dim = 64
    q = torch.randn(head_dim, 16, device="cuda").T
    k = torch.randn(16, head_dim, device="cuda")
    cos = torch.randn(16, head_dim // 2, device="cuda")
    sin = torch.randn(16, head_dim // 2, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        tritium.rope(q, k, cos, sin)


@cuda_required
def test_rope_rejects_odd_head_dim() -> None:
    q = torch.randn(4, 33, device="cuda")
    k = torch.randn(4, 33, device="cuda")
    cos = torch.randn(4, 16, device="cuda")
    sin = torch.randn(4, 16, device="cuda")
    with pytest.raises(ValueError, match="even"):
        tritium.rope(q, k, cos, sin)
