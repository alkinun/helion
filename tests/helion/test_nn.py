import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _utils import cuda_required

import helion


@cuda_required
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_forward_matches_torch(dtype: torch.dtype) -> None:
    layer = helion.Linear(64, 32, bias=True).to(device="cuda", dtype=dtype)
    ref = nn.Linear(64, 32).to(device="cuda", dtype=dtype)
    ref.weight.data.copy_(layer.weight.data)
    ref.bias.data.copy_(layer.bias.data)

    x = torch.randn(8, 64, device="cuda", dtype=dtype)
    torch.testing.assert_close(layer(x), ref(x), rtol=2e-2, atol=2e-2)


@cuda_required
def test_linear_autograd() -> None:
    layer = helion.Linear(32, 16, bias=False, device="cuda", dtype=torch.float32)
    x = torch.randn(4, 32, device="cuda", requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert layer.weight.grad is not None


@cuda_required
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_forward(dtype: torch.dtype) -> None:
    norm = helion.RMSNorm(64).to(device="cuda", dtype=dtype)
    x = torch.randn(8, 64, device="cuda", dtype=dtype)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(
        dtype
    )
    torch.testing.assert_close(norm(x), ref, rtol=2e-2, atol=2e-2)


@cuda_required
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layernorm_forward(dtype: torch.dtype) -> None:
    norm = helion.LayerNorm(64).to(device="cuda", dtype=dtype)
    x = torch.randn(8, 64, device="cuda", dtype=dtype)
    ref = torch.nn.functional.layer_norm(x, (64,), norm.weight, norm.bias, eps=1e-5)
    torch.testing.assert_close(norm(x), ref, rtol=2e-2, atol=2e-2)


@cuda_required
def test_layernorm_autograd() -> None:
    norm = helion.LayerNorm(64, device="cuda", dtype=torch.float32)
    x = torch.randn(8, 64, device="cuda", requires_grad=True)
    norm(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert norm.weight.grad is not None
    assert norm.bias.grad is not None


@cuda_required
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_softmax_forward(dtype: torch.dtype) -> None:
    layer = helion.Softmax().to(device="cuda", dtype=dtype)
    x = torch.randn(8, 64, device="cuda", dtype=dtype)
    ref = torch.softmax(x.float(), dim=-1).to(dtype)
    torch.testing.assert_close(layer(x), ref, rtol=2e-2, atol=2e-2)


@cuda_required
def test_softmax_autograd() -> None:
    layer = helion.Softmax().to(device="cuda", dtype=torch.float32)
    x = torch.randn(8, 64, device="cuda", requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


@cuda_required
def test_swiglu_forward() -> None:
    act = helion.SwiGLU()
    x = torch.randn(8, 32, device="cuda", dtype=torch.float32)
    gate = torch.randn(8, 32, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(act(x, gate), F.silu(x) * gate)


@cuda_required
def test_dropout_training_applies() -> None:
    layer = helion.Dropout(p=0.5).to(device="cuda", dtype=torch.float32)
    layer.train()
    x = torch.randn(8192, device="cuda", dtype=torch.float32)
    out = layer(x)
    keep_rate = (out != 0).to(torch.float32).mean()
    torch.testing.assert_close(
        keep_rate, torch.tensor(0.5, device="cuda"), rtol=0.05, atol=0.02
    )
    kept = out != 0
    torch.testing.assert_close(out[kept], x[kept] * 2.0)


@cuda_required
def test_dropout_eval_is_identity() -> None:
    layer = helion.Dropout(p=0.5).to(device="cuda", dtype=torch.float32)
    layer.eval()
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    assert layer(x) is x


@cuda_required
def test_dropout_p_zero_is_identity() -> None:
    layer = helion.Dropout(p=0.0).to(device="cuda", dtype=torch.float32)
    layer.train()
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    assert layer(x) is x


def test_dropout_rejects_invalid_p() -> None:
    with pytest.raises(ValueError, match="0 <= p < 1"):
        helion.Dropout(p=1.0)


@cuda_required
def test_residual_rmsnorm_returns_pair() -> None:
    norm = helion.ResidualRMSNorm(64).to(device="cuda", dtype=torch.float32)
    delta = torch.randn(4, 64, device="cuda")
    residual = torch.randn(4, 64, device="cuda")
    out, residual_out = norm(delta, residual)
    assert out.shape == delta.shape
    assert residual_out.shape == delta.shape
    torch.testing.assert_close(residual_out, delta + residual)


@cuda_required
def test_attention_forward_shape() -> None:
    attn = helion.Attention(hidden_size=128, n_heads=4, head_dim=32, max_seq_len=64).to(
        device="cuda", dtype=torch.float32
    )
    x = torch.randn(2, 16, 128, device="cuda")
    out = attn(x)
    assert out.shape == x.shape


@cuda_required
@pytest.mark.parametrize("batch", [1, 2])
def test_attention_forward_cached_matches_prefix(batch: int) -> None:
    torch.manual_seed(0)
    seq_len = 4
    attn = helion.Attention(hidden_size=128, n_heads=4, head_dim=32, max_seq_len=16).to(
        device="cuda", dtype=torch.float32
    )
    x = torch.randn(batch, seq_len, 128, device="cuda")
    k_cache = torch.empty(batch, 4, seq_len, 32, device="cuda")
    v_cache = torch.empty_like(k_cache)

    for pos in range(seq_len):
        out = attn.forward_cached(x[:, pos : pos + 1], k_cache, v_cache, pos)
        ref = attn(x[:, : pos + 1])[:, -1:]
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
