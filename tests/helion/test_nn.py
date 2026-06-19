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
def test_swiglu_forward() -> None:
    act = helion.SwiGLU()
    x = torch.randn(8, 32, device="cuda", dtype=torch.float32)
    gate = torch.randn(8, 32, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(act(x, gate), F.silu(x) * gate)


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
