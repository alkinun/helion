"""Quick tour of Helion modules and training utilities.

Run:
    python examples/modules_quickstart.py
"""

from __future__ import annotations

import torch
from common import require_cuda

import helion


def main() -> None:
    device = require_cuda()
    helion.seed_all(123)
    dtype = torch.bfloat16

    x = torch.randn(4, 16, 64, device=device, dtype=dtype, requires_grad=True)
    residual = torch.zeros_like(x)

    linear = helion.Linear(64, 128, bias=True, device=device, dtype=dtype)
    rms = helion.RMSNorm(64, device=device, dtype=dtype)
    layer_norm = helion.LayerNorm(64, device=device, dtype=dtype)
    residual_norm = helion.ResidualRMSNorm(64, device=device, dtype=dtype)
    softmax = helion.Softmax()
    swiglu = helion.SwiGLU()
    dropout = helion.Dropout(p=0.1)
    attention = helion.Attention(64, n_heads=4, head_dim=16, max_seq_len=16).to(
        device=device,
        dtype=dtype,
    )

    normed = rms(x)
    normed = layer_norm(normed)
    attn_out = attention(normed)
    gate, up = linear(attn_out).chunk(2, dim=-1)
    activated = swiglu(gate, up)
    out, updated_residual = residual_norm(dropout(activated), residual)
    probs = softmax(out.float())
    loss = probs.square().mean()
    loss.backward()

    params = list(linear.parameters())
    opt = helion.SGD(params, lr=1e-2)
    grad_norm = helion.clip_grad_norm(params, max_norm=1.0)
    opt.step()

    print(f"attention output: {tuple(attn_out.shape)}")
    print(f"residual output:  {tuple(updated_residual.shape)}")
    print(f"loss: {loss.item():.6f} grad_norm: {grad_norm.item():.4f}")


if __name__ == "__main__":
    main()
