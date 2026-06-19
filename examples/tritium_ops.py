"""Use Tritium kernels directly.

Helion is the friendly layer for models. Tritium is still public when you want
the raw fused primitives: elementwise ops, matmul, normalization, attention,
RoPE, losses, dropout, and optimizer update kernels.

Run:
    python examples/tritium_ops.py
"""

from __future__ import annotations

import torch
from common import require_cuda

import tritium


def main() -> None:
    device = require_cuda()
    dtype = torch.bfloat16
    torch.manual_seed(0)

    x = torch.randn(32, 64, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn_like(x)
    w = torch.randn(64, 96, device=device, dtype=dtype, requires_grad=True)
    norm_weight = torch.ones(64, device=device, dtype=dtype, requires_grad=True)
    norm_bias = torch.zeros(64, device=device, dtype=dtype, requires_grad=True)

    summed = tritium.add(x, y)
    activated = tritium.add_gelu(summed, y)
    logits = tritium.matmul(activated, w)
    targets = torch.randint(logits.shape[-1], (logits.shape[0],), device=device)
    loss = tritium.cross_entropy(logits, targets)
    loss.backward()

    normalized = tritium.layernorm(x.detach(), norm_weight, norm_bias)
    rms = tritium.rmsnorm(x.detach(), norm_weight)
    residual_normed, residual = tritium.residual_rmsnorm(x.detach(), y, norm_weight)
    probs = tritium.softmax(logits.detach())
    dropped = tritium.dropout(x.detach(), p=0.2)
    swiglu = tritium.swiglu(x.detach(), y)
    relu = tritium.relu(x.detach())
    add_relu = tritium.add_relu(x.detach(), y)

    batch, heads, seq, head_dim = 2, 4, 16, 32
    q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    attn = tritium.attention(q, k, v, is_causal=True)

    flat_q = q.transpose(1, 2).reshape(batch * seq, heads, head_dim).contiguous()
    flat_k = k.transpose(1, 2).reshape(batch * seq, heads, head_dim).contiguous()
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device) / half))
    freqs = torch.outer(torch.arange(seq, device=device), inv_freq)
    position_ids = torch.arange(seq, device=device).repeat(batch)
    q_rope, k_rope = tritium.rope(
        flat_q,
        flat_k,
        freqs.cos(),
        freqs.sin(),
        position_ids=position_ids,
    )

    hidden = torch.randn(4, 8, 64, device=device, dtype=dtype, requires_grad=True)
    lm_head = torch.randn(128, 64, device=device, dtype=dtype, requires_grad=True)
    lm_targets = torch.randint(128, (4, 8), device=device)
    fused_loss = tritium.linear_cross_entropy(hidden, lm_head, lm_targets)
    fused_loss.backward()

    param = torch.randn(128, device=device, dtype=dtype)
    grad = torch.randn_like(param)
    exp_avg = torch.zeros_like(param, dtype=torch.float32)
    exp_avg_sq = torch.zeros_like(param, dtype=torch.float32)
    tritium.sgd_step(param, grad, lr=1e-2)
    tritium.adamw_step(param, grad, exp_avg, exp_avg_sq, lr=1e-3, step=1)

    print(f"matmul logits:          {tuple(logits.shape)}")
    print(f"attention:              {tuple(attn.shape)}")
    print(f"rope q/k:               {tuple(q_rope.shape)} / {tuple(k_rope.shape)}")
    print(f"cross entropy:          {loss.item():.4f}")
    print(f"linear cross entropy:   {fused_loss.item():.4f}")
    print(f"normalization outputs:  {normalized.shape}, {rms.shape}, {residual.shape}")
    print(f"elementwise outputs:    {relu.shape}, {add_relu.shape}, {swiglu.shape}")
    print(f"dropout/probs checksum: {dropped.float().mean():.4f}, {probs[0].sum():.4f}")


if __name__ == "__main__":
    main()
