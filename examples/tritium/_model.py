"""Shared model and op backends for the Tritium examples.

``TinyLM`` is a small Llama-style decoder. It is parameterized by a *backend*
that supplies the norm/rope/activation/loss ops, so the same architecture can be
run with either Tritium kernels or vanilla PyTorch for an apples-to-apples speed
comparison. ``TritiumBackend`` and ``TorchBackend`` are functionally equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import tritium

EPS = 1e-6


@dataclass
class Config:
    vocab_size: int = 256
    n_heads: int = 4
    head_dim: int = 32
    d_ff: int = 512
    n_layers: int = 2
    max_seq_len: int = 128

    @property
    def hidden(self) -> int:
        return self.n_heads * self.head_dim


class TritiumBackend:
    @staticmethod
    def residual_rmsnorm(
        delta: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return tritium.residual_rmsnorm(delta, residual, weight, eps)

    @staticmethod
    def attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True
    ) -> torch.Tensor:
        return tritium.attention(q, k, v, is_causal=is_causal)

    @staticmethod
    def rope(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return tritium.rope(q, k, cos, sin, position_ids=pos)

    @staticmethod
    def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return tritium.swiglu(gate, up)

    @staticmethod
    def linear_cross_entropy(
        hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return tritium.linear_cross_entropy(hidden, weight, target)


class TorchBackend:
    @staticmethod
    def residual_rmsnorm(
        delta: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_out = delta + residual
        normed = F.rms_norm(residual_out, (residual_out.shape[-1],), weight, eps)
        return normed, residual_out

    @staticmethod
    def attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    @staticmethod
    def rope(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = cos[pos].unsqueeze(1)
        s = sin[pos].unsqueeze(1)
        half = q.shape[-1] // 2

        def rot(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)

        return rot(q), rot(k)

    @staticmethod
    def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up

    @staticmethod
    def linear_cross_entropy(
        hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        logits = F.linear(hidden, weight)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), target.reshape(-1)
        )


class TritiumAdamW:
    """AdamW backed by ``tritium.adamw_step`` (per-parameter, in-place)."""

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {
            p: (
                torch.zeros_like(p, dtype=torch.float32),
                torch.zeros_like(p, dtype=torch.float32),
            )
            for p in self.params
        }
        self._step = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self._step += 1
        for p in self.params:
            if p.grad is None:
                continue
            exp_avg, exp_avg_sq = self.state[p]
            if not p.is_contiguous():
                p.data = p.data.contiguous()
            tritium.adamw_step(
                p.data,
                p.grad,
                exp_avg,
                exp_avg_sq,
                lr=self.lr,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
                weight_decay=self.weight_decay,
                step=self._step,
            )


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config, backend) -> None:
        super().__init__()
        h, nh, hd, ff = cfg.hidden, cfg.n_heads, cfg.head_dim, cfg.d_ff
        self.attn_norm = nn.Parameter(torch.ones(h))
        self.ffn_norm = nn.Parameter(torch.ones(h))
        self.wq = nn.Linear(h, nh * hd, bias=False)
        self.wk = nn.Linear(h, nh * hd, bias=False)
        self.wv = nn.Linear(h, nh * hd, bias=False)
        self.wo = nn.Linear(nh * hd, h, bias=False)
        self.w_gate = nn.Linear(h, ff, bias=False)
        self.w_up = nn.Linear(h, ff, bias=False)
        self.w_down = nn.Linear(ff, h, bias=False)

        half = hd // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half).float() / half))
        t = torch.arange(cfg.max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)
        self.n_heads = nh
        self.head_dim = hd
        self.b = backend

    def _attention(self, normed: torch.Tensor, seqlen: int) -> torch.Tensor:
        b, s, _ = normed.shape
        nh, hd = self.n_heads, self.head_dim
        q = self.wq(normed).view(b, s, nh, hd)
        k = self.wk(normed).view(b, s, nh, hd)
        v = self.wv(normed).view(b, s, nh, hd)

        q_flat = q.reshape(b * s, nh, hd).contiguous()
        k_flat = k.reshape(b * s, nh, hd).contiguous()
        pos = torch.arange(seqlen, device=normed.device).repeat(b)
        q_rot, k_rot = self.b.rope(
            q_flat, k_flat, self.rope_cos[:seqlen], self.rope_sin[:seqlen], pos
        )

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, s, nh, hd).transpose(1, 2)

        attn = self.b.attention(
            to_heads(q_rot), to_heads(k_rot), to_heads(v), is_causal=True
        )
        attn = attn.transpose(1, 2).reshape(b, s, nh * hd).contiguous()
        return self.wo(attn)

    def _ffn(self, normed: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(normed)
        up = self.w_up(normed)
        return self.w_down(self.b.swiglu(gate, up))

    def forward(
        self, delta: torch.Tensor, residual: torch.Tensor, seqlen: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed, residual = self.b.residual_rmsnorm(delta, residual, self.attn_norm, EPS)
        attn_out = self._attention(normed, seqlen)
        normed, residual = self.b.residual_rmsnorm(
            attn_out, residual, self.ffn_norm, EPS
        )
        ffn_out = self._ffn(normed)
        return ffn_out, residual


class TinyLM(nn.Module):
    def __init__(self, cfg: Config, backend, dtype: torch.dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, backend) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.Parameter(torch.ones(cfg.hidden))
        self.lm_head = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden))
        nn.init.normal_(self.lm_head, std=0.02)
        self.b = backend
        self.to(dtype)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        seqlen = tokens.shape[1]
        x = self.embed(tokens)

        residual = torch.zeros_like(x)
        delta = x
        for block in self.blocks:
            delta, residual = block(delta, residual, seqlen)

        hidden, _ = self.b.residual_rmsnorm(delta, residual, self.final_norm, EPS)
        return self.b.linear_cross_entropy(hidden, self.lm_head, targets)


def make_batch(
    cfg: Config, batch: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(cfg.vocab_size, (batch, cfg.max_seq_len), device=device)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
