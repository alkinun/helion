"""End-to-end training reference for Helion.

A tiny Llama-style decoder built entirely from Helion modules and Tritium fused
ops. Every compute operation — matmul, attention, rmsnorm, swiglu, rope,
cross-entropy, optimizer — runs through a Triton kernel. No PyTorch compute ops.

Run:
    python examples/train_reference.py
    python examples/train_reference.py --steps 20 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import dataclasses

import torch
import torch.nn as nn

import helion
import tritium


@dataclasses.dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    head_dim: int = 32
    d_ff: int = 512
    n_layers: int = 2
    max_seq_len: int = 128

    @property
    def hidden(self) -> int:
        return self.n_heads * self.head_dim


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        h = cfg.hidden
        self.attn_norm = helion.ResidualRMSNorm(h)
        self.ffn_norm = helion.ResidualRMSNorm(h)
        self.attn = helion.Attention(h, cfg.n_heads, cfg.head_dim, cfg.max_seq_len)
        self.w_gate = helion.Linear(h, cfg.d_ff, bias=False)
        self.w_up = helion.Linear(h, cfg.d_ff, bias=False)
        self.w_down = helion.Linear(cfg.d_ff, h, bias=False)
        self.act = helion.SwiGLU()

    def forward(
        self,
        delta: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed, residual = self.attn_norm(delta, residual)
        attn_out = self.attn(normed)
        normed, residual = self.ffn_norm(attn_out, residual)
        ffn_out = self.w_down(self.act(self.w_gate(normed), self.w_up(normed)))
        return ffn_out, residual


class TinyLM(nn.Module):
    def __init__(self, cfg: Config, dtype: torch.dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = helion.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = helion.ResidualRMSNorm(cfg.hidden)
        self.lm_head = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden))
        nn.init.normal_(self.lm_head, std=0.02)
        self.to(dtype)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)

        residual = torch.zeros_like(x)
        delta = x
        for block in self.blocks:
            delta, residual = block(delta, residual)

        hidden, _ = self.final_norm(delta, residual)
        return tritium.linear_cross_entropy(hidden, self.lm_head, targets)


def make_batch(
    cfg: Config,
    batch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(cfg.vocab_size, (batch, cfg.max_seq_len), device=device)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    cfg = Config()

    model = TinyLM(cfg, dtype=dtype).to(device=device, dtype=dtype)
    opt = helion.AdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Helion training reference: {cfg.n_layers} layers, d={cfg.hidden}, "
        f"{args.dtype}, {n_params} params"
    )

    model.train()
    tokens, targets = make_batch(cfg, args.batch, device)
    losses: list[float] = []
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        loss = model(tokens, targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"  step {step:2d}  loss {losses[-1]:.4f}")

    if len(losses) >= 5:
        first, last = losses[0], losses[-1]
        chance = float(torch.log(torch.tensor(float(cfg.vocab_size))))
        print(f"\nloss: start {first:.4f} -> end {last:.4f} (chance ~ {chance:.3f})")
        if last < first - 0.5:
            print("LOSS DECREASED: Helion components compose correctly.")
        else:
            print("WARNING: loss did not decrease enough to confirm learning.")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
