"""Compare whole training-step speed across backends.

Same model architecture in every backend; only the primitives differ. Times a
full training step (forward + backward + optimizer). Answers: how does building
the step out of Tritium kernels (via Helion modules) compare to PyTorch eager
and torch.compile?

    python examples/train_compare.py
    python examples/train_compare.py --dtype bfloat16 --layers 6
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton.testing

import helion
import tritium


@dataclasses.dataclass
class Config:
    vocab_size: int = 4096
    n_heads: int = 8
    head_dim: int = 64
    d_ff: int = 1408
    n_layers: int = 4
    max_seq_len: int = 128

    @property
    def hidden(self) -> int:
        return self.n_heads * self.head_dim


class HelionBlock(nn.Module):
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

    def forward(self, delta, residual):
        normed, residual = self.attn_norm(delta, residual)
        delta = self.attn(normed)
        normed, residual = self.ffn_norm(delta, residual)
        return self.w_down(self.act(self.w_gate(normed), self.w_up(normed))), residual


class HelionModel(nn.Module):
    def __init__(self, cfg: Config, dtype: torch.dtype) -> None:
        super().__init__()
        self.embed = helion.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList([HelionBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = helion.ResidualRMSNorm(cfg.hidden)
        self.lm_head = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden))
        nn.init.normal_(self.lm_head, std=0.02)
        self.to(dtype)

    def forward(self, tokens, targets):
        x = self.embed(tokens)
        residual = torch.zeros_like(x)
        delta = x
        for block in self.blocks:
            delta, residual = block(delta, residual)
        hidden, _ = self.final_norm(delta, residual)
        return tritium.linear_cross_entropy(hidden, self.lm_head, targets)


class TorchBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        h = cfg.hidden
        self.attn_norm = nn.RMSNorm(h)
        self.ffn_norm = nn.RMSNorm(h)
        self.wq = nn.Linear(h, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(h, cfg.n_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(h, cfg.n_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, h, bias=False)
        self.w_gate = nn.Linear(h, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(h, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, h, bias=False)
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim

    def forward(self, x):
        b, s, _ = x.shape
        nh, hd = self.n_heads, self.head_dim
        normed = self.attn_norm(x)

        q = self.wq(normed).view(b, s, nh, hd).transpose(1, 2)
        k = self.wk(normed).view(b, s, nh, hd).transpose(1, 2)
        v = self.wv(normed).view(b, s, nh, hd).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(b, s, nh * hd)
        delta = self.wo(attn)

        normed = self.ffn_norm(delta)
        gate = self.w_gate(normed)
        up = self.w_up(normed)
        return self.w_down(F.silu(gate) * up)


class TorchModel(nn.Module):
    def __init__(self, cfg: Config, dtype: torch.dtype) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList([TorchBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.hidden)
        self.lm_head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        self.to(dtype)

    def forward(self, tokens, targets):
        x = self.embed(tokens)
        for block in self.blocks:
            x = x + block(x)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def bench_step(model, opt, tokens, targets, repeats=5) -> float:
    def step():
        opt.zero_grad()
        model(tokens, targets).backward()
        opt.step()

    step()
    torch.cuda.synchronize()
    times = (float(triton.testing.do_bench(step)) for _ in range(repeats))
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=1408)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    cfg = Config(
        vocab_size=args.vocab,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        d_ff=args.d_ff,
        n_layers=args.layers,
        max_seq_len=args.seq,
    )
    torch.manual_seed(123)
    tokens = torch.randint(cfg.vocab_size, (args.batch, args.seq), device=device)
    targets = torch.randint(cfg.vocab_size, (args.batch, args.seq), device=device)

    h = cfg.hidden
    n_params = (
        args.layers * (2 * h + 4 * h * h + 3 * h * args.d_ff) + cfg.vocab_size * h
    )
    print(
        f"model: {args.layers}L d={h} ff={args.d_ff} | "
        f"batch×seq={args.batch}×{args.seq} vocab={args.vocab} | {args.dtype}"
    )
    print(f"~{n_params / 1e6:.1f}M params\n")

    results = {}

    torch.manual_seed(0)
    model = HelionModel(cfg, dtype).to(device)
    opt = helion.AdamW(model.parameters(), lr=args.lr)
    ms = bench_step(model, opt, tokens, targets)
    results["helion"] = ms
    print(f"  {'helion':14s}  {ms:8.3f} ms/step")

    torch.manual_seed(0)
    model = TorchModel(cfg, dtype).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ms = bench_step(model, opt, tokens, targets)
    results["torch eager"] = ms
    print(f"  {'torch eager':14s}  {ms:8.3f} ms/step")

    torch.manual_seed(0)
    model = torch.compile(TorchModel(cfg, dtype).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ms = bench_step(model, opt, tokens, targets)
    results["torch.compile"] = ms
    print(f"  {'torch.compile':14s}  {ms:8.3f} ms/step")

    base = results["torch eager"]
    print("\nspeedup vs torch eager:")
    for name, ms in results.items():
        print(f"  {name:14s}  {base / ms:5.2f}x")


if __name__ == "__main__":
    main()
