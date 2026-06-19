"""Compare full training-step speed across Helion and PyTorch backends.

Times forward, backward, and optimizer update for a small decoder-only language
model. Helion uses Helion modules plus Tritium fused loss and optimizer; the
PyTorch baselines use equivalent eager and optional torch.compile modules.

Run:
    python examples/train_compare.py
    python examples/train_compare.py --layers 6 --hidden 512 --compile
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton.testing
from common import (
    TinyLanguageModel,
    TransformerConfig,
    parameter_count,
    random_lm_batch,
    require_cuda,
)

import helion


class TorchBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        h = cfg.hidden_size
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        nh, hd = self.n_heads, self.head_dim
        residual = x
        x = self.attn_norm(x)
        q = self.wq(x).view(b, s, nh, hd).transpose(1, 2)
        k = self.wk(x).view(b, s, nh, hd).transpose(1, 2)
        v = self.wv(x).view(b, s, nh, hd).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(b, s, nh * hd)
        x = residual + self.wo(attn)

        residual = x
        x = self.ffn_norm(x)
        x = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        return residual + x


class TorchLanguageModel(nn.Module):
    def __init__(self, cfg: TransformerConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([TorchBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.to(dtype)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))


def bench_step(
    model: nn.Module,
    opt: object,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    repeats: int,
) -> float:
    def step() -> None:
        opt.zero_grad()
        model(tokens, targets).backward()
        opt.step()

    step()
    torch.cuda.synchronize()
    times = (float(triton.testing.do_bench(step)) for _ in range(repeats))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--ffn", type=int, default=1408)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = require_cuda()
    dtype = getattr(torch, args.dtype)
    cfg = TransformerConfig(
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        n_heads=args.heads,
        head_dim=args.head_dim,
        d_ff=args.ffn,
        n_layers=args.layers,
        max_seq_len=args.seq,
    )
    tokens, targets = random_lm_batch(args.vocab, args.batch, args.seq, device)

    print(
        f"step benchmark: {args.layers}L hidden={args.hidden} ffn={args.ffn} "
        f"batch={args.batch} seq={args.seq} vocab={args.vocab} dtype={args.dtype}"
    )

    results: dict[str, float] = {}
    torch.manual_seed(0)
    model = TinyLanguageModel(cfg, dtype).to(device=device, dtype=dtype)
    opt = helion.AdamW(model.parameters(), lr=args.lr)
    results["helion"] = bench_step(model, opt, tokens, targets, args.repeats)
    print(f"helion params: {parameter_count(model):,}")

    torch.manual_seed(0)
    model = TorchLanguageModel(cfg, dtype).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    results["torch eager"] = bench_step(model, opt, tokens, targets, args.repeats)

    if args.compile:
        torch.manual_seed(0)
        compiled = torch.compile(TorchLanguageModel(cfg, dtype).to(device=device))
        opt = torch.optim.AdamW(compiled.parameters(), lr=args.lr)
        results["torch.compile"] = bench_step(
            compiled, opt, tokens, targets, args.repeats
        )

    base = results["torch eager"]
    print("\nmedian step time:")
    for name, ms in results.items():
        print(f"  {name:14s} {ms:8.3f} ms  speedup={base / ms:5.2f}x")


if __name__ == "__main__":
    main()
