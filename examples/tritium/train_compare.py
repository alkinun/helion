"""Compare whole training-step speed across backends.

Same ``_model.TinyLM`` architecture in every backend; only the ops differ. Times
a full training step (forward + backward + optimizer). Answers: how does building
the step out of Tritium kernels compare to PyTorch eager and ``torch.compile``?

    python examples/train_compare.py
    python examples/train_compare.py --dtype bfloat16 --layers 6 --vocab 16384
"""

from __future__ import annotations

import argparse
import statistics

import torch
import triton.testing
from _model import Config, TinyLM, TorchBackend, TritiumAdamW, TritiumBackend


def bench_step(model, opt, tokens, targets, repeats: int = 5) -> float:
    def step() -> None:
        opt.zero_grad()
        model(tokens, targets).backward()
        opt.step()

    step()
    torch.cuda.synchronize()
    times = (float(triton.testing.do_bench(step)) for _ in range(repeats))
    return statistics.median(times)


def build(cfg, backend, dtype, device):
    torch.manual_seed(0)
    model = TinyLM(cfg, backend, dtype=dtype).to(device=device, dtype=dtype)
    if backend is TritiumBackend:
        opt = TritiumAdamW(model.parameters(), lr=3e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    return model, opt


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
        f"batch x seq={args.batch}x{args.seq} vocab={args.vocab} | {args.dtype}"
    )
    print(f"~{n_params / 1e6:.1f}M params\n")

    results: dict[str, float] = {}

    model, opt = build(cfg, TritiumBackend, dtype, device)
    ms = bench_step(model, opt, tokens, targets)
    results["tritium"] = ms
    print(f"  {'tritium':14s}  {ms:8.3f} ms/step")

    model, opt = build(cfg, TorchBackend, dtype, device)
    ms = bench_step(model, opt, tokens, targets)
    results["torch eager"] = ms
    print(f"  {'torch eager':14s}  {ms:8.3f} ms/step")

    model, opt = build(cfg, TorchBackend, dtype, device)
    model = torch.compile(model)
    ms = bench_step(model, opt, tokens, targets)
    results["torch.compile"] = ms
    print(f"  {'torch.compile':14s}  {ms:8.3f} ms/step")

    base = results["torch eager"]
    print("\nspeedup vs torch eager:")
    for name, ms in results.items():
        print(f"  {name:14s}  {base / ms:5.2f}x")


if __name__ == "__main__":
    main()
