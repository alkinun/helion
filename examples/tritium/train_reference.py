"""Minimal end-to-end training reference for Tritium.

A tiny Llama-style decoder (shared ``_model.TinyLM``) trained with the
Tritium backend: every differentiable op — residual_rmsnorm, rope, attention,
swiglu, linear_cross_entropy, adamw_step — runs through a Triton kernel.

Run:
    python examples/train_reference.py
    python examples/train_reference.py --steps 20 --dtype bfloat16
"""

from __future__ import annotations

import argparse

import torch
from _model import Config, TinyLM, TritiumAdamW, TritiumBackend, make_batch


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

    model = TinyLM(cfg, TritiumBackend, dtype=dtype).to(device=device, dtype=dtype)
    opt = TritiumAdamW(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Tritium training reference: {cfg.n_layers} layers, d={cfg.hidden}, "
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
        chance = torch.log(
            torch.tensor(float(cfg.vocab_size), dtype=torch.float32)
        ).item()
        print(f"\nloss: start {first:.4f} -> end {last:.4f} (chance ~ {chance:.3f})")
        if last < first - 0.5:
            print("LOSS DECREASED: Tritium ops compose correctly through autograd.")
        else:
            print("WARNING: loss did not decrease enough to confirm learning.")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
