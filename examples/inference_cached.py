"""Cached autoregressive inference with Helion Attention.forward_cached.

This example trains a tiny character model on a short in-memory corpus for a few
steps, then generates one token at a time through KV caches. For a real run,
use train_shakespeare.py with more steps.

Run:
    python examples/inference_cached.py
"""

from __future__ import annotations

import argparse

import torch
from common import (
    CharTokenizer,
    TinyLanguageModel,
    TransformerConfig,
    generate_cached,
    require_cuda,
    text_batch,
)

import helion

CORPUS = (
    "HELION:\n"
    "A small model can still show the shape of training and inference.\n"
    "TRITIUM:\n"
    "The kernels do the heavy lifting underneath a clean API.\n"
) * 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--gen-len", type=int, default=120)
    args = parser.parse_args()

    device = require_cuda()
    dtype = torch.bfloat16
    helion.seed_all(7)

    tokenizer = CharTokenizer(CORPUS)
    data = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)
    cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        n_heads=4,
        head_dim=32,
        d_ff=512,
        n_layers=2,
        max_seq_len=128,
    )
    model = TinyLanguageModel(cfg, dtype=dtype).to(device=device, dtype=dtype)
    opt = helion.AdamW(model.parameters(), lr=3e-3)

    model.train()
    for step in range(args.steps):
        opt.zero_grad()
        x, y = text_batch(data, batch_size=8, seq_len=64, device=device)
        loss = model(x, y)
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step {step:3d} loss={loss.item():.4f}")

    print("\n=== cached generation ===")
    print(generate_cached(model, tokenizer, "HELION:\n", args.gen_len, temperature=0.7))


if __name__ == "__main__":
    main()
