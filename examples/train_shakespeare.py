"""Train a character-level language model on tiny-shakespeare.

This is the practical example: real data loading, a decoder-only Helion model,
LR scheduling, clipping, periodic cached autoregressive generation, and optional
checkpointing.

Run:
    python examples/train_shakespeare.py
    python examples/train_shakespeare.py --steps 2000 --layers 6 --hidden 384
"""

from __future__ import annotations

import argparse

import torch
from common import (
    CharTokenizer,
    TinyLanguageModel,
    TransformerConfig,
    generate_cached,
    load_shakespeare,
    parameter_count,
    require_cuda,
    text_batch,
)

import helion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--ffn", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--gen-interval", type=int, default=200)
    parser.add_argument("--gen-len", type=int, default=200)
    parser.add_argument("--checkpoint-out", type=str, default="")
    args = parser.parse_args()

    device = require_cuda()
    dtype = getattr(torch, args.dtype)
    helion.seed_all(42)

    text = load_shakespeare()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden,
        n_heads=args.heads,
        head_dim=args.head_dim,
        d_ff=args.ffn,
        n_layers=args.layers,
        max_seq_len=args.seq,
        dropout_p=args.dropout,
    )
    model = TinyLanguageModel(cfg, dtype=dtype).to(device=device, dtype=dtype)
    opt = helion.AdamW(model.parameters(), lr=args.lr)
    scheduler = helion.CosineLR(args.lr, args.warmup, args.steps)

    print(
        f"dataset: {len(data):,} chars vocab={tokenizer.vocab_size}\n"
        f"model: {cfg.n_layers}L hidden={cfg.hidden_size} ffn={cfg.d_ff} "
        f"params={parameter_count(model):,} dtype={args.dtype}"
    )

    meter = helion.AverageMeter()
    model.train()
    for step in range(args.steps):
        opt.lr = scheduler(step)
        opt.zero_grad()
        x, y = text_batch(data, args.batch, cfg.max_seq_len, device)
        loss = model(x, y)
        loss.backward()
        grad_norm = helion.clip_grad_norm(list(model.parameters()), max_norm=1.0)
        opt.step()
        meter.update(loss.item(), n=args.batch)

        if step % 50 == 0 or step == args.steps - 1:
            print(
                f"step {step:5d} loss={meter.avg:.4f} "
                f"lr={opt.lr:.2e} grad_norm={grad_norm.item():.2f}"
            )
            meter.reset()

        if step > 0 and step % args.gen_interval == 0:
            sample = generate_cached(model, tokenizer, "ROMEO:\n", args.gen_len, 0.7)
            print(f"\n--- sample at step {step} ---\n{sample}\n")

    print("\n=== final sample ===")
    print(generate_cached(model, tokenizer, "ROMEO:\n", args.gen_len, 0.7))

    if args.checkpoint_out:
        helion.save_checkpoint(
            args.checkpoint_out,
            model=model,
            optimizer=opt,
            step=args.steps - 1,
            vocab=tokenizer.chars,
            config=cfg,
            dtype=args.dtype,
        )
        print(f"saved checkpoint: {args.checkpoint_out}")


if __name__ == "__main__":
    main()
