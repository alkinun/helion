"""Train a tiny Helion language model on synthetic tokens.

This is the compact end-to-end training reference. It shows the high-level
Helion API: modules, fused linear cross-entropy, AdamW, LR scheduling,
gradient accumulation, optional AMP, activation checkpointing, EMA, gradient
clipping, metric tracking, and checkpoint save/load.

Run:
    python examples/train_reference.py
    python examples/train_reference.py --steps 50 --micro-batches 4 --amp
    python examples/train_reference.py --checkpoint-out /tmp/helion_tiny.pt
"""

from __future__ import annotations

import argparse

import torch
from common import (
    TinyLanguageModel,
    TransformerConfig,
    parameter_count,
    random_lm_batch,
    require_cuda,
)

import helion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro-batches", type=int, default=1)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint-blocks", action="store_true")
    parser.add_argument("--checkpoint-out", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    device = require_cuda()
    helion.seed_all(0)
    dtype = getattr(torch, args.dtype)
    cfg = TransformerConfig(
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        n_heads=args.heads,
        head_dim=args.head_dim,
        d_ff=args.ffn,
        n_layers=args.layers,
        max_seq_len=args.seq,
        dropout_p=args.dropout,
        checkpoint_blocks=args.checkpoint_blocks,
    )

    model = TinyLanguageModel(cfg, dtype=dtype).to(device=device, dtype=dtype)
    opt = helion.AdamW(model.parameters(), lr=args.lr)
    scheduler = helion.CosineLR(
        args.lr,
        warmup_steps=args.warmup,
        total_steps=args.steps,
    )
    scaler = helion.GradScaler(enabled=args.amp and dtype == torch.float16)
    accum = helion.GradientAccumulator(opt, num_micro_batches=args.micro_batches)
    ema = helion.EMA(model)

    start_step = 0
    if args.resume:
        ckpt = helion.load_checkpoint(args.resume, model=model, optimizer=opt)
        ckpt.restore_rng()
        start_step = 0 if ckpt.step is None else ckpt.step + 1
        print(f"resumed {args.resume} at step {start_step}")

    print(
        f"model: {cfg.n_layers}L hidden={cfg.hidden_size} heads={cfg.n_heads} "
        f"seq={cfg.max_seq_len} params={parameter_count(model):,} dtype={args.dtype}"
    )

    meter = helion.AverageMeter()
    model.train()
    accum.reset()
    for step in range(start_step, args.steps):
        opt.lr = scheduler(step)
        for _ in range(args.micro_batches):
            x, y = random_lm_batch(cfg.vocab_size, args.batch, cfg.max_seq_len, device)
            with helion.autocast(dtype=dtype, enabled=args.amp):
                loss = model(x, y)
            accum.backward(scaler.scale(loss))
            meter.update(loss.item(), n=args.batch)

        scaler.unscale_(opt)
        grad_norm = helion.clip_grad_norm(list(model.parameters()), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        ema.update()
        accum.reset()

        print(
            f"step {step + 1:3d}/{args.steps} "
            f"loss={meter.avg:.4f} lr={opt.lr:.2e} grad_norm={grad_norm.item():.2f}"
        )

    if args.checkpoint_out:
        helion.save_checkpoint(
            args.checkpoint_out,
            model=model,
            optimizer=opt,
            step=args.steps - 1,
            config=cfg,
            dtype=args.dtype,
        )
        print(f"saved checkpoint: {args.checkpoint_out}")

    was_training = model.training
    model.eval()
    with ema.swapped():
        x, y = random_lm_batch(cfg.vocab_size, args.batch, cfg.max_seq_len, device)
        ema_loss = model(x, y).item()
    if was_training:
        model.train()
    print(f"ema validation loss on a fresh random batch: {ema_loss:.4f}")


if __name__ == "__main__":
    main()
