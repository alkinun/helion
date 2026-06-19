"""Train a character-level language model on tiny-shakespeare.

This is the practical example: real data loading, a decoder-only Helion model,
train/validation splitting, LR scheduling, clipping, periodic cached sampling,
and optional checkpointing.

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
    evaluate_lm_loss,
    generate_cached,
    load_shakespeare,
    parameter_count,
    require_cuda,
    split_lm_data,
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
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=20,
        help="Validation batches per eval. Use 0 for the full validation split.",
    )
    parser.add_argument("--gen-interval", type=int, default=200)
    parser.add_argument("--gen-len", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=str, default="")
    parser.add_argument("--best-checkpoint-out", type=str, default="")
    args = parser.parse_args()

    device = require_cuda()
    dtype = getattr(torch, args.dtype)
    helion.seed_all(42)

    text = load_shakespeare()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    try:
        train_data, val_data = split_lm_data(data, args.val_fraction)
    except ValueError as exc:
        raise SystemExit(f"--val-fraction {exc}") from exc
    if len(train_data) <= args.seq + 1:
        raise SystemExit("training split is too small for --seq")
    if len(val_data) <= args.seq + 1:
        raise SystemExit("validation split is too small for --seq")

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
        f"split: train={len(train_data):,} val={len(val_data):,} "
        f"val_fraction={args.val_fraction:.1%}\n"
        f"model: {cfg.n_layers}L hidden={cfg.hidden_size} ffn={cfg.d_ff} "
        f"params={parameter_count(model):,} dtype={args.dtype}"
    )

    meter = helion.AverageMeter()
    best_val_loss = float("inf")
    model.train()
    for step in range(args.steps):
        opt.lr = scheduler(step)
        opt.zero_grad()
        x, y = text_batch(train_data, args.batch, cfg.max_seq_len, device)
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

        should_eval = args.eval_interval > 0 and (
            step % args.eval_interval == 0 or step == args.steps - 1
        )
        if should_eval:
            val_loss, val_ppl, val_tokens, val_batches = evaluate_lm_loss(
                model,
                val_data,
                batch_size=args.batch,
                seq_len=cfg.max_seq_len,
                device=device,
                max_batches=args.eval_batches,
            )
            print(
                f"val  {step:5d} loss={val_loss:.4f} ppl={val_ppl:.2f} "
                f"tokens={val_tokens:,} batches={val_batches}"
            )
            if args.best_checkpoint_out and val_loss < best_val_loss:
                best_val_loss = val_loss
                helion.save_checkpoint(
                    args.best_checkpoint_out,
                    model=model,
                    optimizer=opt,
                    step=step,
                    vocab=tokenizer.chars,
                    config=cfg,
                    dtype=args.dtype,
                    val_fraction=args.val_fraction,
                    val_loss=val_loss,
                    val_ppl=val_ppl,
                )
                print(f"saved best checkpoint: {args.best_checkpoint_out}")

        should_sample = (
            step > 0
            and args.gen_interval > 0
            and args.gen_len > 0
            and step % args.gen_interval == 0
        )
        if should_sample:
            sample = generate_cached(
                model,
                tokenizer,
                "ROMEO:\n",
                args.gen_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed + step,
            )
            print(f"\n--- sample at step {step} ---\n{sample}\n")

    if args.gen_len > 0:
        print("\n=== final sample ===")
        print(
            generate_cached(
                model,
                tokenizer,
                "ROMEO:\n",
                args.gen_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed + args.steps,
            )
        )

    if args.checkpoint_out:
        helion.save_checkpoint(
            args.checkpoint_out,
            model=model,
            optimizer=opt,
            step=args.steps - 1,
            vocab=tokenizer.chars,
            config=cfg,
            dtype=args.dtype,
            val_fraction=args.val_fraction,
        )
        print(f"saved checkpoint: {args.checkpoint_out}")


if __name__ == "__main__":
    main()
