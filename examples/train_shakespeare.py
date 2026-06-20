"""Train a character-level language model on tiny-shakespeare.

This is the practical example: real data loading, a decoder-only Helion model,
train/validation splitting, LR scheduling, clipping, periodic cached sampling,
and optional checkpointing.

Run:
    python examples/train_shakespeare.py
    python examples/train_shakespeare.py --steps 2000 --layers 6 --hidden 384 \
        --latest-checkpoint latest.pt --best-checkpoint best.pt
    python examples/train_shakespeare.py --resume latest.pt --steps 4000 \
        --latest-checkpoint latest.pt --best-checkpoint best.pt
"""

from __future__ import annotations

import argparse
from typing import Any

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


def checkpoint_metadata(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if "config" not in metadata or "vocab" not in metadata:
        raise SystemExit(
            f"{path} is missing config/vocab metadata; expected a "
            "train_shakespeare.py checkpoint."
        )
    return metadata


def choose_config(
    args: argparse.Namespace,
    tokenizer: CharTokenizer,
    metadata: dict[str, Any] | None,
) -> TransformerConfig:
    if metadata is not None:
        cfg = metadata["config"]
        overrides = {
            "--seq": (args.seq, cfg.max_seq_len),
            "--hidden": (args.hidden, cfg.hidden_size),
            "--heads": (args.heads, cfg.n_heads),
            "--head-dim": (args.head_dim, cfg.head_dim),
            "--ffn": (args.ffn, cfg.d_ff),
            "--layers": (args.layers, cfg.n_layers),
            "--dropout": (args.dropout, cfg.dropout_p),
        }
        for flag, (value, expected) in overrides.items():
            if value is not None and value != expected:
                raise SystemExit(
                    f"{flag}={value} conflicts with resumed checkpoint value "
                    f"{expected}. Omit architecture flags when using --resume."
                )
        if list(metadata["vocab"]) != tokenizer.chars:
            raise SystemExit("checkpoint tokenizer vocabulary does not match dataset")
        return cfg

    return TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden or 256,
        n_heads=args.heads or 4,
        head_dim=args.head_dim or 64,
        d_ff=args.ffn or 1024,
        n_layers=args.layers or 4,
        max_seq_len=args.seq or 256,
        dropout_p=0.0 if args.dropout is None else args.dropout,
    )


def save_training_checkpoint(
    path: str,
    *,
    model: TinyLanguageModel,
    optimizer: helion.AdamW,
    step: int,
    tokenizer: CharTokenizer,
    config: TransformerConfig,
    dtype: str,
    val_fraction: float,
    best_val_loss: float,
    val_loss: float | None = None,
    val_ppl: float | None = None,
) -> None:
    metadata: dict[str, Any] = {
        "vocab": tokenizer.chars,
        "config": config,
        "dtype": dtype,
        "val_fraction": val_fraction,
        "best_val_loss": best_val_loss,
    }
    if val_loss is not None:
        metadata["val_loss"] = val_loss
    if val_ppl is not None:
        metadata["val_ppl"] = val_ppl

    helion.save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=step,
        **metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--dtype", type=str, default="")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=None)
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
    parser.add_argument("--latest-checkpoint", type=str, default="")
    parser.add_argument("--best-checkpoint", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save latest every N steps. The final step is always saved.",
    )
    args = parser.parse_args()

    device = require_cuda()
    helion.seed_all(args.seed)

    text = load_shakespeare()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    resume_metadata = checkpoint_metadata(args.resume) if args.resume else None
    cfg = choose_config(args, tokenizer, resume_metadata)
    dtype_name = args.dtype or (
        resume_metadata.get("dtype", "bfloat16")
        if resume_metadata is not None
        else "bfloat16"
    )
    dtype = getattr(torch, dtype_name)
    val_fraction = args.val_fraction
    if val_fraction is None:
        val_fraction = (
            resume_metadata.get("val_fraction", 0.1)
            if resume_metadata is not None
            else 0.1
        )

    try:
        train_data, val_data = split_lm_data(data, val_fraction)
    except ValueError as exc:
        raise SystemExit(f"--val-fraction {exc}") from exc
    if len(train_data) <= cfg.max_seq_len + 1:
        raise SystemExit("training split is too small for --seq")
    if len(val_data) <= cfg.max_seq_len + 1:
        raise SystemExit("validation split is too small for --seq")

    model = TinyLanguageModel(cfg, dtype=dtype).to(device=device, dtype=dtype)
    opt = helion.AdamW(model.parameters(), lr=args.lr)
    scheduler = helion.CosineLR(args.lr, args.warmup, args.steps)

    start_step = 0
    best_val_loss = float("inf")
    if resume_metadata is not None:
        ckpt = helion.load_checkpoint(
            args.resume,
            model=model,
            optimizer=opt,
            map_location="cpu",
        )
        ckpt.restore_rng()
        start_step = 0 if ckpt.step is None else ckpt.step + 1
        best_val_loss = resume_metadata.get(
            "best_val_loss",
            resume_metadata.get("val_loss", best_val_loss),
        )
        print(f"resumed checkpoint: {args.resume} at step {start_step}")

    if start_step >= args.steps:
        raise SystemExit(
            f"checkpoint is already at step {start_step - 1}; "
            f"increase --steps above {start_step} to continue training."
        )

    print(
        f"dataset: {len(data):,} chars vocab={tokenizer.vocab_size}\n"
        f"split: train={len(train_data):,} val={len(val_data):,} "
        f"val_fraction={val_fraction:.1%}\n"
        f"model: {cfg.n_layers}L hidden={cfg.hidden_size} ffn={cfg.d_ff} "
        f"params={parameter_count(model):,} dtype={dtype_name}\n"
        f"steps: {start_step} -> {args.steps}"
    )

    meter = helion.AverageMeter()
    model.train()
    last_val_loss: float | None = None
    last_val_ppl: float | None = None
    for step in range(start_step, args.steps):
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
            last_val_loss = val_loss
            last_val_ppl = val_ppl
            if args.best_checkpoint and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_training_checkpoint(
                    args.best_checkpoint,
                    model=model,
                    optimizer=opt,
                    step=step,
                    tokenizer=tokenizer,
                    config=cfg,
                    dtype=dtype_name,
                    val_fraction=val_fraction,
                    best_val_loss=best_val_loss,
                    val_loss=val_loss,
                    val_ppl=val_ppl,
                )
                print(f"saved best checkpoint: {args.best_checkpoint}")

        should_save_latest = (
            args.latest_checkpoint
            and args.save_interval > 0
            and step > start_step
            and step % args.save_interval == 0
        )
        if should_save_latest:
            save_training_checkpoint(
                args.latest_checkpoint,
                model=model,
                optimizer=opt,
                step=step,
                tokenizer=tokenizer,
                config=cfg,
                dtype=dtype_name,
                val_fraction=val_fraction,
                best_val_loss=best_val_loss,
                val_loss=last_val_loss,
                val_ppl=last_val_ppl,
            )
            print(f"saved latest checkpoint: {args.latest_checkpoint}")

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

    if args.latest_checkpoint:
        save_training_checkpoint(
            args.latest_checkpoint,
            model=model,
            optimizer=opt,
            step=args.steps - 1,
            tokenizer=tokenizer,
            config=cfg,
            dtype=dtype_name,
            val_fraction=val_fraction,
            best_val_loss=best_val_loss,
            val_loss=last_val_loss,
            val_ppl=last_val_ppl,
        )
        print(f"saved latest checkpoint: {args.latest_checkpoint}")


if __name__ == "__main__":
    main()
