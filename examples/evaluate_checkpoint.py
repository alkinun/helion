"""Evaluate a saved Helion Shakespeare checkpoint.

Reports mean next-character cross-entropy and perplexity on a held-out tail
split of tiny-shakespeare. Use this with checkpoints produced by
train_shakespeare.py.

Run:
    python examples/evaluate_checkpoint.py --checkpoint model.pt
    python examples/evaluate_checkpoint.py --checkpoint model.pt --max-batches 200
"""

from __future__ import annotations

import argparse

import torch
from common import (
    TinyLanguageModel,
    evaluate_lm_loss,
    load_shakespeare,
    require_cuda,
    split_lm_data,
    tokenizer_from_vocab,
)

import helion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Held-out tail fraction. Defaults to checkpoint metadata or 0.1.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Maximum eval batches to run. Use 0 for the full validation split.",
    )
    parser.add_argument("--dtype", type=str, default="")
    args = parser.parse_args()

    device = require_cuda()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    cfg = metadata["config"]
    dtype_name = args.dtype or metadata.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)
    val_fraction = args.val_fraction
    if val_fraction is None:
        val_fraction = metadata.get("val_fraction", 0.1)

    tokenizer = tokenizer_from_vocab(metadata["vocab"])
    data = torch.tensor(tokenizer.encode(load_shakespeare()), dtype=torch.long)
    try:
        _, val_data = split_lm_data(data, val_fraction)
    except ValueError as exc:
        raise SystemExit(f"--val-fraction {exc}") from exc
    if len(val_data) <= cfg.max_seq_len + 1:
        raise SystemExit(
            "validation split is too small for the checkpoint sequence length"
        )

    model = TinyLanguageModel(cfg, dtype=dtype).to(device=device, dtype=dtype)
    ckpt = helion.load_checkpoint(args.checkpoint, model=model, map_location=device)
    try:
        mean_loss, ppl, total_tokens, total_batches = evaluate_lm_loss(
            model,
            val_data,
            batch_size=args.batch,
            seq_len=cfg.max_seq_len,
            device=device,
            max_batches=args.max_batches,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    step = "unknown" if ckpt.step is None else str(ckpt.step)
    print(f"checkpoint: {args.checkpoint}")
    print(f"step:       {step}")
    print(f"split:      last {val_fraction:.1%} of tiny-shakespeare")
    print(f"batches:    {total_batches}")
    print(f"tokens:     {total_tokens:,}")
    print(f"loss:       {mean_loss:.4f}")
    print(f"ppl:        {ppl:.2f}")


if __name__ == "__main__":
    main()
