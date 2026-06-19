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
import math

import torch
from common import (
    TinyLanguageModel,
    load_shakespeare,
    require_cuda,
    tokenizer_from_vocab,
)

import helion


def iter_eval_batches(
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    max_batches: int,
):
    starts = torch.arange(0, len(data) - seq_len - 1, seq_len)
    if max_batches > 0:
        starts = starts[: batch_size * max_batches]

    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        if len(batch_starts) == 0:
            break
        x = torch.stack([data[i : i + seq_len] for i in batch_starts])
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in batch_starts])
        yield x.to(device), y.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Maximum eval batches to run. Use 0 for the full validation split.",
    )
    parser.add_argument("--dtype", type=str, default="")
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must satisfy 0 < val_fraction < 1")

    device = require_cuda()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    cfg = metadata["config"]
    dtype_name = args.dtype or metadata.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)

    tokenizer = tokenizer_from_vocab(metadata["vocab"])
    data = torch.tensor(tokenizer.encode(load_shakespeare()), dtype=torch.long)
    split = int(len(data) * (1.0 - args.val_fraction))
    val_data = data[split:]
    if len(val_data) <= cfg.max_seq_len + 1:
        raise SystemExit(
            "validation split is too small for the checkpoint sequence length"
        )

    model = TinyLanguageModel(cfg, dtype=dtype).to(device=device, dtype=dtype)
    ckpt = helion.load_checkpoint(args.checkpoint, model=model, map_location=device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    with torch.inference_mode():
        for x, y in iter_eval_batches(
            val_data,
            batch_size=args.batch,
            seq_len=cfg.max_seq_len,
            device=device,
            max_batches=args.max_batches,
        ):
            loss = model(x, y)
            tokens = y.numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
            total_batches += 1

    if total_tokens == 0:
        raise SystemExit("no validation batches were produced")

    mean_loss = total_loss / total_tokens
    ppl = math.exp(mean_loss)
    step = "unknown" if ckpt.step is None else str(ckpt.step)
    print(f"checkpoint: {args.checkpoint}")
    print(f"step:       {step}")
    print(f"split:      last {args.val_fraction:.1%} of tiny-shakespeare")
    print(f"batches:    {total_batches}")
    print(f"tokens:     {total_tokens:,}")
    print(f"loss:       {mean_loss:.4f}")
    print(f"ppl:        {ppl:.2f}")


if __name__ == "__main__":
    main()
