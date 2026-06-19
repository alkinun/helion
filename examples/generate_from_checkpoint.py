"""Load a saved Helion character model and generate from a prompt.

Use this with checkpoints produced by train_shakespeare.py.

Run:
    python examples/train_shakespeare.py --steps 1000 \
        --checkpoint-out /tmp/helion_shakespeare.pt
    python examples/generate_from_checkpoint.py \
        --checkpoint /tmp/helion_shakespeare.pt --prompt "ROMEO:\n"
"""

from __future__ import annotations

import argparse

import torch
from common import CharTokenizer, TinyLanguageModel, generate_cached, require_cuda

import helion


def tokenizer_from_vocab(chars: list[str]) -> CharTokenizer:
    tokenizer = CharTokenizer("".join(chars))
    tokenizer.chars = chars
    tokenizer.stoi = {ch: i for i, ch in enumerate(chars)}
    tokenizer.itos = {i: ch for i, ch in enumerate(chars)}
    tokenizer.vocab_size = len(chars)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dtype", type=str, default="")
    args = parser.parse_args()

    device = require_cuda()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    cfg = metadata["config"]
    dtype_name = args.dtype or metadata.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name)

    model = TinyLanguageModel(cfg, dtype=dtype).to(device=device, dtype=dtype)
    optimizer = helion.AdamW(model.parameters(), lr=1e-3)
    ckpt = helion.load_checkpoint(
        args.checkpoint,
        model=model,
        optimizer=optimizer,
        map_location=device,
    )
    tokenizer = tokenizer_from_vocab(ckpt.metadata["vocab"])

    print(
        generate_cached(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
