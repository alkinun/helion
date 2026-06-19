"""Train a character-level language model on Shakespeare and generate text.

Every compute operation runs through Triton kernels via Helion modules and
Tritium ops. This is the end-to-end validation: if the model produces
recognizable Shakespeare-like text, every kernel in the stack is correct.

    python examples/train_shakespeare.py
    python examples/train_shakespeare.py --steps 2000 --layers 6 --d-model 384
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import urllib.request

import torch
import torch.nn as nn

import helion
import tritium

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclasses.dataclass
class Config:
    vocab_size: int = 0
    d_model: int = 256
    n_heads: int = 4
    head_dim: int = 64
    d_ff: int = 1024
    n_layers: int = 4
    max_seq_len: int = 256

    @property
    def hidden(self) -> int:
        return self.n_heads * self.head_dim


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_shakespeare(data_dir: str = "/tmp/helion_data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(path):
        print("downloading tiny-shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path) as f:
        return f.read()


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        h = cfg.hidden
        self.attn_norm = helion.ResidualRMSNorm(h)
        self.ffn_norm = helion.ResidualRMSNorm(h)
        self.attn = helion.Attention(h, cfg.n_heads, cfg.head_dim, cfg.max_seq_len)
        self.w_gate = helion.Linear(h, cfg.d_ff, bias=False)
        self.w_up = helion.Linear(h, cfg.d_ff, bias=False)
        self.w_down = helion.Linear(cfg.d_ff, h, bias=False)
        self.act = helion.SwiGLU()

    def forward(
        self,
        delta: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed, residual = self.attn_norm(delta, residual)
        delta = self.attn(normed)
        normed, residual = self.ffn_norm(delta, residual)
        return self.w_down(self.act(self.w_gate(normed), self.w_up(normed))), residual

    def forward_cached(
        self,
        delta: torch.Tensor,
        residual: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed, residual = self.attn_norm(delta, residual)
        delta = self.attn.forward_cached(normed, k_cache, v_cache, pos)
        normed, residual = self.ffn_norm(delta, residual)
        return self.w_down(self.act(self.w_gate(normed), self.w_up(normed))), residual


class LanguageModel(nn.Module):
    def __init__(self, cfg: Config, dtype: torch.dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = helion.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = helion.ResidualRMSNorm(cfg.hidden)
        self.lm_head = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden))
        nn.init.normal_(self.lm_head, std=0.02)
        self.to(dtype)

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        residual = torch.zeros_like(x)
        delta = x
        for block in self.blocks:
            delta, residual = block(delta, residual)
        hidden, _ = self.final_norm(delta, residual)
        return hidden

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_hidden(tokens)
        return tritium.linear_cross_entropy(hidden, self.lm_head, targets)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def generate(
    model: LanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
) -> str:
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    max_len = model.cfg.max_seq_len
    ids = tokenizer.encode(prompt)

    caches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block in model.blocks:
        attn = block.attn
        k_cache = torch.zeros(
            1, attn.n_kv_heads, max_len, attn.head_dim, device=device, dtype=dtype
        )
        v_cache = torch.zeros(
            1, attn.n_kv_heads, max_len, attn.head_dim, device=device, dtype=dtype
        )
        caches.append((k_cache, v_cache))

    generated: list[int] = []
    total_len = min(len(ids) + max_new_tokens, max_len)

    for pos in range(total_len):
        input_id = ids[pos] if pos < len(ids) else generated[pos - len(ids)]
        token = torch.tensor([[input_id]], dtype=torch.long, device=device)
        x = model.embed(token)

        residual = torch.zeros_like(x)
        delta = x
        for i, block in enumerate(model.blocks):
            k_cache, v_cache = caches[i]
            delta, residual = block.forward_cached(
                delta, residual, k_cache, v_cache, pos
            )

        hidden, _ = model.final_norm(delta, residual)

        if pos >= len(ids) - 1:
            logits = tritium.matmul(hidden.reshape(1, -1), model.lm_head.T).reshape(
                1, -1
            )
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)

    model.train()
    return prompt + tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--gen-interval", type=int, default=200)
    parser.add_argument("--gen-len", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)

    text = load_shakespeare()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"dataset: {len(data)} chars, vocab: {tokenizer.vocab_size}")

    cfg = Config(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        d_ff=args.d_ff,
        n_layers=args.layers,
        max_seq_len=args.seq,
    )

    model = LanguageModel(cfg, dtype=dtype).to(device=device, dtype=dtype)
    opt = helion.AdamW(model.parameters(), lr=args.lr)
    scheduler = helion.CosineLR(args.lr, args.warmup, args.steps)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg.n_layers}L d={cfg.hidden} ff={cfg.d_ff} | {n_params} params\n")

    model.train()
    for step in range(args.steps):
        opt.lr = scheduler(step)
        opt.zero_grad()

        x, y = get_batch(data, args.batch, cfg.max_seq_len, device)
        loss = model(x, y)
        loss.backward()
        helion.clip_grad_norm(list(model.parameters()), max_norm=1.0)
        opt.step()

        if step % 50 == 0 or step == args.steps - 1:
            lr_str = f"lr={opt.lr:.5f}"
            print(f"  step {step:4d}  loss {loss.item():.4f}  {lr_str}")

        if step > 0 and step % args.gen_interval == 0:
            sample = generate(model, tokenizer, "\n", args.gen_len, temperature=0.7)
            print(f"\n--- sample at step {step} ---\n{sample}\n")

    print("\n=== final generation ===")
    sample = generate(model, tokenizer, "ROMEO:\n", 400, temperature=0.7)
    print(sample)


if __name__ == "__main__":
    main()
