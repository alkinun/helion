from __future__ import annotations

import dataclasses
import math
import os
import urllib.request

import torch
import torch.nn as nn

import helion
import tritium

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclasses.dataclass
class TransformerConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    n_heads: int = 4
    head_dim: int = 32
    d_ff: int = 512
    n_layers: int = 2
    max_seq_len: int = 128
    n_kv_heads: int | None = None
    dropout_p: float = 0.0
    checkpoint_blocks: bool = False


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def tokenizer_from_vocab(chars: list[str]) -> CharTokenizer:
    tokenizer = CharTokenizer("".join(chars))
    tokenizer.chars = chars
    tokenizer.stoi = {ch: i for i, ch in enumerate(chars)}
    tokenizer.itos = {i: ch for i, ch in enumerate(chars)}
    tokenizer.vocab_size = len(chars)
    return tokenizer


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        h = cfg.hidden_size
        self.attn_norm = helion.ResidualRMSNorm(h)
        self.ffn_norm = helion.ResidualRMSNorm(h)
        self.attn = helion.Attention(
            h,
            cfg.n_heads,
            cfg.head_dim,
            cfg.max_seq_len,
            n_kv_heads=cfg.n_kv_heads,
        )
        self.w_gate = helion.Linear(h, cfg.d_ff, bias=False)
        self.w_up = helion.Linear(h, cfg.d_ff, bias=False)
        self.w_down = helion.Linear(cfg.d_ff, h, bias=False)
        self.act = helion.SwiGLU()
        self.dropout = helion.Dropout(cfg.dropout_p)

    def forward(
        self,
        delta: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed, residual = self.attn_norm(delta, residual)
        delta = self.dropout(self.attn(normed))
        normed, residual = self.ffn_norm(delta, residual)
        ffn = self.w_down(self.act(self.w_gate(normed), self.w_up(normed)))
        return self.dropout(ffn), residual

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
        ffn = self.w_down(self.act(self.w_gate(normed), self.w_up(normed)))
        return ffn, residual


class TinyLanguageModel(nn.Module):
    def __init__(self, cfg: TransformerConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = helion.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = helion.ResidualRMSNorm(cfg.hidden_size)
        self.lm_head = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden_size))
        nn.init.normal_(self.lm_head, std=0.02)
        self.to(dtype)

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        delta = self.embed(tokens)
        residual = torch.zeros_like(delta)
        for block in self.blocks:
            if self.cfg.checkpoint_blocks and self.training:
                delta, residual = helion.checkpoint(block, delta, residual)
            else:
                delta, residual = block(delta, residual)
        hidden, _ = self.final_norm(delta, residual)
        return hidden

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_hidden(tokens)
        return tritium.linear_cross_entropy(hidden, self.lm_head, targets)

    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_hidden(tokens)
        flat = hidden.reshape(-1, hidden.shape[-1])
        logits = tritium.matmul(flat, self.lm_head.T.contiguous())
        return logits.view(*hidden.shape[:-1], self.cfg.vocab_size)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit(
            "Helion examples require CUDA because Tritium uses Triton kernels."
        )
    return torch.device("cuda")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def random_lm_batch(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(vocab_size, (batch_size, seq_len + 1), device=device)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def load_shakespeare(data_dir: str = "/tmp/helion_data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tinyshakespeare.txt")
    if not os.path.exists(path):
        print("downloading tiny-shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path) as f:
        return f.read()


def text_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


def split_lm_data(
    data: torch.Tensor,
    val_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 < val_fraction < 1")
    split = int(len(data) * (1.0 - val_fraction))
    return data[:split], data[split:]


def iter_sequential_lm_batches(
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    max_batches: int = 0,
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


def evaluate_lm_loss(
    model: nn.Module,
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    max_batches: int = 0,
) -> tuple[float, float, int, int]:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    with torch.inference_mode():
        for x, y in iter_sequential_lm_batches(
            data,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            max_batches=max_batches,
        ):
            loss = model(x, y)
            tokens = y.numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
            total_batches += 1

    if was_training:
        model.train()
    if total_tokens == 0:
        raise ValueError("no evaluation batches were produced")

    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(mean_loss), total_tokens, total_batches


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0 or not token_ids:
        return logits
    unique_ids = torch.tensor(sorted(set(token_ids)), device=logits.device)
    penalized = logits.clone()
    selected = penalized[unique_ids]
    penalized[unique_ids] = torch.where(
        selected < 0,
        selected * penalty,
        selected / penalty,
    )
    return penalized


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seen_token_ids: list[int],
    generator: torch.Generator | None,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must satisfy 0 < top_p <= 1")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")

    logits = logits.float().reshape(-1)
    logits = _apply_repetition_penalty(logits, seen_token_ids, repetition_penalty)
    logits = logits / temperature

    if top_k > 0 and top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < threshold, -float("inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative_probs > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits = logits.scatter(
            0,
            sorted_indices[remove],
            torch.full_like(sorted_logits[remove], -float("inf")),
        )

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).item()


@torch.no_grad()
def generate_cached(
    model: TinyLanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty.")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must satisfy 0 < top_p <= 1")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")

    was_training = model.training
    model.eval()

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    max_len = model.cfg.max_seq_len
    ids = tokenizer.encode(prompt)[-max_len:]
    prompt = tokenizer.decode(ids)
    new_tokens = max(0, min(max_new_tokens, max_len - len(ids) + 1))

    caches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for block in model.blocks:
        attn = block.attn
        k_cache = torch.empty(
            1,
            attn.n_kv_heads,
            max_len,
            attn.head_dim,
            device=device,
            dtype=dtype,
        )
        v_cache = torch.empty_like(k_cache)
        caches.append((k_cache, v_cache))

    generated: list[int] = []
    for pos in range(len(ids) + new_tokens - 1):
        input_id = ids[pos] if pos < len(ids) else generated[pos - len(ids)]
        token = torch.tensor([[input_id]], dtype=torch.long, device=device)

        delta = model.embed(token)
        residual = torch.zeros_like(delta)
        for block, (k_cache, v_cache) in zip(model.blocks, caches, strict=True):
            delta, residual = block.forward_cached(
                delta,
                residual,
                k_cache,
                v_cache,
                pos,
            )
        hidden, _ = model.final_norm(delta, residual)

        if pos >= len(ids) - 1:
            logits = tritium.matmul_vec(
                hidden.reshape(-1),
                model.lm_head.T.contiguous(),
            )
            next_id = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seen_token_ids=[*ids, *generated],
                generator=generator,
            )
            generated.append(next_id)

    if was_training:
        model.train()
    return prompt + tokenizer.decode(generated)
