from __future__ import annotations

import torch
import torch.nn as nn

import tritium


class Embedding(nn.Module):
    weight: nn.Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


class Linear(nn.Module):
    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        nn.init.normal_(self.weight, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *leading_dims, in_features = x.shape
        x = x.reshape(-1, in_features)
        x = tritium.matmul(x, self.weight.T)
        if self.bias is not None:
            x = x + self.bias
        return x.reshape(*leading_dims, self.out_features)


class RMSNorm(nn.Module):
    weight: nn.Parameter

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tritium.rmsnorm(x, self.weight, self.eps)


class LayerNorm(nn.Module):
    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tritium.layernorm(x, self.weight, self.bias, self.eps)


class SwiGLU(nn.Module):
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return tritium.swiglu(gate, up)


class Dropout(nn.Module):
    p: float

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability p must satisfy 0 <= p < 1, got {p}.")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        return tritium.dropout(x, self.p)


class ResidualRMSNorm(nn.Module):
    weight: nn.Parameter

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(
        self,
        delta: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return tritium.residual_rmsnorm(delta, residual, self.weight, self.eps)


class Attention(nn.Module):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        n_kv_heads: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim
        kv_dim = self.n_kv_heads * head_dim
        q_dim = n_heads * head_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        self.wq = Linear(hidden_size, q_dim, bias=False, **factory_kwargs)
        self.wk = Linear(hidden_size, kv_dim, bias=False, **factory_kwargs)
        self.wv = Linear(hidden_size, kv_dim, bias=False, **factory_kwargs)
        self.wo = Linear(q_dim, hidden_size, bias=False, **factory_kwargs)

        half = head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, **factory_kwargs) / half))
        t = torch.arange(max_seq_len, **factory_kwargs)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        nh = self.n_heads
        nkv = self.n_kv_heads
        hd = self.head_dim

        q = self.wq(x).view(b, s, nh, hd)
        k = self.wk(x).view(b, s, nkv, hd)
        v = self.wv(x).view(b, s, nkv, hd)

        q_flat = q.reshape(b * s, nh, hd).contiguous()
        k_flat = k.reshape(b * s, nkv, hd).contiguous()
        rope_args = (self.rope_cos[:s], self.rope_sin[:s])
        position_ids = None if b == 1 else torch.arange(s, device=x.device).repeat(b)
        if nh == nkv:
            q_rot, k_rot = tritium.rope(
                q_flat, k_flat, *rope_args, position_ids=position_ids
            )
        else:
            q_rot, _ = tritium.rope(
                q_flat, q_flat, *rope_args, position_ids=position_ids
            )
            k_rot, _ = tritium.rope(
                k_flat, k_flat, *rope_args, position_ids=position_ids
            )

        def to_heads(t: torch.Tensor, n: int) -> torch.Tensor:
            return t.view(b, s, n, hd).transpose(1, 2)

        out = tritium.attention(
            to_heads(q_rot, nh),
            to_heads(k_rot, nkv),
            to_heads(v, nkv),
            is_causal=True,
        )
        out = out.transpose(1, 2).reshape(b, s, nh * hd).contiguous()
        return self.wo(out)

    def forward_cached(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos: int,
    ) -> torch.Tensor:
        b = x.shape[0]
        nh, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.wq(x).view(b, 1, nh, hd)
        k_new = self.wk(x).view(b, 1, nkv, hd)
        v_new = self.wv(x).view(b, 1, nkv, hd)

        rope_args = (self.rope_cos[: pos + 1], self.rope_sin[: pos + 1])
        q_rope = q.reshape(b, nh, hd)
        k_rope = k_new.reshape(b, nkv, hd)
        pos_ids = None
        position_offset = pos
        if b != 1:
            pos_ids = torch.full((b,), pos, device=x.device, dtype=torch.int32)
            position_offset = 0
        if nh == nkv:
            q_rot, k_rot = tritium.rope(
                q_rope,
                k_rope,
                *rope_args,
                position_ids=pos_ids,
                position_offset=position_offset,
            )
        else:
            q_rot, _ = tritium.rope(
                q_rope,
                q_rope,
                *rope_args,
                position_ids=pos_ids,
                position_offset=position_offset,
            )
            k_rot, _ = tritium.rope(
                k_rope,
                k_rope,
                *rope_args,
                position_ids=pos_ids,
                position_offset=position_offset,
            )

        k_cache[:, :, pos] = k_rot
        v_cache[:, :, pos] = v_new.view(b, nkv, hd)

        kv_len = pos + 1
        q_heads = q_rot.view(b, nh, 1, hd)
        out = tritium.attention(
            q_heads,
            k_cache[:, :, :kv_len],
            v_cache[:, :, :kv_len],
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(b, 1, nh * hd)
        return self.wo(out)
