from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from ._utils import (
    FLOAT_DTYPES,
    INDEX_DTYPES,
    as_triton_kernel,
    check_contiguous,
    check_cuda_tensor,
    check_supported_dtype,
    requires_autograd,
)

SUPPORTED_DTYPES = FLOAT_DTYPES
SUPPORTED_POSITION_DTYPES = INDEX_DTYPES
MAX_ROPE_BLOCK_SIZE = 65536


@triton.jit
def _rope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    position_ids_ptr,
    q_out_ptr,
    k_out_ptr,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    position_offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    USE_POSITION_IDS: tl.constexpr,
    INVERSE: tl.constexpr,
):
    token = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_dim

    if USE_POSITION_IDS:
        position = tl.load(position_ids_ptr + token) + position_offset
    else:
        position = token + position_offset

    cos = tl.load(cos_ptr + position * half_dim + offsets, mask=mask, other=1.0).to(
        tl.float32
    )
    sin = tl.load(sin_ptr + position * half_dim + offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    base = (token * n_heads + head) * head_dim
    if INTERLEAVED:
        first_offsets = 2 * offsets
        second_offsets = first_offsets + 1
    else:
        first_offsets = offsets
        second_offsets = offsets + half_dim

    q_first = tl.load(q_ptr + base + first_offsets, mask=mask, other=0.0).to(tl.float32)
    q_second = tl.load(q_ptr + base + second_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    k_first = tl.load(k_ptr + base + first_offsets, mask=mask, other=0.0).to(tl.float32)
    k_second = tl.load(k_ptr + base + second_offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    if INVERSE:
        q_out_first = q_first * cos + q_second * sin
        q_out_second = q_second * cos - q_first * sin
        k_out_first = k_first * cos + k_second * sin
        k_out_second = k_second * cos - k_first * sin
    else:
        q_out_first = q_first * cos - q_second * sin
        q_out_second = q_second * cos + q_first * sin
        k_out_first = k_first * cos - k_second * sin
        k_out_second = k_second * cos + k_first * sin

    tl.store(q_out_ptr + base + first_offsets, q_out_first, mask=mask)
    tl.store(q_out_ptr + base + second_offsets, q_out_second, mask=mask)
    tl.store(k_out_ptr + base + first_offsets, k_out_first, mask=mask)
    tl.store(k_out_ptr + base + second_offsets, k_out_second, mask=mask)


def _check_position_ids(
    position_ids: torch.Tensor | None,
    q: torch.Tensor,
    n_tokens: int,
) -> None:
    if position_ids is None:
        return

    check_cuda_tensor("position_ids", position_ids)
    check_supported_dtype("position_ids", position_ids, SUPPORTED_POSITION_DTYPES)

    if position_ids.device != q.device:
        raise ValueError(
            f"Device mismatch: position_ids is on {position_ids.device}, "
            f"q is on {q.device}."
        )

    if position_ids.shape != (n_tokens,):
        raise ValueError(
            f"Shape mismatch: position_ids has shape {tuple(position_ids.shape)}, "
            f"expected ({n_tokens},)."
        )

    check_contiguous("position_ids", position_ids, "for tritium.rope")


def _check_rope_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    check_cuda_tensor("q", q)
    check_cuda_tensor("k", k)
    check_cuda_tensor("cos", cos)
    check_cuda_tensor("sin", sin)
    check_supported_dtype("q", q, SUPPORTED_DTYPES)
    check_supported_dtype("k", k, SUPPORTED_DTYPES)
    check_supported_dtype("cos", cos, SUPPORTED_DTYPES)
    check_supported_dtype("sin", sin, SUPPORTED_DTYPES)

    if q.shape != k.shape:
        raise ValueError(
            f"Shape mismatch: q has shape {tuple(q.shape)}, "
            f"k has shape {tuple(k.shape)}."
        )

    if q.dtype != k.dtype:
        raise ValueError(
            f"Dtype mismatch: q has dtype {q.dtype}, k has dtype {k.dtype}."
        )

    for name, tensor in (("k", k), ("cos", cos), ("sin", sin)):
        if tensor.device != q.device:
            raise ValueError(
                f"Device mismatch: {name} is on {tensor.device}, q is on {q.device}."
            )

    if q.dim() == 2:
        n_tokens, head_dim = q.shape
        n_heads = 1
    elif q.dim() == 3:
        n_tokens, n_heads, head_dim = q.shape
    else:
        raise ValueError(
            "q and k must be shaped (n_tokens, head_dim) or "
            "(n_tokens, n_heads, head_dim)."
        )

    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for tritium.rope.")

    half_dim = head_dim // 2
    block_size = triton.next_power_of_2(half_dim)
    if block_size > MAX_ROPE_BLOCK_SIZE:
        raise ValueError(
            "Unsupported head_dim: "
            f"next_power_of_2({half_dim}) is {block_size}, "
            f"which exceeds {MAX_ROPE_BLOCK_SIZE}."
        )

    if cos.dim() != 2 or sin.dim() != 2:
        raise ValueError(
            "cos and sin must be 2D tensors shaped (max_position, head_dim / 2)."
        )

    if cos.shape != sin.shape:
        raise ValueError(
            f"Shape mismatch: cos has shape {tuple(cos.shape)}, "
            f"sin has shape {tuple(sin.shape)}."
        )

    if cos.shape[1] != half_dim:
        raise ValueError(
            f"Shape mismatch: cos last dimension is {cos.shape[1]}, "
            f"expected {half_dim}."
        )

    check_contiguous("q", q, "for tritium.rope")
    check_contiguous("k", k, "for tritium.rope")
    check_contiguous("cos", cos, "for tritium.rope")
    check_contiguous("sin", sin, "for tritium.rope")

    _check_position_ids(position_ids, q, n_tokens)
    return n_tokens, n_heads, head_dim, half_dim


def _rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    position_offset: int,
    interleaved: bool,
    inverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_tokens, n_heads, head_dim, half_dim = _check_rope_inputs(
        q,
        k,
        cos,
        sin,
        position_ids,
    )

    if position_offset < 0:
        raise ValueError("position_offset must be non-negative.")

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    if q.numel() == 0:
        return q_out, k_out

    block_size = triton.next_power_of_2(half_dim)
    kernel = as_triton_kernel(_rope_kernel)
    kernel[(n_tokens, n_heads)](
        q,
        k,
        cos,
        sin,
        q if position_ids is None else position_ids,
        q_out,
        k_out,
        n_heads,
        head_dim,
        half_dim,
        position_offset,
        BLOCK_SIZE=block_size,
        INTERLEAVED=interleaved,
        USE_POSITION_IDS=position_ids is not None,
        INVERSE=inverse,
    )

    return q_out, k_out


def _rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None,
    position_offset: int,
    interleaved: bool,
) -> None:
    n_tokens, n_heads, head_dim, half_dim = _check_rope_inputs(
        q,
        k,
        cos,
        sin,
        position_ids,
    )

    if position_offset < 0:
        raise ValueError("position_offset must be non-negative.")

    if q.numel() == 0:
        return

    block_size = triton.next_power_of_2(half_dim)
    kernel = as_triton_kernel(_rope_kernel)
    kernel[(n_tokens, n_heads)](
        q,
        k,
        cos,
        sin,
        q if position_ids is None else position_ids,
        q,
        k,
        n_heads,
        head_dim,
        half_dim,
        position_offset,
        BLOCK_SIZE=block_size,
        INTERLEAVED=interleaved,
        USE_POSITION_IDS=position_ids is not None,
        INVERSE=False,
    )


class _RoPEAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor | None,
        position_offset: int,
        interleaved: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_out, k_out = _rope_forward(
            q,
            k,
            cos,
            sin,
            position_ids,
            position_offset,
            interleaved,
            inverse=False,
        )
        ctx.save_for_backward(cos, sin)
        ctx.position_ids = position_ids
        ctx.position_offset = position_offset
        ctx.interleaved = interleaved
        return q_out, k_out

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None, None]:
        dq_out, dk_out = grad_outputs
        cos, sin = ctx.saved_tensors

        dq: torch.Tensor | None = None
        dk: torch.Tensor | None = None
        if dq_out is not None or dk_out is not None:
            if dq_out is None:
                if dk_out is None:
                    raise RuntimeError("unreachable")
                dq_out = torch.zeros_like(dk_out)
            if dk_out is None:
                dk_out = torch.zeros_like(dq_out)

            if not dq_out.is_contiguous():
                dq_out = dq_out.contiguous()
            if not dk_out.is_contiguous():
                dk_out = dk_out.contiguous()

            dq, dk = _rope_forward(
                dq_out,
                dk_out,
                cos,
                sin,
                ctx.position_ids,
                ctx.position_offset,
                ctx.interleaved,
                inverse=True,
            )

        if not ctx.needs_input_grad[0]:
            dq = None
        if not ctx.needs_input_grad[1]:
            dk = None
        return dq, dk, None, None, None, None, None


def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
    position_offset: int = 0,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to q and k over the last dimension."""
    if not requires_autograd(q, k):
        return _rope_forward(
            q,
            k,
            cos,
            sin,
            position_ids,
            position_offset,
            interleaved,
            inverse=False,
        )

    return _RoPEAutograd.apply(
        q,
        k,
        cos,
        sin,
        position_ids,
        position_offset,
        interleaved,
    )


def rope_(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    position_ids: torch.Tensor | None = None,
    position_offset: int = 0,
    interleaved: bool = False,
) -> None:
    """Apply rotary embeddings to q and k in-place."""
    _rope_inplace(q, k, cos, sin, position_ids, position_offset, interleaved)
