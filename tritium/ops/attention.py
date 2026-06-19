from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from ._utils import (
    FLOAT_DTYPES,
    as_triton_kernel,
    check_cuda_tensor,
    check_supported_dtype,
    requires_autograd,
)

SUPPORTED_DTYPES = FLOAT_DTYPES

ATTENTION_CONFIGS = [
    triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 32}, num_warps=4, num_stages=2),
]


@triton.autotune(
    configs=ATTENTION_CONFIGS,
    key=["q_seq_len", "kv_seq_len", "HEAD_DIM", "IS_CAUSAL"],
)
@triton.jit
def _attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    n_heads,
    n_kv_heads,
    q_seq_len,
    kv_seq_len,
    scale,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // n_heads
    head_id = pid_bh % n_heads
    kv_head = head_id // GROUP_SIZE

    q_bh = batch_id * stride_qb + head_id * stride_qh
    k_bh = batch_id * stride_kb + kv_head * stride_kh
    v_bh = batch_id * stride_vb + kv_head * stride_vh
    o_bh = batch_id * stride_ob + head_id * stride_oh

    d_off = tl.arange(0, HEAD_DIM)
    q_pos = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_pos < q_seq_len

    q = tl.load(
        q_ptr + q_bh + q_pos[:, None] * stride_qs + d_off[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )

    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    if IS_CAUSAL:
        kv_max = (pid_q + 1) * BLOCK_Q
    else:
        kv_max = kv_seq_len

    for kv_start in range(0, kv_max, BLOCK_KV):
        kv_pos = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_pos < kv_seq_len

        k = tl.load(
            k_ptr + k_bh + kv_pos[:, None] * stride_ks + d_off[None, :] * stride_kd,
            mask=kv_mask[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k), input_precision="ieee").to(tl.float32) * scale

        valid = q_mask[:, None] & kv_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (q_pos[:, None] >= kv_pos[None, :])
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            v_ptr + v_bh + kv_pos[:, None] * stride_vs + d_off[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(
            p.to(v.dtype), v, input_precision="ieee"
        ).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]

    tl.store(
        o_ptr + o_bh + q_pos[:, None] * stride_os + d_off[None, :] * stride_od,
        acc,
        mask=q_mask[:, None],
    )

    lse = m_i + tl.log(l_i)
    lse_bh = batch_id * (n_heads * q_seq_len) + head_id * q_seq_len
    tl.store(lse_ptr + lse_bh + q_pos, lse, mask=q_mask)


def _check_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        check_cuda_tensor(name, tensor)
        check_supported_dtype(name, tensor, SUPPORTED_DTYPES)

    if q.dim() != 4:
        raise ValueError(
            "q must be a 4D tensor shaped (batch, n_heads, q_seq_len, head_dim)."
        )

    batch, n_heads, q_seq_len, head_dim = q.shape
    n_kv_heads = k.shape[1] if k.dim() == 4 else 0
    kv_seq_len = k.shape[2] if k.dim() == 4 else 0

    for name, tensor in (("k", k), ("v", v)):
        if tensor.dim() != 4:
            raise ValueError(
                f"{name} must be a 4D tensor shaped "
                "(batch, n_kv_heads, kv_seq_len, head_dim)."
            )
        if tensor.shape[0] != batch:
            raise ValueError(
                f"Shape mismatch: {name} has batch {tensor.shape[0]}, expected {batch}."
            )
        if tensor.shape[2] != kv_seq_len:
            raise ValueError(
                f"Shape mismatch: {name} has seq_len {tensor.shape[2]}, "
                f"expected {kv_seq_len}."
            )
        if tensor.shape[3] != head_dim:
            raise ValueError(
                f"Shape mismatch: {name} has head_dim {tensor.shape[3]}, "
                f"expected {head_dim}."
            )
        if tensor.shape[1] != n_kv_heads:
            raise ValueError(
                f"Shape mismatch: {name} has {tensor.shape[1]} kv heads, "
                f"expected {n_kv_heads}."
            )
        if tensor.dtype != q.dtype:
            raise ValueError(
                f"Dtype mismatch: {name} has dtype {tensor.dtype}, "
                f"q has dtype {q.dtype}."
            )
        if tensor.device != q.device:
            raise ValueError(
                f"Device mismatch: {name} is on {tensor.device}, q is on {q.device}."
            )

    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})."
        )

    if head_dim < 16 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError(f"head_dim must be a power of 2 >= 16, got {head_dim}.")

    return batch, n_heads, n_kv_heads, q_seq_len, kv_seq_len, head_dim


def _attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n_heads, n_kv_heads, q_seq_len, kv_seq_len, head_dim = (
        _check_attention_inputs(q, k, v)
    )

    o = torch.empty_like(q)
    lse = torch.empty((batch, n_heads, q_seq_len), device=q.device, dtype=torch.float32)

    if q_seq_len == 0:
        return o, lse

    scale = head_dim ** (-0.5)
    kernel = as_triton_kernel(_attention_forward_kernel)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (triton.cdiv(q_seq_len, meta["BLOCK_Q"]), batch * n_heads)

    kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        n_heads,
        n_kv_heads,
        q_seq_len,
        kv_seq_len,
        scale,
        HEAD_DIM=head_dim,
        GROUP_SIZE=n_heads // n_kv_heads,
        IS_CAUSAL=is_causal,
    )
    return o, lse


@triton.jit
def _attention_backward_dv_di_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    dv_ptr,
    di_ptr,
    lse_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    n_heads,
    n_kv_heads,
    q_seq_len,
    kv_seq_len,
    scale,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // n_heads
    head_id = pid_bh % n_heads
    kv_head = head_id // GROUP_SIZE

    q_bh = batch_id * stride_qb + head_id * stride_qh
    k_bh = batch_id * stride_kb + kv_head * stride_kh
    v_bh = batch_id * stride_vb + kv_head * stride_vh
    do_bh = batch_id * stride_dob + head_id * stride_doh
    bh = batch_id * n_heads + head_id
    bkh = batch_id * n_kv_heads + kv_head

    d_off = tl.arange(0, HEAD_DIM)
    q_pos = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_pos < q_seq_len

    q = tl.load(
        q_ptr + q_bh + q_pos[:, None] * stride_qs + d_off[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )
    do = tl.load(
        do_ptr + do_bh + q_pos[:, None] * stride_dos + d_off[None, :] * stride_dod,
        mask=q_mask[:, None],
        other=0.0,
    )
    lse = tl.load(lse_ptr + bh * q_seq_len + q_pos, mask=q_mask, other=0.0)

    di = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    if IS_CAUSAL:
        kv_max = (pid_q + 1) * BLOCK_Q
    else:
        kv_max = kv_seq_len

    for kv_start in range(0, kv_max, BLOCK_KV):
        kv_pos = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_pos < kv_seq_len

        k = tl.load(
            k_ptr + k_bh + kv_pos[:, None] * stride_ks + d_off[None, :] * stride_kd,
            mask=kv_mask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + v_bh + kv_pos[:, None] * stride_vs + d_off[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k), input_precision="ieee").to(tl.float32) * scale
        valid = q_mask[:, None] & kv_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (q_pos[:, None] >= kv_pos[None, :])
        p = tl.exp(scores - lse[:, None])
        p = tl.where(valid, p, 0.0)

        dp = tl.dot(do, tl.trans(v), input_precision="ieee").to(tl.float32)
        di += tl.sum(p * dp, axis=1)

        dv_block = tl.dot(tl.trans(p.to(v.dtype)), do, input_precision="ieee").to(
            tl.float32
        )
        tl.atomic_add(
            dv_ptr
            + bkh * kv_seq_len * HEAD_DIM
            + kv_pos[:, None] * HEAD_DIM
            + d_off[None, :],
            dv_block,
            mask=kv_mask[:, None],
        )

    tl.store(di_ptr + bh * q_seq_len + q_pos, di, mask=q_mask)


@triton.jit
def _attention_backward_dq_dk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    dq_ptr,
    dk_ptr,
    di_ptr,
    lse_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    n_heads,
    n_kv_heads,
    q_seq_len,
    kv_seq_len,
    scale,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_id = pid_bh // n_heads
    head_id = pid_bh % n_heads
    kv_head = head_id // GROUP_SIZE

    q_bh = batch_id * stride_qb + head_id * stride_qh
    k_bh = batch_id * stride_kb + kv_head * stride_kh
    v_bh = batch_id * stride_vb + kv_head * stride_vh
    do_bh = batch_id * stride_dob + head_id * stride_doh
    bh = batch_id * n_heads + head_id
    bkh = batch_id * n_kv_heads + kv_head

    d_off = tl.arange(0, HEAD_DIM)
    q_pos = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_pos < q_seq_len

    q = tl.load(
        q_ptr + q_bh + q_pos[:, None] * stride_qs + d_off[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )
    do = tl.load(
        do_ptr + do_bh + q_pos[:, None] * stride_dos + d_off[None, :] * stride_dod,
        mask=q_mask[:, None],
        other=0.0,
    )
    lse = tl.load(lse_ptr + bh * q_seq_len + q_pos, mask=q_mask, other=0.0)
    di_val = tl.load(di_ptr + bh * q_seq_len + q_pos, mask=q_mask, other=0.0)

    dq = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    if IS_CAUSAL:
        kv_max = (pid_q + 1) * BLOCK_Q
    else:
        kv_max = kv_seq_len

    for kv_start in range(0, kv_max, BLOCK_KV):
        kv_pos = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_pos < kv_seq_len

        k = tl.load(
            k_ptr + k_bh + kv_pos[:, None] * stride_ks + d_off[None, :] * stride_kd,
            mask=kv_mask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + v_bh + kv_pos[:, None] * stride_vs + d_off[None, :] * stride_vd,
            mask=kv_mask[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k), input_precision="ieee").to(tl.float32) * scale
        valid = q_mask[:, None] & kv_mask[None, :]
        if IS_CAUSAL:
            valid = valid & (q_pos[:, None] >= kv_pos[None, :])
        p = tl.exp(scores - lse[:, None])
        p = tl.where(valid, p, 0.0)

        dp = tl.dot(do, tl.trans(v), input_precision="ieee").to(tl.float32)
        ds = p * (dp - di_val[:, None])

        ds_cast = ds.to(k.dtype)
        dq += tl.dot(ds_cast, k, input_precision="ieee").to(tl.float32) * scale

        dk_block = (
            tl.dot(tl.trans(ds_cast), q, input_precision="ieee").to(tl.float32) * scale
        )
        tl.atomic_add(
            dk_ptr
            + bkh * kv_seq_len * HEAD_DIM
            + kv_pos[:, None] * HEAD_DIM
            + d_off[None, :],
            dk_block,
            mask=kv_mask[:, None],
        )

    tl.store(
        dq_ptr + bh * q_seq_len * HEAD_DIM + q_pos[:, None] * HEAD_DIM + d_off[None, :],
        dq,
        mask=q_mask[:, None],
    )


def _attention_backward_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n_heads, n_kv_heads, q_seq_len, kv_seq_len, head_dim = (
        _check_attention_inputs(q, k, v)
    )
    scale = head_dim ** (-0.5)

    dq = torch.empty_like(q)
    dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
    dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
    di = torch.empty((batch, n_heads, q_seq_len), device=q.device, dtype=torch.float32)

    if q_seq_len == 0:
        return dq, dk.to(k.dtype), dv.to(v.dtype)

    block_q = 64
    block_kv = 64
    grid = (triton.cdiv(q_seq_len, block_q), batch * n_heads)
    common_args = dict(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        scale=scale,
        HEAD_DIM=head_dim,
        GROUP_SIZE=n_heads // n_kv_heads,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
        IS_CAUSAL=is_causal,
    )

    kernel1 = as_triton_kernel(_attention_backward_dv_di_kernel)
    kernel1[grid](
        q,
        k,
        v,
        do,
        dv,
        di,
        lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *do.stride(),
        **common_args,
    )

    kernel2 = as_triton_kernel(_attention_backward_dq_dk_kernel)
    kernel2[grid](
        q,
        k,
        v,
        do,
        dq,
        dk,
        di,
        lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *do.stride(),
        **common_args,
    )

    return dq, dk.to(k.dtype), dv.to(v.dtype)


class _AttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        o, lse = _attention_forward(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, lse)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        do = grad_outputs[0]
        q, k, v, lse = ctx.saved_tensors
        if not do.is_contiguous():
            do = do.contiguous()

        dq, dk, dv = _attention_backward_impl(do, q, k, v, lse, ctx.is_causal)
        if not ctx.needs_input_grad[0]:
            dq = None
        if not ctx.needs_input_grad[1]:
            dk = None
        if not ctx.needs_input_grad[2]:
            dv = None
        return dq, dk, dv, None


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
) -> torch.Tensor:
    """Compute scaled dot-product attention over the last two dimensions.

    Supports grouped-query attention: q has shape (batch, n_heads, q_seq_len,
    head_dim) while k and v have shape (batch, n_kv_heads, kv_seq_len, head_dim)
    where n_heads must be divisible by n_kv_heads. q_seq_len and kv_seq_len may
    differ, enabling KV-cache inference (q_seq_len=1 with a full cached k/v).
    head_dim must be a power of 2 >= 16.
    """
    if not requires_autograd(q, k, v):
        o, _ = _attention_forward(q, k, v, is_causal)
        return o
    return _AttentionAutograd.apply(q, k, v, is_causal)
