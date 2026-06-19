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

MATMUL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=5
    ),
]


@triton.autotune(configs=MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_mask[None, :]),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_mask[:, None]) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _check_matmul_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    check_cuda_tensor("a", a)
    check_cuda_tensor("b", b)
    check_supported_dtype("a", a, SUPPORTED_DTYPES)
    check_supported_dtype("b", b, SUPPORTED_DTYPES)

    if a.dim() != 2:
        raise ValueError("a must be a 2D tensor shaped (M, K).")

    if b.dim() != 2:
        raise ValueError("b must be a 2D tensor shaped (K, N).")

    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(
            f"Shape mismatch: a has shape {tuple(a.shape)}, "
            f"b has shape {tuple(b.shape)}."
        )

    if a.dtype != b.dtype:
        raise ValueError(
            f"Dtype mismatch: a has dtype {a.dtype}, b has dtype {b.dtype}."
        )

    if a.device != b.device:
        raise ValueError(f"Device mismatch: a is on {a.device}, b is on {b.device}.")

    return m, n, k


def _matmul_grid(m: int, n: int) -> Any:
    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]),
        )

    return grid


def _launch_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    m: int,
    n: int,
    k: int,
) -> None:
    kernel = as_triton_kernel(_matmul_kernel)
    kernel[_matmul_grid(m, n)](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )


def _matmul_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n, k = _check_matmul_inputs(a, b)

    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if m == 0 or n == 0:
        return c

    _launch_matmul(a, b, c, m, n, k)
    return c


def _matmul_backward_impl(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    _, n = b.shape

    da = torch.empty_like(a)
    db = torch.empty_like(b)
    if m == 0 or n == 0 or k == 0:
        return da, db

    kernel = as_triton_kernel(_matmul_kernel)

    kernel[_matmul_grid(m, k)](
        dc,
        b,
        da,
        m,
        k,
        n,
        dc.stride(0),
        dc.stride(1),
        b.stride(1),
        b.stride(0),
        da.stride(0),
        da.stride(1),
    )

    kernel[_matmul_grid(k, n)](
        a,
        dc,
        db,
        k,
        n,
        m,
        a.stride(1),
        a.stride(0),
        dc.stride(0),
        dc.stride(1),
        db.stride(0),
        db.stride(1),
    )

    return da, db


class _MatMulAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return _matmul_forward(a, b)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        dc = grad_outputs[0]
        a, b = ctx.saved_tensors
        if not dc.is_contiguous():
            dc = dc.contiguous()

        da, db = _matmul_backward_impl(dc, a, b)
        if not ctx.needs_input_grad[0]:
            da = None
        if not ctx.needs_input_grad[1]:
            db = None
        return da, db


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the matrix product ``a @ b`` with autograd support."""
    if not requires_autograd(a, b):
        return _matmul_forward(a, b)
    return _MatMulAutograd.apply(a, b)
