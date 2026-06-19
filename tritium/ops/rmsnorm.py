from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from ._utils import (
    FLOAT_DTYPES,
    as_triton_kernel,
    check_contiguous,
    check_cuda_tensor,
    check_supported_dtype,
    requires_autograd,
)

SUPPORTED_DTYPES = FLOAT_DTYPES
MAX_RMSNORM_HIDDEN_NEXT_POW2 = 65536

RMSNORM_BACKWARD_SINGLE_ROW = "single_row"
RMSNORM_BACKWARD_PARTIAL_REDUCE = "partial_reduce"


@triton.jit
def _rmsnorm_forward_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    rstd_ptr,
    n_rows,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    x = tl.load(x_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = x * rstd * weight
    tl.store(out_ptr + row * hidden_size + cols, out, mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _residual_rmsnorm_forward_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    rstd_ptr,
    n_rows,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    x = tl.load(x_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * hidden_size + cols, mask=mask, other=0.0
    ).to(tl.float32)
    z = x + residual
    mean_sq = tl.sum(z * z, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = z * rstd * weight
    tl.store(out_ptr + row * hidden_size + cols, out, mask=mask)
    tl.store(residual_out_ptr + row * hidden_size + cols, z, mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _rmsnorm_dx_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    dx_ptr,
    dweight_partial_ptr,
    n_rows,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    x = tl.load(x_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    weighted = dy * weight
    c = tl.sum(weighted * x, axis=0)
    dx = rstd * weighted - (rstd * rstd * rstd / hidden_size) * x * c
    tl.store(dx_ptr + row * hidden_size + cols, dx, mask=mask)

    contribution = dy * x * rstd
    tl.store(dweight_partial_ptr + row * hidden_size + cols, contribution, mask=mask)


@triton.jit
def _rmsnorm_dweight_reduce_kernel(
    dweight_partial_ptr,
    dweight_ptr,
    n_rows,
    hidden_size,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    col_block = tl.program_id(axis=0)
    col_offs = col_block * BLOCK_COL + tl.arange(0, BLOCK_COL)
    col_mask = col_offs < hidden_size
    acc = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    for row_start in range(0, n_rows, BLOCK_ROW):
        row_offs = row_start + tl.arange(0, BLOCK_ROW)
        row_mask = row_offs < n_rows
        block = tl.load(
            dweight_partial_ptr + row_offs[:, None] * hidden_size + col_offs[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(block, axis=0)
    tl.store(dweight_ptr + col_offs, acc, mask=col_mask)


@triton.jit
def _rmsnorm_dx_atomic_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    dx_ptr,
    dweight_ptr,
    n_rows,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    x = tl.load(x_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    weighted = dy * weight
    c = tl.sum(weighted * x, axis=0)
    dx = rstd * weighted - (rstd * rstd * rstd / hidden_size) * x * c
    tl.store(dx_ptr + row * hidden_size + cols, dx, mask=mask)

    contribution = dy * x * rstd
    tl.atomic_add(dweight_ptr + cols, contribution, mask=mask)


def _check_rmsnorm_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> int:
    check_cuda_tensor("x", x)
    check_cuda_tensor("weight", weight)
    check_supported_dtype("x", x, SUPPORTED_DTYPES)
    check_supported_dtype("weight", weight, SUPPORTED_DTYPES)

    if x.dim() < 1:
        raise ValueError("x must have at least 1 dimension.")
    hidden_size = x.shape[-1]
    if weight.shape != (hidden_size,):
        raise ValueError(
            f"Shape mismatch: weight has shape {tuple(weight.shape)}, "
            f"expected ({hidden_size},)."
        )
    if weight.dtype != x.dtype:
        raise ValueError(
            f"Dtype mismatch: weight has dtype {weight.dtype}, x has dtype {x.dtype}."
        )
    if weight.device != x.device:
        raise ValueError(
            f"Device mismatch: weight is on {weight.device}, x is on {x.device}."
        )

    if triton.next_power_of_2(hidden_size) > MAX_RMSNORM_HIDDEN_NEXT_POW2:
        raise ValueError(
            "Unsupported hidden size: "
            f"next_power_of_2({hidden_size}) exceeds {MAX_RMSNORM_HIDDEN_NEXT_POW2}."
        )

    check_contiguous("x", x, "for tritium.rmsnorm")
    check_contiguous("weight", weight, "for tritium.rmsnorm")
    return hidden_size


def _flat_2d(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    n_rows = x.numel() // x.shape[-1]
    return x.view(n_rows, x.shape[-1]), n_rows, x.shape[-1]


def _rmsnorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = _check_rmsnorm_inputs(x, weight)
    x2d, n_rows, hidden = _flat_2d(x)
    out = torch.empty_like(x2d)
    rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    if n_rows == 0:
        return out.view_as(x), rstd

    block_size = triton.next_power_of_2(hidden_size)
    kernel = as_triton_kernel(_rmsnorm_forward_kernel)
    kernel[(n_rows,)](
        x2d,
        weight,
        out,
        rstd,
        n_rows,
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
    )
    return out.view_as(x), rstd


def _select_rmsnorm_backward_variant(n_rows: int) -> str:
    if n_rows == 1:
        return RMSNORM_BACKWARD_SINGLE_ROW
    return RMSNORM_BACKWARD_PARTIAL_REDUCE


def _rmsnorm_backward_single_row(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = _check_rmsnorm_inputs(x, weight)
    x2d, n_rows, hidden = _flat_2d(x)
    dy2d = dy.contiguous().view(n_rows, hidden)
    dx = torch.empty_like(x2d)
    dweight_partial = torch.empty_like(x2d, dtype=torch.float32)

    block_size = triton.next_power_of_2(hidden_size)
    kernel = as_triton_kernel(_rmsnorm_dx_kernel)
    kernel[(n_rows,)](
        dy2d,
        x2d,
        weight,
        dx,
        dweight_partial,
        n_rows,
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
    )
    return dx.view_as(x), dweight_partial.view_as(x).reshape(hidden_size).to(
        weight.dtype
    )


def _rmsnorm_backward_partial_reduce(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = _check_rmsnorm_inputs(x, weight)
    x2d, n_rows, hidden = _flat_2d(x)
    dy2d = dy.contiguous().view(n_rows, hidden)
    dx = torch.empty_like(x2d)
    dweight_partial = torch.empty_like(x2d, dtype=torch.float32)

    block_size = triton.next_power_of_2(hidden_size)
    dx_kernel = as_triton_kernel(_rmsnorm_dx_kernel)
    dx_kernel[(n_rows,)](
        dy2d,
        x2d,
        weight,
        dx,
        dweight_partial,
        n_rows,
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
    )

    dweight = torch.zeros((hidden_size,), device=x.device, dtype=torch.float32)
    block_col = min(triton.next_power_of_2(hidden_size), 1024)
    reduce_kernel = as_triton_kernel(_rmsnorm_dweight_reduce_kernel)
    reduce_kernel[(triton.cdiv(hidden_size, block_col),)](
        dweight_partial,
        dweight,
        n_rows,
        hidden_size,
        BLOCK_ROW=16,
        BLOCK_COL=block_col,
    )
    return dx.view_as(x), dweight.to(weight.dtype)


def _rmsnorm_backward_atomic(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = _check_rmsnorm_inputs(x, weight)
    x2d, n_rows, hidden = _flat_2d(x)
    dy2d = dy.contiguous().view(n_rows, hidden)
    dx = torch.empty_like(x2d)
    dweight = torch.zeros((hidden_size,), device=x.device, dtype=torch.float32)

    block_size = triton.next_power_of_2(hidden_size)
    kernel = as_triton_kernel(_rmsnorm_dx_atomic_kernel)
    kernel[(n_rows,)](
        dy2d,
        x2d,
        weight,
        dx,
        dweight,
        n_rows,
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
    )
    return dx.view_as(x), dweight.to(weight.dtype)


def rmsnorm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dx, dweight)`` for ``rmsnorm`` recomputing rstd from ``x``."""
    hidden_size = _check_rmsnorm_inputs(x, weight)
    check_cuda_tensor("dy", dy)
    if dy.shape != x.shape:
        raise ValueError(
            f"Shape mismatch: dy has shape {tuple(dy.shape)}, "
            f"x has shape {tuple(x.shape)}."
        )
    if dy.dtype != x.dtype:
        raise ValueError(
            f"Dtype mismatch: dy has dtype {dy.dtype}, x has dtype {x.dtype}."
        )
    if dy.device != x.device:
        raise ValueError(f"Device mismatch: dy is on {dy.device}, x is on {x.device}.")
    check_contiguous("dy", dy, "for tritium.rmsnorm_backward")

    n_rows = dy.numel() // hidden_size
    variant = _select_rmsnorm_backward_variant(n_rows)
    if variant == RMSNORM_BACKWARD_SINGLE_ROW:
        return _rmsnorm_backward_single_row(dy, x, weight, eps)
    return _rmsnorm_backward_partial_reduce(dy, x, weight, eps)


class _RMSNormAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        out, rstd = _rmsnorm_forward(x, weight, eps)
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return None, None, None
        x, weight, rstd = ctx.saved_tensors
        dy = grad_outputs[0]
        if not dy.is_contiguous():
            dy = dy.contiguous()
        dx, dweight = rmsnorm_backward(dy, x, weight, ctx.eps)
        if not ctx.needs_input_grad[0]:
            dx = None
        if not ctx.needs_input_grad[1]:
            dweight = None
        return dx, dweight, None


class _ResidualRMSNormAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_size = _check_rmsnorm_inputs(x, weight)
        check_cuda_tensor("residual", residual)
        if residual.shape != x.shape:
            raise ValueError(
                f"Shape mismatch: residual has shape {tuple(residual.shape)}, "
                f"x has shape {tuple(x.shape)}."
            )
        if residual.dtype != x.dtype:
            raise ValueError(
                f"Dtype mismatch: residual has dtype {residual.dtype}, "
                f"x has dtype {x.dtype}."
            )
        check_contiguous("residual", residual, "for tritium.residual_rmsnorm")

        x2d, n_rows, hidden = _flat_2d(x)
        residual2d = residual.contiguous().view(n_rows, hidden)
        out = torch.empty_like(x2d)
        residual_out = torch.empty_like(x2d)
        rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

        block_size = triton.next_power_of_2(hidden_size)
        kernel = as_triton_kernel(_residual_rmsnorm_forward_kernel)
        kernel[(n_rows,)](
            x2d,
            residual2d,
            weight,
            out,
            residual_out,
            rstd,
            n_rows,
            hidden_size,
            eps,
            BLOCK_SIZE=block_size,
        )
        ctx.save_for_backward(residual_out.view_as(x), weight, rstd)
        ctx.eps = eps
        return out.view_as(x), residual_out.view_as(x)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        if (
            not ctx.needs_input_grad[0]
            and not ctx.needs_input_grad[1]
            and not ctx.needs_input_grad[2]
        ):
            return None, None, None, None
        residual_out, weight, rstd = ctx.saved_tensors
        dy = grad_outputs[0]
        dresidual_out = grad_outputs[1]
        if not dy.is_contiguous():
            dy = dy.contiguous()
        if not dresidual_out.is_contiguous():
            dresidual_out = dresidual_out.contiguous()
        dz, dweight = rmsnorm_backward(dy, residual_out, weight, ctx.eps)
        dz = dz + dresidual_out
        if not ctx.needs_input_grad[0]:
            dz_for_x = None
        else:
            dz_for_x = dz
        if not ctx.needs_input_grad[1]:
            dz_for_residual = None
        else:
            dz_for_residual = dz
        if not ctx.needs_input_grad[2]:
            dweight = None
        return dz_for_x, dz_for_residual, dweight, None


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return ``x * rsqrt(mean(x**2) + eps) * weight`` with autograd support."""
    if not requires_autograd(x, weight):
        out, _ = _rmsnorm_forward(x, weight, eps)
        return out
    return _RMSNormAutograd.apply(x, weight, eps)


def residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(rmsnorm(x + residual), x + residual)`` with autograd support."""
    if not requires_autograd(x, residual, weight):
        hidden_size = _check_rmsnorm_inputs(x, weight)
        x2d, n_rows, hidden = _flat_2d(x)
        residual2d = residual.contiguous().view(n_rows, hidden)
        out = torch.empty_like(x2d)
        residual_out = torch.empty_like(x2d)
        rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
        block_size = triton.next_power_of_2(hidden_size)
        kernel = as_triton_kernel(_residual_rmsnorm_forward_kernel)
        kernel[(n_rows,)](
            x2d,
            residual2d,
            weight,
            out,
            residual_out,
            rstd,
            n_rows,
            hidden_size,
            eps,
            BLOCK_SIZE=block_size,
        )
        return out.view_as(x), residual_out.view_as(x)
    return _ResidualRMSNormAutograd.apply(x, residual, weight, eps)
