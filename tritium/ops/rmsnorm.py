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
RMSNORM_DWEIGHT_COL_BLOCK = 64
RMSNORM_DWEIGHT_ROW_BLOCK = 128
RMSNORM_STAGED_DWEIGHT_MIN_HIDDEN = 512
RMSNORM_STAGED_DWEIGHT_EAGER_MIN_HIDDEN = 1024
RMSNORM_STAGED_DWEIGHT_MIN_ROWS = 64
RMSNORM_STAGED_DWEIGHT_EAGER_MIN_ROWS = 2
RMSNORM_STAGED_DWEIGHT_MAX_ROW_BLOCKS = 1024


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
def _rmsnorm_rstd_kernel(
    x_ptr,
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
    tl.store(rstd_ptr + row, 1.0 / tl.sqrt(mean_sq + eps))


@triton.jit
def _rmsnorm_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    rstd_ptr,
    dx_ptr,
    n_rows,
    hidden_size,
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
    rstd = tl.load(rstd_ptr + row)

    weighted = dy * weight
    c = tl.sum(weighted * x, axis=0)
    dx = rstd * weighted - (rstd * rstd * rstd / hidden_size) * x * c
    tl.store(dx_ptr + row * hidden_size + cols, dx, mask=mask)


@triton.jit
def _rmsnorm_backward_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    rstd_ptr,
    dx_ptr,
    dweight_ptr,
    n_rows,
    hidden_size,
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
    rstd = tl.load(rstd_ptr + row)

    weighted = dy * weight
    c = tl.sum(weighted * x, axis=0)
    dx = rstd * weighted - (rstd * rstd * rstd / hidden_size) * x * c
    tl.store(dx_ptr + row * hidden_size + cols, dx, mask=mask)

    contribution = dy * x * rstd
    tl.atomic_add(dweight_ptr + cols, contribution, mask=mask)


@triton.jit
def _rmsnorm_backward_recompute_kernel(
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


@triton.jit
def _rmsnorm_backward_dx_recompute_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    rstd_ptr,
    dx_ptr,
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
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _rmsnorm_dweight_stage1_kernel(
    dy_ptr,
    x_ptr,
    rstd_ptr,
    partial_ptr,
    n_rows,
    hidden_size,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_block = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)
    rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < hidden_size)

    dy = tl.load(
        dy_ptr + rows[:, None] * hidden_size + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x = tl.load(
        x_ptr + rows[:, None] * hidden_size + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=rows < n_rows, other=0.0).to(tl.float32)
    partial = tl.sum(dy * x * rstd[:, None], axis=0)
    tl.store(
        partial_ptr + row_block * hidden_size + cols,
        partial,
        mask=cols < hidden_size,
    )


@triton.jit
def _rmsnorm_dweight_stage2_kernel(
    partial_ptr,
    dweight_ptr,
    n_row_blocks: tl.constexpr,
    hidden_size,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    col_block = tl.program_id(axis=0)
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (row_blocks[:, None] < n_row_blocks) & (cols[None, :] < hidden_size)
    partial = tl.load(
        partial_ptr + row_blocks[:, None] * hidden_size + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dweight = tl.sum(partial, axis=0)
    tl.store(dweight_ptr + cols, dweight, mask=cols < hidden_size)


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


def _compute_rstd(
    x2d: torch.Tensor, n_rows: int, hidden: int, eps: float
) -> torch.Tensor:
    rstd = torch.empty((n_rows,), device=x2d.device, dtype=torch.float32)
    if n_rows == 0:
        return rstd
    block_size = triton.next_power_of_2(hidden)
    kernel = as_triton_kernel(_rmsnorm_rstd_kernel)
    kernel[(n_rows,)](
        x2d,
        rstd,
        n_rows,
        hidden,
        eps,
        BLOCK_SIZE=block_size,
    )
    return rstd


def _use_staged_dweight(n_rows: int, hidden: int) -> bool:
    n_row_blocks = triton.cdiv(n_rows, RMSNORM_DWEIGHT_ROW_BLOCK)
    if hidden >= RMSNORM_STAGED_DWEIGHT_EAGER_MIN_HIDDEN:
        min_rows = RMSNORM_STAGED_DWEIGHT_EAGER_MIN_ROWS
    elif hidden >= RMSNORM_STAGED_DWEIGHT_MIN_HIDDEN:
        min_rows = RMSNORM_STAGED_DWEIGHT_MIN_ROWS
    else:
        return False

    return n_rows >= min_rows and n_row_blocks <= RMSNORM_STAGED_DWEIGHT_MAX_ROW_BLOCKS


def _rmsnorm_backward_staged_dweight(
    dy2d: torch.Tensor,
    x2d: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    n_rows: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x2d)
    block_size = triton.next_power_of_2(hidden)
    dx_kernel = as_triton_kernel(_rmsnorm_backward_dx_kernel)
    dx_kernel[(n_rows,)](
        dy2d,
        x2d,
        weight,
        rstd,
        dx,
        n_rows,
        hidden,
        BLOCK_SIZE=block_size,
    )

    n_row_blocks = triton.cdiv(n_rows, RMSNORM_DWEIGHT_ROW_BLOCK)
    partial = torch.empty(
        (n_row_blocks, hidden),
        device=x2d.device,
        dtype=torch.float32,
    )
    dweight = torch.empty((hidden,), device=x2d.device, dtype=torch.float32)

    col_blocks = triton.cdiv(hidden, RMSNORM_DWEIGHT_COL_BLOCK)
    stage1 = as_triton_kernel(_rmsnorm_dweight_stage1_kernel)
    stage1[(n_row_blocks, col_blocks)](
        dy2d,
        x2d,
        rstd,
        partial,
        n_rows,
        hidden,
        BLOCK_ROWS=RMSNORM_DWEIGHT_ROW_BLOCK,
        BLOCK_COLS=RMSNORM_DWEIGHT_COL_BLOCK,
        num_warps=4,
    )

    row_block_power = triton.next_power_of_2(n_row_blocks)
    stage2 = as_triton_kernel(_rmsnorm_dweight_stage2_kernel)
    stage2[(col_blocks,)](
        partial,
        dweight,
        n_row_blocks,
        hidden,
        BLOCK_ROW_BLOCKS=row_block_power,
        BLOCK_COLS=RMSNORM_DWEIGHT_COL_BLOCK,
        num_warps=8 if row_block_power >= 256 else 4,
    )

    return dx, dweight.to(weight.dtype)


def _rmsnorm_backward_recompute(
    dy2d: torch.Tensor,
    x2d: torch.Tensor,
    weight: torch.Tensor,
    n_rows: int,
    hidden: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x2d)
    block_size = triton.next_power_of_2(hidden)

    if _use_staged_dweight(n_rows, hidden):
        rstd = torch.empty((n_rows,), device=x2d.device, dtype=torch.float32)
        dx_kernel = as_triton_kernel(_rmsnorm_backward_dx_recompute_kernel)
        dx_kernel[(n_rows,)](
            dy2d,
            x2d,
            weight,
            rstd,
            dx,
            n_rows,
            hidden,
            eps,
            BLOCK_SIZE=block_size,
        )

        n_row_blocks = triton.cdiv(n_rows, RMSNORM_DWEIGHT_ROW_BLOCK)
        partial = torch.empty(
            (n_row_blocks, hidden),
            device=x2d.device,
            dtype=torch.float32,
        )
        dweight = torch.empty((hidden,), device=x2d.device, dtype=torch.float32)

        col_blocks = triton.cdiv(hidden, RMSNORM_DWEIGHT_COL_BLOCK)
        stage1 = as_triton_kernel(_rmsnorm_dweight_stage1_kernel)
        stage1[(n_row_blocks, col_blocks)](
            dy2d,
            x2d,
            rstd,
            partial,
            n_rows,
            hidden,
            BLOCK_ROWS=RMSNORM_DWEIGHT_ROW_BLOCK,
            BLOCK_COLS=RMSNORM_DWEIGHT_COL_BLOCK,
            num_warps=4,
        )

        row_block_power = triton.next_power_of_2(n_row_blocks)
        stage2 = as_triton_kernel(_rmsnorm_dweight_stage2_kernel)
        stage2[(col_blocks,)](
            partial,
            dweight,
            n_row_blocks,
            hidden,
            BLOCK_ROW_BLOCKS=row_block_power,
            BLOCK_COLS=RMSNORM_DWEIGHT_COL_BLOCK,
            num_warps=8 if row_block_power >= 256 else 4,
        )
        return dx, dweight.to(weight.dtype)

    dweight = torch.zeros((hidden,), device=x2d.device, dtype=torch.float32)
    kernel = as_triton_kernel(_rmsnorm_backward_recompute_kernel)
    kernel[(n_rows,)](
        dy2d,
        x2d,
        weight,
        dx,
        dweight,
        n_rows,
        hidden,
        eps,
        BLOCK_SIZE=block_size,
    )
    return dx, dweight.to(weight.dtype)


def _rmsnorm_backward_with_rstd(
    dy2d: torch.Tensor,
    x2d: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    n_rows: int,
    hidden: int,
    *,
    compute_dweight: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dx = torch.empty_like(x2d)
    dweight: torch.Tensor | None = None
    if n_rows == 0:
        if compute_dweight:
            dweight = torch.zeros((hidden,), device=x2d.device, dtype=weight.dtype)
        return dx, dweight

    block_size = triton.next_power_of_2(hidden)
    if compute_dweight:
        if _use_staged_dweight(n_rows, hidden):
            return _rmsnorm_backward_staged_dweight(
                dy2d,
                x2d,
                weight,
                rstd,
                n_rows,
                hidden,
            )

        dweight = torch.zeros((hidden,), device=x2d.device, dtype=torch.float32)
        kernel = as_triton_kernel(_rmsnorm_backward_kernel)
        kernel[(n_rows,)](
            dy2d,
            x2d,
            weight,
            rstd,
            dx,
            dweight,
            n_rows,
            hidden,
            BLOCK_SIZE=block_size,
        )
        dweight = dweight.to(weight.dtype)
    else:
        kernel = as_triton_kernel(_rmsnorm_backward_dx_kernel)
        kernel[(n_rows,)](
            dy2d,
            x2d,
            weight,
            rstd,
            dx,
            n_rows,
            hidden,
            BLOCK_SIZE=block_size,
        )
    return dx, dweight


def rmsnorm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dx, dweight)`` for ``rmsnorm`` recomputing rstd from ``x``."""
    _check_rmsnorm_inputs(x, weight)
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

    x2d, n_rows, hidden = _flat_2d(x)
    dy2d = dy.contiguous().view(n_rows, hidden)
    if n_rows == 0:
        dx = torch.empty_like(x2d)
        dweight = torch.zeros((hidden,), device=x2d.device, dtype=weight.dtype)
    else:
        dx, dweight = _rmsnorm_backward_recompute(
            dy2d,
            x2d,
            weight,
            n_rows,
            hidden,
            eps,
        )
    return dx.view_as(x), dweight


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
        x2d, n_rows, hidden = _flat_2d(x)
        dy2d = dy.view(n_rows, hidden)
        dx, dweight = _rmsnorm_backward_with_rstd(
            dy2d,
            x2d,
            weight,
            rstd,
            n_rows,
            hidden,
            compute_dweight=ctx.needs_input_grad[1],
        )
        dx = dx.view_as(x) if ctx.needs_input_grad[0] else None
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
        x2d, n_rows, hidden = _flat_2d(residual_out)
        dy2d = dy.view(n_rows, hidden)
        dz, dweight = _rmsnorm_backward_with_rstd(
            dy2d,
            x2d,
            weight,
            rstd,
            n_rows,
            hidden,
            compute_dweight=ctx.needs_input_grad[2],
        )
        assert dz is not None
        dz.add_(dresidual_out.view(n_rows, hidden))
        dz = dz.view_as(residual_out)
        dz_for_x = dz if ctx.needs_input_grad[0] else None
        dz_for_residual = dz if ctx.needs_input_grad[1] else None
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
