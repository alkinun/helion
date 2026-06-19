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
MAX_LAYERNORM_HIDDEN_NEXT_POW2 = 65536
LAYERNORM_GRAD_COL_BLOCK = 64
LAYERNORM_GRAD_ROW_BLOCK = 128
LAYERNORM_STAGED_GRAD_MIN_HIDDEN = 512
LAYERNORM_STAGED_GRAD_EAGER_MIN_HIDDEN = 1024
LAYERNORM_STAGED_GRAD_MIN_ROWS = 64
LAYERNORM_STAGED_GRAD_EAGER_MIN_ROWS = 2
LAYERNORM_STAGED_GRAD_MAX_ROW_BLOCKS = 1024


@triton.jit
def _layernorm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
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
    mean = tl.sum(x, axis=0) / hidden_size
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = xc * rstd
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = x_hat * weight + bias
    tl.store(out_ptr + row * hidden_size + cols, out, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layernorm_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
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
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x_hat = (x - mean) * rstd
    g = dy * weight
    c1 = tl.sum(g, axis=0) / hidden_size
    c2 = tl.sum(g * x_hat, axis=0) / hidden_size
    dx = rstd * (g - c1 - x_hat * c2)
    tl.store(dx_ptr + row * hidden_size + cols, dx, mask=mask)


@triton.jit
def _layernorm_backward_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dweight_ptr,
    dbias_ptr,
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
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x_hat = (x - mean) * rstd
    g = dy * weight
    c1 = tl.sum(g, axis=0) / hidden_size
    c2 = tl.sum(g * x_hat, axis=0) / hidden_size
    dx = rstd * (g - c1 - x_hat * c2)
    tl.store(dx_ptr + row * hidden_size + cols, dx, mask=mask)

    tl.atomic_add(dweight_ptr + cols, dy * x_hat, mask=mask)
    tl.atomic_add(dbias_ptr + cols, dy, mask=mask)


@triton.jit
def _layernorm_backward_dx_recompute_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
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
    mean = tl.sum(x, axis=0) / hidden_size
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = xc * rstd
    g = dy * weight
    c1 = tl.sum(g, axis=0) / hidden_size
    c2 = tl.sum(g * x_hat, axis=0) / hidden_size
    dx = rstd * (g - c1 - x_hat * c2)
    tl.store(dx_ptr + row * hidden_size + cols, dx, mask=mask)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _layernorm_backward_recompute_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    dx_ptr,
    dweight_ptr,
    dbias_ptr,
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
    mean = tl.sum(x, axis=0) / hidden_size
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = xc * rstd
    g = dy * weight
    c1 = tl.sum(g, axis=0) / hidden_size
    c2 = tl.sum(g * x_hat, axis=0) / hidden_size
    dx = rstd * (g - c1 - x_hat * c2)
    tl.store(dx_ptr + row * hidden_size + cols, dx, mask=mask)

    tl.atomic_add(dweight_ptr + cols, dy * x_hat, mask=mask)
    tl.atomic_add(dbias_ptr + cols, dy, mask=mask)


@triton.jit
def _layernorm_grad_stage1_kernel(
    dy_ptr,
    x_ptr,
    mean_ptr,
    rstd_ptr,
    partial_dw_ptr,
    partial_db_ptr,
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
    mean = tl.load(mean_ptr + rows, mask=rows < n_rows, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=rows < n_rows, other=0.0).to(tl.float32)
    x_hat = (x - mean[:, None]) * rstd[:, None]

    partial_dw = tl.sum(dy * x_hat, axis=0)
    partial_db = tl.sum(dy, axis=0)
    tl.store(
        partial_dw_ptr + row_block * hidden_size + cols,
        partial_dw,
        mask=cols < hidden_size,
    )
    tl.store(
        partial_db_ptr + row_block * hidden_size + cols,
        partial_db,
        mask=cols < hidden_size,
    )


@triton.jit
def _layernorm_grad_stage2_kernel(
    partial_dw_ptr,
    partial_db_ptr,
    dweight_ptr,
    dbias_ptr,
    n_row_blocks: tl.constexpr,
    hidden_size,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    col_block = tl.program_id(axis=0)
    row_blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (row_blocks[:, None] < n_row_blocks) & (cols[None, :] < hidden_size)
    partial_dw = tl.load(
        partial_dw_ptr + row_blocks[:, None] * hidden_size + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    partial_db = tl.load(
        partial_db_ptr + row_blocks[:, None] * hidden_size + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dweight = tl.sum(partial_dw, axis=0)
    dbias = tl.sum(partial_db, axis=0)
    tl.store(dweight_ptr + cols, dweight, mask=cols < hidden_size)
    tl.store(dbias_ptr + cols, dbias, mask=cols < hidden_size)


def _check_norm_input(x: torch.Tensor) -> int:
    check_cuda_tensor("x", x)
    check_supported_dtype("x", x, SUPPORTED_DTYPES)
    if x.dim() < 1:
        raise ValueError("x must have at least 1 dimension.")
    hidden_size = x.shape[-1]
    if triton.next_power_of_2(hidden_size) > MAX_LAYERNORM_HIDDEN_NEXT_POW2:
        raise ValueError(
            "Unsupported hidden size: "
            f"next_power_of_2({hidden_size}) exceeds "
            f"{MAX_LAYERNORM_HIDDEN_NEXT_POW2}."
        )
    check_contiguous("x", x, "for tritium.layernorm")
    return hidden_size


def _check_affine_param(
    name: str,
    param: torch.Tensor,
    hidden_size: int,
    x: torch.Tensor,
) -> None:
    check_cuda_tensor(name, param)
    check_supported_dtype(name, param, SUPPORTED_DTYPES)
    if param.shape != (hidden_size,):
        raise ValueError(
            f"Shape mismatch: {name} has shape {tuple(param.shape)}, "
            f"expected ({hidden_size},)."
        )
    if param.dtype != x.dtype:
        raise ValueError(
            f"Dtype mismatch: {name} has dtype {param.dtype}, x has dtype {x.dtype}."
        )
    if param.device != x.device:
        raise ValueError(
            f"Device mismatch: {name} is on {param.device}, x is on {x.device}."
        )
    check_contiguous(name, param, "for tritium.layernorm")


def _check_layernorm_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> int:
    hidden_size = _check_norm_input(x)
    _check_affine_param("weight", weight, hidden_size, x)
    _check_affine_param("bias", bias, hidden_size, x)
    return hidden_size


def _flat_2d(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    n_rows = x.numel() // x.shape[-1]
    return x.view(n_rows, x.shape[-1]), n_rows, x.shape[-1]


def _layernorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_size = _check_layernorm_inputs(x, weight, bias)
    x2d, n_rows, hidden = _flat_2d(x)
    out = torch.empty_like(x2d)
    mean = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    if n_rows == 0:
        return out.view_as(x), mean, rstd

    block_size = triton.next_power_of_2(hidden_size)
    kernel = as_triton_kernel(_layernorm_forward_kernel)
    kernel[(n_rows,)](
        x2d,
        weight,
        bias,
        out,
        mean,
        rstd,
        n_rows,
        hidden_size,
        eps,
        BLOCK_SIZE=block_size,
    )
    return out.view_as(x), mean, rstd


def _use_staged_grad(n_rows: int, hidden: int) -> bool:
    n_row_blocks = triton.cdiv(n_rows, LAYERNORM_GRAD_ROW_BLOCK)
    if hidden >= LAYERNORM_STAGED_GRAD_EAGER_MIN_HIDDEN:
        min_rows = LAYERNORM_STAGED_GRAD_EAGER_MIN_ROWS
    elif hidden >= LAYERNORM_STAGED_GRAD_MIN_HIDDEN:
        min_rows = LAYERNORM_STAGED_GRAD_MIN_ROWS
    else:
        return False

    return n_rows >= min_rows and n_row_blocks <= LAYERNORM_STAGED_GRAD_MAX_ROW_BLOCKS


def _run_grad_stage2(
    partial_dw: torch.Tensor,
    partial_db: torch.Tensor,
    n_row_blocks: int,
    hidden: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dweight = torch.empty((hidden,), device=device, dtype=torch.float32)
    dbias = torch.empty((hidden,), device=device, dtype=torch.float32)
    col_blocks = triton.cdiv(hidden, LAYERNORM_GRAD_COL_BLOCK)
    row_block_power = triton.next_power_of_2(n_row_blocks)
    stage2 = as_triton_kernel(_layernorm_grad_stage2_kernel)
    stage2[(col_blocks,)](
        partial_dw,
        partial_db,
        dweight,
        dbias,
        n_row_blocks,
        hidden,
        BLOCK_ROW_BLOCKS=row_block_power,
        BLOCK_COLS=LAYERNORM_GRAD_COL_BLOCK,
        num_warps=8 if row_block_power >= 256 else 4,
    )
    return dweight, dbias


def _run_grad_stage1(
    dy2d: torch.Tensor,
    x2d: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    n_rows: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_row_blocks = triton.cdiv(n_rows, LAYERNORM_GRAD_ROW_BLOCK)
    partial_dw = torch.empty(
        (n_row_blocks, hidden), device=x2d.device, dtype=torch.float32
    )
    partial_db = torch.empty(
        (n_row_blocks, hidden), device=x2d.device, dtype=torch.float32
    )
    col_blocks = triton.cdiv(hidden, LAYERNORM_GRAD_COL_BLOCK)
    stage1 = as_triton_kernel(_layernorm_grad_stage1_kernel)
    stage1[(n_row_blocks, col_blocks)](
        dy2d,
        x2d,
        mean,
        rstd,
        partial_dw,
        partial_db,
        n_rows,
        hidden,
        BLOCK_ROWS=LAYERNORM_GRAD_ROW_BLOCK,
        BLOCK_COLS=LAYERNORM_GRAD_COL_BLOCK,
        num_warps=4,
    )
    return partial_dw, partial_db


def _layernorm_backward_staged_grad(
    dy2d: torch.Tensor,
    x2d: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    n_rows: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x2d)
    block_size = triton.next_power_of_2(hidden)
    dx_kernel = as_triton_kernel(_layernorm_backward_dx_kernel)
    dx_kernel[(n_rows,)](
        dy2d, x2d, weight, mean, rstd, dx, n_rows, hidden, BLOCK_SIZE=block_size
    )

    n_row_blocks = triton.cdiv(n_rows, LAYERNORM_GRAD_ROW_BLOCK)
    partial_dw, partial_db = _run_grad_stage1(dy2d, x2d, mean, rstd, n_rows, hidden)
    dweight, dbias = _run_grad_stage2(
        partial_dw, partial_db, n_row_blocks, hidden, x2d.device
    )
    return dx, dweight.to(weight.dtype), dbias.to(weight.dtype)


def _layernorm_backward_with_stats(
    dy2d: torch.Tensor,
    x2d: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    n_rows: int,
    hidden: int,
    *,
    compute_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    dx = torch.empty_like(x2d)
    dweight: torch.Tensor | None = None
    dbias: torch.Tensor | None = None
    if n_rows == 0:
        if compute_grad:
            dweight = torch.zeros((hidden,), device=x2d.device, dtype=weight.dtype)
            dbias = torch.zeros((hidden,), device=x2d.device, dtype=weight.dtype)
        return dx, dweight, dbias

    block_size = triton.next_power_of_2(hidden)
    if compute_grad:
        if _use_staged_grad(n_rows, hidden):
            return _layernorm_backward_staged_grad(
                dy2d, x2d, weight, mean, rstd, n_rows, hidden
            )

        dweight = torch.zeros((hidden,), device=x2d.device, dtype=torch.float32)
        dbias = torch.zeros((hidden,), device=x2d.device, dtype=torch.float32)
        kernel = as_triton_kernel(_layernorm_backward_kernel)
        kernel[(n_rows,)](
            dy2d,
            x2d,
            weight,
            mean,
            rstd,
            dx,
            dweight,
            dbias,
            n_rows,
            hidden,
            BLOCK_SIZE=block_size,
        )
        dweight = dweight.to(weight.dtype)
        dbias = dbias.to(weight.dtype)
    else:
        kernel = as_triton_kernel(_layernorm_backward_dx_kernel)
        kernel[(n_rows,)](
            dy2d, x2d, weight, mean, rstd, dx, n_rows, hidden, BLOCK_SIZE=block_size
        )
    return dx, dweight, dbias


def _layernorm_backward_recompute(
    dy2d: torch.Tensor,
    x2d: torch.Tensor,
    weight: torch.Tensor,
    n_rows: int,
    hidden: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x2d)
    block_size = triton.next_power_of_2(hidden)

    if _use_staged_grad(n_rows, hidden):
        mean = torch.empty((n_rows,), device=x2d.device, dtype=torch.float32)
        rstd = torch.empty((n_rows,), device=x2d.device, dtype=torch.float32)
        dx_kernel = as_triton_kernel(_layernorm_backward_dx_recompute_kernel)
        dx_kernel[(n_rows,)](
            dy2d,
            x2d,
            weight,
            mean,
            rstd,
            dx,
            n_rows,
            hidden,
            eps,
            BLOCK_SIZE=block_size,
        )

        n_row_blocks = triton.cdiv(n_rows, LAYERNORM_GRAD_ROW_BLOCK)
        partial_dw, partial_db = _run_grad_stage1(dy2d, x2d, mean, rstd, n_rows, hidden)
        dweight, dbias = _run_grad_stage2(
            partial_dw, partial_db, n_row_blocks, hidden, x2d.device
        )
        return dx, dweight.to(weight.dtype), dbias.to(weight.dtype)

    dweight = torch.zeros((hidden,), device=x2d.device, dtype=torch.float32)
    dbias = torch.zeros((hidden,), device=x2d.device, dtype=torch.float32)
    kernel = as_triton_kernel(_layernorm_backward_recompute_kernel)
    kernel[(n_rows,)](
        dy2d,
        x2d,
        weight,
        dx,
        dweight,
        dbias,
        n_rows,
        hidden,
        eps,
        BLOCK_SIZE=block_size,
    )
    return dx, dweight.to(weight.dtype), dbias.to(weight.dtype)


def layernorm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(dx, dweight, dbias)`` for ``layernorm``.

    Statistics (``mean``/``rstd``) are recomputed from ``x``.
    """
    hidden_size = _check_norm_input(x)
    _check_affine_param("weight", weight, hidden_size, x)
    check_cuda_tensor("dy", dy)
    check_supported_dtype("dy", dy, SUPPORTED_DTYPES)
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
    check_contiguous("dy", dy, "for tritium.layernorm_backward")

    x2d, n_rows, hidden = _flat_2d(x)
    dy2d = dy.contiguous().view(n_rows, hidden)
    if n_rows == 0:
        dx = torch.empty_like(x2d)
        dweight = torch.zeros((hidden,), device=x2d.device, dtype=weight.dtype)
        dbias = torch.zeros((hidden,), device=x2d.device, dtype=weight.dtype)
    else:
        dx, dweight, dbias = _layernorm_backward_recompute(
            dy2d, x2d, weight, n_rows, hidden, eps
        )
    return dx.view_as(x), dweight, dbias


class _LayerNormAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        out, mean, rstd = _layernorm_forward(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        return out

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
        x, weight, mean, rstd = ctx.saved_tensors
        dy = grad_outputs[0]
        if not dy.is_contiguous():
            dy = dy.contiguous()
        x2d, n_rows, hidden = _flat_2d(x)
        dy2d = dy.view(n_rows, hidden)
        compute_grad = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        dx, dweight, dbias = _layernorm_backward_with_stats(
            dy2d,
            x2d,
            weight,
            mean,
            rstd,
            n_rows,
            hidden,
            compute_grad=compute_grad,
        )
        dx = dx.view_as(x) if ctx.needs_input_grad[0] else None
        dweight = dweight if ctx.needs_input_grad[1] else None
        dbias = dbias if ctx.needs_input_grad[2] else None
        return dx, dweight, dbias, None


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Return ``((x - mean) * rsqrt(var + eps)) * weight + bias`` with autograd.

    Normalization, ``mean`` and ``var`` (population variance) are computed over
    the last dimension of ``x``. ``weight`` and ``bias`` are per-feature affine
    parameters of shape ``(hidden_size,)``.
    """
    if not requires_autograd(x, weight, bias):
        out, _, _ = _layernorm_forward(x, weight, bias, eps)
        return out
    return _LayerNormAutograd.apply(x, weight, bias, eps)
