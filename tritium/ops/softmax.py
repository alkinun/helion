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
MAX_SOFTMAX_HIDDEN_NEXT_POW2 = 65536


@triton.jit
def _softmax_forward_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    x = tl.load(x_ptr + row * hidden_size + cols, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    row_max = tl.max(x, axis=0)
    numerator = tl.exp(x - row_max)
    numerator = tl.where(mask, numerator, 0.0)
    row_sum = tl.sum(numerator, axis=0)
    out = numerator / row_sum
    tl.store(out_ptr + row * hidden_size + cols, out, mask=mask)


@triton.jit
def _softmax_backward_kernel(
    dy_ptr,
    out_ptr,
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
    dy = tl.load(dy_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    out = tl.load(out_ptr + row * hidden_size + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    c = tl.sum(dy * out, axis=0)
    dx = out * (dy - c)
    tl.store(dx_ptr + row * hidden_size + cols, dx, mask=mask)


def _check_softmax_input(x: torch.Tensor) -> int:
    check_cuda_tensor("x", x)
    check_supported_dtype("x", x, SUPPORTED_DTYPES)
    if x.dim() < 1:
        raise ValueError("x must have at least 1 dimension.")
    hidden_size = x.shape[-1]
    if triton.next_power_of_2(hidden_size) > MAX_SOFTMAX_HIDDEN_NEXT_POW2:
        raise ValueError(
            "Unsupported hidden size: "
            f"next_power_of_2({hidden_size}) exceeds "
            f"{MAX_SOFTMAX_HIDDEN_NEXT_POW2}."
        )
    check_contiguous("x", x, "for tritium.softmax")
    return hidden_size


def _flat_2d(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    n_rows = x.numel() // x.shape[-1]
    return x.view(n_rows, x.shape[-1]), n_rows, x.shape[-1]


def _softmax_forward(x: torch.Tensor) -> torch.Tensor:
    _check_softmax_input(x)
    x2d, n_rows, hidden = _flat_2d(x)
    out = torch.empty_like(x2d)
    if n_rows == 0:
        return out.view_as(x)

    block_size = triton.next_power_of_2(hidden)
    kernel = as_triton_kernel(_softmax_forward_kernel)
    kernel[(n_rows,)](x2d, out, n_rows, hidden, BLOCK_SIZE=block_size)
    return out.view_as(x)


def softmax_backward(dy: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Return ``dx`` for ``softmax`` given the upstream grad and the softmax output.

    Uses the identity ``dx_i = out_i * (dy_i - sum_j(dy_j * out_j))`` where the
    sum is over the last dimension.
    """
    hidden_size = _check_softmax_input(out)
    check_cuda_tensor("dy", dy)
    check_supported_dtype("dy", dy, SUPPORTED_DTYPES)
    if dy.shape != out.shape:
        raise ValueError(
            f"Shape mismatch: dy has shape {tuple(dy.shape)}, "
            f"out has shape {tuple(out.shape)}."
        )
    if dy.dtype != out.dtype:
        raise ValueError(
            f"Dtype mismatch: dy has dtype {dy.dtype}, out has dtype {out.dtype}."
        )
    if dy.device != out.device:
        raise ValueError(
            f"Device mismatch: dy is on {dy.device}, out is on {out.device}."
        )
    check_contiguous("dy", dy, "for tritium.softmax_backward")

    out2d, n_rows, hidden = _flat_2d(out)
    dy2d = dy.contiguous().view(n_rows, hidden)
    dx = torch.empty_like(out2d)
    if n_rows == 0:
        return dx.view_as(out)
    block_size = triton.next_power_of_2(hidden_size)
    kernel = as_triton_kernel(_softmax_backward_kernel)
    kernel[(n_rows,)](dy2d, out2d, dx, n_rows, hidden, BLOCK_SIZE=block_size)
    return dx.view_as(out)


class _SoftmaxAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        out = _softmax_forward(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None]:
        if not ctx.needs_input_grad[0]:
            return (None,)
        (out,) = ctx.saved_tensors
        dy = grad_outputs[0]
        if not dy.is_contiguous():
            dy = dy.contiguous()
        return (softmax_backward(dy, out),)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Return the numerically stable softmax of ``x`` over the last dimension.

    Each row is shifted by its maximum before exponentiation. The reduction
    (``max``/``sum``) is performed in float32 internally. With autograd support.
    """
    if not requires_autograd(x):
        return _softmax_forward(x)
    return _SoftmaxAutograd.apply(x)
