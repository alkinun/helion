from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from ._utils import (
    DEFAULT_BLOCK_SIZE,
    FLOAT_DTYPES,
    as_triton_kernel,
    check_same_shape_dtype_device,
    elementwise_grid,
    requires_autograd,
)

SUPPORTED_DTYPES = FLOAT_DTYPES


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def _add_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_same_shape_dtype_device("x", x, "y", y)
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    n_elements = x.numel()
    kernel = as_triton_kernel(_add_kernel)
    kernel[elementwise_grid(n_elements)](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=DEFAULT_BLOCK_SIZE,
    )
    return out


class _AddAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return _add_forward(x, y)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        grad = grad_outputs[0]
        if not grad.is_contiguous():
            grad = grad.contiguous()
        dx = grad if ctx.needs_input_grad[0] else None
        dy = grad if ctx.needs_input_grad[1] else None
        return dx, dy


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the elementwise sum ``x + y`` with autograd support."""
    if not requires_autograd(x, y):
        return _add_forward(x, y)
    return _AddAutograd.apply(x, y)
