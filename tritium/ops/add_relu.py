from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from ._utils import (
    ELEMENTWISE_BLOCK_SIZE_CONFIGS,
    FLOAT_DTYPES,
    as_triton_kernel,
    autotuned_elementwise_grid,
    check_same_shape_dtype_device,
    requires_autograd,
)

SUPPORTED_DTYPES = FLOAT_DTYPES


@triton.autotune(configs=ELEMENTWISE_BLOCK_SIZE_CONFIGS, key=["n_elements"])
@triton.jit
def _add_relu_forward_kernel(
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
    z = x + y
    out = tl.where(z > 0, z, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(configs=ELEMENTWISE_BLOCK_SIZE_CONFIGS, key=["n_elements"])
@triton.jit
def _add_relu_backward_kernel(
    dy_ptr,
    z_ptr,
    dx_ptr,
    dy_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    grad = tl.where(z > 0, dy, 0.0)
    tl.store(dx_ptr + offsets, grad, mask=mask)
    tl.store(dy_out_ptr + offsets, grad, mask=mask)


def _add_relu_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_same_shape_dtype_device("x", x, "y", y)
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    n_elements = x.numel()
    kernel = as_triton_kernel(_add_relu_forward_kernel)
    kernel[autotuned_elementwise_grid(n_elements)](x, y, out, n_elements)
    return out


def _add_relu_backward_impl(
    dy: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = x + y
    dx = torch.empty_like(x)
    dy_out = torch.empty_like(y)
    if x.numel() == 0:
        return dx, dy_out
    n_elements = x.numel()
    kernel = as_triton_kernel(_add_relu_backward_kernel)
    kernel[autotuned_elementwise_grid(n_elements)](dy, z, dx, dy_out, n_elements)
    return dx, dy_out


class _AddReLUAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = _add_relu_forward(x, y)
        ctx.save_for_backward(x, y)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return None, None
        x, y = ctx.saved_tensors
        dy = grad_outputs[0]
        if not dy.is_contiguous():
            dy = dy.contiguous()
        dx, dy_out = _add_relu_backward_impl(dy, x, y)
        if not ctx.needs_input_grad[0]:
            dx = None
        if not ctx.needs_input_grad[1]:
            dy_out = None
        return dx, dy_out


def add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return ``relu(x + y)`` as a single fused kernel with autograd support."""
    if not requires_autograd(x, y):
        return _add_relu_forward(x, y)
    return _AddReLUAutograd.apply(x, y)
