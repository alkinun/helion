from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from ._utils import (
    DEFAULT_BLOCK_SIZE,
    FLOAT_DTYPES,
    as_triton_kernel,
    check_cuda_tensor,
    check_same_shape_dtype_device,
    check_supported_dtype,
    elementwise_grid,
    requires_autograd,
)

SUPPORTED_DTYPES = FLOAT_DTYPES


@triton.jit
def _gelu_forward_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    out = 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * inner) - 1.0))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _gelu_backward_kernel(
    dy_ptr,
    x_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x * x2)
    t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    d_inner = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x2)
    dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner
    tl.store(dx_ptr + offsets, dy * dgelu, mask=mask)


@triton.jit
def _add_gelu_forward_kernel(
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
    inner = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    out = 0.5 * z * (1.0 + (2.0 * tl.sigmoid(2.0 * inner) - 1.0))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _add_gelu_backward_kernel(
    dy_ptr,
    x_ptr,
    y_ptr,
    dx_ptr,
    dy_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x + y
    z2 = z * z
    inner = 0.7978845608028654 * (z + 0.044715 * z * z2)
    t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    d_inner = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z2)
    dgelu = 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * d_inner
    grad = dy * dgelu
    tl.store(dx_ptr + offsets, grad, mask=mask)
    tl.store(dy_out_ptr + offsets, grad, mask=mask)


def _check_gelu_inputs(x: torch.Tensor) -> None:
    check_cuda_tensor("x", x)
    check_supported_dtype("x", x, SUPPORTED_DTYPES)


def _gelu_forward(x: torch.Tensor) -> torch.Tensor:
    _check_gelu_inputs(x)
    x = x.contiguous()
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    n_elements = x.numel()
    kernel = as_triton_kernel(_gelu_forward_kernel)
    kernel[elementwise_grid(n_elements)](
        x, out, n_elements, BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    return out


def gelu_backward(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return ``dx`` for the tanh-approximation GeLU from ``x``."""
    _check_gelu_inputs(x)
    check_cuda_tensor("dy", dy)
    if dy.shape != x.shape:
        raise ValueError(
            f"Shape mismatch: dy has shape {tuple(dy.shape)}, "
            f"x has shape {tuple(x.shape)}."
        )
    x = x.contiguous()
    dy = dy.contiguous()
    dx = torch.empty_like(x)
    if x.numel() == 0:
        return dx
    n_elements = x.numel()
    kernel = as_triton_kernel(_gelu_backward_kernel)
    kernel[elementwise_grid(n_elements)](
        dy, x, dx, n_elements, BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    return dx


def _add_gelu_forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    check_same_shape_dtype_device("x", x, "y", y)
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    n_elements = x.numel()
    kernel = as_triton_kernel(_add_gelu_forward_kernel)
    kernel[elementwise_grid(n_elements)](
        x, y, out, n_elements, BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    return out


def _add_gelu_backward_impl(
    dy: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x)
    dy_out = torch.empty_like(y)
    if x.numel() == 0:
        return dx, dy_out
    n_elements = x.numel()
    kernel = as_triton_kernel(_add_gelu_backward_kernel)
    kernel[elementwise_grid(n_elements)](
        dy, x, y, dx, dy_out, n_elements, BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    return dx, dy_out


class _GeLUAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        out = _gelu_forward(x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None]:
        if not ctx.needs_input_grad[0]:
            return (None,)
        (x,) = ctx.saved_tensors
        return (gelu_backward(grad_outputs[0], x),)


class _AddGeLUAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = _add_gelu_forward(x, y)
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
        dx, dy_out = _add_gelu_backward_impl(dy, x, y)
        if not ctx.needs_input_grad[0]:
            dx = None
        if not ctx.needs_input_grad[1]:
            dy_out = None
        return dx, dy_out


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Return the tanh-approximation GeLU of ``x`` with autograd support."""
    if not requires_autograd(x):
        return _gelu_forward(x)
    return _GeLUAutograd.apply(x)


def add_gelu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return ``gelu(x + y)`` as a single fused kernel with autograd support."""
    if not requires_autograd(x, y):
        return _add_gelu_forward(x, y)
    return _AddGeLUAutograd.apply(x, y)
