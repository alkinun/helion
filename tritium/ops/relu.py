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
    check_cuda_tensor,
    check_supported_dtype,
    requires_autograd,
)

SUPPORTED_DTYPES = FLOAT_DTYPES


@triton.autotune(configs=ELEMENTWISE_BLOCK_SIZE_CONFIGS, key=["n_elements"])
@triton.jit
def _relu_forward_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(configs=ELEMENTWISE_BLOCK_SIZE_CONFIGS, key=["n_elements"])
@triton.jit
def _relu_backward_kernel(
    dy_ptr,
    x_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dx = tl.where(x > 0, dy, 0.0)
    tl.store(dx_ptr + offsets, dx, mask=mask)


def _check_relu_inputs(x: torch.Tensor) -> None:
    check_cuda_tensor("x", x)
    check_supported_dtype("x", x, SUPPORTED_DTYPES)


def _relu_forward(x: torch.Tensor) -> torch.Tensor:
    _check_relu_inputs(x)
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    n_elements = x.numel()
    kernel = as_triton_kernel(_relu_forward_kernel)
    kernel[autotuned_elementwise_grid(n_elements)](x, out, n_elements)
    return out


def _relu_backward_impl(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    _check_relu_inputs(x)
    check_cuda_tensor("dy", dy)
    if dy.shape != x.shape:
        raise ValueError(
            f"Shape mismatch: dy has shape {tuple(dy.shape)}, "
            f"x has shape {tuple(x.shape)}."
        )
    dx = torch.empty_like(x)
    if x.numel() == 0:
        return dx
    n_elements = x.numel()
    kernel = as_triton_kernel(_relu_backward_kernel)
    kernel[autotuned_elementwise_grid(n_elements)](dy, x, dx, n_elements)
    return dx


class _ReLUAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        out = _relu_forward(x)
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
        dy = grad_outputs[0]
        if not dy.is_contiguous():
            dy = dy.contiguous()
        return (_relu_backward_impl(dy, x),)


def relu(x: torch.Tensor) -> torch.Tensor:
    """Return the elementwise ReLU of ``x`` with autograd support."""
    if not requires_autograd(x):
        return _relu_forward(x)
    return _ReLUAutograd.apply(x)
