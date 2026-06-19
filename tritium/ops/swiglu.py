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
def _swiglu_forward_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    silu = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, silu * gate, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    dy_ptr,
    x_ptr,
    gate_ptr,
    dx_ptr,
    dgate_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(x)
    silu = x * sig
    dsilu = sig * (1.0 + x * (1.0 - sig))
    tl.store(dx_ptr + offsets, dy * gate * dsilu, mask=mask)
    tl.store(dgate_ptr + offsets, dy * silu, mask=mask)


def _swiglu_forward(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    check_same_shape_dtype_device("x", x, "gate", gate)
    x = x.contiguous()
    gate = gate.contiguous()
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    n_elements = x.numel()
    kernel = as_triton_kernel(_swiglu_forward_kernel)
    kernel[elementwise_grid(n_elements)](
        x, gate, out, n_elements, BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    return out


def swiglu_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dx, dgate)`` for ``swiglu`` (``silu(x) * gate``)."""
    check_same_shape_dtype_device("x", x, "gate", gate)
    if dy.shape != x.shape:
        raise ValueError(
            f"Shape mismatch: dy has shape {tuple(dy.shape)}, "
            f"x has shape {tuple(x.shape)}."
        )
    x = x.contiguous()
    gate = gate.contiguous()
    dy = dy.contiguous()
    dx = torch.empty_like(x)
    dgate = torch.empty_like(gate)
    if x.numel() == 0:
        return dx, dgate
    n_elements = x.numel()
    kernel = as_triton_kernel(_swiglu_backward_kernel)
    kernel[elementwise_grid(n_elements)](
        dy, x, gate, dx, dgate, n_elements, BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    return dx, dgate


class _SwiGLUAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        out = _swiglu_forward(x, gate)
        ctx.save_for_backward(x, gate)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return None, None
        x, gate = ctx.saved_tensors
        dx, dgate = swiglu_backward(grad_outputs[0], x, gate)
        if not ctx.needs_input_grad[0]:
            dx = None
        if not ctx.needs_input_grad[1]:
            dgate = None
        return dx, dgate


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Return ``silu(x) * gate`` with autograd support."""
    if not requires_autograd(x, gate):
        return _swiglu_forward(x, gate)
    return _SwiGLUAutograd.apply(x, gate)
