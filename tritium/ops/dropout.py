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
    check_supported_dtype,
    elementwise_grid,
    requires_autograd,
)

SUPPORTED_DTYPES = FLOAT_DTYPES


@triton.jit
def _dropout_forward_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    p,
    scale,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    random = tl.rand(seed, offsets)
    keep = random >= p
    out = tl.where(keep, x * scale, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _dropout_backward_kernel(
    dy_ptr,
    dx_ptr,
    n_elements,
    p,
    scale,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
    random = tl.rand(seed, offsets)
    keep = random >= p
    dx = tl.where(keep, dy * scale, 0.0)
    tl.store(dx_ptr + offsets, dx, mask=mask)


def _check_dropout_prob(p: float) -> None:
    if not 0.0 <= p < 1.0:
        raise ValueError(f"Dropout probability p must satisfy 0 <= p < 1, got {p}.")


def _check_dropout_inputs(name: str, tensor: torch.Tensor, p: float) -> None:
    check_cuda_tensor(name, tensor)
    check_supported_dtype(name, tensor, SUPPORTED_DTYPES)
    _check_dropout_prob(p)


def _dropout_forward(
    x: torch.Tensor,
    p: float,
    seed: int,
) -> torch.Tensor:
    _check_dropout_inputs("x", x, p)
    x = x.contiguous()
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    n_elements = x.numel()
    scale = 1.0 / (1.0 - p)
    kernel = as_triton_kernel(_dropout_forward_kernel)
    kernel[elementwise_grid(n_elements)](
        x, out, n_elements, p, scale, seed, BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    return out


def dropout_backward(
    dy: torch.Tensor,
    p: float,
    seed: int,
) -> torch.Tensor:
    """Return ``dx`` for inverted dropout given the drop probability and seed.

    The dropout mask is recomputed from ``seed`` with the same Philox stream
    used in the forward pass, so no mask needs to be stored.
    """
    _check_dropout_inputs("dy", dy, p)
    dy = dy.contiguous()
    dx = torch.empty_like(dy)
    if dy.numel() == 0:
        return dx
    n_elements = dy.numel()
    scale = 1.0 / (1.0 - p)
    kernel = as_triton_kernel(_dropout_backward_kernel)
    kernel[elementwise_grid(n_elements)](
        dy, dx, n_elements, p, scale, seed, BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    return dx


class _DropoutAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        p: float,
        seed: int,
    ) -> torch.Tensor:
        ctx.p = p
        ctx.seed = seed
        return _dropout_forward(x, p, seed)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None, None
        dy = grad_outputs[0]
        if not dy.is_contiguous():
            dy = dy.contiguous()
        return dropout_backward(dy, ctx.p, ctx.seed), None, None


def dropout(
    x: torch.Tensor,
    p: float = 0.5,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply inverted dropout to ``x`` with drop probability ``p``.

    Kept elements are scaled by ``1 / (1 - p)`` so the output is unbiased in
    expectation. The dropout mask is derived from a Philox RNG seeded by
    ``seed`` (drawn from the global torch RNG when ``None``) and is recomputed
    in the backward pass, so no mask is stored. With autograd support.
    """
    if p == 0:
        return x
    if seed is None:
        seed = int(torch.randint(0, 2**31, (), dtype=torch.int64).item())
    if not requires_autograd(x):
        return _dropout_forward(x, p, seed)
    return _DropoutAutograd.apply(x, p, seed)
