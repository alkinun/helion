from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._utils import (
    DEFAULT_BLOCK_SIZE,
    FLOAT_DTYPES,
    as_triton_kernel,
    check_contiguous,
    check_cuda_tensor,
    check_same_shape_dtype_device,
    check_supported_dtype,
    elementwise_grid,
)

SUPPORTED_DTYPES = FLOAT_DTYPES


@triton.jit
def _sgd_step_kernel(
    param_ptr,
    grad_ptr,
    lr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(param_ptr + offsets, p - lr * g, mask=mask)


def _check_sgd_inputs(
    param: torch.Tensor,
    grad: torch.Tensor,
) -> None:
    check_cuda_tensor("param", param)
    check_cuda_tensor("grad", grad)
    check_supported_dtype("param", param, SUPPORTED_DTYPES)
    check_same_shape_dtype_device("param", param, "grad", grad)
    check_contiguous("param", param, "for tritium.sgd_step")
    check_contiguous("grad", grad, "for tritium.sgd_step")


def sgd_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    lr: float,
) -> None:
    """Update ``param`` in-place with vanilla SGD: ``param -= lr * grad``."""
    _check_sgd_inputs(param, grad)
    n_elements = param.numel()
    if n_elements == 0:
        return
    kernel = as_triton_kernel(_sgd_step_kernel)
    kernel[elementwise_grid(n_elements)](
        param, grad, lr, n_elements, BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
