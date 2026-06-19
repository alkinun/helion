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
    check_supported_dtype,
    elementwise_grid,
)

SUPPORTED_DTYPES = FLOAT_DTYPES


@triton.jit
def _adamw_step_kernel(
    param_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg_sq_ptr,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bias_correction1,
    bias_correction2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    ea = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0)
    eas = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0)

    ea = beta1 * ea + (1.0 - beta1) * g
    eas = beta2 * eas + (1.0 - beta2) * g * g
    tl.store(exp_avg_ptr + offsets, ea, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, eas, mask=mask)

    p = p - lr * weight_decay * p
    denom = tl.sqrt(eas / bias_correction2) + eps
    step_size = lr / bias_correction1
    p = p - step_size * ea / denom
    tl.store(param_ptr + offsets, p, mask=mask)


def _check_adamw_inputs(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
) -> None:
    check_cuda_tensor("param", param)
    check_cuda_tensor("grad", grad)
    check_cuda_tensor("exp_avg", exp_avg)
    check_cuda_tensor("exp_avg_sq", exp_avg_sq)
    check_supported_dtype("param", param, SUPPORTED_DTYPES)
    check_supported_dtype("grad", grad, SUPPORTED_DTYPES)

    if param.shape != grad.shape:
        raise ValueError(
            f"Shape mismatch: param has shape {tuple(param.shape)}, "
            f"grad has shape {tuple(grad.shape)}."
        )
    if exp_avg.shape != param.shape:
        raise ValueError(
            f"Shape mismatch: exp_avg has shape {tuple(exp_avg.shape)}, "
            f"expected {tuple(param.shape)}."
        )
    if exp_avg_sq.shape != param.shape:
        raise ValueError(
            f"Shape mismatch: exp_avg_sq has shape {tuple(exp_avg_sq.shape)}, "
            f"expected {tuple(param.shape)}."
        )

    if grad.dtype != param.dtype:
        raise ValueError(
            f"Dtype mismatch: grad has dtype {grad.dtype}, "
            f"param has dtype {param.dtype}."
        )
    if exp_avg.dtype != torch.float32:
        raise ValueError(f"exp_avg must have dtype float32, got {exp_avg.dtype}.")
    if exp_avg_sq.dtype != torch.float32:
        raise ValueError(f"exp_avg_sq must have dtype float32, got {exp_avg_sq.dtype}.")

    for name, tensor in (
        ("grad", grad),
        ("exp_avg", exp_avg),
        ("exp_avg_sq", exp_avg_sq),
    ):
        if tensor.device != param.device:
            raise ValueError(
                f"Device mismatch: {name} is on {tensor.device}, "
                f"param is on {param.device}."
            )

    check_contiguous("param", param, "for tritium.adamw_step")
    check_contiguous("grad", grad, "for tritium.adamw_step")
    check_contiguous("exp_avg", exp_avg, "for tritium.adamw_step")
    check_contiguous("exp_avg_sq", exp_avg_sq, "for tritium.adamw_step")


def adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    step: int = 1,
) -> None:
    """Update ``param`` in-place with AdamW using fp32 moments."""
    _check_adamw_inputs(param, grad, exp_avg, exp_avg_sq)

    n_elements = param.numel()
    if n_elements == 0:
        return

    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step

    kernel = as_triton_kernel(_adamw_step_kernel)
    kernel[elementwise_grid(n_elements)](
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
        n_elements,
        BLOCK_SIZE=DEFAULT_BLOCK_SIZE,
    )
