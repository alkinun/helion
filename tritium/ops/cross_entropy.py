from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from ._utils import (
    FLOAT_DTYPES,
    INDEX_DTYPES,
    as_triton_kernel,
    check_contiguous,
    check_cuda_tensor,
    check_supported_dtype,
    requires_autograd,
)

SUPPORTED_LOGIT_DTYPES = FLOAT_DTYPES
SUPPORTED_TARGET_DTYPES = INDEX_DTYPES
MAX_VOCAB_BLOCK_SIZE = 131072
LINEAR_CE_BACKWARD_MAX_LOGIT_ELEMENTS = 67_108_864
LINEAR_CE_BACKWARD_TRITON_MAX_LOGIT_ELEMENTS = 65_536
LINEAR_CE_BLOCK_V = 128
LINEAR_CE_FORWARD_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_V": 128, "BLOCK_D": 64}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_V": 128, "BLOCK_D": 128}, num_warps=8),
]
LINEAR_CE_BACKWARD_BLOCK_M = 64
LINEAR_CE_BACKWARD_BLOCK_V = 64
LINEAR_CE_BACKWARD_BLOCK_D = 64


@triton.jit
def _cross_entropy_forward_kernel(
    logits_ptr,
    target_ptr,
    loss_ptr,
    logsumexp_ptr,
    n_rows: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size

    logits = tl.load(
        logits_ptr + row * vocab_size + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - row_max)
    exp_sum = tl.sum(exp_logits, axis=0)
    logsumexp = tl.log(exp_sum) + row_max

    target = tl.load(target_ptr + row)
    target_logit = tl.load(logits_ptr + row * vocab_size + target).to(tl.float32)
    nll = logsumexp - target_logit

    tl.store(logsumexp_ptr + row, logsumexp)
    if n_rows == 1:
        tl.store(loss_ptr, nll)
    else:
        tl.atomic_add(loss_ptr, nll / n_rows, sem="relaxed")


@triton.jit
def _cross_entropy_backward_kernel(
    grad_out_ptr,
    logits_ptr,
    target_ptr,
    logsumexp_ptr,
    dlogits_ptr,
    n_rows: tl.constexpr,
    vocab_size: tl.constexpr,
    grad_denominator: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size

    logits = tl.load(
        logits_ptr + row * vocab_size + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    target = tl.load(target_ptr + row)
    logsumexp = tl.load(logsumexp_ptr + row).to(tl.float32)
    grad_out = tl.load(grad_out_ptr).to(tl.float32)

    softmax = tl.exp(logits - logsumexp)
    grad = softmax - tl.where(offsets == target, 1.0, 0.0)
    grad = grad * (grad_out / grad_denominator)

    tl.store(dlogits_ptr + row * vocab_size + offsets, grad, mask=mask)


@triton.jit
def _cross_entropy_backward_inplace_kernel(
    grad_out_ptr,
    logits_ptr,
    target_ptr,
    logsumexp_ptr,
    n_rows: tl.constexpr,
    vocab_size: tl.constexpr,
    grad_denominator: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < vocab_size

    logits = tl.load(
        logits_ptr + row * vocab_size + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    target = tl.load(target_ptr + row)
    logsumexp = tl.load(logsumexp_ptr + row).to(tl.float32)
    grad_out = tl.load(grad_out_ptr).to(tl.float32)

    softmax = tl.exp(logits - logsumexp)
    grad = softmax - tl.where(offsets == target, 1.0, 0.0)
    grad = grad * (grad_out / grad_denominator)

    tl.store(logits_ptr + row * vocab_size + offsets, grad, mask=mask)


@triton.autotune(
    configs=LINEAR_CE_FORWARD_CONFIGS,
    key=["n_rows", "hidden_size", "vocab_size"],
)
@triton.jit
def _linear_cross_entropy_partial_forward_kernel(
    hidden_ptr,
    weight_ptr,
    target_ptr,
    partial_max_ptr,
    partial_exp_sum_ptr,
    partial_target_logit_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_block = tl.program_id(axis=0)
    vocab_block = tl.program_id(axis=1)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    vocab_offsets = vocab_block * BLOCK_V + tl.arange(0, BLOCK_V)
    hidden_offsets = tl.arange(0, BLOCK_D)
    row_mask = rows < n_rows
    vocab_mask = vocab_offsets < vocab_size

    logits = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
    for hidden_start in range(0, hidden_size, BLOCK_D):
        cols = hidden_start + hidden_offsets
        hidden_mask = cols < hidden_size
        hidden = tl.load(
            hidden_ptr + rows[:, None] * hidden_size + cols[None, :],
            mask=row_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + vocab_offsets[:, None] * hidden_size + cols[None, :],
            mask=vocab_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        logits += tl.dot(hidden, tl.trans(weight), input_precision="ieee")

    valid_mask = row_mask[:, None] & vocab_mask[None, :]
    masked_logits = tl.where(valid_mask, logits, -float("inf"))
    block_max = tl.max(masked_logits, axis=1)
    exp_sum = tl.sum(tl.exp(masked_logits - block_max[:, None]), axis=1)

    target = tl.load(target_ptr + rows, mask=row_mask, other=-1)
    target_logit = tl.max(
        tl.where(vocab_offsets[None, :] == target[:, None], logits, -float("inf")),
        axis=1,
    )
    partial_offsets = vocab_block * n_rows + rows
    tl.store(partial_max_ptr + partial_offsets, block_max, mask=row_mask)
    tl.store(partial_exp_sum_ptr + partial_offsets, exp_sum, mask=row_mask)
    tl.store(partial_target_logit_ptr + partial_offsets, target_logit, mask=row_mask)


@triton.jit
def _linear_cross_entropy_reduce_forward_kernel(
    partial_max_ptr,
    partial_exp_sum_ptr,
    partial_target_logit_ptr,
    loss_ptr,
    logsumexp_ptr,
    n_rows: tl.constexpr,
    n_vocab_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_vocab_blocks
    partial_offsets = offsets * n_rows + row

    partial_max = tl.load(
        partial_max_ptr + partial_offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    row_max = tl.max(partial_max, axis=0)
    partial_exp_sum = tl.load(
        partial_exp_sum_ptr + partial_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    exp_sum = tl.sum(partial_exp_sum * tl.exp(partial_max - row_max), axis=0)
    logsumexp = tl.log(exp_sum) + row_max

    partial_target_logit = tl.load(
        partial_target_logit_ptr + partial_offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    target_logit = tl.max(partial_target_logit, axis=0)
    nll = logsumexp - target_logit

    tl.store(logsumexp_ptr + row, logsumexp)
    if n_rows == 1:
        tl.store(loss_ptr, nll)
    else:
        tl.atomic_add(loss_ptr, nll / n_rows, sem="relaxed")


@triton.jit
def _linear_cross_entropy_backward_kernel(
    grad_out_ptr,
    hidden_ptr,
    weight_ptr,
    target_ptr,
    logsumexp_ptr,
    dhidden_ptr,
    dweight_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    vocab_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_FLOAT32_DOT: tl.constexpr,
    USE_BFLOAT16_GRAD: tl.constexpr,
):
    row_block = tl.program_id(axis=0)
    vocab_block = tl.program_id(axis=1)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    vocab_offsets = vocab_block * BLOCK_V + tl.arange(0, BLOCK_V)
    hidden_offsets = tl.arange(0, BLOCK_D)
    row_mask = rows < n_rows
    vocab_mask = vocab_offsets < vocab_size

    logits = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)
    for hidden_start in range(0, hidden_size, BLOCK_D):
        cols = hidden_start + hidden_offsets
        hidden_mask = cols < hidden_size
        hidden = tl.load(
            hidden_ptr + rows[:, None] * hidden_size + cols[None, :],
            mask=row_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + vocab_offsets[:, None] * hidden_size + cols[None, :],
            mask=vocab_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        logits += tl.dot(hidden, tl.trans(weight), input_precision="ieee")

    target = tl.load(target_ptr + rows, mask=row_mask, other=-1)
    logsumexp = tl.load(logsumexp_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr).to(tl.float32)

    is_target = vocab_offsets[None, :] == target[:, None]
    grad = tl.exp(logits - logsumexp[:, None]) - tl.where(is_target, 1.0, 0.0)
    grad = grad * (grad_out / n_rows)
    grad = tl.where(row_mask[:, None] & vocab_mask[None, :], grad, 0.0)

    for hidden_start in range(0, hidden_size, BLOCK_D):
        cols = hidden_start + hidden_offsets
        hidden_mask = cols < hidden_size
        hidden = tl.load(
            hidden_ptr + rows[:, None] * hidden_size + cols[None, :],
            mask=row_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + vocab_offsets[:, None] * hidden_size + cols[None, :],
            mask=vocab_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )

        if USE_FLOAT32_DOT:
            grad_for_dot = grad
        else:
            if USE_BFLOAT16_GRAD:
                grad_for_dot = grad.to(tl.bfloat16)
            else:
                grad_for_dot = grad.to(tl.float16)

        dhidden = tl.dot(grad_for_dot, weight, input_precision="ieee")
        dweight = tl.dot(tl.trans(grad_for_dot), hidden, input_precision="ieee")
        tl.atomic_add(
            dhidden_ptr + rows[:, None] * hidden_size + cols[None, :],
            dhidden,
            sem="relaxed",
            mask=row_mask[:, None] & hidden_mask[None, :],
        )
        tl.atomic_add(
            dweight_ptr + vocab_offsets[:, None] * hidden_size + cols[None, :],
            dweight,
            sem="relaxed",
            mask=vocab_mask[:, None] & hidden_mask[None, :],
        )


def _check_cross_entropy_inputs(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[int, int]:
    check_cuda_tensor("logits", logits)
    check_cuda_tensor("target", target)
    check_supported_dtype("logits", logits, SUPPORTED_LOGIT_DTYPES)

    check_supported_dtype("target", target, SUPPORTED_TARGET_DTYPES)

    if logits.dim() != 2:
        raise ValueError("logits must be a 2D tensor shaped (n_rows, vocab_size).")

    if target.dim() != 1:
        raise ValueError("target must be a 1D tensor.")

    n_rows, vocab_size = logits.shape
    if target.shape != (n_rows,):
        raise ValueError(
            f"Shape mismatch: target has shape {tuple(target.shape)}, "
            f"expected ({n_rows},)."
        )

    if target.device != logits.device:
        raise ValueError(
            f"Device mismatch: target is on {target.device}, "
            f"logits is on {logits.device}."
        )

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")

    block_size = _next_power_of_2(vocab_size)
    if block_size > MAX_VOCAB_BLOCK_SIZE:
        raise ValueError(
            "Unsupported vocab size: "
            f"next_power_of_2({vocab_size}) is {block_size}, "
            f"which exceeds {MAX_VOCAB_BLOCK_SIZE}."
        )

    check_contiguous("logits", logits, "for tritium.cross_entropy")
    check_contiguous("target", target, "for tritium.cross_entropy")

    return n_rows, vocab_size


def _check_linear_cross_entropy_inputs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
) -> tuple[int, int, int]:
    check_cuda_tensor("hidden", hidden)
    check_cuda_tensor("weight", weight)
    check_cuda_tensor("target", target)
    check_supported_dtype("hidden", hidden, SUPPORTED_LOGIT_DTYPES)
    check_supported_dtype("weight", weight, SUPPORTED_LOGIT_DTYPES)

    check_supported_dtype("target", target, SUPPORTED_TARGET_DTYPES)

    if hidden.dim() < 2:
        raise ValueError(
            "hidden must have at least 2 dimensions shaped (..., hidden_size)."
        )

    if weight.dim() != 2:
        raise ValueError("weight must be a 2D tensor shaped (vocab_size, hidden_size).")

    vocab_size, hidden_size = weight.shape
    if hidden.shape[-1] != hidden_size:
        raise ValueError(
            f"Shape mismatch: hidden has last dimension {hidden.shape[-1]}, "
            f"weight expects {hidden_size}."
        )

    if target.shape != hidden.shape[:-1]:
        raise ValueError(
            f"Shape mismatch: target has shape {tuple(target.shape)}, "
            f"expected {tuple(hidden.shape[:-1])}."
        )

    if weight.dtype != hidden.dtype:
        raise ValueError(
            f"Dtype mismatch: weight has dtype {weight.dtype}, "
            f"hidden has dtype {hidden.dtype}."
        )

    if weight.device != hidden.device:
        raise ValueError(
            f"Device mismatch: weight is on {weight.device}, "
            f"hidden is on {hidden.device}."
        )

    if target.device != hidden.device:
        raise ValueError(
            f"Device mismatch: target is on {target.device}, "
            f"hidden is on {hidden.device}."
        )

    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive.")

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive.")

    check_contiguous("hidden", hidden, "for tritium.linear_cross_entropy")
    check_contiguous("weight", weight, "for tritium.linear_cross_entropy")
    check_contiguous("target", target, "for tritium.linear_cross_entropy")

    n_rows = target.numel()
    return n_rows, hidden_size, vocab_size


def _num_warps(block_size: int, n_rows: int) -> int:
    if block_size >= 32768:
        return 16
    if block_size >= 2048:
        return 8
    if n_rows == 1:
        return 8
    return 1


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _linear_ce_backward_chunk_size(n_rows: int, vocab_size: int) -> int:
    return min(
        n_rows,
        max(1, LINEAR_CE_BACKWARD_MAX_LOGIT_ELEMENTS // vocab_size),
    )


def _cross_entropy_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_rows, vocab_size = _check_cross_entropy_inputs(logits, target)

    loss = torch.empty((), device=logits.device, dtype=torch.float32)
    logsumexp = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    if n_rows != 1:
        loss.zero_()
    if logits.numel() == 0:
        return loss, logsumexp

    block_size = _next_power_of_2(vocab_size)
    kernel = as_triton_kernel(_cross_entropy_forward_kernel)
    kernel[(n_rows,)](
        logits,
        target,
        loss,
        logsumexp,
        n_rows,
        vocab_size,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size, n_rows),
    )

    return loss, logsumexp


def _linear_cross_entropy_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_rows, hidden_size, vocab_size = _check_linear_cross_entropy_inputs(
        hidden,
        weight,
        target,
    )
    hidden_2d = hidden.view(n_rows, hidden_size)
    target_1d = target.view(n_rows)

    loss = torch.empty((), device=hidden.device, dtype=torch.float32)
    logsumexp = torch.empty((n_rows,), device=hidden.device, dtype=torch.float32)
    if n_rows != 1:
        loss.zero_()
    if n_rows == 0:
        return loss, logsumexp

    n_vocab_blocks = _ceil_div(vocab_size, LINEAR_CE_BLOCK_V)
    partial_shape = (n_vocab_blocks, n_rows)
    partial_max = torch.empty(partial_shape, device=hidden.device, dtype=torch.float32)
    partial_exp_sum = torch.empty_like(partial_max)
    partial_target_logit = torch.empty_like(partial_max)

    partial_kernel = as_triton_kernel(_linear_cross_entropy_partial_forward_kernel)
    partial_grid = lambda meta: (  # noqa: E731
        _ceil_div(n_rows, meta["BLOCK_M"]),
        n_vocab_blocks,
    )
    partial_kernel[partial_grid](
        hidden_2d,
        weight,
        target_1d,
        partial_max,
        partial_exp_sum,
        partial_target_logit,
        n_rows,
        hidden_size,
        vocab_size,
    )

    reduce_block_size = _next_power_of_2(n_vocab_blocks)
    reduce_kernel = as_triton_kernel(_linear_cross_entropy_reduce_forward_kernel)
    reduce_kernel[(n_rows,)](
        partial_max,
        partial_exp_sum,
        partial_target_logit,
        loss,
        logsumexp,
        n_rows,
        n_vocab_blocks,
        BLOCK_SIZE=reduce_block_size,
        num_warps=_num_warps(reduce_block_size, n_rows),
    )

    return loss, logsumexp


class _CrossEntropyAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss, logsumexp = _cross_entropy_forward(logits, target)
        ctx.save_for_backward(logits, target, logsumexp)
        return loss

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None

        logits, target, logsumexp = ctx.saved_tensors
        dlogits = cross_entropy_backward(grad_outputs[0], logits, target, logsumexp)
        return dlogits, None


class _LinearCrossEntropyAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        loss, logsumexp = _linear_cross_entropy_forward(hidden, weight, target)
        ctx.hidden_shape = tuple(hidden.shape)
        ctx.save_for_backward(
            hidden.view(target.numel(), hidden.shape[-1]),
            weight,
            target.view(target.numel()),
            logsumexp,
        )
        return loss

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        if not ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            return None, None, None

        hidden, weight, target, logsumexp = ctx.saved_tensors
        dhidden, dweight = linear_cross_entropy_backward(
            grad_outputs[0],
            hidden,
            weight,
            target,
            logsumexp,
        )
        if not ctx.needs_input_grad[0]:
            dhidden = None
        else:
            dhidden = dhidden.view(ctx.hidden_shape)

        if not ctx.needs_input_grad[1]:
            dweight = None

        return dhidden, dweight, None


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean cross-entropy loss for 2D logits without materialized softmax."""
    if not requires_autograd(logits):
        loss, _ = _cross_entropy_forward(logits, target)
        return loss
    return _CrossEntropyAutograd.apply(logits, target)


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return mean cross-entropy loss for ``hidden @ weight.T`` without logits."""
    if not requires_autograd(hidden, weight):
        loss, _ = _linear_cross_entropy_forward(hidden, weight, target)
        return loss
    return _LinearCrossEntropyAutograd.apply(hidden, weight, target)


def _cross_entropy_backward_impl(
    grad_out: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_denominator: int | None = None,
) -> torch.Tensor:
    n_rows, vocab_size = _check_cross_entropy_inputs(logits, target)
    grad_denominator = n_rows if grad_denominator is None else grad_denominator
    check_cuda_tensor("grad_out", grad_out)
    check_cuda_tensor("logsumexp", logsumexp)

    if grad_out.shape != ():
        raise ValueError(
            f"Shape mismatch: grad_out has shape {tuple(grad_out.shape)}, expected ()."
        )

    if logsumexp.shape != (n_rows,):
        raise ValueError(
            f"Shape mismatch: logsumexp has shape {tuple(logsumexp.shape)}, "
            f"expected ({n_rows},)."
        )

    if grad_out.device != logits.device:
        raise ValueError(
            f"Device mismatch: grad_out is on {grad_out.device}, "
            f"logits is on {logits.device}."
        )

    if logsumexp.device != logits.device:
        raise ValueError(
            f"Device mismatch: logsumexp is on {logsumexp.device}, "
            f"logits is on {logits.device}."
        )

    if logsumexp.dtype != torch.float32:
        raise ValueError(
            f"Dtype mismatch: logsumexp has dtype {logsumexp.dtype}, expected float32."
        )

    check_contiguous("logsumexp", logsumexp, "for tritium.cross_entropy")

    dlogits = torch.empty_like(logits)
    if logits.numel() == 0:
        return dlogits

    block_size = _next_power_of_2(vocab_size)
    kernel = as_triton_kernel(_cross_entropy_backward_kernel)
    kernel[(n_rows,)](
        grad_out,
        logits,
        target,
        logsumexp,
        dlogits,
        n_rows,
        vocab_size,
        grad_denominator,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size, n_rows),
    )

    return dlogits


def cross_entropy_backward(
    grad_out: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    logsumexp: torch.Tensor,
) -> torch.Tensor:
    """Return ``dlogits`` from saved per-row logsumexp values."""
    return _cross_entropy_backward_impl(grad_out, logits, target, logsumexp)


def _cross_entropy_backward_inplace_unchecked(
    grad_out: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_denominator: int,
) -> None:
    n_rows, vocab_size = logits.shape
    block_size = _next_power_of_2(vocab_size)
    kernel = as_triton_kernel(_cross_entropy_backward_inplace_kernel)
    kernel[(n_rows,)](
        grad_out,
        logits,
        target,
        logsumexp,
        n_rows,
        vocab_size,
        grad_denominator,
        BLOCK_SIZE=block_size,
        num_warps=_num_warps(block_size, n_rows),
    )


def _linear_cross_entropy_backward_triton(
    grad_out: torch.Tensor,
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    target_1d: torch.Tensor,
    logsumexp: torch.Tensor,
    n_rows: int,
    hidden_size: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dhidden_accum = torch.zeros_like(hidden_2d, dtype=torch.float32)
    dweight_accum = torch.zeros_like(weight, dtype=torch.float32)
    kernel = as_triton_kernel(_linear_cross_entropy_backward_kernel)
    grid = (
        _ceil_div(n_rows, LINEAR_CE_BACKWARD_BLOCK_M),
        _ceil_div(vocab_size, LINEAR_CE_BACKWARD_BLOCK_V),
    )
    kernel[grid](
        grad_out,
        hidden_2d,
        weight,
        target_1d,
        logsumexp,
        dhidden_accum,
        dweight_accum,
        n_rows,
        hidden_size,
        vocab_size,
        BLOCK_M=LINEAR_CE_BACKWARD_BLOCK_M,
        BLOCK_V=LINEAR_CE_BACKWARD_BLOCK_V,
        BLOCK_D=LINEAR_CE_BACKWARD_BLOCK_D,
        USE_FLOAT32_DOT=hidden_2d.dtype == torch.float32,
        USE_BFLOAT16_GRAD=hidden_2d.dtype == torch.bfloat16,
        num_warps=4,
    )
    return dhidden_accum.to(hidden_2d.dtype), dweight_accum.to(weight.dtype)


def linear_cross_entropy_backward(
    grad_out: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    logsumexp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gradients for ``linear_cross_entropy`` from saved logsumexp."""
    n_rows, hidden_size, vocab_size = _check_linear_cross_entropy_inputs(
        hidden,
        weight,
        target,
    )
    check_cuda_tensor("grad_out", grad_out)
    check_cuda_tensor("logsumexp", logsumexp)

    if grad_out.shape != ():
        raise ValueError(
            f"Shape mismatch: grad_out has shape {tuple(grad_out.shape)}, expected ()."
        )

    if logsumexp.shape != (n_rows,):
        raise ValueError(
            f"Shape mismatch: logsumexp has shape {tuple(logsumexp.shape)}, "
            f"expected ({n_rows},)."
        )

    if grad_out.device != hidden.device:
        raise ValueError(
            f"Device mismatch: grad_out is on {grad_out.device}, "
            f"hidden is on {hidden.device}."
        )

    if logsumexp.device != hidden.device:
        raise ValueError(
            f"Device mismatch: logsumexp is on {logsumexp.device}, "
            f"hidden is on {hidden.device}."
        )

    if logsumexp.dtype != torch.float32:
        raise ValueError(
            f"Dtype mismatch: logsumexp has dtype {logsumexp.dtype}, expected float32."
        )

    check_contiguous("logsumexp", logsumexp, "for tritium.linear_cross_entropy")

    hidden_2d = hidden.view(n_rows, hidden_size)
    target_1d = target.view(n_rows)
    dhidden = torch.empty_like(hidden_2d)
    if n_rows == 0:
        return dhidden, torch.zeros_like(weight)

    if n_rows * vocab_size <= LINEAR_CE_BACKWARD_TRITON_MAX_LOGIT_ELEMENTS:
        return _linear_cross_entropy_backward_triton(
            grad_out,
            hidden_2d,
            weight,
            target_1d,
            logsumexp,
            n_rows,
            hidden_size,
            vocab_size,
        )

    dweight_accum = torch.zeros_like(weight)
    chunk_size = _linear_ce_backward_chunk_size(n_rows, vocab_size)
    with torch.no_grad():
        first_chunk = True
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            hidden_chunk = hidden_2d[start:end]
            target_chunk = target_1d[start:end]
            logsumexp_chunk = logsumexp[start:end]

            logits_chunk = torch.nn.functional.linear(hidden_chunk, weight)
            _cross_entropy_backward_inplace_unchecked(
                grad_out,
                logits_chunk,
                target_chunk,
                logsumexp_chunk,
                grad_denominator=n_rows,
            )
            torch.mm(logits_chunk, weight, out=dhidden[start:end])
            if first_chunk:
                torch.mm(logits_chunk.t(), hidden_chunk, out=dweight_accum)
                first_chunk = False
            else:
                torch.addmm(
                    dweight_accum,
                    logits_chunk.t(),
                    hidden_chunk,
                    beta=1.0,
                    alpha=1.0,
                    out=dweight_accum,
                )

    return dhidden, dweight_accum
