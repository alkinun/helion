from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Final, cast

import torch
import triton

DEFAULT_BLOCK_SIZE: Final[int] = 1024
FLOAT_DTYPES: Final = (torch.float16, torch.bfloat16, torch.float32)
INDEX_DTYPES: Final = (torch.int32, torch.int64)
ELEMENTWISE_BLOCK_SIZE_CONFIGS: Final = [
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
]


def as_triton_kernel(kernel: object) -> Any:
    """Keep Triton's launch syntax isolated from static type checkers."""
    return cast(Any, kernel)


def elementwise_grid(
    n_elements: int, block_size: int = DEFAULT_BLOCK_SIZE
) -> tuple[int]:
    return (cast(int, triton.cdiv(n_elements, block_size)),)


def autotuned_elementwise_grid(
    n_elements: int,
) -> Callable[[dict[str, Any]], tuple[int]]:
    def grid(meta: dict[str, Any]) -> tuple[int]:
        return (cast(int, triton.cdiv(n_elements, meta["BLOCK_SIZE"])),)

    return grid


def check_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor.")


def check_contiguous(name: str, tensor: torch.Tensor, reason: str) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous {reason}.")


def format_dtypes(dtypes: Iterable[torch.dtype]) -> str:
    return ", ".join(str(dtype).replace("torch.", "") for dtype in dtypes)


def check_supported_dtype(
    name: str,
    tensor: torch.Tensor,
    supported_dtypes: Iterable[torch.dtype],
) -> None:
    if tensor.dtype not in supported_dtypes:
        supported = format_dtypes(supported_dtypes)
        raise ValueError(f"{name} must have one of these dtypes: {supported}.")


def check_same_shape_dtype_device(
    lhs_name: str,
    lhs: torch.Tensor,
    rhs_name: str,
    rhs: torch.Tensor,
) -> None:
    check_cuda_tensor(lhs_name, lhs)
    check_cuda_tensor(rhs_name, rhs)

    if lhs.shape != rhs.shape:
        raise ValueError(
            f"Shape mismatch: {lhs_name} has shape {tuple(lhs.shape)}, "
            f"{rhs_name} has shape {tuple(rhs.shape)}."
        )

    if lhs.dtype != rhs.dtype:
        raise ValueError(
            f"Dtype mismatch: {lhs_name} has dtype {lhs.dtype}, "
            f"{rhs_name} has dtype {rhs.dtype}."
        )

    if lhs.device != rhs.device:
        raise ValueError(
            f"Device mismatch: {lhs_name} is on {lhs.device}, "
            f"{rhs_name} is on {rhs.device}."
        )


def requires_autograd(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)
