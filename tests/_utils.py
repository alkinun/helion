from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for this test.",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    return 2e-2, 2e-2


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    accumulated: bool = False,
) -> None:
    rtol, atol = tolerances(actual.dtype)
    if accumulated and actual.dtype != torch.float32:
        rtol, atol = 3e-2, 3e-2
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
