import pytest
import torch
from _utils import DTYPES, assert_close, cuda_required

import tritium


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [1, 128, 4096])
def test_adamw_step_matches_torch_reference(dtype: torch.dtype, n: int) -> None:
    lr, betas, eps, wd = 3e-4, (0.9, 0.999), 1e-8, 0.01

    param = torch.randn(n, device="cuda", dtype=dtype)
    grad = torch.randn(n, device="cuda", dtype=dtype)
    exp_avg = torch.zeros(n, device="cuda", dtype=torch.float32)
    exp_avg_sq = torch.zeros(n, device="cuda", dtype=torch.float32)

    ref = torch.optim.AdamW(
        [param.detach().float().clone().requires_grad_(True)],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=wd,
    )
    ref.param_groups[0]["params"][0].grad = grad.float()
    ref.step()

    tritium.adamw_step(
        param,
        grad,
        exp_avg,
        exp_avg_sq,
        lr=lr,
        beta1=betas[0],
        beta2=betas[1],
        eps=eps,
        weight_decay=wd,
        step=1,
    )
    assert_close(param, ref.param_groups[0]["params"][0].data.to(dtype))


@cuda_required
def test_adamw_step_rejects_non_fp32_moments() -> None:
    param = torch.randn(8, device="cuda", dtype=torch.float32)
    grad = torch.randn(8, device="cuda", dtype=torch.float32)
    bad = torch.zeros(8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="exp_avg must have dtype float32"):
        tritium.adamw_step(param, grad, bad, bad, lr=1e-3, step=1)


def test_adamw_step_rejects_cpu_tensors() -> None:
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.adamw_step(
            torch.randn(4),
            torch.randn(4),
            torch.zeros(4),
            torch.zeros(4),
            lr=1e-3,
            step=1,
        )
