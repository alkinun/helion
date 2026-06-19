import torch
from _utils import cuda_required

import helion


@cuda_required
def test_sgd_step_decreases_loss() -> None:
    torch.manual_seed(0)
    p = torch.randn(64, device="cuda", requires_grad=True)
    p.grad = torch.randn(64, device="cuda")
    before = p.detach().clone()
    opt = helion.SGD([p], lr=1e-1)
    opt.step()
    assert not torch.equal(before, p.data)


@cuda_required
def test_sgd_weight_decay_matches_reference() -> None:
    torch.manual_seed(0)
    p = torch.randn(64, device="cuda", requires_grad=True)
    p.grad = torch.randn(64, device="cuda")
    before = p.detach().clone()
    grad = p.grad.detach().clone()

    opt = helion.SGD([p], lr=1e-1, weight_decay=0.01)
    opt.step()

    ref = before - 1e-1 * (grad + 0.01 * before)
    torch.testing.assert_close(p.data, ref)


@cuda_required
def test_adamw_step_matches_torch() -> None:
    torch.manual_seed(0)
    p = torch.randn(64, device="cuda")
    grad = torch.randn(64, device="cuda")

    ref = torch.optim.AdamW(
        [p.detach().clone().requires_grad_(True)], lr=3e-4, weight_decay=0.01
    )
    ref.param_groups[0]["params"][0].grad = grad
    ref.step()

    p2 = p.detach().clone().requires_grad_(True)
    p2.grad = grad
    opt = helion.AdamW([p2], lr=3e-4, weight_decay=0.01)
    opt.step()

    torch.testing.assert_close(
        p2.data, ref.param_groups[0]["params"][0].data, rtol=2e-2, atol=2e-2
    )


@cuda_required
def test_adamw_zero_grad_clears() -> None:
    p = torch.randn(16, device="cuda", requires_grad=True)
    opt = helion.AdamW([p], lr=1e-3)
    p.grad = torch.randn(16, device="cuda")
    opt.zero_grad()
    assert p.grad is None


@cuda_required
def test_adamw_reduces_loss_on_quadratic() -> None:
    torch.manual_seed(0)
    target = torch.randn(32, device="cuda")
    p = torch.randn(32, device="cuda", requires_grad=True)
    opt = helion.AdamW([p], lr=1e-2)

    def loss() -> torch.Tensor:
        return ((p - target) ** 2).sum()

    first = loss().item()
    for _ in range(20):
        opt.zero_grad()
        loss().backward()
        opt.step()
    last = loss().item()
    assert last < first
