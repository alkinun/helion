import pytest
import torch
from _utils import cuda_required

import helion


class _DummyOpt:
    """Minimal optimizer stub exposing the ``params`` / ``step`` interface."""

    def __init__(self, params):
        self.params = list(params)
        self.steps = 0

    def step(self):
        self.steps += 1


def test_grad_scaler_default_scale() -> None:
    assert helion.GradScaler().get_scale() == 2.0**16
    assert helion.GradScaler(init_scale=128.0).get_scale() == 128.0


def test_grad_scaler_scale_multiplies_loss() -> None:
    scaler = helion.GradScaler(init_scale=8.0)
    loss = torch.tensor(3.0)
    torch.testing.assert_close(scaler.scale(loss), torch.tensor(24.0))


def test_grad_scaler_scale_handles_tuple() -> None:
    scaler = helion.GradScaler(init_scale=2.0)
    a, b = scaler.scale((torch.tensor(1.0), torch.tensor(4.0)))
    torch.testing.assert_close(a, torch.tensor(2.0))
    torch.testing.assert_close(b, torch.tensor(8.0))


def test_grad_scaler_unscale_divides_grads() -> None:
    x = torch.randn(8, requires_grad=True)
    opt = _DummyOpt([x])
    scaler = helion.GradScaler(init_scale=8.0)

    scaler.scale((x * x).sum()).backward()
    scaler.unscale_(opt)

    torch.testing.assert_close(x.grad, 2.0 * x.detach())


def test_grad_scaler_step_runs_when_finite() -> None:
    x = torch.randn(8, requires_grad=True)
    opt = _DummyOpt([x])
    scaler = helion.GradScaler(init_scale=2.0)

    scaler.scale((x * x).sum()).backward()
    scaler.step(opt)

    assert opt.steps == 1


def test_grad_scaler_step_skips_on_inf() -> None:
    p = torch.randn(8, requires_grad=True)
    p.grad = torch.full_like(p, float("inf"))
    opt = _DummyOpt([p])
    scaler = helion.GradScaler(init_scale=4.0)

    assert scaler.step(opt) is None
    assert opt.steps == 0


def test_grad_scaler_unscale_then_step() -> None:
    x = torch.randn(8, requires_grad=True)
    opt = _DummyOpt([x])
    scaler = helion.GradScaler(init_scale=2.0)

    scaler.scale((x * x).sum()).backward()
    scaler.unscale_(opt)
    scaler.step(opt)

    assert opt.steps == 1


def test_grad_scaler_double_unscale_raises() -> None:
    x = torch.randn(4, requires_grad=True)
    opt = _DummyOpt([x])
    scaler = helion.GradScaler()

    scaler.scale((x * x).sum()).backward()
    scaler.unscale_(opt)
    with pytest.raises(RuntimeError, match="already been called"):
        scaler.unscale_(opt)


def test_grad_scaler_update_backoff_on_overflow() -> None:
    p = torch.randn(4, requires_grad=True)
    p.grad = torch.tensor([float("inf"), 0.0, 0.0, 0.0])
    opt = _DummyOpt([p])
    scaler = helion.GradScaler(init_scale=100.0, backoff_factor=0.5)

    scaler.step(opt)
    scale_before = scaler.get_scale()
    scaler.update()

    assert scaler.get_scale() == scale_before * 0.5


def test_grad_scaler_update_grows_after_interval() -> None:
    x = torch.randn(4, requires_grad=True)
    opt = _DummyOpt([x])
    scaler = helion.GradScaler(init_scale=10.0, growth_interval=3, growth_factor=2.0)

    for _ in range(3):
        x.grad = torch.ones_like(x)
        scaler.step(opt)
        scaler.update()

    assert scaler.get_scale() == 20.0


def test_grad_scaler_update_resets_growth_tracker_on_overflow() -> None:
    x = torch.randn(4, requires_grad=True)
    finite_opt = _DummyOpt([x])
    inf_p = torch.randn(4, requires_grad=True)
    inf_opt = _DummyOpt([inf_p])

    scaler = helion.GradScaler(init_scale=10.0, growth_interval=3, growth_factor=2.0)

    for _ in range(2):
        x.grad = torch.ones_like(x)
        scaler.step(finite_opt)
        scaler.update()

    inf_p.grad = torch.tensor([float("inf"), 0.0, 0.0, 0.0])
    scaler.step(inf_opt)
    scaler.update()

    for _ in range(2):
        x.grad = torch.ones_like(x)
        scaler.step(finite_opt)
        scaler.update()
    assert scaler.get_scale() == 10.0 * 0.5

    x.grad = torch.ones_like(x)
    scaler.step(finite_opt)
    scaler.update()
    assert scaler.get_scale() == (10.0 * 0.5) * 2.0


def test_grad_scaler_update_accepts_new_scale() -> None:
    scaler = helion.GradScaler(init_scale=10.0)
    scaler.update(new_scale=128.0)
    assert scaler.get_scale() == 128.0


def test_grad_scaler_disabled_is_passthrough() -> None:
    x = torch.randn(4, requires_grad=True)
    opt = _DummyOpt([x])
    scaler = helion.GradScaler(enabled=False)

    loss = (x * x).sum()
    torch.testing.assert_close(scaler.scale(loss), loss)

    scaler.step(opt)
    assert opt.steps == 1
    scaler.update()


def test_grad_scaler_rejects_invalid_factors() -> None:
    with pytest.raises(ValueError, match="growth_factor"):
        helion.GradScaler(growth_factor=1.0)
    with pytest.raises(ValueError, match="backoff_factor"):
        helion.GradScaler(backoff_factor=1.0)


def test_autocast_enables_flag() -> None:
    assert torch.is_autocast_enabled() is False
    with helion.autocast(dtype=torch.bfloat16):
        assert torch.is_autocast_enabled() is True
    assert torch.is_autocast_enabled() is False


def test_autocast_disabled_is_noop() -> None:
    with helion.autocast(dtype=torch.bfloat16, enabled=False):
        assert torch.is_autocast_enabled() is False


@cuda_required
def test_grad_scaler_unscale_correct_with_adamw_fp16() -> None:
    torch.manual_seed(0)
    p = torch.randn(64, device="cuda", dtype=torch.float16, requires_grad=True)
    opt = helion.AdamW([p], lr=1e-3)
    scaler = helion.GradScaler(init_scale=8.0)

    scaler.scale((p * p).sum()).backward()
    ref = (2.0 * p.detach()).float()
    scaler.unscale_(opt)

    torch.testing.assert_close(p.grad.float(), ref, rtol=2e-2, atol=2e-2)


@cuda_required
def test_grad_scaler_skips_step_on_real_fp16_overflow() -> None:
    torch.manual_seed(0)
    p = torch.randn(64, device="cuda", dtype=torch.float16, requires_grad=True)
    opt = helion.AdamW([p], lr=1e-3)
    # A huge scale forces the backward to produce non-finite fp16 gradients.
    scaler = helion.GradScaler(init_scale=2.0**30)
    before = p.detach().clone()

    scaler.scale((p * p * 1e8).sum()).backward()
    scaler.step(opt)
    scaler.update()

    assert torch.equal(p.data, before)
    assert scaler.get_scale() < 2.0**30


@cuda_required
def test_grad_scaler_reduces_loss_with_adamw() -> None:
    torch.manual_seed(0)
    target = torch.randn(64, device="cuda")
    p = torch.randn(64, device="cuda", requires_grad=True)
    opt = helion.AdamW([p], lr=1e-2)
    scaler = helion.GradScaler(init_scale=128.0)

    def loss() -> torch.Tensor:
        return ((p - target) ** 2).sum()

    first = loss().item()
    for _ in range(30):
        opt.zero_grad()
        scaler.scale(loss()).backward()
        scaler.step(opt)
        scaler.update()

    assert loss().item() < first
