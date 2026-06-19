import pytest
import torch
from _utils import cuda_required

import helion


class _DummyOpt:
    """Minimal optimizer stub exposing params / step / zero_grad."""

    def __init__(self, params):
        self.params = list(params)
        self.steps = 0

    def step(self):
        self.steps += 1

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# --------------------------------------------------------------------------- #
# clip_grad_norm
# --------------------------------------------------------------------------- #


def test_clip_grad_norm_returns_total_norm() -> None:
    p = torch.randn(64, requires_grad=True)
    p.grad = torch.randn(64)
    expected = p.grad.norm(dtype=torch.float32)
    total = helion.clip_grad_norm([p], max_norm=10.0)
    torch.testing.assert_close(total, expected)


def test_clip_grad_norm_clips_when_exceeding_max() -> None:
    p = torch.randn(64, requires_grad=True)
    p.grad = torch.randn(64) * 100.0
    helion.clip_grad_norm([p], max_norm=1.0)
    torch.testing.assert_close(
        p.grad.norm(dtype=torch.float32),
        torch.tensor(1.0),
        rtol=1e-4,
        atol=1e-4,
    )


def test_clip_grad_norm_leaves_small_grads_untouched() -> None:
    p = torch.randn(64, requires_grad=True)
    g = torch.randn(64) * 0.01
    p.grad = g.clone()
    helion.clip_grad_norm([p], max_norm=100.0)
    torch.testing.assert_close(p.grad, g)


def test_clip_grad_norm_handles_no_grads() -> None:
    p = torch.randn(64, requires_grad=True)
    total = helion.clip_grad_norm([p], max_norm=1.0)
    assert torch.equal(total, torch.tensor(0.0))


# --------------------------------------------------------------------------- #
# CosineLR
# --------------------------------------------------------------------------- #


def test_cosine_lr_warmup_is_linear() -> None:
    sched = helion.CosineLR(base_lr=1.0, warmup_steps=10, total_steps=100)
    assert sched(0) == pytest.approx(0.0)
    assert sched(5) == pytest.approx(0.5)
    assert sched(10) == pytest.approx(1.0)


def test_cosine_lr_decays_after_warmup() -> None:
    sched = helion.CosineLR(base_lr=1.0, warmup_steps=0, total_steps=100)
    assert sched(0) == pytest.approx(1.0)
    assert sched(100) == pytest.approx(0.0)
    assert sched(50) < sched(0)
    assert sched(50) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# LinearLR
# --------------------------------------------------------------------------- #


def test_linear_lr_warmup_is_linear() -> None:
    sched = helion.LinearLR(base_lr=1.0, warmup_steps=10, total_steps=100)
    assert sched(0) == pytest.approx(0.0)
    assert sched(5) == pytest.approx(0.5)
    assert sched(10) == pytest.approx(1.0)


def test_linear_lr_decays_linearly_to_zero() -> None:
    sched = helion.LinearLR(base_lr=2.0, warmup_steps=0, total_steps=100)
    assert sched(0) == pytest.approx(2.0)
    assert sched(50) == pytest.approx(1.0)
    assert sched(100) == pytest.approx(0.0)


def test_linear_lr_clamped_at_zero_past_total() -> None:
    sched = helion.LinearLR(base_lr=1.0, warmup_steps=0, total_steps=100)
    assert sched(150) == 0.0
    assert sched(1000) == 0.0


def test_linear_lr_with_warmup_transitions_smoothly() -> None:
    sched = helion.LinearLR(base_lr=1.0, warmup_steps=10, total_steps=110)
    assert sched(10) == pytest.approx(1.0)
    assert sched(60) == pytest.approx(0.5)
    assert sched(110) == pytest.approx(0.0)


def test_linear_lr_warmup_matches_cosine_warmup() -> None:
    # The two schedules share an identical linear warmup phase.
    lin = helion.LinearLR(base_lr=3e-4, warmup_steps=50, total_steps=500)
    cos = helion.CosineLR(base_lr=3e-4, warmup_steps=50, total_steps=500)
    for step in (0, 1, 10, 25, 49):
        assert lin(step) == pytest.approx(cos(step))


# --------------------------------------------------------------------------- #
# GradientAccumulator
# --------------------------------------------------------------------------- #


def test_accumulator_rejects_invalid_n() -> None:
    with pytest.raises(ValueError, match="num_micro_batches"):
        helion.GradientAccumulator(_DummyOpt([]), num_micro_batches=0)


def test_accumulator_backward_scales_by_one_over_n() -> None:
    x = torch.randn(8, requires_grad=True)
    accum = helion.GradientAccumulator(_DummyOpt([x]), num_micro_batches=4)
    accum.backward((x * x).sum())
    torch.testing.assert_close(x.grad, (1.0 / 4) * 2.0 * x.detach())


def test_accumulator_ready_signal() -> None:
    x = torch.randn(4, requires_grad=True)
    accum = helion.GradientAccumulator(_DummyOpt([x]), num_micro_batches=3)
    accum.backward((x * x).sum())
    assert accum.ready is False
    accum.backward((x * x).sum())
    assert accum.ready is False
    accum.backward((x * x).sum())
    assert accum.ready is True


def test_accumulator_reset_clears_grads_and_count() -> None:
    x = torch.randn(4, requires_grad=True)
    opt = _DummyOpt([x])
    accum = helion.GradientAccumulator(opt, num_micro_batches=2)
    accum.backward((x * x).sum())
    accum.reset()
    assert x.grad is None
    assert accum.count == 0
    assert accum.ready is False


def test_accumulator_step_runs_optimizer_and_resets() -> None:
    x = torch.randn(4, requires_grad=True)
    opt = _DummyOpt([x])
    accum = helion.GradientAccumulator(opt, num_micro_batches=2)
    accum.backward((x * x).sum())
    accum.backward((x * x).sum())
    assert accum.ready
    accum.step()
    assert opt.steps == 1
    assert x.grad is None
    assert accum.ready is False


def test_accumulator_matches_full_batch_gradient() -> None:
    n = 4
    micro = [torch.randn(8, requires_grad=True) for _ in range(n)]
    accum = helion.GradientAccumulator(_DummyOpt(micro), num_micro_batches=n)
    for x in micro:
        accum.backward((x * x).sum())

    # accumulated grad at each micro-batch equals the full-batch mean gradient
    full = torch.cat([x.detach() for x in micro]).requires_grad_(True)
    ((full * full).sum() / n).backward()
    full_grad = full.grad.view(n, 8)
    for i, x in enumerate(micro):
        torch.testing.assert_close(x.grad, full_grad[i])


def test_accumulator_composes_with_grad_scaler() -> None:
    n = 4
    micro = [torch.randn(8, requires_grad=True) for _ in range(n)]
    opt = _DummyOpt(micro)
    accum = helion.GradientAccumulator(opt, num_micro_batches=n)
    scaler = helion.GradScaler(init_scale=8.0)

    for x in micro:
        accum.backward(scaler.scale((x * x).sum()))
    scaler.unscale_(opt)

    for x in micro:
        torch.testing.assert_close(x.grad, (2.0 / n) * x.detach())


def test_accumulator_reset_after_scaler_step() -> None:
    x = torch.randn(8, requires_grad=True)
    opt = _DummyOpt([x])
    accum = helion.GradientAccumulator(opt, num_micro_batches=2)
    scaler = helion.GradScaler(init_scale=4.0)

    accum.backward(scaler.scale((x * x).sum()))
    accum.backward(scaler.scale((x * x).sum()))
    assert accum.ready

    scaler.step(opt)
    scaler.update()
    accum.reset()

    assert x.grad is None
    assert accum.count == 0


@cuda_required
def test_accumulator_matches_full_batch_adamw_step() -> None:
    torch.manual_seed(7)
    data = torch.randn(8, 32, device="cuda")

    def make():
        torch.manual_seed(123)  # identical init for both runs
        w = torch.randn(32, device="cuda", requires_grad=True)
        return w, helion.AdamW([w], lr=1e-2)

    def loss_fn(w, batch):
        return ((w.unsqueeze(0) - batch) ** 2).sum(dim=1).mean()

    # full batch
    wf, optf = make()
    optf.zero_grad()
    loss_fn(wf, data).backward()
    optf.step()

    # accumulated over 4 micro-batches of 2 (mean-reduced per chunk)
    wa, opta = make()
    accum = helion.GradientAccumulator(opta, num_micro_batches=4)
    opta.zero_grad()
    for chunk in data.chunk(4):
        accum.backward(loss_fn(wa, chunk))
    accum.step()

    torch.testing.assert_close(wa.detach(), wf.detach(), rtol=1e-5, atol=1e-5)
