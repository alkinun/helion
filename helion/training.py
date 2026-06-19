from __future__ import annotations

import math

import torch
import torch.nn as nn

from .optim import _Optimizer


def clip_grad_norm(
    params: list[nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    """Clip gradient norm in-place and return the pre-clip total norm."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total_norm = torch.norm(torch.stack([g.norm(dtype=torch.float32) for g in grads]))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)
    return total_norm


class GradientAccumulator:
    """Accumulate gradients over several micro-batches to fake a larger batch.

    Divides each micro-batch loss by ``num_micro_batches`` so the summed
    gradient equals the mean over the micro-batches -- the same gradient a
    single full-batch step would produce (when each micro-batch loss is itself
    reduced by mean).  The accumulator scales, counts, and zeros; the caller
    owns the optimizer step so it composes with :func:`clip_grad_norm` and
    :class:`~helion.amp.GradScaler`.

        accum = helion.GradientAccumulator(opt, num_micro_batches=4)
        opt.zero_grad()
        for batch in loader:
            accum.backward(loss_fn(batch))        # loss * 1/4, accumulated
            if accum.ready:                       # every 4th micro-batch
                helion.clip_grad_norm(params, 1.0)
                opt.step()
                accum.reset()

    With mixed precision, feed the already-scaled loss and let the scaler own
    the step::

        accum.backward(scaler.scale(loss_fn(batch)))
        if accum.ready:
            helion.clip_grad_norm(params, 1.0)
            scaler.step(opt)
            scaler.update()
            accum.reset()
    """

    def __init__(self, optimizer: _Optimizer, num_micro_batches: int) -> None:
        if num_micro_batches < 1:
            raise ValueError(
                f"num_micro_batches must be >= 1, got {num_micro_batches}."
            )
        self.optimizer = optimizer
        self.num_micro_batches = num_micro_batches
        self.scale = 1.0 / num_micro_batches
        self.count = 0

    def backward(self, loss: torch.Tensor) -> None:
        """Scale ``loss`` by ``1 / num_micro_batches`` and accumulate grads."""
        (loss * self.scale).backward()
        self.count += 1

    @property
    def ready(self) -> bool:
        """True once ``num_micro_batches`` micro-batches have been accumulated."""
        return self.count >= self.num_micro_batches

    def step(self) -> None:
        """Run ``optimizer.step()`` then :meth:`reset` (zero grads + counter)."""
        self.optimizer.step()
        self.reset()

    def reset(self) -> None:
        """Zero gradients and reset the micro-batch counter (no optimizer step)."""
        self.optimizer.zero_grad()
        self.count = 0


class CosineLR:
    """Cosine LR schedule with linear warmup."""

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


class LinearLR:
    """Linear LR schedule with linear warmup.

    Mirrors :class:`CosineLR`: a linear warmup from 0 to ``base_lr``, followed by
    a linear decay from ``base_lr`` down to 0 at ``total_steps``.  The decay is
    clamped so the LR never goes negative when training past ``total_steps``.
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.base_lr * (1.0 - progress)


class AverageMeter:
    """Track a running (weighted) mean of scalar values.

    Handy for logging loss and metrics across a training epoch::

        meter = helion.AverageMeter()
        for batch in loader:
            loss = ...
            meter.update(loss.item(), n=batch_size)
        print(f"epoch loss: {meter.avg:.4f}")

    ``val`` holds the most recent sample, ``sum``/``count`` the weighted totals,
    and :attr:`avg` the running mean (``0.0`` before any update).
    """

    val: float
    sum: float
    count: int

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all statistics back to their empty defaults."""
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float | torch.Tensor, n: int = 1) -> None:
        """Record ``val`` weighted by ``n`` items (defaults to a single sample)."""
        self.val = float(val)
        self.sum += self.val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(val={self.val:.4g}, avg={self.avg:.4g}, n={self.count})"
