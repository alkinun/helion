from __future__ import annotations

import math

import torch
import torch.nn as nn


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
