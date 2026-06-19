from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

import tritium


class _Optimizer:
    def __init__(self, params: Iterable[nn.Parameter]) -> None:
        self.params = list(params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


class AdamW(_Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state: dict[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] = {
            p: (
                torch.zeros_like(p, dtype=torch.float32),
                torch.zeros_like(p, dtype=torch.float32),
            )
            for p in self.params
        }
        self._step = 0

    @torch.no_grad()
    def step(self) -> None:
        self._step += 1
        for p in self.params:
            if p.grad is None:
                continue
            exp_avg, exp_avg_sq = self.state[p]
            if not p.is_contiguous():
                p.data = p.data.contiguous()
            tritium.adamw_step(
                p.data,
                p.grad.contiguous(),
                exp_avg,
                exp_avg_sq,
                lr=self.lr,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
                weight_decay=self.weight_decay,
                step=self._step,
            )


class SGD(_Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if not p.is_contiguous():
                p.data = p.data.contiguous()
            grad = grad.contiguous()
            tritium.sgd_step(p.data, grad, self.lr, weight_decay=self.weight_decay)
