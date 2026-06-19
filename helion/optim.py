from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

import tritium


class _Optimizer:
    def __init__(self, params: Iterable[nn.Parameter]) -> None:
        self.params = list(params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError


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

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "state": [
                {"exp_avg": m, "exp_avg_sq": v}
                for m, v in (self.state[p] for p in self.params)
            ],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._step = state_dict["step"]
        self.lr = state_dict["lr"]
        self.beta1 = state_dict["beta1"]
        self.beta2 = state_dict["beta2"]
        self.eps = state_dict["eps"]
        self.weight_decay = state_dict["weight_decay"]

        saved = state_dict["state"]
        if len(saved) != len(self.params):
            raise ValueError(
                f"Optimizer state has {len(saved)} param entries but the "
                f"optimizer has {len(self.params)} params."
            )
        self.state = {
            p: (
                entry["exp_avg"].to(device=p.device),
                entry["exp_avg_sq"].to(device=p.device),
            )
            for p, entry in zip(self.params, saved, strict=True)
        }


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

    def state_dict(self) -> dict[str, Any]:
        return {"lr": self.lr, "weight_decay": self.weight_decay}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.lr = state_dict["lr"]
        self.weight_decay = state_dict["weight_decay"]
