from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of a model's parameters.

    Maintains a shadow copy of each parameter updated each step via

        shadow <- decay * shadow + (1 - decay) * param

    The averaged weights are typically smoother and evaluate better than the
    live weights; they are standard for diffusion models and LLM fine-tuning.

    A decay warmup,

        effective_decay = min(decay, (1 + t) / (10 + t)),

    keeps the shadow tracking closely during the first iterations instead of
    lingering near the random initialization.  Only parameters are averaged
    (``model.parameters()``); create the :class:`EMA` after the model is on its
    target device/dtype.

    Typical use::

        ema = helion.EMA(model, decay=0.9999)
        # after every optimizer step:
        ema.update()
        # evaluate on the averaged weights:
        with ema.swapped():
            evaluate(model)
    """

    _params: list[nn.Parameter]
    shadow_params: list[torch.Tensor]

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must satisfy 0 < decay < 1, got {decay}.")
        self.decay = decay
        self.num_updates = 0
        self._params = [p for p in model.parameters()]
        if not self._params:
            raise ValueError("model has no parameters to average.")
        self.shadow_params = [p.detach().clone() for p in self._params]

    def _decay_rate(self) -> float:
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    def update(self, model: nn.Module | None = None) -> None:
        """Move each shadow parameter toward the model's current parameters."""
        self.num_updates += 1
        decay = self._decay_rate()
        params = self._params if model is None else [p for p in model.parameters()]
        if len(params) != len(self.shadow_params):
            raise ValueError(
                f"Model has {len(params)} params but EMA tracks "
                f"{len(self.shadow_params)}."
            )
        inv_decay = 1.0 - decay
        for shadow, p in zip(self.shadow_params, params, strict=True):
            shadow.mul_(decay).add_(p.detach(), alpha=inv_decay)

    @contextlib.contextmanager
    def swapped(self):
        """Evaluate the model using the averaged weights.

        Replaces each parameter's data with its shadow for the duration of the
        block, restoring the live weights on exit (including on error).
        """
        backup = [p.detach().clone() for p in self._params]
        try:
            for p, shadow in zip(self._params, self.shadow_params, strict=True):
                p.data.copy_(shadow)
            yield
        finally:
            for p, saved in zip(self._params, backup, strict=True):
                p.data.copy_(saved)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": list(self.shadow_params),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved = state_dict["shadow"]
        if len(saved) != len(self.shadow_params):
            raise ValueError(
                f"EMA state has {len(saved)} shadow params but the model has "
                f"{len(self.shadow_params)}."
            )
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = [
            t.to(device=p.device) for t, p in zip(saved, self._params, strict=True)
        ]
