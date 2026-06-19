from __future__ import annotations

from typing import Any, Protocol

import torch


class _OptimizerLike(Protocol):
    """Minimal optimizer shape required by :class:`GradScaler`.

    Any Helion optimizer (an ``_Optimizer`` subclass) satisfies this; it only
    needs a ``params`` list and a ``step`` method.
    """

    params: list[Any]

    def step(self, *args: Any, **kwargs: Any) -> Any: ...


def _scale_value(outputs: Any, scale: float) -> Any:
    if isinstance(outputs, torch.Tensor):
        return outputs * scale
    if isinstance(outputs, (list, tuple)):
        scaled = [_scale_value(x, scale) for x in outputs]
        return type(outputs)(scaled)
    return outputs


def _first_grad_device(params: list[Any]) -> torch.device | None:
    for p in params:
        grad = getattr(p, "grad", None)
        if grad is not None:
            return grad.device
    return None


class GradScaler:
    """Scale a loss to prevent float16 gradient underflow.

    The narrow dynamic range of float16 flushes many gradients to zero during
    the backward pass.  ``GradScaler`` multiplies the loss by a large factor so
    gradients stay representable, then divides them back before the optimizer
    step (:meth:`unscale_`).  When backward produces non-finite gradients the
    step is skipped and the scale is reduced; after a run of stable steps the
    scale grows again.  bfloat16 shares float32's exponent range and therefore
    does not need a scaler.

    Typical use::

        scaler = helion.GradScaler()
        for ...:
            opt.zero_grad()
            with helion.autocast(dtype=torch.float16):
                loss = model(...)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be > 1.0.")
        if not 0.0 < backoff_factor < 1.0:
            raise ValueError("backoff_factor must satisfy 0 < backoff_factor < 1.")
        self._enabled = enabled
        self._scale = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0
        # Per-step state.
        self._unscaled: set[int] = set()
        self._found_inf: torch.Tensor | None = None

    def get_scale(self) -> float:
        return self._scale

    def scale(self, outputs: Any) -> Any:
        """Multiply ``outputs`` (a loss tensor, or list/tuple thereof) by the scale."""
        if not self._enabled:
            return outputs
        return _scale_value(outputs, self._scale)

    def unscale_(self, optimizer: _OptimizerLike) -> None:
        """Divide the optimizer's gradients by the current scale, in place.

        Records whether any gradient became non-finite so :meth:`step` can skip
        the update.  Calling this twice between :meth:`update` calls raises.
        """
        if not self._enabled:
            return
        opt_id = id(optimizer)
        if opt_id in self._unscaled:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the "
                "last update()."
            )
        self._unscaled.add(opt_id)

        device = _first_grad_device(optimizer.params)
        if device is None:
            return
        if self._found_inf is None or self._found_inf.device != device:
            self._found_inf = torch.zeros((), device=device, dtype=torch.float32)
        found_inf = self._found_inf
        found_inf.zero_()

        inv_scale = 1.0 / self._scale
        for p in optimizer.params:
            grad = getattr(p, "grad", None)
            if grad is None:
                continue
            grad.mul_(inv_scale)
            found_inf += torch.logical_not(torch.isfinite(grad)).any().float()

    def step(self, optimizer: _OptimizerLike, *args: Any, **kwargs: Any) -> Any:
        """Unscale (if needed) then run ``optimizer.step`` unless grads overflowed."""
        if not self._enabled:
            return optimizer.step(*args, **kwargs)
        if id(optimizer) not in self._unscaled:
            self.unscale_(optimizer)
        if self._found_inf is None or self._found_inf.item() == 0.0:
            return optimizer.step(*args, **kwargs)
        return None

    def update(self, new_scale: float | None = None) -> None:
        """Adjust the scale for the next step and clear per-step bookkeeping."""
        if not self._enabled:
            return
        self._unscaled.clear()

        if new_scale is not None:
            self._scale = float(new_scale)
            self._growth_tracker = 0
            return

        overflow = self._found_inf is not None and self._found_inf.item() > 0.0
        if overflow:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0


class autocast:
    """Context manager declaring a mixed-precision forward region.

    Wraps :class:`torch.autocast` for CUDA.  Helion's Triton kernels take their
    compute dtype from their inputs, so the usual way to run a model in low
    precision is ``model.to(dtype)``; this manager makes torch-level operations
    inside the region (loss reductions, residual adds, masks, ...) honour
    ``dtype`` and exposes the active precision via
    :func:`torch.is_autocast_enabled`.

    For float16, pair with :class:`GradScaler`::

        with helion.autocast(dtype=torch.float16):
            loss = model(...)
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        *,
        enabled: bool = True,
        device_type: str = "cuda",
    ) -> None:
        self._inner = torch.autocast(
            device_type=device_type, dtype=dtype, enabled=enabled
        )

    def __enter__(self) -> autocast:
        self._inner.__enter__()
        return self

    def __exit__(self, *exc: object) -> object:
        return self._inner.__exit__(*exc)
