from __future__ import annotations

import dataclasses
import os
from typing import Any

import torch
import torch.nn as nn


@dataclasses.dataclass
class Checkpoint:
    """Metadata restored by :func:`load_checkpoint`.

    Attributes:
        step: the training step saved with the checkpoint, or ``None``.
        rng_state: captured RNG state, or ``None`` if none was saved.
        metadata: any extra keyword values passed to :func:`save_checkpoint`.
    """

    step: int | None = None
    rng_state: dict[str, Any] | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def restore_rng(self) -> None:
        """Restore the saved CPU + CUDA RNG state. No-op if none was saved."""
        if self.rng_state is None:
            return
        cpu = self.rng_state.get("cpu")
        if cpu is not None:
            torch.set_rng_state(cpu)
        for device, state in zip(
            self.rng_state.get("devices", []),
            self.rng_state.get("cuda", []),
            strict=True,
        ):
            torch.cuda.set_rng_state(state, device)


def _snapshot_rng() -> dict[str, Any]:
    devices = list(range(torch.cuda.device_count()))
    return {
        "cpu": torch.get_rng_state(),
        "devices": devices,
        "cuda": [torch.cuda.get_rng_state(d) for d in devices],
    }


def _atomic_save(payload: dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Write ``payload`` to ``path`` via a temp file then atomic replace."""
    tmp = f"{path}.incomplete"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module | None = None,
    optimizer: Any = None,
    step: int | None = None,
    rng_state: bool = True,
    **metadata: Any,
) -> None:
    """Persist model/optimizer state and training metadata to ``path``.

    Writes to a temporary file in the same directory and atomically replaces the
    destination, so an interrupted save never leaves a half-written checkpoint.
    Any keyword arguments beyond those listed are stored verbatim and returned on
    :func:`load_checkpoint` as ``Checkpoint.metadata``.
    """
    payload: dict[str, Any] = {}
    if model is not None:
        payload["model"] = model.state_dict()
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if step is not None:
        payload["step"] = step
    if rng_state:
        payload["rng_state"] = _snapshot_rng()
    if metadata:
        payload["metadata"] = metadata
    _atomic_save(payload, path)


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module | None = None,
    optimizer: Any = None,
    map_location: Any = None,
    strict: bool = True,
) -> Checkpoint:
    """Restore state saved by :func:`save_checkpoint` and return its metadata.

    ``model`` and ``optimizer`` (when given) are updated in place.  Returns a
    :class:`Checkpoint` exposing ``step``, ``rng_state`` (call
    :meth:`Checkpoint.restore_rng` to apply it), and any saved ``metadata``.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model" in payload:
        model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return Checkpoint(
        step=payload.get("step"),
        rng_state=payload.get("rng_state"),
        metadata=payload.get("metadata", {}),
    )
