from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _cuda_device_indices(tensors: tuple[torch.Tensor, ...]) -> list[int]:
    """Distinct CUDA device indices referenced by ``tensors`` (sorted)."""
    seen: set[int] = set()
    for t in tensors:
        if t.is_cuda:
            seen.add(t.device.index or 0)
    return sorted(seen)


class _CheckpointFunction(torch.autograd.Function):
    """Recompute the wrapped function's activations during ``backward``.

    This is the reentrant activation-checkpoint primitive. ``forward`` runs the
    function under ``no_grad`` so no intermediate graph is retained; only the
    tensor inputs are saved. ``backward`` detaches those inputs, re-runs the
    function under ``enable_grad`` to rebuild the subgraph, and propagates the
    upstream gradients through it.
    """

    @staticmethod
    def forward(
        ctx: Any,
        function: Callable[..., Any],
        preserve_rng_state: bool,
        *args: Any,
    ) -> Any:
        ctx.function = function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.is_tensor_flags = [torch.is_tensor(a) for a in args]
        ctx.n_args = len(args)

        tensor_inputs = [a for a in args if torch.is_tensor(a)]
        ctx.non_tensor_args = [
            (i, a) for i, a in enumerate(args) if not torch.is_tensor(a)
        ]
        ctx.save_for_backward(*tensor_inputs)

        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.fwd_gpu_devices = _cuda_device_indices(tuple(tensor_inputs))
            ctx.fwd_gpu_states = [
                torch.cuda.get_rng_state(d) for d in ctx.fwd_gpu_devices
            ]

        with torch.no_grad():
            outputs = function(*args)
        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        saved = list(ctx.saved_tensors)

        # Rebuild the argument tuple: detached tensor inputs (with grad enabled
        # so a fresh graph is recorded) interleaved with the stashed non-tensors.
        recompute_args: list[Any] = [None] * ctx.n_args
        recompute_inputs: list[torch.Tensor] = []
        t_idx = 0
        for i, is_tensor in enumerate(ctx.is_tensor_flags):
            if is_tensor:
                leaf = saved[t_idx].detach()
                leaf.requires_grad_(True)
                recompute_args[i] = leaf
                recompute_inputs.append(leaf)
                t_idx += 1
        for i, value in ctx.non_tensor_args:
            recompute_args[i] = value

        if ctx.preserve_rng_state:
            with torch.random.fork_rng(devices=ctx.fwd_gpu_devices):
                torch.set_rng_state(ctx.fwd_cpu_state)
                for device, state in zip(
                    ctx.fwd_gpu_devices, ctx.fwd_gpu_states, strict=True
                ):
                    torch.cuda.set_rng_state(state, device)
                with torch.enable_grad():
                    outputs = ctx.function(*recompute_args)
        else:
            with torch.enable_grad():
                outputs = ctx.function(*recompute_args)

        if isinstance(outputs, tuple):
            outputs_tuple = outputs
        else:
            outputs_tuple = (outputs,)
        torch.autograd.backward(outputs_tuple, grad_outputs)

        grads: list[Any] = []
        for is_tensor in ctx.is_tensor_flags:
            if is_tensor:
                leaf = recompute_inputs.pop(0)
                grads.append(leaf.grad)
            else:
                grads.append(None)
        # Leading ``None``s correspond to ``function`` and ``preserve_rng_state``.
        return (None, None, *grads)


def checkpoint(
    function: Callable[..., Any],
    *args: Any,
    preserve_rng_state: bool = True,
) -> Any:
    """Recompute ``function(*args)`` during the backward pass to save memory.

    The forward pass runs under ``no_grad``, so only the inputs to ``function``
    are retained for backward; its intermediate activations are discarded and
    rebuilt by re-running the forward under ``enable_grad``. This trades extra
    compute for a large reduction in activation memory, the standard trade-off
    for activation (gradient) checkpointing.

    Random state is saved on entry and restored before the recompute, so
    stochastic ops such as :class:`helion.Dropout` reproduce the exact same mask
    on the second pass. Set ``preserve_rng_state=False`` only when the wrapped
    function is deterministic.

    Returns whatever ``function`` returns (a tensor or a tuple of tensors).
    """
    if not torch.is_grad_enabled():
        return function(*args)
    if not any(torch.is_tensor(a) and a.requires_grad for a in args):
        return function(*args)
    return _CheckpointFunction.apply(function, preserve_rng_state, *args)
