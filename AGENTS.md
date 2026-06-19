# AGENTS.md

Guidance for humans and AI agents working on this repo.

## What this is

A clean Triton-native LLM training stack with two layers:

- **`tritium/`** — the kernel library. Triton kernels for LLM training primitives,
  one file per op, each with full forward + backward + autograd.
- **`helion/`** — the training stack. `torch.nn.Module` subclasses and optimizer
  classes that compose Tritium ops into a standard, composable training API.

`helion` contains **no kernels**. If a kernel primitive is missing, add it to
`tritium/`, then wrap it in `helion/` if a module wrapper adds value.

## Commands

```bash
pip install -e ".[dev]"                  # install both packages with dev deps
pytest                                   # run tests (CUDA tests skip on CPU-only machines)
pytest tests/tritium/test_rmsnorm.py -q  # run a single suite
ruff check .                             # lint (must pass)
ruff format --check .                    # verify formatting (must pass)
ruff format .                            # fix formatting
python -m benchmarks.run_all             # run all benchmarks (CUDA required)
python examples/helion/train_reference.py
```

Run `ruff check .`, `ruff format --check .`, and `pytest` before considering any
change done. CUDA is required for most tests and all benchmarks.

## Layout

- `tritium/ops/<op>.py` — one Triton kernel file per op
- `tritium/__init__.py` — Tritium public API + `__all__`
- `tritium/ops/_utils.py` — shared validation/grid/dtype helpers
- `helion/nn.py` — neural-network modules (`Linear`, `RMSNorm`,
  `ResidualRMSNorm`, `SwiGLU`, `Embedding`, `Attention`)
- `helion/optim.py` — optimizers (`AdamW`, `SGD`) backed by Tritium kernels
- `helion/training.py` — training utilities (`CosineLR`, `clip_grad_norm`)
- `tests/_utils.py` — shared test helpers (`cuda_required`, tolerances)
- `tests/tritium/`, `tests/helion/` — per-package tests
- `benchmarks/bench_<op>.py` — `triton.testing` perf reports vs eager + compile
- `benchmarks/results/` — committed benchmark numbers
- `examples/tritium/`, `examples/helion/` — per-package end-to-end examples

## The op pattern (follow it exactly)

Every Tritium op is built in four layers.

- `tritium/ops/add.py` — minimal example (no backward kernel).
- `tritium/ops/swiglu.py` — canonical full example: kernel + backward kernel +
  autograd + public forward + public backward.
- `tritium/ops/matmul.py` — pattern for GEMM ops (tiled, stride-based, one kernel
  reused for forward and backward via transposed strides).
- `tritium/ops/rmsnorm.py` — pattern for reductions (multi-kernel, dispatch).

1. **Kernel(s)** — `@triton.jit` functions, pure compute, no validation. Use
   `@triton.autotune` when tile size matters. **Never autotune an in-place kernel**
   (autotuning re-runs it and corrupts state) — see `sgd.py` / `adamw.py`.
2. **Validation + forward impl** — `_check_<op>_inputs(...)` using the `_utils`
   validators, then `_xxx_forward(...)` which allocates output, handles the
   empty-tensor early return, launches via `as_triton_kernel`, and returns.
3. **Autograd** — `_XxxAutograd(torch.autograd.Function)` with `forward`/`backward`.
   `backward` must return `None` for inputs where `ctx.needs_input_grad[i]` is
   false, and make grad outputs contiguous before launching kernels.
4. **Public function** — `xxx(...)` with a docstring, gated by
   `requires_autograd(...)`: when grad isn't needed, call the forward impl
   directly to skip autograd overhead. Add an `xxx_backward(...)` public function
   only when an explicit/manual backward is useful.

## The module pattern (follow it exactly)

Every Helion module is a `torch.nn.Module` subclass that delegates compute to
Tritium. `helion/nn.py` `Linear` is the canonical example: allocate parameters
with `torch.empty`, initialize with `nn.init.*`, and call a Tritium op in
`forward`.

1. **Parameters** — `nn.Parameter(torch.empty(...))` for each learnable tensor.
   Initialize with `nn.init.normal_(..., std=0.02)` (weights) or
   `torch.zeros`/`torch.ones` (biases, norm weights).
2. **Forward** — call the Tritium op. No PyTorch compute ops except tensor
   allocation (`torch.empty`, `torch.zeros`, `torch.ones`) and indexing.
3. **Autograd** — handled automatically by Tritium's `torch.autograd.Function`
   wrappers. Helion modules never define custom backward methods.

## Shared kernel helpers (`tritium/ops/_utils.py`)

- dtype constants: `FLOAT_DTYPES`, `INDEX_DTYPES`
- validation: `check_cuda_tensor`, `check_contiguous`, `check_supported_dtype`,
  `check_same_shape_dtype_device`
- autograd gating: `requires_autograd(*tensors)` — true only when
  `torch.is_grad_enabled()` and at least one tensor has `requires_grad`
- kernel launch: `as_triton_kernel(kernel)` — keeps Triton's `kernel[grid](...)`
  syntax isolated from static type checkers
- grids: `elementwise_grid`, `autotuned_elementwise_grid`

## Conventions

- `from __future__ import annotations` at the top of every module.
- No code comments. Docstrings only on public functions/classes.
- Error messages name the offending tensor and include shape/dtype/device, e.g.
  `f"Shape mismatch: dy has shape {tuple(dy.shape)}, x has shape {tuple(x.shape)}."`
- Public kernel functions accept CUDA contiguous tensors only; validate and raise
  `ValueError` early.
- Supported float dtypes: fp16, bf16, fp32 (`FLOAT_DTYPES`). Compute in fp32
  inside kernels (`.to(tl.float32)`), store back in the input dtype. Accumulate
  reductions (e.g. `dweight`) in fp32 buffers.
- Empty-tensor inputs get an early return after allocating correctly-shaped
  outputs.
- Autograd `backward` signatures take `*grad_outputs` and index into it, so they
  tolerate missing grads from downstream.

## Tests

- Parametrize over `DTYPES = [fp32, fp16, bf16]` and sizes spanning small,
  realistic, and edge cases.
- Write a pure-PyTorch reference and compare with `torch.testing.assert_close`
  using per-dtype tolerances (see `tests/_utils.py`).
- Mark CUDA-dependent tests with `@cuda_required` (from `tests/_utils.py`).
- Cover: numerical correctness (forward + backward), autograd integration, and
  rejection cases (CPU tensor, non-contiguous, bad dtype, shape mismatch, too-large).

## Benchmarks

- Use `triton.testing.perf_report` + `triton.testing.Benchmark`; compare against
  PyTorch eager and `torch.compile` (see `benchmarks/_utils.py`).
- Report the median over repeated runs via `bench_ms` (control with
  `TRITIUM_BENCH_REPEATS`).
- Benchmark realistic shapes the op will actually see, plus edge sizes.
- Store committed results under `benchmarks/results/`.

## When to add something new

1. **New kernel primitive** → add it to `tritium/ops/<op>.py` following the op
   pattern, export it from `tritium/__init__.py`, then wrap it in `helion/` if a
   module wrapper adds value.
2. **New module** → add it to `helion/nn.py` following the module pattern, backed
   by the corresponding Tritium op.
3. **New optimizer** → add it to `helion/optim.py`, backed by the corresponding
   Tritium step kernel.
