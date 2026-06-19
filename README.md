# Helion

Clean Triton-native LLM training stack. Two layers in one repo:

| Layer | Package | What lives there |
|---|---|---|
| **Kernels** | `tritium/` | All Triton kernels (`matmul`, `rmsnorm`, `swiglu`, `rope`, `attention`, `cross_entropy`, `linear_cross_entropy`, `adamw_step`, `sgd_step`, ...). One file per op, autograd-enabled. |
| **Training stack** | `helion/` | `torch.nn.Module` subclasses and optimizer classes that compose Tritium ops. No kernels. |

If a kernel primitive is missing, add it to `tritium/`, then wrap it in `helion/`.

## Philosophy

Every forward pass, backward pass, and optimizer step runs through a Triton kernel — no PyTorch compute ops, no `torch.compile`. PyTorch is used only for tensor allocation, autograd graph construction, and cuBLAS dispatch where Tritium defers to the shared compute floor. The result is a stack legible enough to read in a day: a person can understand exactly what every line does.

Priorities, strictly ordered:

1. **Cleanliness** — minimal public surface, consistent patterns, no speculative complexity.
2. **Speed** — delegate all compute to Triton kernels; no `torch.compile` baggage.
3. **Features** — add modules/ops only when a real workload needs them.

## Performance

Tritium kernels achieve `torch.compile`-class performance without the baggage of `torch.compile` itself. The truth: no magic perf wins — ~75% of every training step is cuBLAS GEMMs + attention, and everyone shares that compute floor. Tritium's advantage is **avoiding `torch.compile`'s baggage**: no compile latency, no recompilation on shape changes, no graph-capture constraints. Committed numbers live under `benchmarks/results/`.

## Install

```bash
pip install -e ".[dev]"
```

This installs both the `tritium` (kernels) and `helion` (modules) packages.

## Test

```bash
pytest
```

CUDA tests skip automatically on CPU-only machines.

## Benchmark

```bash
python -m benchmarks.run_all            # all ops
python -m benchmarks.bench_matmul       # one op
TRITIUM_BENCH_REPEATS=7 python -m benchmarks.run_all
```

## Examples

Tritium end-to-end (every op is a Triton kernel):

```bash
python examples/tritium/train_reference.py --steps 20 --dtype bfloat16
python examples/tritium/train_compare.py --layers 6     # vs eager + torch.compile
```

Helion end-to-end (modules + optimizers built on Tritium):

```bash
python examples/helion/train_reference.py --steps 20 --dtype bfloat16
python examples/helion/train_shakespeare.py             # char-LM that generates text
```

## Contributing

See [`AGENTS.md`](AGENTS.md) for the op pattern, module pattern, conventions, and commands. CI runs `ruff` and `pytest` on every push.
