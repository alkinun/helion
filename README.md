# Helion

Helion is a small LLM training stack built around two layers:

- `tritium`: handwritten Triton kernels for training and inference primitives.
- `helion`: a compact PyTorch-like layer over those kernels for modules,
  optimizers, and training utilities.

Project priorities, in order:

1. Cleanliness
2. Optimization
3. Being featureful

The codebase intentionally favors a small, readable API surface over broad
coverage. New features should start with clear interfaces and tests before
additional tuning.

## Layout

- `tritium/ops/`: Triton-backed primitive operations.
- `tritium/`: public kernel package exports and compatibility shims.
- `helion/`: module, optimizer, and training abstractions built on Tritium.
- `tests/`: CUDA correctness tests for Tritium primitives and Helion wrappers.
- `examples/`: small training scripts and reference comparisons.

## Development

Install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

Run the standard checks:

```bash
ruff check .
ruff format --check .
pytest -q
```

Most tests require CUDA and are skipped automatically when CUDA is unavailable.
