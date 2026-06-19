from __future__ import annotations

import os
import statistics

import torch
import triton.testing

PROVIDER_LINE_VALS = ["torch_eager", "torch_compile", "tritium"]
PROVIDER_LINE_NAMES = ["PyTorch eager", "torch.compile", "Tritium"]
PROVIDER_STYLES = [("green", "-"), ("blue", "-"), ("red", "-")]


def bench_repeats() -> int:
    return int(os.environ.get("TRITIUM_BENCH_REPEATS", "5"))


def bench_ms(fn, repeats: int | None = None) -> float:
    if repeats is None:
        repeats = bench_repeats()
    torch.cuda.synchronize()
    times = [float(triton.testing.do_bench(fn)) for _ in range(repeats)]
    return statistics.median(times)


def compile_fn(fn):
    try:
        import torch._functorch.config as _functorch_config

        _functorch_config.donated_buffer = False
    except AttributeError:
        pass
    return torch.compile(fn)


def gbps(bytes_moved: int, ms: float) -> float:
    return bytes_moved / (ms * 1e-3) / 1e9


def tflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmarks.")
