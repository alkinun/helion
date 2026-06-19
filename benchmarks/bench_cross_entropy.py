import torch
import triton
import triton.testing

import tritium
from benchmarks._utils import (
    PROVIDER_LINE_NAMES,
    PROVIDER_LINE_VALS,
    PROVIDER_STYLES,
    bench_ms,
    compile_fn,
    gbps,
    require_cuda,
)

SHAPES = [(1, 1024), (16, 1024), (128, 1024), (16, 8192), (16, 32768)]


def _torch_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits.float(), target)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "vocab_size"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="cross-entropy-forward-performance",
        args={},
    )
)
def benchmark(n_rows: int, vocab_size: int, provider: str) -> float:
    require_cuda()
    logits = torch.randn(n_rows, vocab_size, device="cuda", dtype=torch.float16)
    target = torch.randint(vocab_size, (n_rows,), device="cuda")

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_cross_entropy(logits, target))
    elif provider == "torch_compile":
        f = compile_fn(_torch_cross_entropy)
        ms = bench_ms(lambda: f(logits, target))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.cross_entropy(logits, target))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return gbps(logits.numel() * logits.element_size(), ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)
