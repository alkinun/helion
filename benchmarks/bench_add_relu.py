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

SIZES = [2**i for i in range(10, 26)]


def _torch_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=SIZES,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="add-relu-performance",
        args={},
    )
)
def benchmark(n: int, provider: str) -> float:
    require_cuda()
    x = torch.randn(n, device="cuda", dtype=torch.float16)
    y = torch.randn(n, device="cuda", dtype=torch.float16)

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_add_relu(x, y))
    elif provider == "torch_compile":
        f = compile_fn(_torch_add_relu)
        ms = bench_ms(lambda: f(x, y))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.add_relu(x, y))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return gbps(3 * x.numel() * x.element_size(), ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)
