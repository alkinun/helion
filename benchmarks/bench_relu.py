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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=SIZES,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="GB/s",
        plot_name="relu-performance",
        args={},
    )
)
def benchmark(n: int, provider: str) -> float:
    require_cuda()
    x = torch.randn(n, device="cuda", dtype=torch.float16)

    if provider == "torch_eager":
        ms = bench_ms(lambda: torch.relu(x))
    elif provider == "torch_compile":
        f = compile_fn(torch.relu)
        ms = bench_ms(lambda: f(x))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.relu(x))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return gbps(2 * x.numel() * x.element_size(), ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)
