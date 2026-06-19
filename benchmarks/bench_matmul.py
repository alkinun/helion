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
    require_cuda,
    tflops,
)

SIZES = [
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
    2560,
    3072,
    3328,
    3584,
    4096,
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=SIZES,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(size: int, provider: str) -> float:
    require_cuda()
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)
    b = torch.randn(size, size, device="cuda", dtype=torch.float16)

    if provider == "torch_eager":
        ms = bench_ms(lambda: torch.matmul(a, b))
    elif provider == "torch_compile":
        f = compile_fn(torch.matmul)
        ms = bench_ms(lambda: f(a, b))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.matmul(a, b))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return tflops(2 * size * size * size, ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)
