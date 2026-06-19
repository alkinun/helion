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


def _torch_gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=SIZES,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="gelu-forward-performance",
        args={},
    )
)
def benchmark(n: int, provider: str) -> float:
    require_cuda()
    x = torch.randn(n, device="cuda", dtype=torch.float16)

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_gelu(x))
    elif provider == "torch_compile":
        f = compile_fn(_torch_gelu)
        ms = bench_ms(lambda: f(x))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.gelu(x))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return gbps(2 * x.numel() * x.element_size(), ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=SIZES,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="gelu-backward-performance",
        args={},
    )
)
def backward_benchmark(n: int, provider: str) -> float:
    require_cuda()
    x = torch.randn(n, device="cuda", dtype=torch.float16, requires_grad=True)
    dy = torch.randn(n, device="cuda", dtype=torch.float16)

    if provider == "torch_eager":
        out = _torch_gelu(x)
        ms = bench_ms(lambda: out.backward(dy, retain_graph=True))
    elif provider == "torch_compile":
        out = compile_fn(_torch_gelu)(x)
        ms = bench_ms(lambda: out.backward(dy, retain_graph=True))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.gelu_backward(dy, x.detach()))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return gbps(3 * x.numel() * x.element_size(), ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)
    backward_benchmark.run(print_data=True, show_plots=False)
