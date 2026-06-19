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

SHAPE_VALS = [
    (1, 1024),
    (16, 1024),
    (128, 1024),
    (1024, 1024),
    (4096, 1024),
    (128, 2048),
    (128, 4096),
    (128, 8192),
    (128, 11008),
    (4096, 11008),
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "hidden_size"],
        x_vals=SHAPE_VALS,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="softmax-forward-performance",
        args={},
    )
)
def benchmark(n_rows: int, hidden_size: int, provider: str) -> float:
    require_cuda()

    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)

    if provider == "torch_eager":
        ms = bench_ms(lambda: torch.softmax(x.float(), dim=-1).to(x.dtype))
    elif provider == "torch_compile":
        fn = compile_fn(lambda t: torch.softmax(t.float(), dim=-1).to(t.dtype))
        fn(x)
        torch.cuda.synchronize()
        ms = bench_ms(lambda: fn(x))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.softmax(x))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Logical bytes: read x, write output.
    bytes_moved = 2 * x.numel() * x.element_size()
    return gbps(bytes_moved, ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "hidden_size"],
        x_vals=SHAPE_VALS,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="softmax-backward-performance",
        args={},
    )
)
def backward_benchmark(n_rows: int, hidden_size: int, provider: str) -> float:
    require_cuda()

    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    out = torch.softmax(x.float(), dim=-1).to(x.dtype)
    dy = torch.randn_like(x)

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_softmax_backward(dy, out))
    elif provider == "torch_compile":
        fn = compile_fn(_torch_softmax_backward)
        fn(dy, out)
        torch.cuda.synchronize()
        ms = bench_ms(lambda: fn(dy, out))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.softmax_backward(dy, out))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Logical bytes: read dy/out, write dx.
    bytes_moved = 3 * x.numel() * x.element_size()
    return gbps(bytes_moved, ms)


def _torch_softmax_backward(
    dy: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    dy_float = dy.float()
    out_float = out.float()
    c = (dy_float * out_float).sum(dim=-1, keepdim=True)
    return (out_float * (dy_float - c)).to(out.dtype)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
    backward_benchmark.run(print_data=True, show_plots=True)
