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
]


def _torch_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.float()
    mean = x_float.mean(dim=-1, keepdim=True)
    var = x_float.var(dim=-1, keepdim=True, correction=0)
    x_hat = (x_float - mean) * torch.rsqrt(var + eps)
    return (x_hat * weight.float() + bias.float()).to(x.dtype)


def _torch_layernorm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dy_float = dy.float()
    x_float = x.float()
    weight_float = weight.float()
    mean = x_float.mean(dim=-1, keepdim=True)
    var = x_float.var(dim=-1, keepdim=True, correction=0)
    rstd = torch.rsqrt(var + eps)
    x_hat = (x_float - mean) * rstd
    g = dy_float * weight_float
    c1 = g.mean(dim=-1, keepdim=True)
    c2 = (g * x_hat).mean(dim=-1, keepdim=True)
    dx = rstd * (g - c1 - x_hat * c2)
    dweight = (dy_float * x_hat).sum(dim=0)
    dbias = dy_float.sum(dim=0)
    return dx.to(x.dtype), dweight.to(weight.dtype), dbias.to(weight.dtype)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "hidden_size"],
        x_vals=SHAPE_VALS,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="layernorm-forward-performance",
        args={},
    )
)
def benchmark(n_rows: int, hidden_size: int, provider: str) -> float:
    require_cuda()

    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    bias = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    eps = 1e-5

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_layernorm(x, weight, bias, eps))
    elif provider == "torch_compile":
        compiled = compile_fn(_torch_layernorm)
        compiled(x, weight, bias, eps)
        torch.cuda.synchronize()
        ms = bench_ms(lambda: compiled(x, weight, bias, eps))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.layernorm(x, weight, bias, eps))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Logical bytes: read x/weight/bias, write output.
    bytes_moved = (2 * x.numel() + 2 * weight.numel()) * x.element_size()
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
        plot_name="layernorm-backward-performance",
        args={},
    )
)
def backward_benchmark(n_rows: int, hidden_size: int, provider: str) -> float:
    require_cuda()

    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    dy = torch.randn_like(x)
    eps = 1e-5

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_layernorm_backward(dy, x, weight, eps))
    elif provider == "torch_compile":
        compiled = compile_fn(_torch_layernorm_backward)
        compiled(dy, x, weight, eps)
        torch.cuda.synchronize()
        ms = bench_ms(lambda: compiled(dy, x, weight, eps))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.layernorm_backward(dy, x, weight, eps))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Logical bytes: read dy/x/weight, write dx/dweight/dbias.
    bytes_moved = (3 * x.numel() + 3 * weight.numel()) * x.element_size()
    return gbps(bytes_moved, ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
    backward_benchmark.run(print_data=True, show_plots=True)
