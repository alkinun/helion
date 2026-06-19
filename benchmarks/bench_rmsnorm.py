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
from tritium.ops.rmsnorm import (
    _rmsnorm_backward_atomic,
    _rmsnorm_backward_partial_reduce,
)


def _torch_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    rstd = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * rstd * weight.float()).to(x.dtype)


def _torch_rmsnorm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_float = x.float()
    dy_float = dy.float()
    weight_float = weight.float()
    hidden_size = x.shape[-1]
    rstd = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    dot = (dy_float * weight_float * x_float).sum(dim=-1, keepdim=True)
    dx = dy_float * weight_float * rstd - x_float * dot * rstd.pow(3) / hidden_size
    dweight = (dy_float * x_float * rstd).sum(dim=0)
    return dx.to(x.dtype), dweight.to(weight.dtype)


def _torch_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = (x.float() + residual.float()).to(x.dtype)
    out = _torch_rmsnorm(residual_out, weight, eps)
    return out, residual_out


def _torch_residual_rmsnorm_backward(
    dy: torch.Tensor,
    dresidual_out: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, residual_out = _torch_residual_rmsnorm(x, residual, weight, eps)
    dz, dweight = _torch_rmsnorm_backward(dy, residual_out, weight, eps)
    dz = dz + dresidual_out
    return dz.to(x.dtype), dz.to(residual.dtype), dweight


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "hidden_size"],
        x_vals=[
            (1, 1024),
            (16, 1024),
            (128, 1024),
            (1024, 1024),
            (4096, 1024),
            (128, 2048),
            (128, 4096),
            (128, 8192),
        ],
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="rmsnorm-forward-performance",
        args={},
    )
)
def benchmark(n_rows: int, hidden_size: int, provider: str) -> float:
    require_cuda()

    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    eps = 1e-6

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_rmsnorm(x, weight, eps))
    elif provider == "torch_compile":
        compiled_rmsnorm = compile_fn(_torch_rmsnorm)
        compiled_rmsnorm(x, weight, eps)
        torch.cuda.synchronize()
        ms = bench_ms(lambda: compiled_rmsnorm(x, weight, eps))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.rmsnorm(x, weight, eps))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Logical bytes: read x, read weight, write output.
    bytes_moved = (2 * x.numel() + weight.numel()) * x.element_size()
    return gbps(bytes_moved, ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "hidden_size"],
        x_vals=[
            (1, 1024),
            (16, 1024),
            (128, 1024),
            (1024, 1024),
            (4096, 1024),
            (128, 2048),
            (128, 4096),
            (128, 8192),
        ],
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="residual-rmsnorm-forward-performance",
        args={},
    )
)
def residual_benchmark(n_rows: int, hidden_size: int, provider: str) -> float:
    require_cuda()

    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    eps = 1e-6

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_residual_rmsnorm(x, residual, weight, eps))
    elif provider == "torch_compile":
        compiled_residual_rmsnorm = compile_fn(_torch_residual_rmsnorm)
        compiled_residual_rmsnorm(x, residual, weight, eps)
        torch.cuda.synchronize()
        ms = bench_ms(lambda: compiled_residual_rmsnorm(x, residual, weight, eps))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.residual_rmsnorm(x, residual, weight, eps))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Logical bytes: read x/residual/weight, write output/residual_out.
    bytes_moved = (4 * x.numel() + weight.numel()) * x.element_size()
    return gbps(bytes_moved, ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "hidden_size"],
        x_vals=[
            (1, 1024),
            (16, 1024),
            (128, 1024),
            (1024, 1024),
            (4096, 1024),
            (128, 2048),
            (128, 4096),
            (128, 8192),
        ],
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="rmsnorm-backward-performance",
        args={},
    )
)
def backward_benchmark(n_rows: int, hidden_size: int, provider: str) -> float:
    require_cuda()

    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    dy = torch.randn_like(x)
    eps = 1e-6

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_rmsnorm_backward(dy, x, weight, eps))
    elif provider == "torch_compile":
        compiled_rmsnorm_backward = compile_fn(_torch_rmsnorm_backward)
        compiled_rmsnorm_backward(dy, x, weight, eps)
        torch.cuda.synchronize()
        ms = bench_ms(lambda: compiled_rmsnorm_backward(dy, x, weight, eps))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.rmsnorm_backward(dy, x, weight, eps))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Logical bytes: read dy/x/weight, write dx/dweight.
    bytes_moved = (3 * x.numel() + x.numel() + weight.numel()) * x.element_size()
    return gbps(bytes_moved, ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "hidden_size"],
        x_vals=[
            (1, 1024),
            (16, 1024),
            (128, 1024),
            (1024, 1024),
            (4096, 1024),
            (128, 2048),
            (128, 4096),
            (128, 8192),
        ],
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="residual-rmsnorm-backward-performance",
        args={},
    )
)
def residual_backward_benchmark(
    n_rows: int,
    hidden_size: int,
    provider: str,
) -> float:
    require_cuda()

    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    dy = torch.randn_like(x)
    dresidual_out = torch.randn_like(x)
    eps = 1e-6

    if provider == "torch_eager":
        ms = bench_ms(
            lambda: _torch_residual_rmsnorm_backward(
                dy,
                dresidual_out,
                x,
                residual,
                weight,
                eps,
            )
        )
    elif provider == "torch_compile":
        compiled_residual_rmsnorm_backward = compile_fn(
            _torch_residual_rmsnorm_backward
        )
        compiled_residual_rmsnorm_backward(dy, dresidual_out, x, residual, weight, eps)
        torch.cuda.synchronize()
        ms = bench_ms(
            lambda: compiled_residual_rmsnorm_backward(
                dy,
                dresidual_out,
                x,
                residual,
                weight,
                eps,
            )
        )
    elif provider == "tritium":
        x = x.detach().requires_grad_(True)
        residual = residual.detach().requires_grad_(True)
        weight = weight.detach().requires_grad_(True)

        def run_tritium_backward() -> None:
            x.grad = None
            residual.grad = None
            weight.grad = None
            out, residual_out = tritium.residual_rmsnorm(x, residual, weight, eps)
            torch.autograd.backward((out, residual_out), (dy, dresidual_out))

        run_tritium_backward()
        torch.cuda.synchronize()
        ms = bench_ms(run_tritium_backward)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Logical bytes: read dy/dresidual_out/x/residual/weight, write grads.
    bytes_moved = (7 * x.numel() + weight.numel()) * x.element_size()
    return gbps(bytes_moved, ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "hidden_size"],
        x_vals=[
            (1, 1024),
            (16, 1024),
            (128, 1024),
            (1024, 1024),
            (4096, 1024),
            (128, 2048),
            (128, 4096),
            (128, 8192),
        ],
        line_arg="variant",
        line_vals=["partial_reduce", "atomic"],
        line_names=["partial + reduce", "atomic"],
        styles=[("green", "-"), ("red", "-")],
        ylabel="effective GB/s",
        plot_name="rmsnorm-backward-variant-performance",
        args={},
    )
)
def backward_variant_benchmark(
    n_rows: int,
    hidden_size: int,
    variant: str,
) -> float:
    require_cuda()

    x = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)
    dy = torch.randn_like(x)
    eps = 1e-6

    if variant == "partial_reduce":
        ms = bench_ms(
            lambda: _rmsnorm_backward_partial_reduce(
                dy,
                x,
                weight,
                eps,
            )
        )
    elif variant == "atomic":
        ms = bench_ms(lambda: _rmsnorm_backward_atomic(dy, x, weight, eps))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Logical bytes: read dy/x/weight, write dx/dweight.
    bytes_moved = (3 * x.numel() + x.numel() + weight.numel()) * x.element_size()
    return gbps(bytes_moved, ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)
    residual_benchmark.run(print_data=True, show_plots=True)
    backward_benchmark.run(print_data=True, show_plots=True)
    residual_backward_benchmark.run(print_data=True, show_plots=True)
    backward_variant_benchmark.run(print_data=True, show_plots=True)
