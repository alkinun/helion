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
LR = 1e-2


def _torch_sgd_step(param: torch.Tensor, grad: torch.Tensor, lr: float) -> None:
    torch.add(param, grad, alpha=-lr, out=param)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=SIZES,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="GB/s",
        plot_name="sgd-performance",
        args={},
    )
)
def benchmark(n: int, provider: str) -> float:
    require_cuda()
    param = torch.randn(n, device="cuda", dtype=torch.float16)
    grad = torch.randn(n, device="cuda", dtype=torch.float16)

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_sgd_step(param, grad, LR))
    elif provider == "torch_compile":
        f = compile_fn(_torch_sgd_step)
        f(param, grad, LR)
        torch.cuda.synchronize()
        ms = bench_ms(lambda: f(param, grad, LR))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.sgd_step(param, grad, LR))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return gbps(3 * param.numel() * param.element_size(), ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)
