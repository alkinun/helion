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
LR, BETA1, BETA2, EPS, WD = 3e-4, 0.9, 0.999, 1e-8, 0.01


def _torch_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
) -> None:
    exp_avg.mul_(BETA1).add_(grad, alpha=1 - BETA1)
    exp_avg_sq.mul_(BETA2).addcmul_(grad, grad, value=1 - BETA2)
    bias1 = 1 - BETA1**step
    bias2 = 1 - BETA2**step
    step_size = LR / bias1
    denom = (exp_avg_sq / bias2).sqrt_().add_(EPS)
    if WD != 0:
        param.mul_(1 - LR * WD)
    param.addcdiv_(exp_avg, denom, value=-step_size)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=SIZES,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="effective GB/s",
        plot_name="adamw-performance",
        args={},
    )
)
def benchmark(n: int, provider: str) -> float:
    require_cuda()
    param = torch.randn(n, device="cuda", dtype=torch.float16)
    grad = torch.randn(n, device="cuda", dtype=torch.float16)
    exp_avg = torch.zeros(n, device="cuda", dtype=torch.float32)
    exp_avg_sq = torch.zeros(n, device="cuda", dtype=torch.float32)
    step = 10

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_adamw_step(param, grad, exp_avg, exp_avg_sq, step))
    elif provider == "torch_compile":
        f = compile_fn(_torch_adamw_step)
        f(param, grad, exp_avg, exp_avg_sq, step)
        torch.cuda.synchronize()
        ms = bench_ms(lambda: f(param, grad, exp_avg, exp_avg_sq, step))
    elif provider == "tritium":
        ms = bench_ms(
            lambda: tritium.adamw_step(
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                lr=LR,
                beta1=BETA1,
                beta2=BETA2,
                eps=EPS,
                weight_decay=WD,
                step=step,
            )
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return gbps(7 * param.numel() * param.element_size(), ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)
