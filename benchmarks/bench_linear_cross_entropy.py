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
)

SHAPES = [
    (16, 512, 1024),
    (128, 512, 1024),
    (128, 1024, 8192),
    (128, 2048, 32768),
    (512, 4096, 32000),
    (512, 4096, 128256),
    (1024, 4096, 32000),
]


def _torch_linear_cross_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden, weight).float(), target
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "hidden_size", "vocab_size"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="ms",
        plot_name="linear-cross-entropy-forward-performance",
        args={},
    )
)
def benchmark(n_rows: int, hidden_size: int, vocab_size: int, provider: str) -> float:
    require_cuda()
    hidden = torch.randn(n_rows, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float16)
    target = torch.randint(vocab_size, (n_rows,), device="cuda")

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_linear_cross_entropy(hidden, weight, target))
    elif provider == "torch_compile":
        f = compile_fn(_torch_linear_cross_entropy)
        ms = bench_ms(lambda: f(hidden, weight, target))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.linear_cross_entropy(hidden, weight, target))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_rows", "hidden_size", "vocab_size"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=PROVIDER_LINE_VALS,
        line_names=PROVIDER_LINE_NAMES,
        styles=PROVIDER_STYLES,
        ylabel="ms",
        plot_name="linear-cross-entropy-backward-performance",
        args={},
    )
)
def backward_benchmark(
    n_rows: int, hidden_size: int, vocab_size: int, provider: str
) -> float:
    require_cuda()
    hidden = torch.randn(
        n_rows, hidden_size, device="cuda", dtype=torch.float16, requires_grad=True
    )
    weight = torch.randn(
        vocab_size, hidden_size, device="cuda", dtype=torch.float16, requires_grad=True
    )
    target = torch.randint(vocab_size, (n_rows,), device="cuda")
    grad_out = torch.tensor(1.0, device="cuda")

    if provider == "torch_eager":
        loss = _torch_linear_cross_entropy(hidden, weight, target)
        ms = bench_ms(lambda: loss.backward(grad_out, retain_graph=True))
    elif provider == "torch_compile":
        f = compile_fn(_torch_linear_cross_entropy)
        loss = f(hidden, weight, target)
        ms = bench_ms(lambda: loss.backward(grad_out, retain_graph=True))
    elif provider == "tritium":
        loss = tritium.linear_cross_entropy(hidden.detach(), weight.detach(), target)
        logsumexp = torch.logsumexp(
            torch.nn.functional.linear(hidden.detach(), weight.detach()).float(), dim=-1
        )

        def run() -> None:
            tritium.linear_cross_entropy_backward(
                grad_out, hidden.detach(), weight.detach(), target, logsumexp
            )

        run()
        torch.cuda.synchronize()
        ms = bench_ms(run)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ms


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)
    backward_benchmark.run(print_data=True, show_plots=False)
