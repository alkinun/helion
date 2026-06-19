import torch
import triton
import triton.testing

import tritium
from benchmarks._utils import bench_ms, gbps, require_cuda

SHAPES = [(128, 32, 64), (512, 32, 64), (1024, 32, 128), (2048, 32, 128)]
LINE_VALS = ["torch_eager", "tritium", "tritium_inplace"]
LINE_NAMES = ["PyTorch eager", "Tritium", "Tritium in-place"]
STYLES = [("green", "-"), ("red", "-"), ("orange", "-")]


def _make_cos_sin(
    max_seq_len: int, head_dim: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().contiguous(), freqs.sin().contiguous()


def _torch_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    n_tokens = q.shape[0]
    half = q.shape[-1] // 2
    c = cos[:n_tokens].unsqueeze(1)
    s = sin[:n_tokens].unsqueeze(1)

    def rot(t: torch.Tensor) -> torch.Tensor:
        t1, t2 = t[..., :half], t[..., half:]
        return torch.cat((t1 * c - t2 * s, t2 * c + t1 * s), dim=-1)

    return rot(q), rot(k)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_tokens", "n_heads", "head_dim"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="effective GB/s",
        plot_name="rope-performance",
        args={},
    )
)
def benchmark(n_tokens: int, n_heads: int, head_dim: int, provider: str) -> float:
    require_cuda()
    q = torch.randn(n_tokens, n_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(n_tokens, n_heads, head_dim, device="cuda", dtype=torch.float16)
    cos, sin = _make_cos_sin(n_tokens, head_dim, "cuda")

    if provider == "torch_eager":
        ms = bench_ms(lambda: _torch_rope(q, k, cos, sin))
    elif provider == "tritium":
        ms = bench_ms(lambda: tritium.rope(q, k, cos, sin))
    elif provider == "tritium_inplace":
        q_mut = q.clone()
        k_mut = k.clone()

        def run() -> None:
            tritium.rope_(q_mut, k_mut, cos, sin)

        run()
        torch.cuda.synchronize()
        ms = bench_ms(run)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    bytes_moved = (
        4 * n_tokens * n_heads * head_dim + 2 * n_tokens * (head_dim // 2)
    ) * q.element_size()
    return gbps(bytes_moved, ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=False)
