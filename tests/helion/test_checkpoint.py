import torch
from _utils import cuda_required

import helion


def test_checkpoint_forward_matches_direct() -> None:
    x = torch.randn(4, 8)

    direct = x * x + 1.0
    out = helion.checkpoint(lambda t: t * t + 1.0, x)

    assert out.requires_grad is False
    torch.testing.assert_close(out, direct)


def test_checkpoint_backward_grads_match() -> None:
    def fn(t: torch.Tensor) -> torch.Tensor:
        return (t * t + 1.0).sum()

    x = torch.randn(4, 8, requires_grad=True)
    fn(x).backward()
    grad_direct = x.grad.clone()

    x.grad = None
    helion.checkpoint(fn, x).backward()

    assert x.grad is not None
    torch.testing.assert_close(x.grad, grad_direct)


def test_checkpoint_multiple_inputs_grads_match() -> None:
    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ((a * b) + a + b).sum()

    a = torch.randn(4, requires_grad=True)
    b = torch.randn(4, requires_grad=True)
    fn(a, b).backward()
    ga, gb = a.grad.clone(), b.grad.clone()

    a.grad = None
    b.grad = None
    helion.checkpoint(fn, a, b).backward()

    torch.testing.assert_close(a.grad, ga)
    torch.testing.assert_close(b.grad, gb)


def test_checkpoint_tuple_output_grads_match() -> None:
    def fn(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return t * 2.0, t * 3.0

    x = torch.randn(4, requires_grad=True)
    o1, o2 = fn(x)
    (o1.sum() + o2.sum()).backward()
    grad_direct = x.grad.clone()

    x.grad = None
    o1, o2 = helion.checkpoint(fn, x)
    (o1.sum() + o2.sum()).backward()

    torch.testing.assert_close(x.grad, grad_direct)


def test_checkpoint_accepts_non_tensor_args() -> None:
    def fn(t: torch.Tensor, scale: float, bias: torch.Tensor) -> torch.Tensor:
        return (t * scale + bias).sum()

    x = torch.randn(4, requires_grad=True)
    bias = torch.zeros(4)

    fn(x, 2.0, bias).backward()
    grad_direct = x.grad.clone()

    x.grad = None
    helion.checkpoint(fn, x, 2.0, bias).backward()

    torch.testing.assert_close(x.grad, grad_direct)


def test_checkpoint_grad_disabled_runs_directly() -> None:
    x = torch.randn(4, requires_grad=True)
    with torch.no_grad():
        out = helion.checkpoint(lambda t: t * t, x)

    assert out.requires_grad is False
    torch.testing.assert_close(out, x * x)


def test_checkpoint_input_without_grad_runs_directly() -> None:
    x = torch.randn(4)
    out = helion.checkpoint(lambda t: t * t, x)

    assert out.requires_grad is False
    torch.testing.assert_close(out, x * x)


@cuda_required
def test_checkpoint_helion_residual_block_tuple() -> None:
    norm = helion.ResidualRMSNorm(64, device="cuda", dtype=torch.float32)
    linear = helion.Linear(64, 64, bias=False, device="cuda", dtype=torch.float32)

    def block(
        delta: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed, res = norm(delta, residual)
        return linear(normed), res

    delta = torch.randn(4, 64, device="cuda", requires_grad=True)
    residual = torch.randn(4, 64, device="cuda", requires_grad=True)
    out_delta, out_res = helion.checkpoint(block, delta, residual)

    (out_delta.sum() + out_res.sum()).backward()

    assert delta.grad is not None
    assert residual.grad is not None
    assert linear.weight.grad is not None


@cuda_required
def test_checkpoint_preserves_dropout_mask() -> None:
    linear = helion.Linear(64, 64, bias=False, device="cuda", dtype=torch.float32)
    dropout = helion.Dropout(p=0.3).to(device="cuda")

    def fn(x: torch.Tensor) -> torch.Tensor:
        return dropout(linear(x)).sum()

    torch.manual_seed(7)
    x1 = torch.randn(8, 64, device="cuda", requires_grad=True)
    fn(x1).backward()
    grad_direct = x1.grad.clone()

    torch.manual_seed(7)
    x2 = torch.randn(8, 64, device="cuda", requires_grad=True)
    helion.checkpoint(fn, x2).backward()

    assert x2.grad is not None
    torch.testing.assert_close(x2.grad, grad_direct, rtol=1e-5, atol=1e-5)


@cuda_required
def test_checkpoint_reduces_retained_memory() -> None:
    weight = torch.randn(1024, 1024, device="cuda")

    def fn(x: torch.Tensor) -> torch.Tensor:
        for _ in range(8):
            x = torch.relu(x @ weight)
        return x.sum()

    x = torch.randn(512, 1024, device="cuda", requires_grad=True)

    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    out = fn(x)
    non_ckpt_retained = torch.cuda.memory_allocated() - base
    del out
    x.grad = None

    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    out = helion.checkpoint(fn, x)
    ckpt_retained = torch.cuda.memory_allocated() - base
    del out

    assert ckpt_retained < non_ckpt_retained // 2
