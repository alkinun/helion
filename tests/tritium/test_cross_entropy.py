import pytest
import torch
import torch.nn.functional as F
from _utils import DTYPES, assert_close, cuda_required

import tritium

VOCAB_SIZES = [16, 128, 1024]


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_cross_entropy_forward(
    dtype: torch.dtype, vocab_size: int, n_rows: int
) -> None:
    logits = torch.randn(n_rows, vocab_size, device="cuda", dtype=dtype)
    target = torch.randint(vocab_size, (n_rows,), device="cuda")

    loss = tritium.cross_entropy(logits, target)
    ref = F.cross_entropy(logits.float(), target)
    torch.testing.assert_close(loss, ref, rtol=2e-2, atol=2e-3)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_cross_entropy_backward(dtype: torch.dtype) -> None:
    vocab_size, n_rows = 256, 64
    logits = torch.randn(n_rows, vocab_size, device="cuda", dtype=dtype)
    target = torch.randint(vocab_size, (n_rows,), device="cuda")
    grad_out = torch.randn((), device="cuda", dtype=torch.float32)

    logsumexp = torch.logsumexp(logits.float(), dim=-1)
    dlogits = tritium.cross_entropy_backward(grad_out, logits, target, logsumexp)

    logits_ref = logits.detach().float().clone().requires_grad_(True)
    F.cross_entropy(logits_ref, target).backward(grad_out)
    assert_close(dlogits, logits_ref.grad.to(dtype))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_cross_entropy_autograd(dtype: torch.dtype) -> None:
    vocab_size, n_rows = 128, 32
    logits = torch.randn(
        n_rows, vocab_size, device="cuda", dtype=dtype, requires_grad=True
    )
    target = torch.randint(vocab_size, (n_rows,), device="cuda")

    loss = tritium.cross_entropy(logits, target)
    loss.backward()

    logits_ref = logits.detach().float().clone().requires_grad_(True)
    F.cross_entropy(logits_ref, target).backward()
    assert_close(logits.grad, logits_ref.grad.to(dtype))


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab_size", [128, 1024])
@pytest.mark.parametrize("n_rows", [1, 16, 128])
def test_linear_cross_entropy_forward(
    dtype: torch.dtype, vocab_size: int, n_rows: int
) -> None:
    hidden_size = 64
    hidden = torch.randn(n_rows, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype)
    target = torch.randint(vocab_size, (n_rows,), device="cuda")

    loss = tritium.linear_cross_entropy(hidden, weight, target)
    ref = F.cross_entropy(F.linear(hidden, weight).float(), target)
    torch.testing.assert_close(loss, ref, rtol=2e-2, atol=2e-3)


@cuda_required
@pytest.mark.parametrize("dtype", DTYPES)
def test_linear_cross_entropy_autograd(dtype: torch.dtype) -> None:
    hidden_size, vocab_size, n_rows = 64, 256, 32
    hidden = torch.randn(
        n_rows, hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    weight = torch.randn(
        vocab_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True
    )
    target = torch.randint(vocab_size, (n_rows,), device="cuda")

    loss = tritium.linear_cross_entropy(hidden, weight, target)
    loss.backward()

    hidden_ref = hidden.detach().float().clone().requires_grad_(True)
    weight_ref = weight.detach().float().clone().requires_grad_(True)
    F.cross_entropy(F.linear(hidden_ref, weight_ref), target).backward()
    assert_close(hidden.grad, hidden_ref.grad.to(dtype))
    assert_close(weight.grad, weight_ref.grad.to(dtype), accumulated=True)


@cuda_required
def test_linear_cross_entropy_accepts_3d_hidden() -> None:
    hidden_size, vocab_size = 64, 128
    hidden = torch.randn(4, 16, hidden_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float32)
    target = torch.randint(vocab_size, (4, 16), device="cuda")

    loss = tritium.linear_cross_entropy(hidden, weight, target)
    ref = F.cross_entropy(
        F.linear(hidden, weight).view(-1, vocab_size), target.view(-1)
    )
    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-4)


def test_cross_entropy_rejects_cpu_tensors() -> None:
    with pytest.raises(ValueError, match="CUDA tensor"):
        tritium.cross_entropy(torch.randn(4, 16), torch.randint(16, (4,)))


@cuda_required
def test_cross_entropy_rejects_non_contiguous() -> None:
    logits = torch.randn(4, 16, device="cuda").T
    target = torch.randint(4, (16,), device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        tritium.cross_entropy(logits, target)
