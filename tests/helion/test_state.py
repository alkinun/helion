import os

import pytest
import torch
import torch.nn as nn
from _utils import cuda_required

import helion


def _clone_params(params):
    return [p.detach().clone().requires_grad_(p.requires_grad) for p in params]


# --------------------------------------------------------------------------- #
# optimizer state_dict / load_state_dict
# --------------------------------------------------------------------------- #


@cuda_required
def test_adamw_state_dict_roundtrip_preserves_state() -> None:
    torch.manual_seed(0)
    p = torch.randn(32, device="cuda", requires_grad=True)
    p.grad = torch.randn(32, device="cuda")
    opt = helion.AdamW([p], lr=3e-4, weight_decay=0.01)
    opt.step()
    opt.step()

    state = opt.state_dict()
    p2 = _clone_params([p])[0]
    p2.grad = p.grad.detach().clone()
    opt2 = helion.AdamW([p2], lr=1e-1)
    opt2.load_state_dict(state)

    assert opt2.lr == opt.lr
    assert opt2.beta1 == opt.beta1
    assert opt2.beta2 == opt.beta2
    assert opt2.eps == opt.eps
    assert opt2.weight_decay == opt.weight_decay
    assert opt2._step == opt._step

    m1, v1 = opt.state[p]
    m2, v2 = opt2.state[p2]
    torch.testing.assert_close(m2, m1)
    torch.testing.assert_close(v2, v1)
    assert m2.device == p2.device


@cuda_required
def test_sgd_state_dict_roundtrip_preserves_hyperparams() -> None:
    p = torch.randn(8, device="cuda", requires_grad=True)
    p.grad = torch.randn(8, device="cuda")
    opt = helion.SGD([p], lr=1e-2, weight_decay=0.05)

    state = opt.state_dict()
    p2 = _clone_params([p])[0]
    opt2 = helion.SGD([p2], lr=1.0)
    opt2.load_state_dict(state)

    assert opt2.lr == 1e-2
    assert opt2.weight_decay == 0.05


@cuda_required
def test_adamw_load_state_dict_rejects_param_count_mismatch() -> None:
    p = torch.randn(8, device="cuda", requires_grad=True)
    opt = helion.AdamW([p], lr=1e-3)
    opt.step()
    state = opt.state_dict()

    opt2 = helion.AdamW(
        [torch.randn(8, device="cuda"), torch.randn(8, device="cuda")], lr=1e-3
    )
    with pytest.raises(ValueError, match="param entries"):
        opt2.load_state_dict(state)


@cuda_required
def test_adamw_load_state_dict_moves_state_to_param_device() -> None:
    p = torch.randn(8, device="cuda", requires_grad=True)
    p.grad = torch.randn(8, device="cuda")
    opt = helion.AdamW([p], lr=1e-3)
    opt.step()

    state = opt.state_dict()
    # Move saved state tensors to CPU to simulate a cross-device reload.
    for entry in state["state"]:
        entry["exp_avg"] = entry["exp_avg"].cpu()
        entry["exp_avg_sq"] = entry["exp_avg_sq"].cpu()

    p2 = torch.randn(8, device="cuda", requires_grad=True)
    opt2 = helion.AdamW([p2], lr=1e-3)
    opt2.load_state_dict(state)

    m, v = opt2.state[p2]
    assert m.device.type == "cuda"
    assert v.device.type == "cuda"


# --------------------------------------------------------------------------- #
# save_checkpoint / load_checkpoint
# --------------------------------------------------------------------------- #


def _make_model():
    m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    for p in m.parameters():
        p.data.fill_(0.5)
    return m


def test_save_load_checkpoint_roundtrips_model(tmp_path) -> None:
    model = _make_model()
    path = tmp_path / "ckpt.pt"
    helion.save_checkpoint(path, model=model, step=42)

    fresh = _make_model()
    for p in fresh.parameters():
        torch.testing.assert_close(p, torch.full_like(p, 0.5))

    # perturb so we can confirm the load actually overwrites
    for p in fresh.parameters():
        p.data.fill_(0.0)

    ckpt = helion.load_checkpoint(path, model=fresh)
    assert ckpt.step == 42

    for (a_name, a), (b_name, b) in zip(
        model.state_dict().items(), fresh.state_dict().items(), strict=True
    ):
        assert a_name == b_name
        torch.testing.assert_close(b, a)
    assert not (tmp_path / "ckpt.pt.incomplete").exists()


def test_save_load_checkpoint_roundtrips_metadata(tmp_path) -> None:
    path = tmp_path / "ckpt.pt"
    helion.save_checkpoint(
        path,
        model=_make_model(),
        step=7,
        epoch=3,
        config={"layers": 4, "name": "run-a"},
    )

    ckpt = helion.load_checkpoint(path)
    assert ckpt.step == 7
    assert ckpt.metadata["epoch"] == 3
    assert ckpt.metadata["config"] == {"layers": 4, "name": "run-a"}


def test_save_load_checkpoint_strict_rejects_mismatch(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    path = tmp_path / "ckpt.pt"
    helion.save_checkpoint(path, model=model)

    smaller = nn.Sequential(nn.Linear(8, 8))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        helion.load_checkpoint(path, model=smaller, strict=True)


def test_save_load_checkpoint_strict_false_allows_mismatch(tmp_path) -> None:
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    path = tmp_path / "ckpt.pt"
    helion.save_checkpoint(path, model=model)

    smaller = nn.Sequential(nn.Linear(8, 8))
    helion.load_checkpoint(path, model=smaller, strict=False)
    torch.testing.assert_close(smaller[0].weight, model[0].weight)


def test_save_checkpoint_is_atomic_no_leftover(tmp_path) -> None:
    path = tmp_path / "ckpt.pt"
    helion.save_checkpoint(path, model=_make_model(), step=1)
    assert path.exists()
    assert not (tmp_path / "ckpt.pt.incomplete").exists()


def test_save_load_checkpoint_rng_state_roundtrip(tmp_path) -> None:
    torch.manual_seed(123)
    path = tmp_path / "ckpt.pt"
    helion.save_checkpoint(path, rng_state=True)  # snapshot RNG at S0

    draw_a = torch.randn(4)  # advances past S0
    torch.randn(1000)  # advance further so a naive draw would differ

    ckpt = helion.load_checkpoint(path)
    assert ckpt.rng_state is not None
    ckpt.restore_rng()  # back to S0

    draw_b = torch.randn(4)  # reproduces draw_a
    torch.testing.assert_close(draw_b, draw_a)


def test_save_checkpoint_rng_disabled_has_none(tmp_path) -> None:
    path = tmp_path / "ckpt.pt"
    helion.save_checkpoint(path, rng_state=False)
    ckpt = helion.load_checkpoint(path)
    assert ckpt.rng_state is None
    ckpt.restore_rng()  # must be a no-op, not raise


def test_load_checkpoint_missing_keys_are_skipped(tmp_path) -> None:
    path = tmp_path / "ckpt.pt"
    # checkpoint with only metadata
    helion.save_checkpoint(path, step=5, note="meta-only")
    ckpt = helion.load_checkpoint(path, model=_make_model())
    assert ckpt.step == 5
    assert ckpt.metadata["note"] == "meta-only"


# --------------------------------------------------------------------------- #
# full resume equivalence
# --------------------------------------------------------------------------- #


class _Quadratic(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.randn(n))

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        return ((self.p - target) ** 2).sum()


@cuda_required
def test_checkpoint_resume_matches_continuous_training(tmp_path) -> None:
    target = torch.randn(32, device="cuda")

    def make() -> tuple[_Quadratic, helion.AdamW]:
        torch.manual_seed(123)  # identical init for every instance
        m: _Quadratic = _Quadratic(32).to(device="cuda")
        return m, helion.AdamW(list(m.parameters()), lr=1e-2)

    def train(m: _Quadratic, opt: helion.AdamW, count: int) -> torch.Tensor:
        for _ in range(count):
            opt.zero_grad()
            m(target).backward()
            opt.step()
        return m.p.detach().clone()

    # uninterrupted baseline
    mc, optc = make()
    continuous = train(mc, optc, 10)

    # run 4 steps, checkpoint both model + optimizer, reload, continue
    mi, opti = make()
    train(mi, opti, 4)
    path = tmp_path / "resume.pt"
    helion.save_checkpoint(path, model=mi, optimizer=opti, step=4, rng_state=False)

    mr, optr = make()
    helion.load_checkpoint(path, model=mr, optimizer=optr)
    resumed = train(mr, optr, 6)

    torch.testing.assert_close(resumed, continuous, rtol=1e-5, atol=1e-5)


def test_os_pathlike_accepted(tmp_path) -> None:
    path = os.path.join(str(tmp_path), "ckpt.pt")
    helion.save_checkpoint(path, step=1)
    ckpt = helion.load_checkpoint(path)
    assert ckpt.step == 1
