import pytest
import torch
import torch.nn as nn
from _utils import cuda_required

import helion


def _perturb(model: nn.Module, delta: float = 1.0) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p += delta


def test_ema_shadow_starts_as_copy_of_params() -> None:
    model = nn.Linear(4, 4)
    ema = helion.EMA(model, decay=0.99)
    for shadow, p in zip(ema.shadow_params, model.parameters(), strict=True):
        torch.testing.assert_close(shadow, p.detach())


def test_ema_first_update_uses_warmup_decay() -> None:
    model = nn.Linear(4, 4)
    ema = helion.EMA(model, decay=0.9999)
    initial = [p.detach().clone() for p in model.parameters()]
    _perturb(model, delta=1.0)
    ema.update()

    decay = min(0.9999, 2.0 / 11.0)
    for shadow, p0 in zip(ema.shadow_params, initial, strict=True):
        expected = decay * p0 + (1.0 - decay) * (p0 + 1.0)
        torch.testing.assert_close(shadow, expected)
    assert ema.num_updates == 1


def test_ema_decay_approaches_configured_after_many_updates() -> None:
    model = nn.Linear(4, 4)
    ema = helion.EMA(model, decay=0.9)
    ema.num_updates = 100_000
    assert ema._decay_rate() == pytest.approx(0.9)


def test_ema_update_tracks_model_changes() -> None:
    model = nn.Linear(8, 8)
    ema = helion.EMA(model, decay=0.5)
    # Force the configured decay to dominate (skip warmup) by setting t large.
    ema.num_updates = 100_000
    _perturb(model, delta=2.0)
    prev = [s.clone() for s in ema.shadow_params]
    ema.update()
    for shadow, before in zip(ema.shadow_params, prev, strict=True):
        # shadow moved halfway toward (before + 2): new = 0.5*before + 0.5*(before+2)
        expected = 0.5 * before + 0.5 * (before + 2.0)
        torch.testing.assert_close(shadow, expected)


def test_ema_swapped_uses_shadow_then_restores() -> None:
    model = nn.Linear(4, 4)
    ema = helion.EMA(model, decay=0.9)
    live = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for s in ema.shadow_params:
            s.zero_()

    with ema.swapped():
        for p in model.parameters():
            torch.testing.assert_close(p, torch.zeros_like(p))

    for p, original in zip(model.parameters(), live, strict=True):
        torch.testing.assert_close(p, original)


def test_ema_swapped_restores_on_exception() -> None:
    model = nn.Linear(4, 4)
    ema = helion.EMA(model, decay=0.9)
    live = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for s in ema.shadow_params:
            s.zero_()

    with pytest.raises(RuntimeError, match="boom"):
        with ema.swapped():
            raise RuntimeError("boom")

    for p, original in zip(model.parameters(), live, strict=True):
        torch.testing.assert_close(p, original)


def test_ema_state_dict_roundtrip() -> None:
    model = nn.Linear(4, 4)
    ema = helion.EMA(model, decay=0.99)
    _perturb(model, delta=1.0)
    ema.update()
    ema.update()

    state = ema.state_dict()
    ema2 = helion.EMA(model, decay=0.5)
    ema2.load_state_dict(state)

    assert ema2.decay == 0.99
    assert ema2.num_updates == 2
    for a, b in zip(ema.shadow_params, ema2.shadow_params, strict=True):
        torch.testing.assert_close(a, b)


def test_ema_load_rejects_param_count_mismatch() -> None:
    ema = helion.EMA(nn.Linear(4, 4), decay=0.9)
    state = ema.state_dict()
    ema2 = helion.EMA(nn.Linear(4, 4, bias=False), decay=0.9)
    with pytest.raises(ValueError, match="shadow params"):
        ema2.load_state_dict(state)


def test_ema_rejects_invalid_decay() -> None:
    with pytest.raises(ValueError, match="decay"):
        helion.EMA(nn.Linear(4, 4), decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        helion.EMA(nn.Linear(4, 4), decay=0.0)


def test_ema_rejects_paramless_model() -> None:
    with pytest.raises(ValueError, match="no parameters"):
        helion.EMA(nn.Identity())


@cuda_required
def test_ema_integration_with_helion_model() -> None:
    model = helion.Linear(64, 64, device="cuda", dtype=torch.float32)
    ema = helion.EMA(model, decay=0.99)
    ema.num_updates = 100_000  # use configured decay

    live = model.weight.detach().clone()
    with torch.no_grad():
        model.weight += 0.5
    ema.update()

    # shadow has moved partway from the old live value toward the new one
    assert not torch.equal(ema.shadow_params[0], live)
    assert not torch.equal(ema.shadow_params[0], model.weight.detach())

    with ema.swapped():
        torch.testing.assert_close(model.weight, ema.shadow_params[0])
    torch.testing.assert_close(model.weight, live + 0.5)


@cuda_required
def test_ema_checkpoints_via_save_checkpoint_metadata(tmp_path) -> None:
    model = helion.Linear(32, 32, device="cuda")
    ema = helion.EMA(model, decay=0.99)
    with torch.no_grad():
        model.weight += 1.0
    ema.update()

    path = tmp_path / "ckpt.pt"
    helion.save_checkpoint(path, model=model, step=1, ema_state=ema.state_dict())

    fresh = helion.Linear(32, 32, device="cuda")
    ckpt = helion.load_checkpoint(path, model=fresh)
    ema2 = helion.EMA(fresh, decay=0.5)
    ema2.load_state_dict(ckpt.metadata["ema_state"])

    assert ema2.num_updates == 1
    for a, b in zip(ema.shadow_params, ema2.shadow_params, strict=True):
        torch.testing.assert_close(a, b)
