"""B3 tests for C51 — checkpoint round-trip + auto-detect + config-boundary (docs/c51_value_head_design.md).

B3 makes a full c51 RUN work: state_checkpoint stamps the distributional config + carries the atoms head;
from_bc_checkpoint AUTO-DETECTS a saved c51 critic (so mp_collect rebuild / resume / collapse-revert
rebuild the matching head); the build_train_configs c51 guard is removed (c51 is now wired end-to-end).
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import torch

from conftest import write_attn_ckpt
from v_dance.encoders.state_encoder import get_state_dim
from v_dance.selfplay.actor_critic import ActorCritic

STATE_DIM = get_state_dim()


def _c51_ac(tmp_path, n_atoms=51, name="bc.pt"):
    return ActorCritic.from_bc_checkpoint(write_attn_ckpt(tmp_path / name, seed=3),
                                          n_value_atoms=n_atoms)


def _scalar_ac(tmp_path, name="bc_s.pt"):
    return ActorCritic.from_bc_checkpoint(write_attn_ckpt(tmp_path / name, seed=3))


def test_state_checkpoint_stamps_c51_config(tmp_path):
    c = _c51_ac(tmp_path, 51).state_checkpoint()["config"]
    assert c["n_value_atoms"] == 51 and c["v_min"] == -1.0 and c["v_max"] == 1.0
    # scalar critic stamps NOTHING (back-compat)
    s = _scalar_ac(tmp_path).state_checkpoint()["config"]
    assert "n_value_atoms" not in s


def test_from_bc_checkpoint_autodetects_saved_c51(tmp_path):
    ac = _c51_ac(tmp_path, 51)
    p = tmp_path / "c51_ac.pt"
    ac.save(p)                                            # verify=True -> the (scalar) policy must round-trip
    reloaded = ActorCritic.from_bc_checkpoint(p)          # NO n_value_atoms passed -> must auto-detect
    assert reloaded.critic.is_c51 and reloaded.critic.support.numel() == 51
    assert float(reloaded.critic.support[0]) == -1.0 and float(reloaded.critic.support[-1]) == 1.0


def test_checkpoint_roundtrip_c51_value_matches(tmp_path):
    ac = _c51_ac(tmp_path, 51)
    # perturb the atoms head so the saved critic differs from a fresh scalar-tied init
    with torch.no_grad():
        h = ac.critic.net.value_atoms_head
        h.weight.add_(0.05 * torch.randn_like(h.weight))
        h.bias.add_(0.05 * torch.randn_like(h.bias))
    x = torch.randn(8, STATE_DIM)
    with torch.no_grad():
        before = ac.value_pm(x).clone()
    p = tmp_path / "c51_ac.pt"
    ac.save(p)
    ac2 = ActorCritic.from_bc_checkpoint(p)               # auto-detect builds the matching c51 critic
    ac2.restore_from(p)                                   # load the trained critic_state (atoms + support)
    with torch.no_grad():
        after = ac2.value_pm(x)
    assert torch.allclose(before, after, atol=1e-5)       # the trained distribution round-trips exactly


def test_scalar_checkpoint_still_scalar(tmp_path):
    ac = _scalar_ac(tmp_path)
    p = tmp_path / "scalar_ac.pt"
    ac.save(p)
    assert not ActorCritic.from_bc_checkpoint(p).critic.is_c51


def test_build_train_configs_accepts_c51_now(tmp_path):
    pytest.importorskip("torch")
    from v_dance.selfplay.generation import build_train_configs
    ppo_cfg, _ = build_train_configs(ppo_overrides={"value_loss_mode": "c51", "n_atoms": 51})
    assert ppo_cfg.value_loss_mode == "c51" and ppo_cfg.n_atoms == 51
    # sanity guard still rejects a degenerate atom count
    with pytest.raises(SystemExit):
        build_train_configs(ppo_overrides={"value_loss_mode": "c51", "n_atoms": 1})
