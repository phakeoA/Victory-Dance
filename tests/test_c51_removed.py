"""Cleanup pass 2 step 3 (2026-10-04): the C51 distributional critic is gone.

A checkpoint or resume snapshot that still carries a C51 critic is REFUSED with a clear message (instead of a strict
load_state_dict error deep inside RL); its policy still serves through model_io; the scalar critic round-trips with no
C51 stamp; the run config and the CLI no longer accept the C51 knobs."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
import torch

from conftest import write_attn_ckpt
from v_dance.rl.actor_critic import ActorCritic


def _stamp_c51(path, *, stamp: bool, head: bool):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if stamp:
        ck["config"]["n_value_atoms"] = 51
    if head:
        ck["critic_state"] = {"net.value_atoms_head.weight": torch.zeros(51, 64),
                              "support": torch.linspace(-1.0, 1.0, 51)}
    torch.save(ck, path)
    return path


@pytest.mark.parametrize("stamp,head", [(True, False), (False, True), (True, True)])
def test_a_c51_checkpoint_is_refused_for_rl_but_its_policy_still_serves(tmp_path, stamp, head):
    from v_dance.play import model_io
    p = _stamp_c51(write_attn_ckpt(tmp_path / "c51.pt", seed=3), stamp=stamp, head=head)
    with pytest.raises(ValueError, match="C51"):
        ActorCritic.from_bc_checkpoint(p)
    model, heads = model_io.load_bc_policy(p)              # serving reads the (scalar) policy only
    assert model is not None and heads is not None


def test_restore_from_refuses_a_c51_checkpoint(tmp_path):
    ac = ActorCritic.from_bc_checkpoint(write_attn_ckpt(tmp_path / "bc.pt", seed=3))
    p = _stamp_c51(write_attn_ckpt(tmp_path / "c51.pt", seed=3), stamp=False, head=True)
    with pytest.raises(ValueError, match="C51"):
        ac.restore_from(p)


def test_the_scalar_critic_round_trips_with_no_c51_stamp(tmp_path):
    ac = ActorCritic.from_bc_checkpoint(write_attn_ckpt(tmp_path / "bc.pt", seed=3))
    ck = ac.state_checkpoint(generation=1)
    assert not {"n_value_atoms", "v_min", "v_max"} & set(ck["config"])
    assert not any("value_atoms" in k or k == "support" for k in ck["critic_state"])
    out = tmp_path / "gen1.pt"
    ac.save(out)
    ActorCritic.from_bc_checkpoint(out).restore_from(out)   # loads back cleanly


def test_resume_refuses_a_c51_snapshot(tmp_path):
    from v_dance.rl.ppo import PPOConfig
    from v_dance.rl.trainer import PPOTrainer, TrainConfig
    from v_dance.selfplay import resume as RS
    from v_dance.selfplay.generation import GenerationHistory
    from v_dance.selfplay.league import OpponentLeague
    ac = ActorCritic.from_bc_checkpoint(write_attn_ckpt(tmp_path / "bc.pt", seed=3))
    tr = PPOTrainer(ac, ppo_cfg=PPOConfig(), train_cfg=TrainConfig())
    p = RS.save_snapshot(tmp_path / "snap.pt", actor_critic=ac, trainer=tr,
                         league=OpponentLeague(latest_path=str(tmp_path / "bc.pt")),
                         history=GenerationHistory(), ppo_cfg=tr.cfg, train_cfg=tr.tcfg)
    RS.load_into(p, actor_critic=ac, trainer=tr)            # a scalar snapshot resumes
    snap = torch.load(p, map_location="cpu", weights_only=False)
    snap["ac_state"]["critic.net.value_atoms_head.weight"] = torch.zeros(51, 64)
    torch.save(snap, p)
    with pytest.raises(ValueError, match="C51"):
        RS.load_into(p, actor_critic=ac, trainer=tr)


def test_the_c51_knobs_are_gone_from_the_run_config(tmp_path):
    import dataclasses
    from v_dance.rl.ppo import PPOConfig
    from v_dance.selfplay import generation as GEN
    assert not {"n_atoms", "v_min", "v_max"} & {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = tmp_path / "c51_run.json"
    cfg.write_text(json.dumps({"ppo": {"value_loss_mode": "bce", "n_atoms": 51}}), encoding="utf-8")
    with pytest.raises(SystemExit):                          # a closed-run config fails LOUD, not silently
        GEN._load_run_config(str(cfg), set())


def test_the_cli_rejects_c51_and_no_longer_lists_value_atoms():
    import os
    import subprocess
    import sys
    env = {**os.environ, "PYTHONUTF8": "1"}
    bad = subprocess.run([sys.executable, "-X", "utf8", "-m", "v_dance.selfplay.generation",
                          "--value-loss-mode", "c51", "--help"], capture_output=True, text=True, env=env)
    assert bad.returncode == 2 and "invalid choice" in bad.stderr      # rejected while parsing, before --help
    hlp = subprocess.run([sys.executable, "-X", "utf8", "-m", "v_dance.selfplay.generation", "--help"],
                         capture_output=True, text=True, env=env)
    assert hlp.returncode == 0 and "--value-atoms" not in hlp.stdout and "c51" not in hlp.stdout
