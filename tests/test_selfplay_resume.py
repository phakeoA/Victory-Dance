"""Task 3c.4: resumability — snapshot round-trip, the chunked==continuous determinism
guarantee (the whole point), and the graceful StopController.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
from v_dance.encoders.state_encoder import (get_state_dim, get_action_dim, get_gimmick_dim,  # noqa: E402
                           get_state_layout_version)
from conftest import write_attn_ckpt  # noqa: E402
from v_dance.selfplay.actor_critic import ActorCritic  # noqa: E402
from v_dance.selfplay.collector import TrajectoryCollector  # noqa: E402
from v_dance.selfplay.reward import place_terminal_reward  # noqa: E402
from v_dance.selfplay.schema import PASS_ACTION  # noqa: E402
from v_dance.selfplay import policy_eval as pe  # noqa: E402
from v_dance.selfplay.ppo import PPOConfig  # noqa: E402
from v_dance.selfplay.trainer import PPOTrainer, TrainConfig  # noqa: E402
from v_dance.selfplay.league import OpponentLeague  # noqa: E402
from v_dance.selfplay.generation import GenerationHistory, GenerationRecord, GenConfig  # noqa: E402
from v_dance.selfplay import resume as RS  # noqa: E402

SD, AD, GD = get_state_dim(), get_action_dim(), get_gimmick_dim()


def _write_ckpt(tmp_path, name="bc.pt"):
    return write_attn_ckpt(tmp_path / name)


def _ac(ckpt):
    return ActorCritic.from_bc_checkpoint(ckpt)


def _trainer(ac, seed=7):
    return PPOTrainer(ac, PPOConfig(), TrainConfig(minibatch_size=4, ppo_epochs=3,
                                                   actor_lr=1e-3, critic_lr=1e-3), seed=seed)


def _trajs(ac, seed=0):
    out = []
    for w in (True, False):
        c = TrajectoryCollector(f"g{seed}{w}", "p1" if w else "p2")
        rng = np.random.default_rng(seed + int(w))
        for t in range(4):
            st_state = rng.standard_normal(SD).astype(np.float32)
            step = type("S", (), {"state": st_state})
            lp, _e, v = pe.evaluate_actions(ac, [type("T", (), {
                "state": st_state, "action_s0": t % AD, "action_s1": PASS_ACTION,
                "gimmick_s0": 0, "gimmick_s1": 0, "mask_s0": [1] * AD, "mask_s1": None,
                "gmask_s0": None, "gmask_s1": None})()])
            c.add_step(state=st_state, action_s0=t % AD, action_s1=PASS_ACTION,
                       mask_s0=[1] * AD, logprob=float(lp[0].detach()),
                       value=float(v[0].detach()), turn=t + 1)
        tr = c.finish(own_team=["a"] * 4, opp_team=["b"] * 4, tp_bring=[0, 1, 2, 3],
                      tp_leads=[0, 1], won=w, terminal_type="win" if w else "loss")
        place_terminal_reward(tr)
        out.append(tr)
    return out


def _actor_params(ac):
    return [p.detach().clone() for p in ac.actor_parameters()]


# ── snapshot round-trip ───────────────────────────────────────────────────────
def test_snapshot_round_trip(tmp_path):
    ckpt = _write_ckpt(tmp_path)
    ac = _ac(ckpt); tr = _trainer(ac)
    trajs = _trajs(_ac(ckpt))
    tr.ppo_update(trajs)                                   # advance opt + rng + counters

    league = OpponentLeague(latest_path=str(ckpt))
    league.admit("gen0", "gen0.pt", 0, 1200.0)
    league.record_result("gen0", True)
    history = GenerationHistory()
    history.add(GenerationRecord(0, 8, 5, 10, 1200.0, "promote", True))
    history.best_path, history.best_scripted = "gen0.pt", (5, 10)

    p = RS.save_snapshot(tmp_path / "resume.pt", actor_critic=ac, trainer=tr,
                         league=league, history=history, seed=0)
    assert p.exists() and not (tmp_path / "resume.pt.tmp").exists()   # atomic write cleaned up

    ac2 = _ac(ckpt); tr2 = _trainer(ac2)
    league2, history2, snap = RS.load_into(p, actor_critic=ac2, trainer=tr2)
    # weights, counters, rng, league, history all restored
    for a, b in zip(ac.actor_parameters(), ac2.actor_parameters()):
        assert torch.allclose(a, b, atol=1e-7)
    assert tr2.updates == tr.updates and tr2.warmups == tr.warmups
    assert tr2.rng.bit_generator.state == tr.rng.bit_generator.state
    assert league2.to_obj() == league.to_obj()
    assert history2.to_obj() == history.to_obj()
    assert snap["generation"] == 1 and snap["format"] == RS.SNAPSHOT_FORMAT


# ── the gold guarantee: chunked == continuous ─────────────────────────────────
def test_chunked_equals_continuous(tmp_path):
    ckpt = _write_ckpt(tmp_path)
    trajs = _trajs(_ac(ckpt))

    # continuous: two updates back to back
    ac_c = _ac(ckpt); tr_c = _trainer(ac_c)
    tr_c.ppo_update(trajs)
    tr_c.ppo_update(trajs)
    cont = _actor_params(ac_c)

    # chunked: update -> snapshot -> fresh objects -> resume -> update
    ac_a = _ac(ckpt); tr_a = _trainer(ac_a)
    tr_a.ppo_update(trajs)
    snap = RS.save_snapshot(tmp_path / "r.pt", actor_critic=ac_a, trainer=tr_a,
                            league=OpponentLeague(latest_path=str(ckpt)),
                            history=GenerationHistory(), seed=0)
    ac_b = _ac(ckpt); tr_b = _trainer(ac_b)            # fresh from the SAME ckpt
    RS.load_into(snap, actor_critic=ac_b, trainer=tr_b)
    tr_b.ppo_update(trajs)
    chunk = _actor_params(ac_b)

    # bit-for-bit identical weights -> pausing costs only wall-clock (sec 17)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(cont, chunk))
    # and they genuinely MOVED from the start (the test isn't trivially true)
    start = _actor_params(_ac(ckpt))
    assert any(not torch.allclose(s, c, atol=1e-6) for s, c in zip(start, cont))


# ── graceful stop ─────────────────────────────────────────────────────────────
def test_stop_controller_time_budget():
    t = {"v": 1000.0}
    s = RS.StopController(max_hours=1.0, install_signal=False, clock=lambda: t["v"])
    assert not s.should_stop()
    t["v"] += 3599.0                                       # just under an hour
    assert not s.should_stop() and s.time_left() == pytest.approx(1.0, abs=1.0)
    t["v"] += 2.0                                          # past the hour
    assert s.should_stop()


def test_stop_controller_request():
    s = RS.StopController(install_signal=False)
    assert not s.should_stop()
    s.request_stop()
    assert s.should_stop()
    assert s.time_left() is None                           # no budget set
