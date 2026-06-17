"""Task 3c.1a: self-play recording helpers (offline).

Tests the pure recording logic - resolve_action (pass detection), record_decision
(behaviour log-prob + critic value captured into a collector), finalize_trajectory
(reward placement + zero-sum-ready), terminal_type_for - without a live battle. The
battle glue (mask rebuild from a live battle) is covered by the Phase-0 smoke (3c.1b).
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
from v_dance.models.bc_model import BCPolicy  # noqa: E402
from v_dance.selfplay.actor_critic import ActorCritic  # noqa: E402
from v_dance.selfplay.collector import TrajectoryCollector, assert_zero_sum  # noqa: E402
from v_dance.selfplay.schema import PASS_ACTION  # noqa: E402
from v_dance.selfplay import game_runner as G  # noqa: E402

SD, AD, GD = get_state_dim(), get_action_dim(), get_gimmick_dim()


def _ac(tmp_path):
    torch.manual_seed(0)
    m = BCPolicy(state_dim=SD, action_dim=AD, hidden_dims=(8, 4), dropout=0.0, gimmick_dim=GD)
    p = tmp_path / "bc.pt"
    torch.save({"model_state": m.state_dict(), "config": {
        "state_dim": SD, "action_dim": AD, "hidden_dims": (8, 4), "dropout": 0.0,
        "heads": ("our_a", "our_b"), "gimmick_dim": GD, "gimmick_trained": True,
        "value_trained": True, "state_layout_version": get_state_layout_version()}}, p)
    return ActorCritic.from_bc_checkpoint(p)


def _state(seed=0):
    return np.random.default_rng(seed).standard_normal(SD).astype(np.float32)


# ── resolve_action (pass detection) ───────────────────────────────────────────
def test_resolve_action():
    assert G.resolve_action(None, [1] * AD) == PASS_ACTION           # no decision
    assert G.resolve_action(3, None) == PASS_ACTION                  # no mask
    assert G.resolve_action(3, [0] * AD) == PASS_ACTION              # all-illegal slot
    assert G.resolve_action(0, [1] + [0] * (AD - 1)) == 0            # real action 0 (not a pass)
    assert G.resolve_action(5, [1] * AD) == 5


# ── record_decision ───────────────────────────────────────────────────────────
def test_record_decision_captures_logprob_value_masks(tmp_path):
    ac = _ac(tmp_path)
    c = TrajectoryCollector("g", "p1")
    G.record_decision(c, ac, state=_state(1), a0=3, a1=7, g0=0, g1=0,
                      mask0=[1] * AD, mask1=[1] * AD, gmask0=[1, 0], gmask1=[1, 0],
                      decision_type="turn", turn=4, tau=1.0)
    tr = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0, 1, 2, 3],
                  tp_leads=[0, 1], won=True, terminal_type="win")
    t = tr.transitions[0]
    assert t.action_s0 == 3 and t.action_s1 == 7 and t.turn == 4
    assert t.mask_s0 == [1] * AD and t.gmask_s0 == [1, 0]
    assert -1.0 <= t.value <= 1.0                                    # value_pm space
    assert t.logprob < 0.0                                           # log of a probability


def test_record_decision_pass_slot(tmp_path):
    ac = _ac(tmp_path)
    c = TrajectoryCollector("g", "p1")
    # slot 1 has an all-zero mask -> recorded as PASS, contributes nothing
    G.record_decision(c, ac, state=_state(2), a0=2, a1=0, mask0=[1] * AD, mask1=[0] * AD,
                      decision_type="turn", turn=1)
    t = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0], tp_leads=[0],
                 won=True, terminal_type="win").transitions[0]
    assert t.action_s0 == 2 and t.action_s1 == PASS_ACTION


# ── terminal type + finalize + zero-sum ───────────────────────────────────────
def test_terminal_type_for():
    assert G.terminal_type_for(True, False) == "win"
    assert G.terminal_type_for(False, True) == "loss"
    assert G.terminal_type_for(False, False) == "draw"              # true tie


def _one_step_collector(role, seed):
    c = TrajectoryCollector("battle-1", role)
    c.add_step(state=np.zeros(SD, np.float32), action_s0=0, action_s1=PASS_ACTION,
               mask_s0=[1] * AD, logprob=-1.0, value=0.0, turn=5)
    return c


def test_finalize_places_reward_and_is_zero_sum():
    cp1 = _one_step_collector("p1", 1)
    cp2 = _one_step_collector("p2", 2)
    win = G.finalize_trajectory(cp1, won=True, terminal_type="win",
                                own_team=["a", "b", "c", "d"], n_turns=5)
    loss = G.finalize_trajectory(cp2, won=False, terminal_type="loss",
                                 own_team=["e", "f", "g", "h"], n_turns=5)
    assert win.transitions[-1].reward == +1.0 and win.transitions[-1].done
    assert loss.transitions[-1].reward == -1.0
    assert win.meta.n_turns == 5 and loss.meta.n_turns == 5
    assert_zero_sum(win, loss)                                       # the pair is symmetric


def test_finalize_n_turns_uses_battle_clock_not_step_count():
    # one perspective skipped a (non-model) turn -> fewer steps, but n_turns is the
    # battle's final turn so the symmetry clock still matches.
    c = TrajectoryCollector("battle-2", "p1")
    c.add_step(state=np.zeros(SD, np.float32), action_s0=0, action_s1=PASS_ACTION,
               mask_s0=[1] * AD, turn=2)
    tr = G.finalize_trajectory(c, won=True, terminal_type="win", own_team=["a"], n_turns=9)
    assert tr.meta.n_turns == 9 and len(tr.transitions) == 1        # clock != step count


# ── the live player is wired (class shape) ────────────────────────────────────
def test_selfplay_player_overrides_hooks():
    from v_dance.play.player import VGCPlayer
    assert issubclass(G.SelfPlayVGCPlayer, VGCPlayer)
    assert G.SelfPlayVGCPlayer._record_rl_decision is not VGCPlayer._record_rl_decision
    assert G.SelfPlayVGCPlayer._battle_finished_callback is not VGCPlayer._battle_finished_callback


# ── pairing + Phase-0 report (pure) ───────────────────────────────────────────
def _game(tag, p1_won, n=3, turn=5):
    cp1, cp2 = TrajectoryCollector(tag, "p1"), TrajectoryCollector(tag, "p2")
    for c in (cp1, cp2):
        for i in range(n):
            c.add_step(state=np.zeros(SD, np.float32), action_s0=0, action_s1=PASS_ACTION,
                       mask_s0=[1] * AD, logprob=-1.0, value=0.0, turn=i + 1)
    t1 = G.finalize_trajectory(cp1, won=p1_won, terminal_type="win" if p1_won else "loss",
                               own_team=["a"], n_turns=turn)
    t2 = G.finalize_trajectory(cp2, won=not p1_won, terminal_type="loss" if p1_won else "win",
                               own_team=["b"], n_turns=turn)
    return t1, t2


def _games(pattern, turn=5):
    return [_game(f"b{i}", w, turn=turn) for i, w in enumerate(pattern)]


def test_pair_trajectories_matches_by_tag_and_drops_unmatched():
    t1a, t2a = _game("x", True)
    t1b, t2b = _game("y", False)
    z1, _z2 = _game("z", True)
    pairs = G.pair_trajectories({"x": t1a, "y": t1b, "z": z1}, {"x": t2a, "y": t2b})
    assert len(pairs) == 2                                  # z (only one side) dropped


def test_phase0_report_pass():
    rep = G.phase0_report(_games([True, False] * 3), {}, min_games=4)
    assert rep["PASS"]
    assert rep["model_driven"] == 1.0 and rep["symmetry_ok"]            # no non-model executions
    assert rep["duplicate_steps"] == 0 and rep["data_clean"]
    assert rep["p1_win_rate"] == pytest.approx(0.5) and rep["balance_ok"]


def test_phase0_flags_low_model_driven():
    # 6 games x 2 trajs x 3 steps = 36 executed model steps; 100 non-model executions
    rep = G.phase0_report(_games([True, False] * 3), {"retry_default": 100}, min_games=4)
    assert rep["non_model_executed"] == 100
    assert rep["model_driven"] == pytest.approx(36 / 136)
    assert not rep["model_driven_ok"] and not rep["PASS"]


def test_phase0_flags_duplicate_steps():
    import copy
    t1, t2 = _game("b", True)
    t1.transitions.append(copy.deepcopy(t1.transitions[0]))            # same (turn,dtype) dup
    assert G.duplicate_turn_steps(t1) == 1
    rep = G.phase0_report([(t1, t2)], {}, min_games=1)
    assert rep["duplicate_steps"] == 1 and not rep["data_clean"] and not rep["PASS"]


def test_phase0_resample_rate_is_informational():
    rep = G.phase0_report(_games([True, False] * 3), {"rejected_resample": 12}, min_games=4)
    assert rep["resample_count"] == 12
    assert rep["resample_rate"] == pytest.approx(12 / (36 + 12))       # of model turns
    assert rep["PASS"]                                                 # resamples don't gate


def test_phase0_flags_symmetry_failure():
    t1, t2 = _game("b", True)
    t2.meta.n_turns = 99                                    # break the shared clock
    rep = G.phase0_report([(t1, t2)], {"model": 10}, min_games=1)
    assert rep["symmetry_failures"] == 1 and not rep["symmetry_ok"] and not rep["PASS"]


def test_phase0_flags_p1_imbalance():
    rep = G.phase0_report(_games([True] * 6), {"model": 600}, min_games=4)
    assert rep["p1_win_rate"] == 1.0 and not rep["balance_ok"] and not rep["PASS"]


def test_phase0_flags_too_few_games():
    rep = G.phase0_report(_games([True, False]), {"model": 200}, min_games=200)
    assert not rep["enough_games"] and not rep["PASS"]


def test_phase0_flags_terminal_unclean():
    t1, t2 = _game("b", True)
    t1.transitions[0].reward = 0.5                          # non-terminal reward must be 0
    rep = G.phase0_report([(t1, t2)], {"model": 10}, min_games=1)
    assert not rep["terminal_clean"] and not rep["PASS"]


def test_collector_dedup_keeps_only_executed_pick():
    """Simulate a rejection re-call: the prior step is replaced by the accepted resample
    so the trajectory holds only the executed decision (the 3c.1c de-dup mechanism)."""
    c = TrajectoryCollector("g", "p1")
    c.add_step(state=np.zeros(SD, np.float32), action_s0=3, action_s1=PASS_ACTION,
               mask_s0=[1] * AD, turn=5, decision_type="turn")           # rejected pick
    last = c.last_step()
    assert last is not None and last.turn == 5 and last.action_s0 == 3
    c.pop_step()                                                         # re-call -> drop it
    c.add_step(state=np.zeros(SD, np.float32), action_s0=12, action_s1=PASS_ACTION,
               mask_s0=[1] * AD, turn=5, decision_type="turn")           # accepted resample
    tr = c.finish(own_team=["a"], opp_team=["b"], tp_bring=[0], tp_leads=[0],
                  won=True, terminal_type="win")
    assert len(tr.transitions) == 1 and tr.transitions[0].action_s0 == 12
    assert G.duplicate_turn_steps(tr) == 0


def test_print_phase0_report_runs(capsys):
    G.print_phase0_report(G.phase0_report(_games([True, False] * 3),
                                          {"rejected_resample": 5}, min_games=4))
    out = capsys.readouterr().out
    assert "Phase-0" in out and "MODEL-DRIVEN" in out and "duplicate-turn" in out and "PASS" in out


# ── 3c.6e-3: live spectate feed (player publishes its active battle room) ───────
def test_report_active_publishes_battle_room(tmp_path):
    import types
    from v_dance.selfplay.status import LiveStatus, read_status
    ls = LiveStatus(tmp_path / "status.json")
    fake = types.SimpleNamespace(_status=ls, username="LG0x1")
    battle = types.SimpleNamespace(battle_tag="battle-gen9-77", players=["LG0x1", "LG1x2"], turn=5)
    G.SelfPlayVGCPlayer._report_active(fake, battle)          # unbound, fake self (no live Player)
    s = read_status(ls.path)
    assert s["active_battles"] == [{"tag": "battle-gen9-77", "p1": "LG0x1", "p2": "LG1x2", "turn": 5}]


def test_report_active_noop_without_status():
    import types
    battle = types.SimpleNamespace(battle_tag="battle-x", players=[], turn=1)
    # no status wired -> returns cleanly, nothing to assert beyond "does not raise"
    G.SelfPlayVGCPlayer._report_active(types.SimpleNamespace(_status=None), battle)


def test_report_active_writes_live_log(tmp_path):
    import json
    import types
    from v_dance.selfplay.status import LiveStatus
    ls = LiveStatus(tmp_path / "status.json")
    fake = types.SimpleNamespace(_status=ls, username="LG0x1",
                                 _proto_log={"battle-gen9-77": ["|turn|1", "|move|p1a: A|Tackle|p2a: B"]})
    battle = types.SimpleNamespace(battle_tag="battle-gen9-77", players=["LG0x1", "LG1x2"], turn=1)
    G.SelfPlayVGCPlayer._report_active(fake, battle)
    d = json.loads((tmp_path / "live_log.json").read_text(encoding="utf-8"))
    assert d["tag"] == "battle-gen9-77" and d["n_lines"] == 2 and d["log"][0] == "|turn|1"
