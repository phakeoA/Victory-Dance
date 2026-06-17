"""End-to-end integration smoke for the self-play data + GAE pipeline (3a + 3b.2).

Chains EVERY module on a synthetic 2-perspective game:
  collector -> assert_zero_sum -> reward.prepare_batch -> store (write/read round-trip)
  -> store.assert_terminal_rewards_clean -> gae.compute_batch_gae -> diagnostics.summarize.
Catches integration bugs (lost fields on serialize, GAE mis-reading rewards, etc.) that
per-module unit tests can miss.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "local_battle"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from self_play.collector import TrajectoryCollector, assert_zero_sum  # noqa: E402
from self_play.reward import prepare_batch  # noqa: E402
from self_play.store import write_trajectories, read_trajectories, assert_terminal_rewards_clean  # noqa: E402
from self_play.gae import compute_batch_gae  # noqa: E402
from self_play import diagnostics  # noqa: E402

STATE_DIM = 7
P1_TEAM = ["zoroark", "ditto", "charizard", "whimsicott", "sneasler", "basculegion"]
P2_TEAM = ["kingambit", "sylveon", "talonflame", "tyranitar", "steelix", "rotom"]


def _play_one_game(battle_id, p1_wins, n_turns=4):
    """Simulate one self-play game from BOTH perspectives -> two Trajectories."""
    c1 = TrajectoryCollector(battle_id, "p1")
    c2 = TrajectoryCollector(battle_id, "p2")
    for t in range(1, n_turns + 1):
        # distinct states/actions per side; values + logprobs as a real player would log
        c1.add_step(state=np.full(STATE_DIM, 0.1 * t, np.float32), action_s0=t % 12,
                    action_s1=-1, gimmick_s0=(1 if t == 1 else 0), logprob=-0.4 * t,
                    value=0.6, decision_type=("replacement" if t == n_turns else "turn"),
                    turn=t)
        c2.add_step(state=np.full(STATE_DIM, 0.2 * t, np.float32), action_s0=(t + 1) % 12,
                    action_s1=3, logprob=-0.5 * t, value=0.4, turn=t)
    tr1 = c1.finish(own_team=P1_TEAM, opp_team=P2_TEAM, tp_bring=[0, 1, 2, 3],
                    tp_leads=[0, 1], won=p1_wins, terminal_type="win" if p1_wins else "loss",
                    n_turns=n_turns)
    tr2 = c2.finish(own_team=P2_TEAM, opp_team=P1_TEAM, tp_bring=[0, 1, 2, 3],
                    tp_leads=[0, 1], won=not p1_wins, terminal_type="loss" if p1_wins else "win",
                    n_turns=n_turns)
    return tr1, tr2


def test_full_pipeline_round_trips_and_trains(tmp_path):
    # 1) collect a few games (both perspectives each), zero-sum-checked
    raw = []
    for i, p1_wins in enumerate([True, False, True]):
        tr1, tr2 = _play_one_game(f"g{i}", p1_wins)
        assert_zero_sum(tr1, tr2)                         # sec 6 symmetry holds
        raw += [tr1, tr2]

    # 2) finalise the batch: place terminal rewards, hard-fail guard (100% model-driven)
    batch = prepare_batch(raw, source_counts={"model": 100, "forced_switch_model": 8})
    assert len(batch) == 6                                # nothing dropped (no fallback)

    # 3) persist to the Type-C store and read back (serialization round-trip)
    buf = tmp_path / "selfplay.jsonl"
    assert write_trajectories(buf, batch) == 6
    back = read_trajectories(buf, expected_state_dim=STATE_DIM)
    assert len(back) == 6
    # states + key fields survive the round-trip
    for a, b in zip(batch, back):
        assert a.meta == b.meta
        for x, y in zip(a.transitions, b.transitions):
            assert np.allclose(x.state, y.state)
            assert (x.action_s0, x.gimmick_s0, x.logprob, x.value, x.reward, x.decision_type) == \
                   (y.action_s0, y.gimmick_s0, y.logprob, y.value, y.reward, y.decision_type)

    # 4) data-integrity: sparse rewards, terminal in range
    assert_terminal_rewards_clean(back)

    # 5) GAE over the read-back batch -> standardized advantages + return targets
    adv, ret = compute_batch_gae(back)
    n_steps = sum(len(t.transitions) for t in back)
    assert adv.shape == (n_steps,) and ret.shape == (n_steps,)
    assert np.isfinite(adv).all() and np.isfinite(ret).all()
    assert adv.mean() == pytest.approx(0.0, abs=1e-4)     # standardized across the batch

    # 6) diagnostics: overall ~50% (mirrored outcomes), per-bring tallied
    summary = diagnostics.summarize(back, min_games=1)
    assert summary["overall"]["games"] == 6
    assert summary["overall"]["win_rate"] == pytest.approx(0.5)  # 3 wins / 6 decisive
    assert summary["n_brings"] == 2                        # one bring per team
