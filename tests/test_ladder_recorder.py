"""W3b-0 (2026-09-02) — the ladder trajectory recorder (docs/w3b_ladder_ppo_design.md §4).

Every MODEL decision of the online bot becomes a self-play-schema Transition (state, actions,
gimmicks, EFFECTIVE masks, value from the served value head, logprob PLACEHOLDER), each game is
sealed with the ±1 terminal reward and per-game metadata (arm, pinned, τ, top-p, pair decode,
adapt-rules, opponent, ratings, logprob_valid=false), and appended to a TrajectoryStore.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from v_dance.play.ladder_recorder import LadderRecorder, terminal_type_for
from v_dance.selfplay.schema import PASS_ACTION
from v_dance.selfplay.store import read_trajectories

FMT = "gen9championsvgc2026regmb"
DIM = 8


def _player(**kw):
    p = SimpleNamespace(_sampling_masks={}, _record_masks=False, _last_value=0.7, _tp_decision={},
                        _temperature=0.3, _top_p=1.0, _model=SimpleNamespace(_pair_decode=True))
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _battle(tag, turn=1, won=None, lost=None):
    mon = lambda s: SimpleNamespace(species=s)  # noqa: E731
    return SimpleNamespace(battle_tag=tag, turn=turn, won=won, lost=lost, player_role="p1",
                           teampreview_team=[mon("Charizard"), mon("Garchomp"), mon("Incineroar"),
                                             mon("Kingambit"), mon("Whimsicott"), mon("Basculegion")],
                           teampreview_opponent_team=[mon("Tyranitar"), mon("Toxapex")],
                           team={}, opponent_team={}, opponent_username="Rival", rating=1200,
                           opponent_rating=1180)


def _builders():
    calls = []

    def legal(b, slot):
        calls.append(("legal", slot))
        return [1] * 16

    def repl(b, slot):
        calls.append(("repl", slot))
        return [0] * 12 + [1] * 4

    def gim(b, slot):
        calls.append(("gim", slot))
        return [1, 0, 0]
    return (legal, repl, gim), calls


def _state(i):
    return np.full(DIM, float(i), dtype=np.float32)


def _recorder(tmp_path, player=None, **kw):
    b, calls = _builders()
    kw.setdefault("mask_builders", b)
    kw.setdefault("now", lambda: 1_700_000_000.0)
    rec = LadderRecorder(player or _player(), tmp_path / "rl" / f"{FMT}" / "s1.jsonl",
                         session_id="s1", fmt=FMT, **kw)
    return rec, calls


def test_hooks_are_installed_and_only_model_decisions_are_recorded(tmp_path: Path):
    p = _player()
    rec, calls = _recorder(tmp_path, p)
    assert p._record_masks is True and isinstance(p._sampling_masks, dict)
    tag = f"battle-{FMT}-1"
    b = _battle(tag, turn=1)
    p._sampling_masks[(tag, "turn")] = ([1, 0] * 8, [0, 1] * 8)       # the sampler's EFFECTIVE masks
    p._record_rl_decision(b, _state(1), 3, 5, 0, 1, "model", "turn")   # bound hook → recorder
    p._record_rl_decision(b, _state(9), 0, 0, 0, 0, "retry_default", "turn")   # not the model's pick
    assert rec.steps == 1 and rec.failed == 0
    c = rec._collectors[tag]
    st = c.last_step()
    assert st.action_s0 == 3 and st.action_s1 == 5 and st.gimmick_s1 == 1
    assert st.mask_s0 == [1, 0] * 8 and st.mask_s1 == [0, 1] * 8    # from the stash, not rebuilt
    assert st.value == pytest.approx(2 * 0.7 - 1) and st.logprob == 0.0
    assert ("gim", 0) in calls and ("legal", 0) not in calls          # gimmick masks built, legal from stash
    assert (tag, "turn") not in p._sampling_masks                      # stash consumed


def test_rejection_recall_replaces_and_discard_drops(tmp_path: Path):
    p = _player()
    rec, _ = _recorder(tmp_path, p)
    tag = f"battle-{FMT}-2"
    b = _battle(tag, turn=3)
    rec.record(b, _state(1), 1, 1, 0, 0, "model", "turn")
    rec.record(b, _state(2), 2, 2, 0, 0, "model", "turn")              # same turn → Showdown rejected #1
    assert rec.rejected == 1 and len(rec._collectors[tag]) == 1
    assert rec._collectors[tag].last_step().action_s0 == 2
    rec.discard(b, "turn")                                             # the executed order was not the model's
    assert len(rec._collectors[tag]) == 0
    rec.discard(_battle("nope"), "turn")                               # unknown tag → no-op
    b.turn = 4
    rec.record(b, _state(3), None, 14, 0, 0, "forced_switch_model", "replacement")
    st = rec._collectors[tag].last_step()
    assert st.action_s0 == PASS_ACTION and st.action_s1 == 14 and st.decision_type == "replacement"
    assert st.gmask_s0 is None and st.mask_s1 == [0] * 12 + [1] * 4


def test_finish_seals_the_game_with_terminal_reward_and_metadata(tmp_path: Path):
    p = _player()
    tag = f"battle-{FMT}-3"
    p._tp_decision[tag] = {"bring": [0, 2, 3, 5], "leads": [0, 2],
                           "own_team": ["A", "B", "C", "D", "E", "F"], "opp_team": ["X", "Y"]}
    info = {"arm": "2b_tau03", "pinned": True, "tau": 0.3, "top_p": 1.0, "pair_decode": True}
    rec, _ = _recorder(tmp_path, p, arm_info=lambda t: info if t == tag else None, adapt_rules=True)
    b = _battle(tag, turn=1)
    for turn in (1, 2, 3):
        b.turn = turn
        rec.record(b, _state(turn), turn, turn + 1, 0, 0, "model", "turn")
    b.won, b.lost, b.turn = True, False, 3
    traj = rec.finish(tag, b, won=True, lost=False, opponent="Rival", rating_before=1200,
                      opp_rating_before=1180, turn=3, lane=2)
    assert traj is not None and rec.games == 1 and tag not in rec._collectors
    assert [t.reward for t in traj.transitions] == [0.0, 0.0, 1.0]     # ±1 terminal only
    assert traj.transitions[-1].done and not traj.transitions[0].done
    m = traj.meta
    assert m.won is True and m.terminal_type == "win" and m.n_turns == 3
    assert m.own_team == ["A", "B", "C", "D", "E", "F"] and m.tp_bring == [0, 2, 3, 5] and m.tp_leads == [0, 2]
    s = m.sampling
    assert s["arm"] == "2b_tau03" and s["pinned"] is True and s["tau"] == 0.3 and s["pair_decode"] is True
    assert s["adapt_rules"] is True and s["logprob_valid"] is False and s["source"] == "ladder"
    assert s["opponent"] == "Rival" and s["rating_before"] == 1200 and s["lane"] == 2
    assert s["session"] == "s1" and s["fmt"] == FMT and s["recorded_at"].endswith("Z")
    # on disk, round-trips through the self-play store reader
    back = read_trajectories(rec.path, expected_state_dim=DIM)
    assert len(back) == 1 and back[0].meta.sampling["arm"] == "2b_tau03"
    assert back[0].transitions[-1].reward == 1.0 and back[0].transitions[1].mask_s0 == [1] * 16


def test_loss_and_draw_terminals_and_empty_games(tmp_path: Path):
    p = _player()
    rec, _ = _recorder(tmp_path, p)
    lose, draw = f"battle-{FMT}-4", f"battle-{FMT}-5"
    rec.record(_battle(lose), _state(1), 0, 0, 0, 0, "model", "turn")
    t = rec.finish(lose, _battle(lose), won=False, lost=True)
    assert t.meta.terminal_type == "loss" and t.transitions[-1].reward == -1.0
    rec.record(_battle(draw), _state(1), 0, 0, 0, 0, "model", "turn")
    t = rec.finish(draw, _battle(draw), won=None, lost=None)
    assert t.meta.terminal_type == "draw" and t.transitions[-1].reward == 0.0
    assert rec.finish("battle-never-seen", None) is None and rec.skipped_empty == 1
    assert rec.finish(draw, None) is None and rec.skipped_empty == 2       # sealed once, never twice
    assert rec.games == 2 and len(read_trajectories(rec.path, expected_state_dim=DIM)) == 2


def test_roster_falls_back_to_the_battle_when_no_tp_decision(tmp_path: Path):
    p = _player()
    rec, _ = _recorder(tmp_path, p)
    tag = f"battle-{FMT}-6"
    rec.record(_battle(tag), _state(1), 0, 0, 0, 0, "model", "turn")
    t = rec.finish(tag, _battle(tag, won=True), won=True)
    assert t.meta.own_team[:2] == ["Charizard", "Garchomp"] and t.meta.opp_team == ["Tyranitar", "Toxapex"]
    assert t.meta.tp_bring == [0, 1, 2, 3] and t.meta.tp_leads == [0, 1]   # first-4 default
    s = t.meta.sampling                                                     # no arm_info → player defaults
    assert s["arm"] is None and s["tau"] == 0.3 and s["pair_decode"] is True


def test_hook_failures_never_raise(tmp_path: Path):
    def boom(b, slot):
        raise RuntimeError("mask builder broke")
    p = _player()
    rec, _ = _recorder(tmp_path, p, mask_builders=(boom, boom, boom))
    rec.record(_battle("battle-x"), _state(1), 0, 0, 0, 0, "model", "turn")   # no stash → builder → boom
    assert rec.failed == 1 and rec.steps == 0
    rec2, _ = _recorder(tmp_path, _player(), arm_info=lambda t: (_ for _ in ()).throw(RuntimeError("x")))
    rec2.record(_battle("battle-y"), _state(1), 0, 0, 0, 0, "model", "turn")
    t = rec2.finish("battle-y", _battle("battle-y"), won=True)             # broken arm_info → defaults
    assert t is not None and t.meta.sampling["arm"] is None


def test_banner_summary_and_terminal_helper(tmp_path: Path):
    rec, _ = _recorder(tmp_path)
    assert "ladder RECORDER ACTIVE" in rec.banner() and "logprob placeholder" in rec.banner()
    assert set(rec.summary()) >= {"games", "steps", "rejected", "failed", "path"}
    assert terminal_type_for(True, False) == "win" and terminal_type_for(False, True) == "loss"
    assert terminal_type_for(None, None) == "draw"
    rec.close()
