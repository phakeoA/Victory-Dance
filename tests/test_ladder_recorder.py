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
    assert "ladder RECORDER ACTIVE" in rec.banner() and "behaviour log-prob" in rec.banner()
    assert set(rec.summary()) >= {"games", "steps", "rejected", "failed", "path"}
    assert terminal_type_for(True, False) == "win" and terminal_type_for(False, True) == "loss"
    assert terminal_type_for(None, None) == "draw"
    rec.close()


def test_private_room_suffix_is_stripped_from_the_sealed_battle_id(tmp_path: Path):
    """Live check 09-02: a private game's poke-env tag carries the room password suffix; the bench
    rows and the bandit key by the BASE tag, so the trajectory must too."""
    p = _player()
    rec, _ = _recorder(tmp_path, p)
    full = f"battle-{FMT}-4242-4mdjaya3vkjdrkm917wb3jfjbkg2q6qpw"
    rec.record(_battle(full), _state(1), 0, 0, 0, 0, "model", "turn")
    t = rec.finish(full, _battle(full, won=True), won=True)
    assert t.meta.battle_id == f"battle-{FMT}-4242"
    assert LadderRecorder.base_tag(">" + full) == f"battle-{FMT}-4242"


# ── W3b-1a (2026-09-02): the behaviour log-prob from the sampler ─────────────
def _lp(lp0, lp1, tau=0.3, top_p=1.0, exact=True):
    return {"logp": (lp0, lp1), "tau": tau, "top_p": top_p, "pair": True, "first": 0,
            "cond_on": 3, "exact": exact}


def test_sampler_logprob_is_summed_per_step_and_a_clean_tau_arm_seals_valid(tmp_path: Path):
    p = _player()
    tag = f"battle-{FMT}-11"
    info = {"arm": "2b_tau03", "pinned": False, "tau": 0.3, "top_p": 1.0, "pair_decode": True,
            "adapt_rules": False}
    rec, _ = _recorder(tmp_path, p, arm_info=lambda t: info)
    assert isinstance(p._sampling_logp, dict)                          # installed by the recorder
    b = _battle(tag, turn=1)
    p._sampling_logp[(tag, "turn")] = _lp(-0.4, -1.1)
    rec.record(b, _state(1), 3, 5, 0, 0, "model", "turn")
    assert rec._collectors[tag].last_step().logprob == pytest.approx(-1.5)
    assert rec._collectors[tag].last_step().pair_first == 0             # W3b-1b: the decode order
    assert (tag, "turn") not in p._sampling_logp                       # consumed
    b.turn = 2
    p._sampling_logp[(tag, "turn")] = _lp(None, -0.2)                  # slot 0 had no pick → 0
    rec.record(b, _state(2), None, 4, 0, 0, "model", "turn")
    assert rec._collectors[tag].last_step().logprob == pytest.approx(-0.2)
    b.turn = 3                                                         # forced replacement: argmax → 0
    p._sampling_logp[(tag, "replacement")] = {"logp": (0.0, None), "tau": 0.0, "top_p": 1.0,
                                              "pair": False, "first": None, "cond_on": None, "exact": True}
    rec.record(b, _state(3), 13, None, 0, 0, "forced_switch_model", "replacement")
    assert rec._collectors[tag].last_step().logprob == 0.0
    t = rec.finish(tag, _battle(tag, won=True), won=True)
    s = t.meta.sampling
    assert s["logprob_valid"] is True and s["logprob_reason"] is None
    assert s["logprob_source"] == "sampler" and s["logprob_inexact_steps"] == 0
    assert s["turn_steps"] == 2 and s["replacement_steps"] == 1
    assert s["gimmick_sampled"] is False and s["replacement_sampled"] is False
    assert s["logprob_site"] == "model_io.masked_sample_logp"
    assert rec.lp_steps == 2 and rec.valid_games == 1 and rec.summary()["valid_games"] == 1
    back = read_trajectories(rec.path, expected_state_dim=DIM)
    assert back[0].transitions[0].logprob == pytest.approx(-1.5) and back[0].meta.sampling["logprob_valid"]


def test_logprob_verdict_names_every_reason_a_game_is_not_trainable():
    V = LadderRecorder.logprob_verdict
    clean = {"turn": 3, "sampler": 3, "missing": 0, "inexact": 1, "repl": 0, "taus": {0.3}, "top_ps": {1.0}}
    v = V(clean, tau=0.3, top_p=1.0, adapt_rules=False)
    assert v["logprob_valid"] and v["logprob_inexact_steps"] == 1     # inexact steps stay trainable
    assert "argmax" in V(dict(clean, taus={0.0}), tau=0.0, top_p=1.0, adapt_rules=False)["logprob_reason"]
    assert "top-p" in V(clean, tau=0.3, top_p=0.9, adapt_rules=False)["logprob_reason"]
    assert "adapt-rules" in V(clean, tau=0.3, top_p=1.0, adapt_rules=True)["logprob_reason"]
    v = V(dict(clean, sampler=2, missing=1), tau=0.3, top_p=1.0, adapt_rules=False)
    assert not v["logprob_valid"] and "without a sampler" in v["logprob_reason"] and v["logprob_source"] == "mixed"
    v = V(dict(clean, taus={0.2, 0.3}), tau=0.3, top_p=1.0, adapt_rules=False)
    assert "changed mid-game" in v["logprob_reason"]
    v = V(dict(clean, taus={0.5}), tau=0.3, top_p=1.0, adapt_rules=False)
    assert "differs from the arm" in v["logprob_reason"]
    v = V(None, tau=0.3, top_p=1.0, adapt_rules=False)
    assert not v["logprob_valid"] and v["logprob_source"] == "placeholder" and "no turn decisions" in v["logprob_reason"]
    v = V(dict(clean, top_ps={0.9}), tau=0.3, top_p=1.0, adapt_rules=False)
    assert "sampler used top-p" in v["logprob_reason"]


def test_placeholder_steps_without_a_sampler_logprob_seal_invalid(tmp_path: Path):
    p = _player()
    tag = f"battle-{FMT}-12"
    info = {"arm": "2b_tau03", "tau": 0.3, "top_p": 1.0, "pair_decode": True, "adapt_rules": False}
    rec, _ = _recorder(tmp_path, p, arm_info=lambda t: info)
    b = _battle(tag, turn=1)
    rec.record(b, _state(1), 1, 2, 0, 0, "model", "turn")              # nothing stashed → placeholder
    assert rec._collectors[tag].last_step().logprob == 0.0
    t = rec.finish(tag, _battle(tag, won=True), won=True)
    s = t.meta.sampling
    assert s["logprob_valid"] is False and "without a sampler log-prob" in s["logprob_reason"]
    assert s["logprob_source"] == "placeholder" and rec.valid_games == 0


def test_rejection_recall_and_discard_unbook_the_logprob_accounting(tmp_path: Path):
    p = _player()
    tag = f"battle-{FMT}-13"
    info = {"arm": "2b_tau03", "tau": 0.3, "top_p": 1.0, "pair_decode": True, "adapt_rules": False}
    rec, _ = _recorder(tmp_path, p, arm_info=lambda t: info)
    b = _battle(tag, turn=1)
    p._sampling_logp[(tag, "turn")] = _lp(-0.3, -0.3, exact=False)
    rec.record(b, _state(1), 1, 1, 0, 0, "model", "turn")              # rejected by Showdown …
    p._sampling_logp[(tag, "turn")] = _lp(-0.7, -0.1)
    rec.record(b, _state(2), 2, 2, 0, 0, "model", "turn")              # … re-call replaces it
    assert rec.rejected == 1 and rec.lp_steps == 1 and rec.lp_inexact == 0
    assert rec._lp[tag]["turn"] == 1 and rec._lp[tag]["inexact"] == 0
    rec.discard(b, "turn")                                             # the executed order was not the model's
    assert rec.lp_steps == 0 and rec._lp[tag]["turn"] == 0
    t = rec.finish(tag, _battle(tag, won=True), won=True)
    assert t is None                                                   # nothing left to seal



def test_empty_slot_action_zero_under_an_all_zero_mask_is_recorded_as_pass(tmp_path: Path):
    """2026-09-03: the live player encodes an EMPTY / fainted active slot as action 0 (``_select_actions``:
    None -> 0) and its effective mask is all-zero; the schema means PASS_ACTION (the self-play rule,
    ``game_runner.resolve_action``). The first W3b update died in ``assert_actions_legal`` on exactly
    this ("action_s1=0 illegal under mask_s1=[0]*16"). A REAL action 0 under a legal mask stays 0."""
    p = _player(_record_masks=True)
    rec, _ = _recorder(tmp_path, p)
    tag = f"battle-{FMT}-31"
    b = _battle(tag, turn=1)
    p._sampling_masks[(tag, "turn")] = ([1] * 16, [0] * 16)            # slot 1 empty: effective mask all-zero
    rec.record(b, _state(1), 3, 0, 0, 0, "model", "turn")
    st = rec._collectors[tag].last_step()
    assert st.action_s0 == 3 and st.action_s1 == PASS_ACTION and list(st.mask_s1) == [0] * 16
    b.turn = 2
    p._sampling_masks[(tag, "turn")] = ([1] * 16, [1] * 16)
    rec.record(b, _state(2), 0, 0, 0, 0, "model", "turn")              # a real action 0 (legal) stays 0
    st = rec._collectors[tag].last_step()
    assert st.action_s0 == 0 and st.action_s1 == 0
    b.turn = 3                                                         # replacement: slot 0 fainted, bench exhausted
    p._sampling_masks[(tag, "replacement")] = ([0] * 16, None)
    rec.record(b, _state(3), 0, None, 0, 0, "forced_switch_model", "replacement")
    st = rec._collectors[tag].last_step()
    assert st.action_s0 == PASS_ACTION and st.action_s1 == PASS_ACTION
    assert rec.steps == 3 and rec.failed == 0
    t = rec.finish(tag, _battle(tag, won=True), won=True)
    back = read_trajectories(rec.path, expected_state_dim=DIM)
    assert [(x.action_s0, x.action_s1) for x in back[0].transitions] == [(3, PASS_ACTION), (0, 0),
                                                                          (PASS_ACTION, PASS_ACTION)]
    assert t.meta.sampling["turn_steps"] == 2 and t.meta.sampling["replacement_steps"] == 1
