"""Regression tests for the 2026-06-27 BROAD full-codebase audit (16 confirmed findings).

Covers the cleanly-unit-testable fixes:
  #1  phase0_report now flags a WIN-PROB [0,1] value batch (the dormant looks_like_winprob detector, wired in).
  #2  the policy/critic aliasing assertion actually fires (was `id(x) is not id(y)`, always True).
  #14 corpus_qa fails (exit 1) on a 0-file scan instead of silently reading as clean.
(#3/#4/#5/#6/#7/#8/#11/#15/#16 are covered by the suite staying green + inspection; #9/#10/#13 surfaced for decision.)
"""
from __future__ import annotations

import pytest


# ── #2: policy/critic aliasing guard actually fires ───────────────────────────
def test_actor_critic_aliasing_assertion_fires():
    pytest.importorskip("torch")
    import torch.nn as nn
    from v_dance.selfplay.actor_critic import ActorCritic
    shared = nn.Linear(2, 2)
    with pytest.raises(AssertionError):
        ActorCritic(shared, shared, ("our_a",), (), True)   # policy IS critic → must raise now


# ── #1: Phase-0 gate catches win-prob stored as value_pm ──────────────────────
def test_phase0_flags_winprob_value_batch():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    import numpy as np
    from v_dance.encoders.state_encoder import get_state_dim, get_action_dim
    from v_dance.selfplay.collector import TrajectoryCollector
    from v_dance.selfplay import game_runner as G
    from v_dance.selfplay.schema import PASS_ACTION
    SD, AD = get_state_dim(), get_action_dim()

    def _game(tag, p1_won, value):
        cp1, cp2 = TrajectoryCollector(tag, "p1"), TrajectoryCollector(tag, "p2")
        for c in (cp1, cp2):
            for i in range(3):
                c.add_step(state=np.zeros(SD, np.float32), action_s0=0, action_s1=PASS_ACTION,
                           mask_s0=[1] * AD, logprob=-1.0, value=value, turn=i + 1)
        t1 = G.finalize_trajectory(cp1, won=p1_won, terminal_type="win" if p1_won else "loss",
                                   own_team=["a"], n_turns=5)
        t2 = G.finalize_trajectory(cp2, won=not p1_won, terminal_type="loss" if p1_won else "win",
                                   own_team=["b"], n_turns=5)
        return t1, t2

    # 10 games × 2 × 3 = 60 values, all 0.7 ∈ [0,1] with NO negatives → win-prob suspect
    winprob = [_game(f"w{i}", w, value=0.7) for i, w in enumerate([True, False] * 5)]
    rep = G.phase0_report(winprob, {}, min_games=4)
    assert rep["looks_like_winprob"] and not rep["value_space_ok"] and not rep["PASS"]

    # control: a balanced value_pm batch (contains negatives) is NOT flagged
    ok = [_game(f"p{i}", w, value=(0.6 if w else -0.6)) for i, w in enumerate([True, False] * 5)]
    rep_ok = G.phase0_report(ok, {}, min_games=4)
    assert not rep_ok["looks_like_winprob"] and rep_ok["value_space_ok"]


# ── #14: corpus_qa must not read a 0-file scan as clean ───────────────────────
def test_corpus_qa_empty_scan_is_not_clean(tmp_path):
    corpus_qa = pytest.importorskip("v_dance.datatools.corpus_qa")
    rc = corpus_qa.main([str(tmp_path / "does_not_exist")])
    assert rc == 1                       # 0 files scanned → FAIL, not a silent clean exit 0


# ── #9: Illusion Species-Clause — revealed Scenario B (genuine led, disguise entered second) ──
_SCENARIO_B_LOG = """\
|player|p1|alice|101|1500
|player|p2|bob|102|1500
|gen|9
|tier|[Gen 9] Test Doubles
|poke|p1|Dragapult, L50, M|
|poke|p2|Zoroark-Hisui, L50, M|
|poke|p2|Excadrill, L50, F|
|poke|p2|Tyranitar, L50, M|
|teamsize|p1|1
|teamsize|p2|3
|start
|switch|p1a: Dragapult|Dragapult, L50, M|100/100
|switch|p2a: Excadrill|Excadrill, L50, F|100/100
|switch|p2b: Tyranitar|Tyranitar, L50, M|100/100
|turn|1
|move|p2a: Excadrill|Iron Head|p1a: Dragapult
|-damage|p1a: Dragapult|60/100
|move|p1a: Dragapult|Dragon Darts|p2b: Tyranitar
|-damage|p2b: Tyranitar|0 fnt
|faint|p2b: Tyranitar
|switch|p2b: Excadrill|Excadrill, L50, F|100/100
|turn|2
|move|p1a: Dragapult|Shadow Ball|p2b: Excadrill
|-damage|p2b: Excadrill|40/100
|replace|p2b: Zoroark-Hisui|Zoroark-Hisui, L50, M
|-end|p2b: Zoroark-Hisui|Illusion
|move|p2b: Zoroark-Hisui|Night Daze|p1a: Dragapult
|-damage|p1a: Dragapult|0 fnt
|faint|p1a: Dragapult
|win|bob
"""


def test_illusion_species_clause_revealed_scenario_b():
    """#9: the real Excadrill LEADS (p2a), a Zoroark disguised as Excadrill enters SECOND (p2b), then the
    disguise BREAKS via |replace|. The Species-Clause rule first GUESSES the incumbent (real Excadrill) is
    the disguise; the reveal of the NEW arrival must UNDO that so the genuine lead keeps its own moves."""
    pytest.importorskip("poke_env")
    from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
    p = ShowdownReplayParser(_SCENARIO_B_LOG, our_player="p1")
    p.parse()
    seen = p.seen_mons
    assert "p2:Excadrill" in seen and "p2:Zoroark-Hisui" in seen
    # the genuine lead Excadrill keeps Iron Head and is NOT relabeled to Zoroark
    assert "Iron Head" in seen["p2:Excadrill"].revealed_moves
    assert "Iron Head" not in seen["p2:Zoroark-Hisui"].revealed_moves
    # the disguise (p2b) is the Zoroark and owns its post-reveal move
    assert "Night Daze" in seen["p2:Zoroark-Hisui"].revealed_moves
