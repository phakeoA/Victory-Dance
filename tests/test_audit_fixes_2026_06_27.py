"""Regression tests for the 2026-06-27 full-project bug/gap audit (multi-agent: 10 confirmed findings).

Each test fails without its fix. Covers the behavioural findings:
  #1  phase0_report MODEL-DRIVEN% counts ALL executed non-model sources (blocklist), not a 3-key whitelist.
  #4  BattleHost._ended is bounded (the late-frame gate keeps only recent tags).
  #5  balanced class-weights pool move-slot counts per bucket when move-order augmentation is on.
  #7  bulk_parse_replays exports BOTH perspectives for non-Type-B replays w/o --team (server.py /export parity).
  #9  a cancelled browser challenge clears the stale `pending`.
  #10 phazing |drag| switch-ins are NOT emitted as voluntary turn-start switch labels.
(The leak fixes #2/#3/#8 and the stale-default #6 are covered by the suite staying green + manual inspection.)
"""
from __future__ import annotations

import pytest


# ── #1: phase0_report counts every executed non-model source (blocklist) ──────
def test_phase0_counts_all_nonmodel_sources_not_just_whitelist():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    import numpy as np
    from v_dance.encoders.state_encoder import get_state_dim, get_action_dim
    from v_dance.selfplay.collector import TrajectoryCollector
    from v_dance.selfplay import game_runner as G
    from v_dance.selfplay.schema import PASS_ACTION
    SD, AD = get_state_dim(), get_action_dim()

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

    games = [_game(f"b{i}", w) for i, w in enumerate([True, False] * 3)]   # 6 games × 2 × 3 = 36 model steps
    # forced_default / forced_switch / forfeit are executed NON-model turns the OLD 3-key whitelist missed.
    rep = G.phase0_report(games, {"forced_default": 5, "forced_switch": 4, "forfeit": 1}, min_games=4)
    assert rep["non_model_executed"] == 10                          # 5+4+1, none silently dropped
    assert rep["model_driven"] == pytest.approx(36 / 46)
    assert not rep["model_driven_ok"]                              # 0.78 < 0.99 → correctly flagged
    # bookkeeping counters stay excluded from the denominator (a resample burst can't trip the guard)
    rep2 = G.phase0_report(games, {"rejected_resample": 99, "abandon_forfeit": 7}, min_games=4)
    assert rep2["non_model_executed"] == 0 and rep2["model_driven"] == 1.0


# ── #4: BattleHost._ended is bounded ──────────────────────────────────────────
def test_battle_host_ended_set_is_bounded():
    pytest.importorskip("poke_env")
    from v_dance.play.browser.battle_host import BattleHost
    h = BattleHost(team=None, model_path=None, team_chooser_path=None)
    for i in range(700):
        h.end_battle(f"battle-x-{i}")
    assert len(h._ended) <= 512                  # bounded, not one tag per battle forever
    assert "battle-x-699" in h._ended            # the most-recent (only source of late frames) retained
    assert "battle-x-0" not in h._ended          # the oldest evicted


# ── #5: balanced class-weights pool move slots per bucket under augmentation ───
def test_class_weights_pool_move_slots_when_augmenting():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from v_dance.training.train_bc import compute_class_weights

    def mk(idx, n):
        return [{"targets": {"our_a": idx}} for _ in range(n)]
    # bucket 0 move slots: idx0 (slot0) common; idx3/6/9 (slots1-3) rare; idx12 a switch class.
    examples = mk(0, 100) + mk(3, 4) + mk(6, 4) + mk(9, 4) + mk(12, 20)
    w_aug, _ = compute_class_weights(examples, 16, augment_move_order=True)
    w_raw, _ = compute_class_weights(examples, 16, augment_move_order=False)
    # under augmentation the 4 bucket-0 move slots have EQUAL effective frequency → equal weights
    assert w_aug[0] == pytest.approx(w_aug[3])
    assert w_aug[3] == pytest.approx(w_aug[6])
    assert w_aug[6] == pytest.approx(w_aug[9])
    # without pooling the raw skew mis-weights them (common slot0 down, rare slot1 up)
    assert w_raw[0] < w_raw[3]
    # switch classes (≥12) are permutation-invariant → identical with or without pooling
    assert w_aug[12] == pytest.approx(w_raw[12])


# ── #7: bulk export keeps BOTH perspectives for non-B without --team ──────────
def test_bulk_players_for_exports_both_perspectives():
    pytest.importorskip("poke_env")
    from v_dance.datatools.bulk_parse_replays import _players_for
    assert _players_for("B", None) == ["p1", "p2"]
    assert _players_for("D", None) == ["p1", "p2"]      # audit: was ["p1"] (silently dropped p2)
    assert _players_for("C", None) == ["p1", "p2"]
    assert _players_for("D", "p2") == ["p2"]            # explicit --players override still wins


# ── #9: a cancelled browser challenge is reconciled out of `pending` ──────────
def test_current_challengers_reports_authoritative_set():
    mod = pytest.importorskip("v_dance.play.play_vs_human_browser")
    cc = mod._current_challengers
    assert cc('|updatechallenges|{"challengesFrom":{"Bob":"gen9foo"}}') == {"bob"}
    assert cc('|updatechallenges|{"challengesFrom":{}}') == set()   # everyone cancelled → empty set
    assert cc("|pm|Bob|Me|hi there") is None                        # no updatechallenges frame → None


# ── #10: phazing |drag| is not a voluntary turn-start switch label ────────────
_HEADER = """\
|player|p1|alice|101|1500
|player|p2|bob|102|1500
|gen|9
|tier|[Gen 9] Test Doubles
|poke|p1|Floette-Eternal, L50, F|
|poke|p1|Sneasler, L50, F|
|poke|p1|Incineroar, L50, F|
|poke|p1|Kingambit, L50, M|
|poke|p2|Aerodactyl, L50, M|
|poke|p2|Meganium, L50, M|
|poke|p2|Kingambit, L50, M|
|poke|p2|Rotom-Wash, L50|
|teamsize|p1|4
|teamsize|p2|4
|start
|switch|p1a: Floette|Floette-Eternal, L50, F|100/100
|switch|p1b: Sneasler|Sneasler, L50, F|100/100
|switch|p2a: Kingambit|Kingambit, L50, M|100/100
|switch|p2b: Aerodactyl|Aerodactyl, L50, M|100/100
"""


def test_phazing_drag_is_not_a_turn_start_switch_label(tmp_path):
    pytest.importorskip("poke_env")
    from v_dance.parser.vod_parser.transitions import replay_to_transitions
    # Turn 1: p2 Kingambit (faster) Dragon Tails our ACTIVE, LIVING Floette out; Incineroar is DRAGGED in
    # (involuntary) BEFORE Floette could move → the drag is our_a's only event this turn.
    body = (
        "|turn|1\n"
        "|move|p2a: Kingambit|Dragon Tail|p1a: Floette\n"
        "|-damage|p1a: Floette|40/100\n"
        "|drag|p1a: Incineroar|Incineroar, L50, F|100/100\n"
        "|move|p1b: Sneasler|Close Combat|p2a: Kingambit\n"
        "|-damage|p2a: Kingambit|50/100\n"
        "|turn|2\n"
        "|move|p1a: Incineroar|Fake Out|p2a: Kingambit\n"
        "|move|p1b: Sneasler|Close Combat|p2a: Kingambit\n"
        "|faint|p2a: Kingambit\n"
        "|win|alice\n"
    )
    html = f'<script class="battle-log-data">{_HEADER}{body}</script>'
    rp = tmp_path / "drag.html"
    rp.write_text(html, encoding="utf-8")

    transitions = replay_to_transitions(rp)
    t1 = [t for t in transitions if t["turn"] == 1 and t["perspective"] == "p1"
          and t.get("decision_type") == "turn"]
    assert t1, "expected a turn-1 p1 turn-decision transition"
    acts = t1[0]["our_actions"]
    # #10: the involuntary drag of our_a (Incineroar) must NOT appear as a turn-start switch label
    assert all(not (a["slot"] == "our_a" and a["action"] == "switch") for a in acts), acts
    # the genuine decision (our_b Sneasler moved) is still present
    assert any(a["slot"] == "our_b" and a["action"] == "move" for a in acts), acts
    # a phaze is NOT a post-faint replacement → no decision_type='replacement' transition for the drag
    repl = [t for t in transitions if t.get("decision_type") == "replacement" and t["perspective"] == "p1"]
    assert not any(a.get("species") == "Incineroar" for t in repl for a in t["our_actions"]), repl
