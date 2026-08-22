"""Regression tests for the 2026-06-27 round-4 audit (9 confirmed; incl. 2 regressions from this session).

Covers the cleanly-unit-testable fixes:
  #1  trainer._mean_stats drops NON-FINITE per-key entries (a NaN opp_ce minibatch no longer poisons the gen mean).
  #2  generation.fs_monitor_counts excludes rejected_resample (bookkeeping, not an edge event).
  #3  gate.cluster_hof_suspects(n=0) returns [] (Python [-0:] gotcha) instead of ALL past champions.
  #4  replay_parser Encore lock-target = the actually-taken last move (A,B,A-on-encore-turn → A, not B).
  #5  replay_parser -sethp (Pain Split) emits a damage_event with hp_pct_delta.
  #8  corpus_qa flags a cross-folder same-basename duplicate replay_id.
(#6 root empty-mask loop guard, #7 TP lead metric mask, #9 gauntlet no-result exit → covered by suite + inspection.)
"""
from __future__ import annotations

import pytest


# ── #1: _mean_stats tolerates per-minibatch NaN (e.g. opp_ce with no valid targets) ──
def test_mean_stats_drops_nonfinite_per_key():
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from v_dance.selfplay.trainer import _mean_stats
    stats = [{"loss": 1.0, "opp_ce": 2.0}, {"loss": 3.0, "opp_ce": float("nan")}]
    out = _mean_stats(stats)
    assert out["loss"] == pytest.approx(2.0)        # finite mean unaffected
    assert out["opp_ce"] == pytest.approx(2.0)      # the NaN entry is dropped, not poisoning the mean
    # all-NaN for a key → NaN (e.g. opp head absent every minibatch)
    import math
    assert math.isnan(_mean_stats([{"opp_ce": float("nan")}, {"opp_ce": float("nan")}])["opp_ce"])


# ── #2: fs-monitor does not count rejected_resample as an edge event ──────────
def test_fs_monitor_excludes_rejected_resample():
    from v_dance.selfplay.generation import fs_monitor_counts
    fs = fs_monitor_counts({"model": 100, "rejected_resample": 40, "games": 10})
    assert fs["total"] == 0                          # a benign resample burst is NOT an edge event
    # a real edge source IS counted
    fs2 = fs_monitor_counts({"model": 100, "forced_default": 3, "abandon_forfeit": 1})
    assert fs2["total"] == 4 and fs2["forced_default"] == 3 and fs2["abandon_forfeit"] == 1


# ── #3: cluster_hof_suspects(n=0) selects ZERO, not all ───────────────────────
def test_cluster_hof_suspects_n_zero_is_empty():
    pytest.importorskip("numpy")
    from v_dance.selfplay.gate import cluster_hof_suspects
    from v_dance.selfplay.league import LeagueSnapshot
    champs = [LeagueSnapshot(f"g{i}", f"g{i}.pt", i, is_champion=True) for i in range(4)]
    assert cluster_hof_suspects(champs, n=0) == []            # was ALL past champions (the [-0:] gotcha)
    assert len(cluster_hof_suspects(champs, n=2)) == 2        # sanity: n=2 still works


# ── #4: Encore lock-target is the actually-taken move (repeat case) ───────────
_ENCORE_REPEAT_LOG = """\
|player|p1|alice|101|1500
|player|p2|bob|102|1500
|gen|9
|tier|[Gen 9] Test Doubles
|poke|p1|Garchomp, L50, M|
|poke|p2|Dragapult, L50, M|
|teamsize|p1|1
|teamsize|p2|1
|start
|switch|p1a: Garchomp|Garchomp, L50, M|100/100
|switch|p2a: Dragapult|Dragapult, L50, M|100/100
|turn|1
|move|p1a: Garchomp|Earthquake|p2a: Dragapult
|-damage|p2a: Dragapult|70/100
|turn|2
|move|p1a: Garchomp|Protect|p1a: Garchomp
|turn|3
|move|p1a: Garchomp|Earthquake|p2a: Dragapult
|-damage|p2a: Dragapult|40/100
|-start|p1a: Garchomp|Encore
|turn|4
|win|alice
"""


def test_encore_lock_target_is_actually_taken_move_on_repeat():
    pytest.importorskip("poke_env")
    from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
    p = ShowdownReplayParser(_ENCORE_REPEAT_LOG, our_player="p1")
    p.parse()
    chomp = next(m for m in p.seen_mons.values() if m.nickname == "Garchomp")
    # Encore on turn 3 (Earthquake turn) must lock Earthquake, NOT stint_moves[-1] (Protect)
    assert chomp.encore_move == "Earthquake"


# ── #5: Pain Split / -sethp emits a damage_event ─────────────────────────────
_PAINSPLIT_LOG = """\
|player|p1|alice|101|1500
|player|p2|bob|102|1500
|gen|9
|tier|[Gen 9] Test Doubles
|poke|p1|Rotom-Wash, L50|
|poke|p2|Dragapult, L50, M|
|teamsize|p1|1
|teamsize|p2|1
|start
|switch|p1a: Rotom|Rotom-Wash, L50|100/100
|switch|p2a: Dragapult|Dragapult, L50, M|40/100
|turn|1
|move|p1a: Rotom|Pain Split|p2a: Dragapult
|-sethp|p2a: Dragapult|70/100|[from] move: Pain Split
|-sethp|p1a: Rotom|70/100|[from] move: Pain Split
|turn|2
|win|alice
"""


def test_sethp_emits_damage_event():
    pytest.importorskip("poke_env")
    from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
    result = ShowdownReplayParser(_PAINSPLIT_LOG, our_player="p1").parse()
    turns = result.get("turns") or []
    de = [e for t in turns for e in (t.get("damage_events") or []) if e.get("event") == "sethp"]
    assert de, "expected at least one sethp damage_event from Pain Split"
    assert all(e.get("hp_pct_delta") is not None for e in de)


# ── #8: corpus_qa flags a cross-folder same-basename duplicate replay_id ──────
def test_corpus_qa_flags_cross_folder_same_basename_dup(tmp_path):
    corpus_qa = pytest.importorskip("v_dance.datatools.corpus_qa")
    import json
    a = tmp_path / "A"; b = tmp_path / "B"
    a.mkdir(); b.mkdir()
    row = {"replay_id": "r123", "state": {}, "our_actions": [], "opp_actions_actual": [],
           "perspective": "p1", "turn": 1, "decision_type": "turn"}
    for d in (a, b):
        (d / "r123.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = corpus_qa.audit_folders([str(a), str(b)])
    assert "r123" in report["duplicate_replay_ids"]   # same basename across folders must be flagged
