"""P1.1/P2.2 — the v2 frozen-champion gate (the user's 'static until proven' ladder).

promotion_gate_v2 keeps the champion FROZEN until the candidate either clears a HIGH bar vs it
(observed mirror win-rate >= 0.55 over >= min_h2h_games, decisively) OR the head-to-head climb
PLATEAUS at a not-losing level (the backstop). Safety reverts take priority: a scripted COLLAPSE,
and (P2.2) a MIRROR-COLLAPSE when the learner erodes significantly BELOW its own frozen champion.
Calibrated on gate_sim (P2.1: the 0.55 bar needs ~360 mirror games; mirror-collapse bar = wilson
upper < 0.45 over >= 360 games so a freshly re-anchored ~50% learner is never falsely reverted).
"""
from __future__ import annotations

import pytest

from v_dance.selfplay.generation import promotion_gate_v2, GateConfigV2, is_plateau

CFG = GateConfigV2()
RISING = [0.50, 0.52, 0.54, 0.56, 0.58, 0.62, 0.64, 0.66, 0.68, 0.70]
FLAT53 = [0.53] * 10        # plateaued NOT-losing but below the 0.55 bar → backstop
FLAT45 = [0.45] * 10        # plateaued but LOSING


def _gate(*, scr=(120, 180), hw=0.55, mir=(270, 360), hist=RISING, champ=True, cfg=CFG):
    return promotion_gate_v2(scripted_wins=scr[0], scripted_games=scr[1], high_water=hw,
                             mirror_wins=mir[0], mirror_games=mir[1], h2h_history=hist,
                             have_champion=champ, cfg=cfg)


# ── gen 0 / no champion ───────────────────────────────────────────────────────
def test_no_champion_auto_promotes():
    v, st = _gate(champ=False)
    assert v == "promote" and st["reason"] == "no_baseline"


# ── the high bar (proof of superiority) ───────────────────────────────────────
def test_clears_high_bar_promotes_beat_champion():
    v, st = _gate(mir=(270, 360))                       # 0.75 over 360 games — decisive
    assert v == "promote" and st["reason"] == "beat_champion"
    assert st["mirror"]["beats_bar"] is True


def test_below_bar_while_climbing_holds():
    v, st = _gate(mir=(190, 360), hist=RISING)          # 0.528 < 0.55, still rising → wait
    assert v == "hold" and st["reason"] == "hold"
    assert st["mirror"]["beats_bar"] is False
    assert st["plateau"]["detected"] is False


def test_high_observed_but_too_few_games_holds():
    """A lucky 76% over only 50 games must NOT crown a champion — the bar needs min_h2h_games."""
    v, st = _gate(mir=(38, 50), hist=[0.7, 0.72])
    assert v == "hold"
    assert st["mirror"]["enough_games"] is False and st["mirror"]["beats_bar"] is False


# ── the plateau backstop ──────────────────────────────────────────────────────
def test_plateau_at_not_losing_level_reanchors():
    v, st = _gate(mir=(190, 360), hist=FLAT53)          # flat ~53% (not losing, below bar) → backstop
    assert v == "promote" and st["reason"] == "plateau_reanchor"
    assert st["plateau"]["detected"] is True and st["plateau"]["backstop"] is True


def test_plateau_while_losing_does_not_reanchor():
    """If the climb has flattened BELOW 50% (the candidate is worse) but not collapsed, do NOT
    re-anchor onto it — hold and let more training / a deeper erosion (mirror-collapse) handle it."""
    v, st = _gate(mir=(162, 360), hist=FLAT45)          # flat ~45% — plateaued but LOSING (not collapsed)
    assert v == "hold"
    assert st["plateau"]["detected"] is True and st["plateau"]["backstop"] is False
    assert st["mirror"]["collapsed"] is False


def test_bar_takes_precedence_over_plateau():
    # decisively winning AND the history is flat-high → promote by the BAR, not the backstop
    v, st = _gate(mir=(270, 360), hist=[0.74] * 10)
    assert v == "promote" and st["reason"] == "beat_champion"


# ── scripted collapse safety (highest priority) ───────────────────────────────
def test_scripted_collapse_reverts():
    v, st = _gate(scr=(72, 180), hw=0.55, mir=(180, 360))   # scripted 0.40, floor 0.49 → collapse
    assert v == "revert" and st["reason"] == "scripted_collapse"
    assert st["scripted"]["collapsed"] is True


def test_collapse_takes_precedence_over_a_bar_clear():
    # even a decisive mirror win can't promote a policy that forgot how to play vs scripts
    v, st = _gate(scr=(72, 180), hw=0.55, mir=(285, 360))   # mirror 0.79 but scripted collapsed
    assert v == "revert" and st["reason"] == "scripted_collapse"


def test_small_scripted_dip_within_floor_does_not_collapse():
    # scripted 0.52 vs high-water 0.55 (floor 0.49): a dip, but NOT below the floor → not a collapse
    v, st = _gate(scr=(94, 180), hw=0.55, mir=(190, 360), hist=RISING)   # 0.522 scripted, 0.528 mirror
    assert st["scripted"]["collapsed"] is False
    assert v == "hold"          # below the bar, still climbing


def test_high_water_none_disables_collapse_floor():
    # no accepted-champion baseline yet (high_water None) → never a scripted collapse
    v, st = _gate(scr=(20, 180), hw=None, mir=(270, 360))
    assert st["scripted"]["collapsed"] is False
    assert v == "promote" and st["reason"] == "beat_champion"   # bar still works


# ── mirror-collapse revert (P2.2 — degradation vs the learner's OWN frozen champion) ──
def test_mirror_collapse_reverts():
    """The learner has eroded to ~0.38 vs its own frozen champion (scripted still fine) → revert."""
    v, st = _gate(scr=(120, 180), hw=0.55, mir=(137, 360), hist=FLAT45)   # 0.38 mirror, scripted ok
    assert v == "revert" and st["reason"] == "mirror_collapse"
    assert st["mirror"]["collapsed"] is True
    assert st["scripted"]["collapsed"] is False


def test_mirror_collapse_needs_enough_games():
    """A bad mirror over too few games must NOT revert (noise) — needs mirror_collapse_min_games."""
    v, st = _gate(mir=(8, 20), hist=FLAT45)             # 0.40 over only 20 games
    assert st["mirror"]["collapsed"] is False
    assert v == "hold"


def test_healthy_reanchor_not_mirror_collapsed():
    """A freshly re-anchored learner sitting ~0.50 vs the new champion is never falsely reverted."""
    v, st = _gate(mir=(180, 360), hist=RISING)          # exactly 0.50 over 360
    assert st["mirror"]["collapsed"] is False
    assert v == "hold"


def test_scripted_collapse_precedence_over_mirror_collapse():
    # both axes collapsed → the reason is scripted_collapse (the objective floor wins precedence)
    v, st = _gate(scr=(72, 180), hw=0.55, mir=(137, 360))   # scripted 0.40 AND mirror 0.38
    assert v == "revert" and st["reason"] == "scripted_collapse"
    assert st["scripted"]["collapsed"] is True and st["mirror"]["collapsed"] is True


# ── champion-lineage Elo (non-saturating observability) ───────────────────────
def test_advance_champion_steps_lineage_elo():
    import math
    from v_dance.selfplay.generation import GenerationHistory
    h = GenerationHistory()
    h.advance_champion("g0.pt", (120, 180), mirror_rate=None, base_elo=1000.0)
    assert h.champion_elo == pytest.approx(1000.0)          # first champion seeds the lineage
    h.advance_champion("g2.pt", (130, 180), mirror_rate=0.72)
    assert h.champion_elo == pytest.approx(1000 + 400 * math.log10(0.72 / 0.28))   # ~+164
    before = h.champion_elo
    h.advance_champion("g5.pt", (130, 180), mirror_rate=0.55)
    assert 0 < h.champion_elo - before < 40                 # a marginal re-anchor adds little


# ── operator alert (unattended-run watchdog) ──────────────────────────────────
def test_operator_alert_collapse_loop_and_stall():
    from v_dance.selfplay.generation import operator_alert, GenerationHistory, GenerationRecord
    collapse = GenerationHistory()
    for g in range(3):
        collapse.add(GenerationRecord(g, 0, 0, 0, None, "revert", False))
    assert "collapse loop" in operator_alert(collapse, revert_limit=3)

    healthy = GenerationHistory()
    healthy.add(GenerationRecord(0, 0, 0, 0, None, "promote", True))
    for g in range(1, 5):
        healthy.add(GenerationRecord(g, 0, 0, 0, None, "hold", False))
    assert operator_alert(healthy, stall_limit=25) is None  # only 4 gens since a promote

    stall = GenerationHistory()
    stall.add(GenerationRecord(0, 0, 0, 0, None, "promote", True))
    for g in range(1, 30):
        stall.add(GenerationRecord(g, 0, 0, 0, None, "hold", False))
    assert "frozen" in operator_alert(stall, stall_limit=25)


# ── shared plateau detector ───────────────────────────────────────────────────
def test_is_plateau_shared_impl_matches_expectations():
    assert is_plateau(RISING, 5, 0.01) is False
    assert is_plateau(FLAT53, 5, 0.01) is True
    assert is_plateau([0.6] * 9, 5) is False            # < 2*window → keep waiting
