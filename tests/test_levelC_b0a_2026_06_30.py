"""
Level C / B0a — white-box stepper state-mutation core (2026-06-30).

Unit tests for the low-level mutation ops in ``v_dance/encoders/white_box_sim.py``: clone, damage→faint,
heal, status, boost, switch-in (+ entry hazards), residuals (sand/status chip, Leftovers, durations), and
the Tier-3 ``UnmodelledLog``.
"""
from __future__ import annotations

from v_dance.encoders import white_box_sim as W


def _mon(species="Charizard", hp=100.0, status=None, item=None, hp_stat=200):
    return {
        "species": species, "base_species": species, "hp_pct": hp, "status": status,
        "is_fainted": False, "boosts": {}, "known_item": item,
        "volatiles": {"protect_counter": 2, "confused": True},
        "stats_estimate": {"mode": "distribution",
                           "stats": {"hp": hp_stat, "atk": 150, "def": 120, "spa": 100, "spd": 100, "spe": 120}},
    }


def _state(our_a=None, opp_a=None, our_bench=None, our_side=None, opp_side=None, field=None):
    return {
        "field": field or {"weather": None, "terrain": None, "trick_room_turns_remaining": 0},
        "side_conditions": {"our_side": our_side or {"tailwind_turns_remaining": 0, "screens": {}},
                            "opp_side": opp_side or {"tailwind_turns_remaining": 0, "screens": {}}},
        "our_active": {"our_a": our_a, "our_b": None},
        "opp_active": {"opp_a": opp_a, "opp_b": None},
        "our_bench": list(our_bench or []), "opp_bench": [],
    }


# ── HP / faint ────────────────────────────────────────────────────────────────
def test_apply_damage_and_faint():
    m = _mon(hp=40.0)
    assert W.apply_damage_pct(m, 25.0) == 25.0 and m["hp_pct"] == 15.0
    assert not W.is_fainted(m)
    dealt = W.apply_damage_pct(m, 30.0)        # overkill → faint, dealt capped at remaining 15
    assert dealt == 15.0 and m["hp_pct"] == 0.0 and m["is_fainted"] is True
    assert W.apply_damage_pct(m, 10.0) == 0.0  # no-op on a fainted mon


def test_apply_heal_caps_and_skips_fainted():
    m = _mon(hp=90.0)
    assert W.apply_heal_pct(m, 25.0) == 10.0 and m["hp_pct"] == 100.0   # capped at 100
    f = _mon(hp=0.0); f["is_fainted"] = True
    assert W.apply_heal_pct(f, 50.0) == 0.0


def test_is_fainted_by_flag_or_zero_hp():
    assert W.is_fainted({"hp_pct": 0.0}) is True
    assert W.is_fainted({"hp_pct": 50.0, "is_fainted": True}) is True
    assert W.is_fainted({"hp_pct": 50.0}) is False
    assert W.is_fainted(None) is False


# ── status / boosts ───────────────────────────────────────────────────────────
def test_set_status():
    m = _mon()
    assert W.set_status(m, "brn") is True and m["status"] == "brn"
    assert W.set_status(m, "par") is False and m["status"] == "brn"   # already statused → no overwrite
    f = _mon(); f["is_fainted"] = True
    assert W.set_status(f, "psn") is False


def test_apply_boost_clamps():
    m = _mon()
    assert W.apply_boost(m, "atk", 2) == 2 and m["boosts"]["atk"] == 2
    assert W.apply_boost(m, "atk", 6) == 4 and m["boosts"]["atk"] == 6   # clamp +6
    assert W.apply_boost(m, "spe", -8) == -6 and m["boosts"]["spe"] == -6  # clamp -6


# ── clone isolation ───────────────────────────────────────────────────────────
def test_clone_state_isolation():
    s = _state(our_a=_mon(hp=100.0))
    c = W.clone_state(s)
    W.apply_damage_pct(c["our_active"]["our_a"], 50.0)
    assert c["our_active"]["our_a"]["hp_pct"] == 50.0
    assert s["our_active"]["our_a"]["hp_pct"] == 100.0     # original untouched


# ── switch-in ─────────────────────────────────────────────────────────────────
def test_switch_in_swaps_resets_and_hazards():
    out = _mon(species="Charizard", hp=80.0)
    out["boosts"] = {"atk": 3}
    inc = _mon(species="Charizard", hp=100.0)              # Fire/Flying → 4× Stealth Rock
    s = _state(our_a=out, our_bench=[inc],
               our_side={"tailwind_turns_remaining": 0, "screens": {}, "stealth_rock": True})
    got = W.switch_in(s, "our_a", 0)
    assert got is inc and s["our_active"]["our_a"] is inc
    assert s["our_bench"] == [out]
    assert out["boosts"] == {} and out["volatiles"]["confused"] is False and out["volatiles"]["protect_counter"] == 0
    assert inc["hp_pct"] < 100.0                            # Stealth Rock chip on entry (4× on Charizard)
    assert W.switch_in(s, "our_a", 9) is None               # bad index → None


# ── residuals ─────────────────────────────────────────────────────────────────
def test_residual_sandstorm_chips_non_immune_only():
    char = _mon(species="Charizard", hp=100.0)             # Fire/Flying → chipped
    chomp = _mon(species="Garchomp", hp=100.0)             # Ground → sand-immune
    s = _state(our_a=char, opp_a=chomp, field={"weather": "Sandstorm", "trick_room_turns_remaining": 0})
    W.apply_residuals(s)
    assert char["hp_pct"] < 100.0 and chomp["hp_pct"] == 100.0


def test_residual_status_chip_and_leftovers():
    burned = _mon(hp=100.0, status="brn")
    lefto = _mon(hp=50.0, item="Leftovers")
    s = _state(our_a=burned, opp_a=lefto)
    W.apply_residuals(s)
    assert abs(burned["hp_pct"] - (100.0 - 100.0 / 16.0)) < 1e-6   # burn chip 1/16
    assert lefto["hp_pct"] > 50.0                                  # Leftovers heal


def test_residual_toxic_logged_and_durations_decrement():
    tox = _mon(hp=100.0, status="tox")
    s = _state(our_a=tox,
               our_side={"tailwind_turns_remaining": 3, "screens": {}},
               field={"weather": None, "trick_room_turns_remaining": 2})
    log = W.UnmodelledLog()
    W.apply_residuals(s, log=log)
    assert tox["hp_pct"] < 100.0                                   # poison chip applied
    assert log["toxic_no_escalation"] == 1                         # Tier-3 noted (never silent)
    assert s["side_conditions"]["our_side"]["tailwind_turns_remaining"] == 2
    assert s["field"]["trick_room_turns_remaining"] == 1


def test_unmodelled_log_note():
    log = W.UnmodelledLog()
    log.note("foo"); log.note("foo", 2); log.note("bar")
    assert log["foo"] == 3 and log["bar"] == 1
