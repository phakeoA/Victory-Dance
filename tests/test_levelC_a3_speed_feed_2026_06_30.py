"""
Level C / A3c — live SPEED feeder unit tests (2026-06-30).

Covers ``live_belief_feed.feed_speed_constraints`` (per-turn speed-tier narrowing from move order) +
the player glue (``_opp_speed_mult_known`` / ``_order_forced_hint`` / ``_prev_turn_proto_segment`` /
``_speed_ctx``):

  * a same-bracket move-vs-move turn records a speed bound in the correct DIRECTION (slower/faster);
  * Trick Room FLIPS the read; a priority-bracket difference yields NO read; ``order_forced`` (proto
    hint) and a repeated move (Instruct/Dancer) make every read bail; identity (Zoroark/Ditto) skips;
  * a sandwich (our_a, opp, our_b) yields TWO informative bounds;
  * the opp's KNOWN multipliers (boost/Tailwind/paralysis/revealed-Scarf) are folded; ``context_known``
    reflects whether item+ability are revealed; the forced-order proto scan works.

The narrowing MATH + the analyzer physics are covered by ``test_levelC_a3_speed_*``; these verify the
live PLUMBING (order → analyzer context → observe) + gating.
"""
from __future__ import annotations

import types

from v_dance.parser.belief_state import BeliefState
from v_dance.parser.match_belief import MatchBelief
from v_dance.parser.vod_parser.pokedex import norm_species
from v_dance.play.live_belief_feed import feed_speed_constraints, _move_speed_props

OPP = "Garchomp"   # any species — the speed feeder records constraints without a belief lookup


def _mb():
    # observe_speed_bound never touches the prior, so a bare BeliefState is enough (narrowing is at block_for)
    try:
        return MatchBelief(BeliefState())
    except Exception:
        import pytest
        pytest.skip("BeliefState unavailable")


def _constraints(mb, sp=OPP):
    obs = mb._mons.get(norm_species(sp))
    return obs.speed_constraints if obs else []


def _move(slot, move, idx):
    return {"event": "move", "user_slot": slot, "move": move, "execution_index": idx}


def _prev(actions, opp_extra=None):
    opp = {"species": OPP, "base_species": OPP}
    if opp_extra:
        opp.update(opp_extra)
    return {
        "state_before_actions": {"p1": {
            "our_active": {"our_a": {"species": "Floette"}, "our_b": {"species": "Sneasler"}},
            "opp_active": {"opp_a": opp, "opp_b": None},
        }},
        "actions": actions,
    }


def _ctx(**kw):
    base = {
        "trick_room": False, "order_forced": False,
        "our": {"our_a": {"eff_speed": 200.0, "ability": None, "hp_frac": 1.0},
                "our_b": {"eff_speed": 100.0, "ability": None, "hp_frac": 1.0}},
        "opp": {"opp_a": {"speed_mult_known": 1.0, "context_known": True,
                          "ability": None, "hp_frac": 1.0}},
    }
    base.update(kw)
    return base


# ── direction ─────────────────────────────────────────────────────────────────
def test_our_first_opp_slower():
    mb = _mb()
    prev = _prev([_move("p1a", "Tackle", 0), _move("p2a", "Tackle", 1)])
    stats = feed_speed_constraints(mb, prev, "p1", _ctx())
    assert stats["used"] == 1
    c = _constraints(mb)
    assert len(c) == 1 and c[0]["faster"] is False and c[0]["threshold"] == 200.0


def test_opp_first_opp_faster():
    mb = _mb()
    prev = _prev([_move("p2a", "Tackle", 0), _move("p1a", "Tackle", 1)])
    stats = feed_speed_constraints(mb, prev, "p1", _ctx())
    assert stats["used"] == 1 and _constraints(mb)[0]["faster"] is True


def test_trick_room_flips():
    mb = _mb()
    prev = _prev([_move("p1a", "Tackle", 0), _move("p2a", "Tackle", 1)])
    feed_speed_constraints(mb, prev, "p1", _ctx(trick_room=True))
    # our moved first, opp second → under TR that means opp is FASTER
    assert _constraints(mb)[0]["faster"] is True


def test_known_multiplier_divided_out():
    mb = _mb()
    prev = _prev([_move("p1a", "Tackle", 0), _move("p2a", "Tackle", 1)])
    ctx = _ctx()
    ctx["opp"]["opp_a"]["speed_mult_known"] = 2.0       # opp had Tailwind → divide the threshold by 2
    feed_speed_constraints(mb, prev, "p1", ctx)
    assert _constraints(mb)[0]["threshold"] == 100.0    # 200 / 2


# ── no-read gates ─────────────────────────────────────────────────────────────
def test_priority_bracket_no_read():
    mb = _mb()
    # our uses +1 priority Quick Attack → moved first by PRIORITY, not speed
    prev = _prev([_move("p1a", "Quick Attack", 0), _move("p2a", "Tackle", 1)])
    stats = feed_speed_constraints(mb, prev, "p1", _ctx())
    assert stats["used"] == 0 and stats["skipped_bracket"] == 1 and not _constraints(mb)


def test_order_forced_bails():
    mb = _mb()
    prev = _prev([_move("p1a", "Tackle", 0), _move("p2a", "Tackle", 1)])
    stats = feed_speed_constraints(mb, prev, "p1", _ctx(order_forced=True))
    assert stats["used"] == 0 and not _constraints(mb)


def test_repeated_move_forces_bail():
    mb = _mb()
    # opp_a moves twice (Instruct/Dancer) → the whole turn's reads bail
    prev = _prev([_move("p1a", "Tackle", 0), _move("p2a", "Tackle", 1), _move("p2a", "Tackle", 2)])
    stats = feed_speed_constraints(mb, prev, "p1", _ctx())
    assert stats["used"] == 0 and not _constraints(mb)


def test_disguised_opp_skipped():
    mb = _mb()
    prev = _prev([_move("p1a", "Tackle", 0), _move("p2a", "Tackle", 1)],
                 opp_extra={"illusion_active": True})
    stats = feed_speed_constraints(mb, prev, "p1", _ctx())
    assert stats["used"] == 0 and stats["skipped_id"] == 1


def test_unknown_our_speed_skipped():
    mb = _mb()
    prev = _prev([_move("p1a", "Tackle", 0), _move("p2a", "Tackle", 1)])
    ctx = _ctx()
    ctx["our"]["our_a"]["eff_speed"] = 0.0      # our speed unknown → no threshold
    stats = feed_speed_constraints(mb, prev, "p1", ctx)
    assert stats["used"] == 0


# ── sandwich → two bounds ─────────────────────────────────────────────────────
def test_sandwich_two_bounds():
    mb = _mb()
    # our_a (idx0), opp (idx1), our_b (idx2): opp slower than our_a (200), faster than our_b (100)
    prev = _prev([_move("p1a", "Tackle", 0), _move("p2a", "Tackle", 1), _move("p1b", "Tackle", 2)])
    stats = feed_speed_constraints(mb, prev, "p1", _ctx())
    assert stats["used"] == 2
    c = sorted(_constraints(mb), key=lambda x: x["threshold"])
    assert c[0]["threshold"] == 100.0 and c[0]["faster"] is True     # vs our_b → faster than 100
    assert c[1]["threshold"] == 200.0 and c[1]["faster"] is False    # vs our_a → slower than 200


def test_role_p2_mapping():
    mb = _mb()
    prev = {
        "state_before_actions": {"p2": {
            "our_active": {"our_a": {"species": "Floette"}, "our_b": None},
            "opp_active": {"opp_a": {"species": OPP, "base_species": OPP}, "opp_b": None},
        }},
        "actions": [_move("p2a", "Tackle", 0), _move("p1a", "Tackle", 1)],
    }
    ctx = {"trick_room": False, "order_forced": False,
           "our": {"our_a": {"eff_speed": 150.0, "ability": None, "hp_frac": 1.0}},
           "opp": {"opp_a": {"speed_mult_known": 1.0, "context_known": True, "ability": None, "hp_frac": 1.0}}}
    stats = feed_speed_constraints(mb, prev, "p2", ctx)
    assert stats["used"] == 1 and _constraints(mb)[0]["faster"] is False


def test_move_speed_props():
    p = _move_speed_props("Quick Attack")
    assert p["priority"] == 1 and p["category"] == "physical" and p["type"].lower() == "normal"
    assert _move_speed_props("Tackle")["priority"] == 0
    assert _move_speed_props("Nonexistent Move XYZ") is None


# ── player glue ───────────────────────────────────────────────────────────────
def _player():
    from v_dance.play.live_vgc_base import SplicingVGCPlayerBase

    class _C(SplicingVGCPlayerBase):
        def _select_actions(self, battle, state_vec):
            return 0, 0, "test"

    p = _C.__new__(_C)
    p._proto_log = {}
    return p


def test_opp_speed_mult_known():
    from v_dance.play.live_vgc_base import SplicingVGCPlayerBase
    f = SplicingVGCPlayerBase._opp_speed_mult_known
    # +1 spe boost → ×1.5
    m = types.SimpleNamespace(boosts={"spe": 1}, status=None, item=None, ability=None,
                              revealed=True, current_hp_fraction=1.0)
    r = f(m, opp_tw=False, weather=None)
    assert abs(r["speed_mult_known"] - 1.5) < 1e-9 and r["context_known"] is False  # item/ability unknown
    # Tailwind ×2 + revealed Choice Scarf ×1.5
    m2 = types.SimpleNamespace(boosts={}, status=None, item="choicescarf", ability="roughskin",
                               revealed=True, current_hp_fraction=1.0)
    r2 = f(m2, opp_tw=True, weather=None)
    assert abs(r2["speed_mult_known"] - 3.0) < 1e-9 and r2["context_known"] is True
    # paralysis ×0.5
    par = types.SimpleNamespace(name="PAR")
    m3 = types.SimpleNamespace(boosts={}, status=par, item=None, ability=None,
                               revealed=True, current_hp_fraction=1.0)
    assert abs(f(m3, opp_tw=False, weather=None)["speed_mult_known"] - 0.5) < 1e-9


def test_order_forced_hint_and_segment():
    p = _player()
    tag = "battle-1"
    p._proto_log[tag] = [
        "|turn|3",
        "|move|p1a: Floette|Protect|p1a: Floette",
        "|move|p2a: Whimsicott|After You|p1b: Sneasler",
        "|turn|4",
    ]
    battle = types.SimpleNamespace(battle_tag=tag, turn=4)
    seg = p._prev_turn_proto_segment(battle)
    assert "After You" in seg and "|turn|" not in seg          # only the T-3 segment
    assert p._order_forced_hint(battle, {}) is True            # After You detected
    # a clean turn → no forced hint
    p._proto_log[tag] = ["|turn|3", "|move|p1a: Floette|Moonblast|p2a: X", "|turn|4"]
    assert p._order_forced_hint(battle, {}) is False


def test_order_forced_hint_catches_each_token():
    """FIX #2: the forced-order proto scan is a GUARANTEED guard (a hit → the read bails), not a soft
    hope — Quick Claw / Quash / Instruct / After You / Dancer in the T-3 segment all trigger it."""
    p = _player()
    tag, battle = "b", types.SimpleNamespace(battle_tag="b", turn=4)
    for line in (
        "|-activate|p2a: Garchomp|item: Quick Claw",
        "|-activate|p1a: Whimsicott|move: Quash|p2a: Garchomp",
        "|-activate|p2a: Amoonguss|move: Instruct",
        "|move|p2a: Whimsicott|After You|p1a: X",
        "|-activate|p2a: Oricorio|ability: Dancer",
    ):
        p._proto_log[tag] = ["|turn|3", line, "|turn|4"]
        assert p._order_forced_hint(battle, {}) is True, line


def test_speed_ctx_assembles():
    p = _player()
    p._encoder = types.SimpleNamespace(_live_effective_speed=lambda m, o, tw, w, mr: (150.0, 1.0))
    our = types.SimpleNamespace(ability=None, revealed=True, current_hp_fraction=1.0)
    opp = types.SimpleNamespace(boosts={}, status=None, item=None, ability=None,
                                revealed=True, current_hp_fraction=1.0)
    battle = types.SimpleNamespace(
        weather=[], fields=[], side_conditions={}, opponent_side_conditions={},
        active_pokemon=[our], opponent_active_pokemon=[opp], battle_tag="b", turn=4)
    ctx = p._speed_ctx(battle, {})
    assert ctx["our"]["our_a"]["eff_speed"] == 150.0
    assert "opp_a" in ctx["opp"] and ctx["opp"]["opp_a"]["speed_mult_known"] == 1.0
    assert ctx["trick_room"] is False and ctx["order_forced"] is False
