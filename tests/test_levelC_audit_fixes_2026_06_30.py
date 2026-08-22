"""
Level C — audit fixes (2026-06-30 bug-scan). Regression tests for the 5 confirmed bugs:

  1. match_belief loglik: a MISSING spread key defaults to the floor (neutral), not 0.0 (which would rank
     un-evaluated spreads as MOST likely and invert the narrowing — reachable for a mega opp).
  2. live_vgc_base: the per-turn belief feed is IDEMPOTENT — a re-issued choose_move for the same turn
     does not double-feed the Bayesian product.
  3. live_belief_feed: the speed feed SKIPS a slot whose current-board mon != the prev-turn mover (so a
     switch/replacement's multipliers aren't applied to the wrong species).
  4. replay_parser: a crit absorbed by a Substitute (no -damage line) does not leak crit=True onto a
     later direct hit on the same slot.
  (5. bulk_parse legacy-twin ordering is a data-tool reorder — verified by inspection, not unit-tested.)
"""
from __future__ import annotations

import types

import pytest

from v_dance.parser.belief_state import BeliefState, dex_base_stats, calc_full_stats
from v_dance.parser.match_belief import MatchBelief, _spread_key
from v_dance.parser.vod_parser.pokedex import norm_species
from v_dance.play.live_belief_feed import feed_speed_constraints


@pytest.fixture(scope="module")
def belief() -> BeliefState:
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"BeliefState unavailable: {e}")


def _species_with_atk_spread(belief):
    for sp in belief.all_pokemon():
        base = dex_base_stats(sp)
        block = belief.belief_block(sp, top_k=5) if base else None
        spreads = (block or {}).get("spreads") or []
        if base and len(spreads) >= 2 and len({
                calc_full_stats(base, s["evs_actual"], s["nature"])["atk"] for s in spreads}) >= 2:
            return sp
    pytest.skip("no species with a varied Atk spread")


# ── Fix 1: a missing loglik key is NEUTRAL (floor), not max-likelihood ─────────
def test_loglik_missing_key_is_neutral_not_max(belief):
    sp = _species_with_atk_spread(belief)
    base = dex_base_stats(sp)
    static = belief.belief_block(sp, top_k=5)
    smin = min(static["spreads"], key=lambda s: calc_full_stats(base, s["evs_actual"], s["nature"])["atk"])
    mb = MatchBelief(belief)
    # ONLY the min-Atk spread is in the map (favoured, loglik 0); EVERY other spread is MISSING.
    mb.observe_damage_loglik(sp, {_spread_key(smin): 0.0})
    narrowed = mb.block_for(sp)
    # with the fix the present (favoured) spread wins → est Atk drops; the bug would let the missing
    # spreads default to 0.0 (>= the present 0.0) and leave the distribution unchanged or inverted.
    assert static["spreads"] != narrowed["spreads"]
    p_static = next(s["p"] for s in static["spreads"] if _spread_key(s) == _spread_key(smin))
    p_narrow = next(s["p"] for s in narrowed["spreads"] if _spread_key(s) == _spread_key(smin))
    assert p_narrow > p_static
    assert narrowed["expected_stats"]["atk"] <= static["expected_stats"]["atk"]


# ── Fix 2: per-turn feed is idempotent ────────────────────────────────────────
def test_feed_idempotent_per_turn(monkeypatch):
    from v_dance.play.live_vgc_base import SplicingVGCPlayerBase
    import v_dance.play.live_belief_feed as LBF
    calls = {"dmg": 0, "spd": 0}
    monkeypatch.setattr(LBF, "feed_damage_constraints",
                        lambda *a, **k: (calls.__setitem__("dmg", calls["dmg"] + 1), {})[1])
    monkeypatch.setattr(LBF, "feed_speed_constraints",
                        lambda *a, **k: (calls.__setitem__("spd", calls["spd"] + 1), {})[1])

    class _C(SplicingVGCPlayerBase):
        def _select_actions(self, b, s):
            return 0, 0, "t"

    p = _C.__new__(_C)
    p._use_match_belief = True
    p._match_belief = {}
    p._belief_fed_turn = {}
    p._encoder = types.SimpleNamespace(belief=object(), level=50)
    battle = types.SimpleNamespace(
        player_role="p1", battle_tag="b1", turn=3, team={}, weather=[], fields=[],
        side_conditions={}, opponent_side_conditions={},
        active_pokemon=[None, None], opponent_active_pokemon=[None, None])
    prev = {"state_before_actions": {"p1": {"our_active": {}, "opp_active": {}}},
            "damage_events": [], "actions": []}
    p._feed_match_belief(battle, prev)
    p._feed_match_belief(battle, prev)        # SAME turn re-entry → must not re-feed
    assert calls["dmg"] == 1 and calls["spd"] == 1
    battle.turn = 4
    p._feed_match_belief(battle, prev)        # a new turn DOES feed
    assert calls["dmg"] == 2 and calls["spd"] == 2


# ── Fix 3: speed feed skips a slot whose current mon != prev-turn mover ────────
def _move(slot, mv, idx):
    return {"event": "move", "user_slot": slot, "move": mv, "execution_index": idx}


def _speed_prev(opp_sp):
    return {
        "state_before_actions": {"p1": {
            "our_active": {"our_a": {"species": "Floette"}, "our_b": None},
            "opp_active": {"opp_a": {"species": opp_sp, "base_species": opp_sp}, "opp_b": None},
        }},
        "actions": [_move("p1a", "Tackle", 0), _move("p2a", "Tackle", 1)],
    }


def _speed_ctx(opp_species):
    return {"trick_room": False, "order_forced": False,
            "our": {"our_a": {"eff_speed": 200.0, "ability": None, "hp_frac": 1.0}},
            "opp": {"opp_a": {"speed_mult_known": 1.0, "context_known": True, "ability": None,
                              "hp_frac": 1.0, "species": opp_species}}}


def test_speed_feed_skips_on_slot_species_change(belief):
    # prev-turn mover = Garchomp, but the slot's CURRENT mon is Iron Hands → skip
    mb = MatchBelief(belief)
    stats = feed_speed_constraints(mb, _speed_prev("Garchomp"), "p1", _speed_ctx(norm_species("Iron Hands")))
    assert stats["used"] == 0 and stats["skipped_ctx"] >= 1
    # same mon still in the slot → the read proceeds
    mb2 = MatchBelief(belief)
    stats2 = feed_speed_constraints(mb2, _speed_prev("Garchomp"), "p1", _speed_ctx(norm_species("Garchomp")))
    assert stats2["used"] == 1


# ── Fix 4: a Substitute-absorbed crit does not leak onto a later hit ───────────
def _header(our, opp):
    return (
        "|player|p1|alice|101|1500\n|player|p2|bob|102|1500\n|gen|9\n|tier|[Gen 9] Test Doubles\n"
        f"|poke|p1|{our}, L50|\n|poke|p1|Pikachu, L50|\n|poke|p2|{opp}, L50|\n|poke|p2|Pikachu, L50|\n"
        "|teamsize|p1|2\n|teamsize|p2|2\n|start\n"
        f"|switch|p1a: {our}|{our}, L50|100/100\n|switch|p1b: Pikachu|Pikachu, L50|100/100\n"
        f"|switch|p2a: {opp}|{opp}, L50|100/100\n|switch|p2b: Pikachu|Pikachu, L50|100/100\n"
    )


def test_crit_does_not_leak_past_substitute(belief):
    from v_dance.encoders.live_state_encoder import prev_turn_from_log_prefix
    opp = next((s for s in belief.all_pokemon() if dex_base_stats(s)), "Garchomp")
    # p1a crits p2a's Substitute (absorbed, NO -damage); then p1b's Tackle hits p2a directly.
    log = _header("Garchomp", opp) + (
        "|turn|1\n"
        f"|move|p1a: Garchomp|Earthquake|p2a: {opp}\n"
        f"|-crit|p2a: {opp}\n"
        f"|-end|p2a: {opp}|Substitute\n"
        f"|move|p1b: Pikachu|Tackle|p2a: {opp}\n"
        f"|-damage|p2a: {opp}|60/100\n"
        "|upkeep\n|turn|2\n"
    )
    prev = prev_turn_from_log_prefix(log, "p1", 2)
    dmg = [e for e in prev["damage_events"] if e.get("event") == "damage" and e.get("source_move") == "Tackle"]
    assert dmg and dmg[0]["crit"] is False        # the pending crit was cleared, not leaked onto Tackle
