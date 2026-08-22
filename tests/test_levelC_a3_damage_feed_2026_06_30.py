"""
Level C / A3c — live DAMAGE feeder unit tests (2026-06-30).

Covers ``live_belief_feed.feed_damage_constraints`` (the per-turn A2 narrowing feeder) + the player
glue (``_our_stats_by_species`` / ``_feed_match_belief`` no-op-when-off):

  * a DEFENSIVE hit (our mon → opp) records an opp Def/SpD constraint; an OFFENSIVE hit (opp → our
    mon) records an opp Atk/SpA constraint — both reached through the feeder's slot/perspective mapping;
  * gates: full-HP-only, non-KO-only, identity (Zoroark/Ditto skipped), same-side (self-hit) skipped,
    status / stat-independent move skipped, missing our-stats skipped;
  * ``_persp_key`` raw-slot → perspective mapping for both roles;
  * ``_our_stats_by_species`` pulls exact stats off the live team; ``_feed_match_belief`` is a true
    no-op when MatchBelief is off.

The narrowing MATH is covered by the A2 unit tests; these verify the live PLUMBING + gating.
"""
from __future__ import annotations

import types

import pytest

from v_dance.parser.belief_state import BeliefState, dex_base_stats
from v_dance.parser.match_belief import MatchBelief
from v_dance.parser.vod_parser.pokedex import norm_species
from v_dance.play.live_belief_feed import feed_damage_constraints, _persp_key


@pytest.fixture(scope="module")
def belief() -> BeliefState:
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"BeliefState unavailable: {e}")


def _opp_species(belief: BeliefState) -> str:
    """An opponent species with both a spread distribution AND base stats (so _modal_full resolves)."""
    for sp in belief.all_pokemon():
        if dex_base_stats(sp) and belief.spread_distribution(sp, top_k=5):
            return sp
    pytest.skip("no species with spread + base stats")


# Garchomp = a clean Ground/Dragon attacker; Earthquake = a standard physical Ground move (bp 100).
OUR = {"species": "Garchomp", "base_species": "Garchomp"}
OUR_STATS = {norm_species("Garchomp"): {"hp": 183, "atk": 180, "def": 115,
                                        "spa": 90, "spd": 105, "spe": 169}}


def _prev_turn(opp_sp, *, dslot, sslot, dspecies, sspecies, move="Earthquake",
               after=60.0, delta=-40.0, opp_extra=None, crit=False):
    opp = {"species": opp_sp, "base_species": opp_sp}
    if opp_extra:
        opp.update(opp_extra)
    return {
        "state_before_actions": {"p1": {
            "our_active": {"our_a": dict(OUR), "our_b": None},
            "opp_active": {"opp_a": opp, "opp_b": None},
        }},
        "damage_events": [{
            "event": "damage", "slot": dslot, "species": dspecies,
            "source_slot": sslot, "source_species": sspecies, "source_move": move,
            "hp_pct_after": after, "hp_pct_delta": delta, "crit": crit,
        }],
    }


def _constraints(mb, sp):
    obs = mb._mons.get(norm_species(sp))
    return obs.damage_constraints if obs else []


# ── DEFENSIVE: our hit → narrow opp bulk ─────────────────────────────────────
def test_defensive_constraint_recorded(belief):
    opp_sp = _opp_species(belief)
    mb = MatchBelief(belief)
    prev = _prev_turn(opp_sp, dslot="p2a", sslot="p1a", dspecies=opp_sp, sspecies="Garchomp")
    stats = feed_damage_constraints(mb, belief, prev, "p1", OUR_STATS)
    assert stats["used_def"] == 1 and stats["used_off"] == 0
    cons = _constraints(mb, opp_sp)
    assert len(cons) == 1 and cons[0]["mode"] == "def" and cons[0]["stat"] == "def"
    # block_for stays a valid, normalised, never-zeroed distribution
    block = mb.block_for(opp_sp)
    ps = [s["p"] for s in (block.get("spreads") or [])]
    assert ps and all(p > 0 for p in ps) and abs(sum(ps) - 1.0) <= 0.02


# ── OFFENSIVE: opp hit → narrow opp offense ──────────────────────────────────
def test_offensive_constraint_recorded(belief):
    opp_sp = _opp_species(belief)
    mb = MatchBelief(belief)
    # opp (p2a) hits OUR Garchomp (p1a) with a physical move
    prev = _prev_turn(opp_sp, dslot="p1a", sslot="p2a", dspecies="Garchomp", sspecies=opp_sp)
    stats = feed_damage_constraints(mb, belief, prev, "p1", OUR_STATS)
    assert stats["used_off"] == 1 and stats["used_def"] == 0
    cons = _constraints(mb, opp_sp)
    assert len(cons) == 1 and cons[0]["mode"] == "off" and cons[0]["stat"] == "atk"


# ── gates ────────────────────────────────────────────────────────────────────
def test_not_full_hp_skipped(belief):
    opp_sp = _opp_species(belief)
    mb = MatchBelief(belief)
    # before = 70 + 30 ... after=40, delta=-30 → before=70 < 95 → skipped
    prev = _prev_turn(opp_sp, dslot="p2a", sslot="p1a", dspecies=opp_sp, sspecies="Garchomp",
                      after=40.0, delta=-30.0)
    stats = feed_damage_constraints(mb, belief, prev, "p1", OUR_STATS)
    assert stats["used_def"] == 0 and stats["skipped_hp"] == 1
    assert not _constraints(mb, opp_sp)


def test_ko_skipped(belief):
    opp_sp = _opp_species(belief)
    mb = MatchBelief(belief)
    prev = _prev_turn(opp_sp, dslot="p2a", sslot="p1a", dspecies=opp_sp, sspecies="Garchomp",
                      after=0.0, delta=-100.0)
    stats = feed_damage_constraints(mb, belief, prev, "p1", OUR_STATS)
    assert stats["used_def"] == 0 and stats["skipped_hp"] == 1


def test_disguised_opp_skipped(belief):
    opp_sp = _opp_species(belief)
    mb = MatchBelief(belief)
    prev = _prev_turn(opp_sp, dslot="p2a", sslot="p1a", dspecies=opp_sp, sspecies="Garchomp",
                      opp_extra={"illusion_active": True})
    stats = feed_damage_constraints(mb, belief, prev, "p1", OUR_STATS)
    assert stats["used_def"] == 0 and stats["skipped_id"] == 1
    assert not _constraints(mb, opp_sp)


def test_same_side_self_hit_skipped(belief):
    opp_sp = _opp_species(belief)
    mb = MatchBelief(belief)
    # both slots ours (a self-inflicted / ally hit) → neither direction applies
    prev = _prev_turn(opp_sp, dslot="p1a", sslot="p1b", dspecies="Garchomp", sspecies="Garchomp")
    stats = feed_damage_constraints(mb, belief, prev, "p1", OUR_STATS)
    assert stats["used_def"] == 0 and stats["used_off"] == 0


def test_status_move_skipped(belief):
    opp_sp = _opp_species(belief)
    mb = MatchBelief(belief)
    prev = _prev_turn(opp_sp, dslot="p2a", sslot="p1a", dspecies=opp_sp, sspecies="Garchomp",
                      move="Thunder Wave")
    stats = feed_damage_constraints(mb, belief, prev, "p1", OUR_STATS)
    assert stats["used_def"] == 0 and stats["skipped_move"] == 1


def test_missing_our_stats_skipped(belief):
    opp_sp = _opp_species(belief)
    mb = MatchBelief(belief)
    prev = _prev_turn(opp_sp, dslot="p2a", sslot="p1a", dspecies=opp_sp, sspecies="Garchomp")
    stats = feed_damage_constraints(mb, belief, prev, "p1", {})   # our exact stats unavailable
    assert stats["used_def"] == 0 and stats["skipped_ctx"] == 1


def test_no_constraint_on_empty_prev(belief):
    mb = MatchBelief(belief)
    assert feed_damage_constraints(mb, belief, None, "p1", OUR_STATS)["used_def"] == 0


# ── perspective slot mapping ──────────────────────────────────────────────────
def test_persp_key_mapping():
    assert _persp_key("p1a", "p1") == ("our_a", True)
    assert _persp_key("p2b", "p1") == ("opp_b", False)
    assert _persp_key("p2a", "p2") == ("our_a", True)
    assert _persp_key("p1a", "p2") == ("opp_a", False)
    assert _persp_key("", "p1") is None
    assert _persp_key("p1", "p1") is None     # too short / no slot letter


def test_role_p2_defensive(belief):
    """own_role = p2: our mon is p2a, opp is p1a — the mapping must flip correctly."""
    opp_sp = _opp_species(belief)
    mb = MatchBelief(belief)
    prev = {
        "state_before_actions": {"p2": {
            "our_active": {"our_a": dict(OUR), "our_b": None},
            "opp_active": {"opp_a": {"species": opp_sp, "base_species": opp_sp}, "opp_b": None},
        }},
        "damage_events": [{
            "event": "damage", "slot": "p1a", "species": opp_sp,
            "source_slot": "p2a", "source_species": "Garchomp", "source_move": "Earthquake",
            "hp_pct_after": 60.0, "hp_pct_delta": -40.0,
        }],
    }
    stats = feed_damage_constraints(mb, belief, prev, "p2", OUR_STATS)
    assert stats["used_def"] == 1


# ── player glue ───────────────────────────────────────────────────────────────
def _player(use_match_belief: bool):
    from v_dance.play.live_vgc_base import SplicingVGCPlayerBase

    class _C(SplicingVGCPlayerBase):
        def _select_actions(self, battle, state_vec):
            return 0, 0, "test"

    p = _C.__new__(_C)
    p._use_match_belief = use_match_belief
    p._match_belief = {}
    p._encoder = types.SimpleNamespace(belief=None, level=50)
    return p


def test_our_stats_by_species():
    from v_dance.play.live_vgc_base import SplicingVGCPlayerBase
    good = types.SimpleNamespace(species="garchomp", base_species="garchomp",
                                 stats={"hp": 183, "atk": 180, "def": 115, "spa": 90, "spd": 105, "spe": 169})
    bad = types.SimpleNamespace(species="ditto", base_species="ditto",
                                stats={"hp": 100, "atk": None, "def": None, "spa": None, "spd": None, "spe": None})
    battle = types.SimpleNamespace(team={"a": good, "b": bad})
    out = SplicingVGCPlayerBase._our_stats_by_species(battle)
    assert out[norm_species("garchomp")]["atk"] == 180
    assert norm_species("ditto") not in out        # incomplete stats skipped


def test_feed_match_belief_noop_when_off():
    p = _player(use_match_belief=False)
    battle = types.SimpleNamespace(player_role="p1", turn=5, battle_tag="b1", team={})
    p._feed_match_belief(battle)                     # must not raise, must create nothing
    assert p._match_belief == {}


# ── real-parser integration: prev_turn_from_log_prefix off-by-one + raw→feeder ──
def _header(our: str, opp: str) -> str:
    return (
        "|player|p1|alice|101|1500\n|player|p2|bob|102|1500\n|gen|9\n|tier|[Gen 9] Test Doubles\n"
        f"|poke|p1|{our}, L50|\n|poke|p1|Pikachu, L50|\n"
        f"|poke|p2|{opp}, L50|\n|poke|p2|Pikachu, L50|\n"
        "|teamsize|p1|2\n|teamsize|p2|2\n|start\n"
        f"|switch|p1a: {our}|{our}, L50|100/100\n|switch|p1b: Pikachu|Pikachu, L50|100/100\n"
        f"|switch|p2a: {opp}|{opp}, L50|100/100\n|switch|p2b: Pikachu|Pikachu, L50|100/100\n"
    )


def test_prev_turn_from_log_prefix_offbyone(belief):
    from v_dance.encoders.live_state_encoder import prev_turn_from_log_prefix
    opp_sp = _opp_species(belief)
    log = _header("Garchomp", opp_sp) + (
        "|turn|1\n"
        f"|move|p1a: Garchomp|Earthquake|p2a: {opp_sp}\n"
        f"|-damage|p2a: {opp_sp}|60/100\n"
        "|upkeep\n|turn|2\n"
    )
    assert prev_turn_from_log_prefix(log, "p1", 1) is None         # no prior turn at turn 1
    prev = prev_turn_from_log_prefix(log, "p1", 2)
    assert prev is not None and prev["turn"] == 1                  # turns[-1] == T-1, not T
    dmg = [e for e in prev.get("damage_events") or [] if e.get("event") == "damage"]
    assert dmg and dmg[0]["slot"] == "p2a" and dmg[0]["source_slot"] == "p1a"
    assert dmg[0]["source_move"] == "Earthquake"


def test_raw_protocol_feeds_defensive_constraint(belief):
    """End-to-end: a real parsed turn (raw Showdown protocol) flows through the feeder and records a
    DEFENSIVE opp constraint — validating the raw turn-dict shape, not just synthetic dicts."""
    from v_dance.encoders.live_state_encoder import prev_turn_from_log_prefix
    opp_sp = _opp_species(belief)
    log = _header("Garchomp", opp_sp) + (
        "|turn|1\n"
        f"|move|p1a: Garchomp|Earthquake|p2a: {opp_sp}\n"
        f"|-damage|p2a: {opp_sp}|60/100\n"
        "|upkeep\n|turn|2\n"
    )
    prev = prev_turn_from_log_prefix(log, "p1", 2)
    mb = MatchBelief(belief)
    stats = feed_damage_constraints(mb, belief, prev, "p1", OUR_STATS)
    assert stats["used_def"] == 1 and stats["used_off"] == 0
    assert len(_constraints(mb, opp_sp)) == 1


# ── FIX #1: crit captured by the parser + folded by the feeder ────────────────
def test_parser_stamps_crit(belief):
    from v_dance.encoders.live_state_encoder import prev_turn_from_log_prefix
    opp_sp = _opp_species(belief)
    crit_log = _header("Garchomp", opp_sp) + (
        "|turn|1\n"
        f"|move|p1a: Garchomp|Earthquake|p2a: {opp_sp}\n"
        f"|-crit|p2a: {opp_sp}\n"
        f"|-damage|p2a: {opp_sp}|60/100\n"
        "|upkeep\n|turn|2\n"
    )
    prev = prev_turn_from_log_prefix(crit_log, "p1", 2)
    dmg = [e for e in prev["damage_events"] if e.get("event") == "damage"]
    assert dmg and dmg[0]["crit"] is True                  # |-crit| before |-damage| → stamped
    # control: no |-crit| line → not a crit
    plain = _header("Garchomp", opp_sp) + (
        f"|turn|1\n|move|p1a: Garchomp|Earthquake|p2a: {opp_sp}\n"
        f"|-damage|p2a: {opp_sp}|60/100\n|upkeep\n|turn|2\n"
    )
    prev2 = prev_turn_from_log_prefix(plain, "p1", 2)
    assert [e for e in prev2["damage_events"] if e.get("event") == "damage"][0]["crit"] is False


def test_crit_scales_reference_and_sharpens_sigma(belief):
    from v_dance.play.live_belief_feed import _live_damage_sigma
    opp_sp = _opp_species(belief)
    mb_c, mb_n = MatchBelief(belief), MatchBelief(belief)
    crit = _prev_turn(opp_sp, dslot="p2a", sslot="p1a", dspecies=opp_sp, sspecies="Garchomp", crit=True)
    none = _prev_turn(opp_sp, dslot="p2a", sslot="p1a", dspecies=opp_sp, sspecies="Garchomp", crit=False)
    feed_damage_constraints(mb_c, belief, crit, "p1", OUR_STATS)
    feed_damage_constraints(mb_n, belief, none, "p1", OUR_STATS)
    cc, cn = _constraints(mb_c, opp_sp)[0], _constraints(mb_n, opp_sp)[0]
    assert cc["mu_ref"] > cn["mu_ref"] * 1.4               # crit folds ~×1.5 into the reference
    # crit is KNOWN either way → σ drops the crit term (opp item/ability unknown → mitigation kept)
    assert abs(cc["sigma"] - _live_damage_sigma(False)) < 1e-9
    assert abs(cn["sigma"] - _live_damage_sigma(False)) < 1e-9


def test_offensive_sigma_is_sharp(belief):
    from v_dance.play.live_belief_feed import _live_damage_sigma
    opp_sp = _opp_species(belief)
    mb = MatchBelief(belief)
    prev = _prev_turn(opp_sp, dslot="p1a", sslot="p2a", dspecies="Garchomp", sspecies=opp_sp)
    feed_damage_constraints(mb, belief, prev, "p1", OUR_STATS)
    # OUR defender → mitigation known → no mitigation term (sharp)
    assert abs(_constraints(mb, opp_sp)[0]["sigma"] - _live_damage_sigma(True)) < 1e-9


# ── FIX #3: ONE parse → (snapshot, prev_turn) ────────────────────────────────
def test_reconstruct_for_decision_single_parse(belief):
    from v_dance.encoders.live_state_encoder import (
        reconstruct_for_decision, opp_snapshot_from_log_prefix, prev_turn_from_log_prefix,
    )
    opp_sp = _opp_species(belief)
    log = _header("Garchomp", opp_sp) + (
        "|turn|1\n"
        f"|move|p1a: Garchomp|Earthquake|p2a: {opp_sp}\n"
        f"|-damage|p2a: {opp_sp}|60/100\n"
        "|upkeep\n|turn|2\n"
    )
    snap, prev = reconstruct_for_decision(log, "p1", 2)
    # identical to the two single-purpose helpers (just one parse instead of two)
    assert prev["turn"] == prev_turn_from_log_prefix(log, "p1", 2)["turn"] == 1
    indiv = opp_snapshot_from_log_prefix(log, "p1", 2)
    assert (snap is None) == (indiv is None)
    if snap is not None:
        assert set(snap.get("opp_active") or {}) == set(indiv.get("opp_active") or {})
    assert reconstruct_for_decision(log, "p1", 99) == (None, None)  # marker absent → nothing
    assert reconstruct_for_decision(log, "p1", 1)[1] is None        # turn 1 → no prior turn
