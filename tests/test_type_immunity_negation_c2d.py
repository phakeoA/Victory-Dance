"""v11 Phase C.2d — TYPE-IMMUNITY NEGATION.

Game rules that turn a per-(move_type → defender_type) 0× TYPING immunity into 1× at decision time:
  1. FORCE-GROUND (Smack Down/Ingrain vol, Iron Ball item) — GROUND vs FLYING + negate Levitate.
  2. RING TARGET (item) — negates ALL the holder's chart immunities.
  3. SCRAPPY / MIND'S EYE (attacker ability) — Normal/Fighting vs Ghost.
  4. FORESIGHT / ODOR SLEUTH (defender volatile 'foresight') — Normal/Fighting vs Ghost.
  5. MIRACLE EYE (defender volatile) — Psychic vs Dark.
  6. THOUSAND ARROWS (move intrinsic ignoreImmunity{Ground}) — GROUND vs FLYING + negate Levitate/Air Balloon.
Earth Eater (absorption) is NOT negated by force-ground; Mold Breaker is orthogonal (typing, not ability).
Value-only (no STATE_DIM/layout change). Mirrors pokemon-showdown sim/pokemon.ts runImmunity/isGrounded.
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    _type_mult, _type_eff_signed_immune, _immunity_negated, _move_ground_pierces,
    _immunity_neg_ctx, _move_immune, StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL,
    move_slots_for_mon, norm_species,
)
from v_dance.parser.vod_parser.battle_models import volatile_flags


# ════════════════════════════ unit: _immunity_negated / _type_mult ════════════════════════════
def test_immunity_negated_predicate():
    assert _immunity_negated("GROUND", "FLYING", frozenset({"force_ground"})) is True
    assert _immunity_negated("POISON", "STEEL", frozenset({"force_ground"})) is False   # force-ground is Flying-only
    assert _immunity_negated("POISON", "STEEL", frozenset({"ringtarget"})) is True       # Ring Target = blanket
    assert _immunity_negated("NORMAL", "GHOST", frozenset({"ghost"})) is True
    assert _immunity_negated("FIGHTING", "GHOST", frozenset({"ghost"})) is True
    assert _immunity_negated("PSYCHIC", "DARK", frozenset({"ghost"})) is False           # ghost token ≠ dark
    assert _immunity_negated("PSYCHIC", "DARK", frozenset({"dark"})) is True
    assert _immunity_negated("GROUND", "FLYING", None) is False                          # fast path


def test_type_mult_neg_keeps_other_type():
    # force-ground EQ vs Fire/Flying (Talonflame): Fire ×2 survives, Flying 0×→1× → 2.0
    assert _type_mult("GROUND", ["FIRE", "FLYING"], frozenset({"force_ground"})) == 2.0
    assert _type_mult("GROUND", ["FIRE", "FLYING"]) == 0.0                               # legacy = immune
    # Ring Target lets Poison hit Steel; force-ground does NOT
    assert _type_mult("POISON", ["STEEL"], frozenset({"ringtarget"})) == 1.0
    assert _type_mult("POISON", ["STEEL"], frozenset({"force_ground"})) == 0.0


def test_type_eff_signed_forwards_neg():
    # grounded EQ vs Fire/Flying → 2× → signed +0.5, NOT immune
    assert _type_eff_signed_immune("GROUND", ["FIRE", "FLYING"], frozenset({"force_ground"})) == (0.5, 0.0)
    assert _type_eff_signed_immune("GROUND", ["FIRE", "FLYING"]) == (-1.0, 1.0)          # legacy immune


def test_move_ground_pierces():
    assert _move_ground_pierces("thousandarrows") is True
    assert _move_ground_pierces("earthquake") is False
    assert _move_ground_pierces(None) is False


def test_future_sight_ignores_all_immunity():
    # Future Sight (standard 120-BP Psychic, ignoreImmunity:true) hits Dark — blanket chart negation.
    # (un-deferred 2026-06-21: was wrongly lumped with nonstandard Bide.)
    from v_dance.encoders.state_encoder import _move_ignores_all_immunity
    assert _move_ignores_all_immunity("futuresight") is True
    assert _move_ignores_all_immunity("psychic") is False
    ctx = _immunity_neg_ctx([{"force_grounded": False, "ring_target": False,
                              "foresight": False, "miracleeye": False}], "noability", "futuresight")
    assert "ringtarget" in ctx[0]                                  # blanket negation token applied


def test_volatile_flags_c2d_keys():
    assert volatile_flags({"foresight"})["foresight"] is True
    assert volatile_flags({"miracleeye"})["miracleeye"] is True
    assert volatile_flags(set())["foresight"] is False
    assert volatile_flags(set())["miracleeye"] is False


# ════════════════════════════ unit: _move_immune force-ground / pierce ════════════════════════════
def _D(**kw):
    base = {"types": ["STEEL", "PSYCHIC"], "ability": None, "ground_immune": False, "force_grounded": False}
    base.update(kw)
    return base


def test_move_immune_force_ground_negates_levitate():
    assert _move_immune("GROUND", _D(ability="levitate"), None, "earthquake") is True            # normal: immune
    assert _move_immune("GROUND", _D(ability="levitate", force_grounded=True), None, "earthquake") is False
    assert _move_immune("GROUND", _D(ability="levitate"), None, "thousandarrows") is False        # TA pierces


def test_move_immune_earth_eater_not_negated():
    # Earth Eater absorbs Ground regardless of grounding — must STAY immune.
    assert _move_immune("GROUND", _D(ability="eartheater"), None, "earthquake") is True
    assert _move_immune("GROUND", _D(ability="eartheater", force_grounded=True), None, "earthquake") is True


def test_move_immune_thousand_arrows_pierces_air_balloon():
    assert _move_immune("GROUND", _D(ability=None, ground_immune=True), None, "thousandarrows") is False
    assert _move_immune("GROUND", _D(ability=None, ground_immune=True), None, "earthquake") is True


def test_move_immune_moldbreaker_orthogonal():
    # Mold Breaker already bypasses Levitate (A.1); C.2d leaves that path unchanged.
    assert _move_immune("GROUND", _D(ability="levitate"), "moldbreaker", "earthquake") is False


# ════════════════════════════ unit: _immunity_neg_ctx token assembly ════════════════════════════
def test_immunity_neg_ctx_tokens():
    rt = {"ring_target": True, "force_grounded": False, "foresight": False, "miracleeye": False}
    fg = {"ring_target": False, "force_grounded": True, "foresight": False, "miracleeye": False}
    # attacker Scrappy → ghost token on every (non-None) slot
    ctx = _immunity_neg_ctx([rt, fg], "scrappy", "earthquake")
    assert ctx[0] == frozenset({"ringtarget", "ghost"})
    assert ctx[1] == frozenset({"force_ground", "ghost"})
    # Mind's Eye behaves identically to Scrappy
    assert _immunity_neg_ctx([rt], "mindseye", "tackle")[0] == frozenset({"ringtarget", "ghost"})
    # no negation + no Scrappy → None (the byte-identical fast path)
    plain = {"ring_target": False, "force_grounded": False, "foresight": False, "miracleeye": False}
    assert _immunity_neg_ctx([plain, None], "intimidate", "earthquake") == [None, None]
    # Thousand Arrows injects force_ground even with no defender volatile
    assert "force_ground" in _immunity_neg_ctx([plain], "intimidate", "thousandarrows")[0]


# ════════════════════════════ end-to-end through the encoder ════════════════════════════
def _mon(species, ability, *, moves=(), volatiles=None, item=None, stats=None):
    m = {
        "species": species, "base_species": species, "hp_pct": 100.0, "seen": True,
        "is_fainted": False, "known_moves": list(moves), "revealed_moves": [], "boosts": {},
        "status": None, "known_ability": ability, "volatiles": volatiles or {},
        "stats_estimate": {"mode": "exact",
                           "stats": stats or {"atk": 130, "spa": 130, "def": 100, "spd": 100,
                                              "hp": 260, "spe": 80}},
    }
    if item is not None:
        m["known_item"] = item
    return m


def _chan(att, dfn, move):
    """(signed, immune, dmax) for `move` of `att` vs `dfn` in slot a."""
    snap = {"our_active": {"our_a": att, "our_b": None}, "opp_active": {"opp_a": dfn, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return float(vec[b + 9]), float(vec[b + 10]), float(vec[b + 13 + 1])   # signed, immune, dmax
    raise AssertionError(move)


def test_e2e_force_ground_flying_keeps_other_type():
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    plain = _chan(chomp, _mon("Talonflame", "Flame Body"), "Earthquake")
    grounded = _chan(chomp, _mon("Talonflame", "Flame Body", volatiles={"force_grounded": True}), "Earthquake")
    assert plain[1] == 1.0 and plain[2] == 0.0                 # Flying immune to Ground
    assert grounded[1] == 0.0 and grounded[2] > 0.0            # grounded → hits
    assert grounded[0] == 0.5                                  # Fire ×2 preserved (not blanket-zeroed)


def test_e2e_thousand_arrows_vs_flying_no_volatile():
    chomp = _mon("Garchomp", "Rough Skin", moves=["Thousand Arrows"])
    res = _chan(chomp, _mon("Talonflame", "Flame Body"), "Thousand Arrows")
    assert res[1] == 0.0 and res[2] > 0.0                      # decision-turn pierce (before smackdown vol)


def test_e2e_thousand_arrows_pierces_levitate():
    chomp = _mon("Garchomp", "Rough Skin", moves=["Thousand Arrows"])
    res = _chan(chomp, _mon("Bronzong", "Levitate"), "Thousand Arrows")
    assert res[1] == 0.0 and res[2] > 0.0


def test_e2e_ring_target_chart():
    # Ring Target lets a Poison move hit a Steel mon for neutral.
    user = _mon("Sneasler", "Poison Touch", moves=["Sludge Bomb"])
    steel = _mon("Heatran", "Flash Fire", item="Ring Target")          # Fire/Steel; Poison vs both → 0×
    res = _chan(user, steel, "Sludge Bomb")
    assert res[1] == 0.0 and res[2] > 0.0


def test_e2e_ring_target_vs_flying():
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    res = _chan(chomp, _mon("Talonflame", "Flame Body", item="Ring Target"), "Earthquake")
    assert res[1] == 0.0 and res[2] > 0.0


def test_e2e_ring_target_does_not_touch_levitate():
    # Ring Target negates TYPE immunities only; Levitate (ability) stays.
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    res = _chan(chomp, _mon("Bronzong", "Levitate", item="Ring Target"), "Earthquake")
    assert res[1] == 1.0 and res[2] == 0.0


def test_e2e_scrappy_normal_vs_ghost():
    user = _mon("Komala", "Scrappy", moves=["Body Slam"])
    plain = _chan(_mon("Komala", "Comatose", moves=["Body Slam"]), _mon("Gengar", "Cursed Body"), "Body Slam")
    scrap = _chan(user, _mon("Gengar", "Cursed Body"), "Body Slam")
    assert plain[1] == 1.0 and plain[2] == 0.0                 # Normal immune to Ghost
    assert scrap[1] == 0.0 and scrap[2] > 0.0                  # Scrappy → hits


def test_e2e_minds_eye_fighting_vs_ghost():
    user = _mon("Basculegion", "Mind's Eye", moves=["Close Combat"])
    res = _chan(user, _mon("Gengar", "Cursed Body"), "Close Combat")
    assert res[1] == 0.0 and res[2] > 0.0                      # Mind's Eye = Scrappy for Normal/Fighting


def test_e2e_foresight_defender_volatile():
    user = _mon("Snorlax", "Thick Fat", moves=["Body Slam"])
    res = _chan(user, _mon("Gengar", "Cursed Body", volatiles={"foresight": True}), "Body Slam")
    assert res[1] == 0.0 and res[2] > 0.0


def test_e2e_future_sight_hits_dark():
    user = _mon("Hatterene", "Magic Bounce", moves=["Future Sight", "Psychic"])
    fs = _chan(user, _mon("Umbreon", "Synchronize"), "Future Sight")
    psy = _chan(user, _mon("Umbreon", "Synchronize"), "Psychic")
    assert psy[1] == 1.0 and psy[2] == 0.0                     # plain Psychic immune to Dark
    assert fs[1] == 0.0 and fs[2] > 0.0                        # Future Sight ignoreImmunity:true → hits


def test_e2e_miracle_eye_psychic_vs_dark():
    user = _mon("Hatterene", "Magic Bounce", moves=["Psychic"])
    plain = _chan(user, _mon("Umbreon", "Synchronize"), "Psychic")
    seen = _chan(user, _mon("Umbreon", "Synchronize", volatiles={"miracleeye": True}), "Psychic")
    assert plain[1] == 1.0 and plain[2] == 0.0                 # Psychic immune to Dark
    assert seen[1] == 0.0 and seen[2] > 0.0                    # Miracle Eye → hits


def test_e2e_miracle_eye_wrong_type_unchanged():
    # dark token must ONLY affect Psychic — a Ghost move vs Dark stays 0.5× (not immune anyway).
    user = _mon("Hatterene", "Magic Bounce", moves=["Shadow Ball"])
    plain = _chan(user, _mon("Umbreon", "Synchronize"), "Shadow Ball")
    seen = _chan(user, _mon("Umbreon", "Synchronize", volatiles={"miracleeye": True}), "Shadow Ball")
    assert seen == plain                                        # unchanged


def test_e2e_force_ground_levitate_channel_and_band():
    # Smack Down on a Levitate mon (Bronzong) → EQ hits in BOTH channel and band.
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    res = _chan(chomp, _mon("Bronzong", "Levitate", volatiles={"force_grounded": True}), "Earthquake")
    assert res[1] == 0.0 and res[2] > 0.0


def test_e2e_control_non_negated_immune():
    # No negation active → byte-identical to pre-C.2d (Flyer immune to Ground).
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    res = _chan(chomp, _mon("Talonflame", "Flame Body"), "Earthquake")
    assert res[1] == 1.0 and res[2] == 0.0
