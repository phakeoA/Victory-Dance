"""15b-feat.1 + robust.2: the shared TP feature extractor (tp_features.py).

The pure mechanic-tag functions and the extractor are tested against SYNTHETIC stub beliefs
(reg-independent, no false-fail / false-confidence on a data swap), with ONE tolerant live
smoke against the pinned M-A data. Structural invariants (parity, no-leak, layout, overlay,
reserved gimmick, determinism, channel order) survive any data update by construction.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
from v_dance.training import tp_features as T  # noqa: E402
from v_dance.parser.belief_state import BeliefState  # noqa: E402

_PIKA = _REPO / "data" / "pikalytics_regma.json"


class _StubBelief:
    """A frozen, in-memory belief — tests the EXTRACTOR, not the live meta."""
    def __init__(self, ability=(), moves=(), usage=10.0, known=True):
        self._ab = [{"name": n, "p": p} for n, p in ability]
        self._mv = [{"name": n, "p": p} for n, p in moves]
        self._u, self._known = usage, known

    def known(self, s): return self._known
    def ability_distribution(self, s, top_k=4): return self._ab[:top_k]
    def move_distribution(self, s, top_k=12): return self._mv[:top_k]
    def usage(self, s): return self._u


# ── pure tag functions (synthetic) ────────────────────────────────────────────
def test_weather_tags_setter_abuser_move_clamp():
    sets, abuse = T.weather_tags([{"name": "Sand Stream", "p": 0.98}], [])
    assert sets[T.WEATHERS.index("sand")] == pytest.approx(0.98) and abuse.sum() == 0.0
    _, abuse2 = T.weather_tags([{"name": "Sand Force", "p": 1.0}], [])
    assert abuse2[T.WEATHERS.index("sand")] == 1.0
    sets3, _ = T.weather_tags([{"name": "Sand Stream", "p": 0.7}], [{"name": "Sandstorm", "p": 0.7}])
    assert sets3[T.WEATHERS.index("sand")] == 1.0                      # 1.4 clamped


def test_terrain_tags_move_setter_and_abuser():
    sets, abuse = T.terrain_tags([], [{"name": "Psychic Terrain", "p": 0.9}])
    assert sets[T.TERRAINS.index("psychic")] == pytest.approx(0.9) and abuse.sum() == 0.0
    _, abuse2 = T.terrain_tags([{"name": "Surge Surfer", "p": 1.0}],
                               [{"name": "Expanding Force", "p": 1.0}, {"name": "Grassy Glide", "p": 1.0}])
    assert abuse2[T.TERRAINS.index("electric")] == 1.0                 # Surge Surfer
    assert abuse2[T.TERRAINS.index("psychic")] == 1.0                  # Expanding Force
    assert abuse2[T.TERRAINS.index("grassy")] == 1.0                   # Grassy Glide


def test_role_tags_moves_ability_clamp():
    r = T.role_tags([{"name": "Intimidate", "p": 1.0}],
                    [{"name": "Follow Me", "p": 0.5}, {"name": "Rage Powder", "p": 0.5}])
    assert r[T.ROLE_TAGS.index("redirect")] == pytest.approx(1.0)
    assert r[T.ROLE_TAGS.index("intimidate")] == 1.0
    r2 = T.role_tags([], [{"name": m, "p": 1.0} for m in ("Icy Wind", "Electroweb", "Thunder Wave")])
    assert r2[T.ROLE_TAGS.index("speed_control")] == 1.0               # 3.0 clamped


# ── layout + channel order ────────────────────────────────────────────────────
def test_layout_consistent():
    assert T.FEAT_DIM == T.OFF_KTERA + T.NUM_TYPES
    assert T.BASE_DIM == T.OFF_USAGE + 1
    f = T.own_mon_features("X", _StubBelief())
    assert f.shape == (T.FEAT_DIM,) and f.dtype == np.float32


def test_channel_order_assert_guards_reorder(monkeypatch):
    T._assert_channel_order()                                          # current order is fine
    monkeypatch.setattr(T, "WEATHERS", ("rain", "sand", "sun", "snow"))
    with pytest.raises(AssertionError):
        T._assert_channel_order()                                     # a reorder must crash loudly


# ── symmetric base: own(known=None) == opp, byte-identical ─────────────────────
def test_own_none_equals_opp():
    b = _StubBelief(ability=[("Sand Stream", 0.98)], moves=[("Protect", 0.9)], usage=5.0)
    assert np.array_equal(T.own_mon_features("X", b, known=None), T.opp_mon_features("X", b))


def test_own_none_equals_opp_across_all_families():
    # non-vacuous parity: light up EVERY family (weather/terrain/spread/immune/reverser/debuff/order)
    # so a future asymmetric edit to _fill_base is caught, not just the Sand Stream path.
    b = _StubBelief(ability=[("Contrary", 0.4), ("Sand Stream", 0.4), ("Illusion", 0.2)],
                    moves=[("Earthquake", 0.9), ("Charm", 0.3), ("Psychic Terrain", 0.4)], usage=7.0)
    for sp in ("Charizard", "Garchomp", "Zoroark-Hisui", "Zzz"):    # typing-immune / ground / order / OOV
        assert np.array_equal(T.own_mon_features(sp, b, known=None), T.opp_mon_features(sp, b)), sp


def test_opp_no_leak_default_and_zero_overlay():
    # tpfeat-v7: ``revealed`` exists for the OTS regime (open sheets ARE
    # preview-visible), but it must DEFAULT to None so every closed-sheet call
    # site can never leak a hidden build into the overlay.
    params = inspect.signature(T.opp_mon_features).parameters
    assert list(params) == ["species", "belief", "revealed"]
    assert params["revealed"].default is None
    assert T.opp_mon_features("X", _StubBelief(ability=[("Sand Stream", 1.0)]))[T.OFF_OWNBIT:].sum() == 0.0
    # and an EXPLICIT reveal rides the overlay (the OTS training/serve pathway)
    f = T.opp_mon_features("X", _StubBelief(),
                           revealed=T.OwnKnown(ability="Sand Stream", moves=["Trick Room"]))
    assert f[T.OFF_OWNBIT] == 1.0
    assert f[T.OFF_KROLES + T.ROLE_TAGS.index("trick_room")] == 1.0


# ── overlay: bit + sharp known build across weather, terrain, gimmick ──────────
def test_overlay_known_build_weather_terrain_mega():
    b = _StubBelief()
    none = T.own_mon_features("Steelix", b, known=None)
    assert none[T.OFF_OWNBIT] == 0.0 and none[T.OFF_OWNBIT:].sum() == 0.0

    k = T.OwnKnown(ability="Sand Force", moves=["Trick Room", "Psychic Terrain"], will_mega=True)
    sharp = T.own_mon_features("Steelix", b, known=k)
    assert sharp[T.OFF_OWNBIT] == 1.0
    assert sharp[T.OFF_KWABUSE + T.WEATHERS.index("sand")] == 1.0        # Sand Force
    assert sharp[T.OFF_KTSETS + T.TERRAINS.index("psychic")] == 1.0      # Psychic Terrain (sets)
    assert sharp[T.OFF_KROLES + T.ROLE_TAGS.index("trick_room")] == 1.0
    assert sharp[T.OFF_KGK + T.GIMMICK_KINDS.index("mega")] == 1.0
    assert np.array_equal(sharp[:T.OFF_OWNBIT], none[:T.OFF_OWNBIT])     # base untouched by overlay


def test_overlay_terrain_abuser_and_speed_clamp():
    b = _StubBelief()
    s1 = T.own_mon_features("X", b, known=T.OwnKnown(moves=["Expanding Force"]))
    assert s1[T.OFF_KTABUSE + T.TERRAINS.index("psychic")] == 1.0
    s2 = T.own_mon_features("X", b, known=T.OwnKnown(moves=["Icy Wind", "Electroweb", "Thunder Wave"]))
    assert s2[T.OFF_KROLES + T.ROLE_TAGS.index("speed_control")] == 1.0


# ── reserved gimmick + determinism + soft tags from stub ──────────────────────
def test_reserved_gimmick_block_zeroed_in_MA():
    f = T.own_mon_features("X", _StubBelief())
    assert f[T.OFF_GK + T.GIMMICK_KINDS.index("none")] == 1.0
    assert f[T.OFF_GK:T.OFF_GK + len(T.GIMMICK_KINDS)].sum() == 1.0
    assert f[T.OFF_TERA:T.OFF_TERA + T.NUM_TYPES].sum() == 0.0           # tera reserved, zero in M-A


# ── canonical name matching (2026-07-23 casing fix — 9 ability + 28 move tags were DEAD) ──
def test_lowercase_id_ability_names_fire_tags():
    sets, _ = T.weather_tags([{"name": "drought", "p": 0.68}], [])       # the Charizard bug
    assert sets[T.WEATHERS.index("sun")] == pytest.approx(0.68)
    _, abuse = T.weather_tags([{"name": "swiftswim", "p": 0.55}], [])
    assert abuse[T.WEATHERS.index("rain")] == pytest.approx(0.55)
    r = T.role_tags([{"name": "intimidate", "p": 1.0}], [])
    assert r[T.ROLE_TAGS.index("intimidate")] == 1.0
    v = T._ability_immune([{"name": "levitate", "p": 0.4}])
    assert v[T._TYPE_IDX["GROUND"]] == pytest.approx(0.4)


def test_lowercase_id_move_names_fire_tags():
    r = T.role_tags([], [{"name": "ragepowder", "p": 0.5}, {"name": "trickroom", "p": 0.25}])
    assert r[T.ROLE_TAGS.index("redirect")] == pytest.approx(0.5)
    assert r[T.ROLE_TAGS.index("trick_room")] == pytest.approx(0.25)
    sets, _ = T.weather_tags([], [{"name": "raindance", "p": 0.3}])
    assert sets[T.WEATHERS.index("rain")] == pytest.approx(0.3)
    sp = T.spread_tags([{"name": "earthquake", "p": 0.7}])
    assert sp[T._TYPE_IDX["GROUND"]] == pytest.approx(0.7)


def test_dup_id_and_display_name_takes_max_not_sum():
    r = T.role_tags([], [{"name": "Trick Room", "p": 0.3}, {"name": "trickroom", "p": 0.2}])
    assert r[T.ROLE_TAGS.index("trick_room")] == pytest.approx(0.3)


# ── mega-stone -> ability augmentation (2026-07-23 fix) ───────────────────────
def _dex_or_skip():
    from v_dance.parser.vod_parser.pokedex import get_pokedex
    dex = get_pokedex()
    if dex is None or not dex.mega_formes_for("Charizard"):
        pytest.skip("pokedex with mega formes unavailable")
    return dex


def test_stone_augmented_abilities_charizard_twin_megas():
    _dex_or_skip()
    aug, p_mega = T._stone_augmented_abilities(
        "Charizard", [{"name": "Blaze", "p": 0.8}],
        [{"name": "Charizardite Y", "p": 0.24}, {"name": "Charizardite X", "p": 0.01}])
    d = {T._canon_name(e["name"]): e["p"] for e in aug}
    assert d["drought"] == pytest.approx(0.24)                           # Y stone => Drought
    assert d["toughclaws"] == pytest.approx(0.01)                        # X stone => Tough Claws
    assert p_mega == pytest.approx(0.25)


def test_stone_augment_max_merges_with_stronger_marginal():
    _dex_or_skip()
    aug, p_mega = T._stone_augmented_abilities(
        "Charizard", [{"name": "drought", "p": 0.68}],
        [{"name": "Charizardite Y", "p": 0.24}])
    d = {T._canon_name(e["name"]): e["p"] for e in aug}
    assert d["drought"] == pytest.approx(0.68)                           # max, not sum/replace
    assert p_mega == pytest.approx(0.24)


class _StubBeliefItems(_StubBelief):
    def __init__(self, ability=(), moves=(), items=(), usage=10.0, known=True):
        super().__init__(ability, moves, usage, known)
        self._it = [{"name": n, "p": p} for n, p in items]

    def item_distribution(self, s, top_k=6):
        return self._it[:top_k]


def test_fill_base_sun_tag_and_mega_prior_from_stub():
    _dex_or_skip()
    b = _StubBeliefItems(ability=(("drought", 0.68), ("Blaze", 0.20)),
                         items=(("Charizardite Y", 0.24),))
    f = T.own_mon_features("Charizard", b)
    assert f[T.OFF_WSETS + T.WEATHERS.index("sun")] == pytest.approx(0.68, abs=1e-6)
    assert f[T.OFF_GK + T.GIMMICK_KINDS.index("mega")] == pytest.approx(0.24, abs=1e-6)
    assert f[T.OFF_GK + T.GIMMICK_KINDS.index("none")] == pytest.approx(0.76, abs=1e-6)


# ── v8 channels ───────────────────────────────────────────────────────────────
def test_v8_intimidate_interaction_tags():
    # punish: the Kingambit case; immune: the Mawile Hyper Cutter case (lowercase = casing-proof)
    assert T.ability_scalar_tag([{"name": "Defiant", "p": 0.99}], T._INTIM_PUNISH_C) == pytest.approx(0.99)
    assert T.ability_scalar_tag([{"name": "hypercutter", "p": 0.22}], T._INTIM_IMMUNE_C) == pytest.approx(0.22)
    assert T.ability_scalar_tag([{"name": "Mirror Armor", "p": 1.0}], T._INTIM_PUNISH_C) == 1.0
    b = _StubBelief(ability=(("Defiant", 0.99),))
    f = T.own_mon_features("X", b)
    assert f[T.OFF_INTIMP] == pytest.approx(0.99)
    assert f[T.OFF_INTIMI] == 0.0


def test_v8_guard_and_trapping_role_tags():
    r = T.role_tags([], [{"name": "Wide Guard", "p": 0.6}, {"name": "quickguard", "p": 0.3},
                         {"name": "infestation", "p": 0.4}])
    assert r[T.ROLE_TAGS.index("wide_guard")] == pytest.approx(0.6)
    assert r[T.ROLE_TAGS.index("quick_guard")] == pytest.approx(0.3)
    assert r[T.ROLE_TAGS.index("trapping")] == pytest.approx(0.4)
    r2 = T.role_tags([{"name": "Shadow Tag", "p": 0.8}], [])            # trapping ability half
    assert r2[T.ROLE_TAGS.index("trapping")] == pytest.approx(0.8)


def test_v8_prio_block_weather_negate_sleep():
    assert T.ability_scalar_tag([{"name": "armortail", "p": 1.0}], T._PRIO_BLOCK_C) == 1.0
    assert T.ability_scalar_tag([{"name": "Cloud Nine", "p": 0.5}], T._WNEG_C) == pytest.approx(0.5)
    assert T.sleep_tag([{"name": "Spore", "p": 0.9}, {"name": "sleeppowder", "p": 0.2}]) == 1.0


def test_v8_item_tags_and_choice_prefix():
    v = T.item_tags([{"name": "Focus Sash", "p": 0.4}, {"name": "Choice Specs", "p": 0.3},
                     {"name": "choicescarf", "p": 0.5}, {"name": "clearamulet", "p": 0.1}])
    assert v[T.ITEM_TAGS.index("focus_sash")] == pytest.approx(0.4)
    assert v[T.ITEM_TAGS.index("choice_lock")] == pytest.approx(0.5)     # max over Choice items
    assert v[T.ITEM_TAGS.index("clear_amulet")] == pytest.approx(0.1)
    assert v[T.ITEM_TAGS.index("safety_goggles")] == 0.0


def test_v8_phys_share_neutral_and_ratio():
    assert T.phys_share([]) == pytest.approx(0.5)                        # no damaging info
    assert T.phys_share([{"name": "Protect", "p": 1.0}]) == pytest.approx(0.5)
    assert T.phys_share([{"name": "Earthquake", "p": 1.0}]) == pytest.approx(1.0)
    assert T.phys_share([{"name": "earthquake", "p": 1.0},
                         {"name": "Shadow Ball", "p": 1.0}]) == pytest.approx(0.5)


def test_v8_expected_speed_base_fallback_and_live():
    _dex_or_skip()
    b = _StubBelief()                                                    # no expected_stats_weighted
    f = T.own_mon_features("Charizard", b)
    assert f[T.OFF_EXPSPE] == pytest.approx(100 / 255.0, abs=1e-6)       # base 100 Spe fallback


def test_v8_overlay_stone_locks_mega_ability_and_items():
    _dex_or_skip()
    k = T.OwnKnown(ability="Blaze", moves=("Protect", "Heat Wave"),
                   item="Charizardite Y", spe=167.0)
    f = T.own_mon_features("Charizard", _StubBelief(known=False), known=k)
    assert f[T.OFF_KWSETS + T.WEATHERS.index("sun")] == 1.0              # stone => Drought, hard
    assert f[T.OFF_KGK + T.GIMMICK_KINDS.index("mega")] == 1.0           # stone implies the mega
    assert f[T.OFF_KEXPSPE] == pytest.approx(167 / 255.0, abs=1e-6)
    assert f[T.OFF_KPHYSSH] == pytest.approx(0.0)                        # Heat Wave = special
    # base ability still rides the union (entry abilities fire pre-mega)
    k2 = T.OwnKnown(ability="Intimidate", item="Mawilite")
    f2 = T.own_mon_features("Mawile", _StubBelief(known=False), known=k2)
    assert f2[T.OFF_KROLES + T.ROLE_TAGS.index("intimidate")] == 1.0
    assert f2[T.OFF_KGK + T.GIMMICK_KINDS.index("mega")] == 1.0


def test_soft_weather_and_terrain_tags_from_stub():
    b = _StubBelief(ability=[("Sand Stream", 0.98)], moves=[("Psychic Terrain", 0.6)])
    f = T.opp_mon_features("X", b)
    assert f[T.OFF_WSETS + T.WEATHERS.index("sand")] == pytest.approx(0.98)
    assert f[T.OFF_TSETS + T.TERRAINS.index("psychic")] == pytest.approx(0.6)
    assert f[T.OFF_HASDATA] == 1.0


def test_no_data_zeros_tags_and_bit():
    f = T.opp_mon_features("Zzz", _StubBelief(known=False))
    assert f[T.OFF_HASDATA] == 0.0 and f[T.OFF_USAGE] == 0.0
    assert f[T.OFF_WSETS:T.OFF_ROLES + 8].sum() == 0.0                   # all soft tags zero


def test_determinism():
    b = _StubBelief(ability=[("Drizzle", 1.0)])
    assert np.array_equal(T.own_mon_features("X", b), T.own_mon_features("X", b))


# ── 15b-feat.spread: spread-move <-> ally-immunity ────────────────────────────
def test_spread_tags_pure():
    g = T._TYPE_IDX[T._canon_type("Ground")]
    e = T._TYPE_IDX[T._canon_type("Electric")]
    assert T.spread_tags([{"name": "Earthquake", "p": 1.0}, {"name": "Protect", "p": 1.0}])[g] == 1.0
    assert T.spread_tags([{"name": "Discharge", "p": 0.5}])[e] == pytest.approx(0.5)
    assert T.spread_tags([{"name": "Protect", "p": 1.0}]).sum() == 0.0     # not a spread move


def test_immune_from_typing_and_ability():
    g = T._TYPE_IDX[T._canon_type("Ground")]
    e = T._TYPE_IDX[T._canon_type("Electric")]
    assert T.immune_tags("Charizard", [])[g] == 1.0                        # Fire/Flying -> immune Ground
    assert T.immune_tags("Garchomp", [])[e] == 1.0                         # Ground -> immune Electric
    assert T.immune_tags("Pikachu", [{"name": "Levitate", "p": 1.0}])[g] == 1.0  # ability immunity


def test_overlay_spread_and_immune():
    b = _StubBelief()
    g = T._TYPE_IDX[T._canon_type("Ground")]
    eq = T.own_mon_features("Garchomp", b, known=T.OwnKnown(moves=["Earthquake"]))
    assert eq[T.OFF_KSPREAD + g] == 1.0
    lv = T.own_mon_features("Pikachu", b, known=T.OwnKnown(ability="Levitate"))
    assert lv[T.OFF_KIMMUNE + g] == 1.0


def test_spread_immunity_synergy_pair_representable():
    # the user's example: EQ Garchomp + Ground-immune Charizard -> both halves light up,
    # so the attention layer (15b-arch.1) can pair them.
    b = _StubBelief()
    g = T._TYPE_IDX[T._canon_type("Ground")]
    eq = T.own_mon_features("Garchomp", b, known=T.OwnKnown(moves=["Earthquake"]))
    charizard = T.opp_mon_features("Charizard", b)                          # public typing immunity
    assert eq[T.OFF_KSPREAD + g] == 1.0 and charizard[T.OFF_IMMUNE + g] == 1.0


# ── 15b-feat.stat: stat-reverser <-> ally-debuff (Contrary + Prankster Charm) ──
def test_stat_reverser_and_ally_debuff_pure():
    assert T.reverser_tag([{"name": "Contrary", "p": 1.0}]) == 1.0
    assert T.reverser_tag([{"name": "Intimidate", "p": 1.0}]) == 0.0
    assert T.ally_debuff_tag([{"name": "Charm", "p": 0.8}]) == pytest.approx(0.8)
    assert T.ally_debuff_tag([{"name": "Charm", "p": 0.7}, {"name": "Fake Tears", "p": 0.7}]) == 1.0  # clamp
    assert T.ally_debuff_tag([{"name": "Protect", "p": 1.0}]) == 0.0


def test_contrary_charm_synergy_pair_representable():
    b = _StubBelief()
    contrary = T.own_mon_features("X", b, known=T.OwnKnown(ability="Contrary"))
    charmer = T.own_mon_features("Whimsicott", b, known=T.OwnKnown(moves=["Charm"]))
    assert contrary[T.OFF_KREVERSER] == 1.0 and contrary[T.OFF_KDEBUFF] == 0.0   # beneficiary half
    assert charmer[T.OFF_KDEBUFF] == 1.0 and charmer[T.OFF_KREVERSER] == 0.0      # enabler half


# ── 15b-feat.order: illusion / imposter order-sensitivity flags ───────────────
def test_order_sensitivity_flags():
    assert T.order_tags([{"name": "Illusion", "p": 1.0}]).tolist() == [1.0, 0.0]
    assert T.order_tags([{"name": "Imposter", "p": 0.5}]).tolist() == [0.0, 0.5]
    assert T.order_tags([{"name": "Levitate", "p": 1.0}]).sum() == 0.0
    b = _StubBelief()
    zoro = T.own_mon_features("Zoroark-Hisui", b, known=T.OwnKnown(ability="Illusion"))
    ditto = T.own_mon_features("Ditto", b, known=T.OwnKnown(ability="Imposter"))
    assert zoro[T.OFF_KORDER + T.ORDER_FLAGS.index("illusion")] == 1.0
    assert ditto[T.OFF_KORDER + T.ORDER_FLAGS.index("imposter")] == 1.0


# ── 15b-feat.defense: signed defensive type-effectiveness profile ─────────────
def _ti(t):
    return T._TYPE_IDX[T._canon_type(t)]


def test_def_eff_profile_signs():
    cz = T.def_eff_profile("Charizard")                 # Fire/Flying
    assert cz[_ti("Rock")] == pytest.approx(1.0)         # 4x weak (2x Fire * 2x Flying)
    assert cz[_ti("Ground")] == pytest.approx(-1.0)      # immune via Flying
    assert cz[_ti("Grass")] == pytest.approx(-1.0)       # 0.25x
    assert cz[_ti("Electric")] == pytest.approx(0.5)     # 2x via Flying
    gc = T.def_eff_profile("Garchomp")                   # Dragon/Ground
    assert gc[_ti("Ice")] == pytest.approx(1.0) and gc[_ti("Electric")] == pytest.approx(-1.0)
    assert T.def_eff_profile("Zzzfakemon").sum() == 0.0  # unknown species -> all zero, no crash


def test_type_complementarity_representable():
    # the user's example: Charizard weak to Rock, Garchomp resists it -> opposite signs at Rock, so
    # the attention layer can learn "A's weakness is B's resistance" (type complementarity).
    cz = T.def_eff_profile("Charizard")
    gc = T.def_eff_profile("Garchomp")
    assert cz[_ti("Rock")] > 0 and gc[_ti("Rock")] < 0


# ── 15b-feat.1b: Pikalytics teammates -> pairwise affinity prior ──────────────
def test_teammate_affinity_matrix():
    class _B:
        _tm = {"A": [{"name": "B", "p": 0.6}], "B": [{"name": "A", "p": 0.4}], "C": []}

        def teammates(self, s, top_k=16):
            return self._tm.get(s, [])

    A = T.teammate_affinity_matrix(["A", "B", "C"], _B(), n=3)
    assert A[0, 1] == pytest.approx(0.5)          # mean of the two directional pcts (0.6, 0.4)
    assert A[1, 0] == pytest.approx(0.5)          # symmetric
    assert A[0, 2] == 0.0 and A[2, 0] == 0.0      # C lists no teammates
    assert A[0, 0] == 0.0 and A.shape == (3, 3)   # zero diagonal


def test_teammate_affinity_live_positive_for_known_pair():
    b = BeliefState(_PIKA)
    A = T.teammate_affinity_matrix(["Whimsicott", "Garchomp", "Incineroar", "Amoonguss", "X", "Y"], b)
    assert A[0, 1] > 0.0                           # Whimsicott + Garchomp co-occur in the M-A meta
    assert np.allclose(A, A.T)


# ── ONE tolerant live smoke (data-dependent, intentionally loose) ─────────────
def test_live_smoke_some_species_sets_sand():
    b = BeliefState(_PIKA)
    assert any(T.opp_mon_features(sp, b)[T.OFF_WSETS + T.WEATHERS.index("sand")] > 0.5
               for sp in b.all_pokemon())                               # the live path produces a sand setter
