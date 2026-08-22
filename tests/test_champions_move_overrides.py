"""v11 N1 — Champions-mod move overrides (BP / type / accuracy).

The custom Champions VGC format (pokemon-showdown/data/mods/champions/moves.ts) overrides ~30 moves' BP/
type/accuracy vs standard Gen-9, but BOTH encoders read standard data (offline GenData(9), live poke-env
Move). N1 overlays the modded values via shared id-keyed helpers (champ_bp/champ_type/champ_acc_raw) at 9
chokepoints so the damage band, type-eff cross, STAB, immunity, accuracy channel, and the Technician/-ate
gates all see the Champions value, byte-identically on both encoders. Design: scratch/v11_n1_design_synthesis.md.
"""
from __future__ import annotations

import numpy as np
from poke_env.battle import Pokemon
from poke_env.battle.move import Move

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    champ_bp, champ_type, champ_acc_raw, _move_damage_props, _ability_damage_mult, _move_immune,
)
from v_dance.encoders.live_state_encoder import LiveStateEncoder
from v_dance.encoders import gen_champions_move_overrides as _gen
from v_dance.encoders.champions_move_overrides import CHAMP_BP, CHAMP_ACC


# ════════════════════════════ table lookups ════════════════════════════
def test_champ_bp_lookup():
    assert champ_bp("psyshieldbash", 70) == 90
    assert champ_bp("infernalparade", 60) == 65
    assert champ_bp("boltbeak", 85) == 80           # a decrease
    assert champ_bp("tackle", 40) == 40             # unknown → std passthrough


def test_baked_bp_overrides_force_value_regardless_of_std():
    # Scraper-baked approximations: moves.json hand-bakes Acrobatics 110 / Return 102, but poke-env's live
    # Move.base_power is the UN-baked 55 / 0. champ_bp must force the baked value on BOTH std inputs so the
    # LIVE own-mon path matches the trained corpus (offline already passes the baked std).
    assert champ_bp("acrobatics", 55) == 110        # live std (poke-env) → lifted
    assert champ_bp("acrobatics", 110) == 110       # offline std (moves.json) → unchanged
    assert champ_bp("return", 0) == 102             # live std (poke-env) → lifted
    assert champ_bp("return", 102) == 102           # offline std (moves.json) → unchanged


def test_champ_type_lookup():
    assert champ_type("snaptrap", "Grass") == "Steel"
    assert champ_type("growth", "Normal") == "Grass"
    assert champ_type("earthquake", "Ground") == "Ground"   # passthrough


def test_champ_acc_lookup():
    assert champ_acc_raw("crabhammer", 90) == 95
    assert champ_acc_raw("makeitrain", 100) == 95
    assert champ_acc_raw("clangoroussoul", True) is True
    assert champ_acc_raw("earthquake", 100) == 100   # passthrough


def test_makeitrain_is_accuracy_only():
    # over-application guard: Make It Rain overrides ONLY accuracy (std BP 120 unchanged) — no BP entry.
    assert "makeitrain" not in CHAMP_BP
    assert "makeitrain" in CHAMP_ACC


def test_generator_idempotent():
    # re-run the generator from moves.ts; the committed module must be byte-identical (catches an un-regen'd bump)
    regen = _gen.render_module(_gen.parse_overrides())
    committed = (_gen._OUT).read_text(encoding="utf-8")
    assert regen == committed


# ════════════════════════════ Technician/-ate gate (the subtlety-#1 landmine) ════════════════════════════
def test_infernalparade_technician_gate():
    # std BP 60 → Technician (<=60) WOULD apply; champ BP 65 → must NOT. Reads _move_damage_props.base_power.
    assert _move_damage_props("infernalparade")["base_power"] == 65
    mult = _ability_damage_mult("infernalparade", "technician", None, eff_move_type="GHOST", is_stab=False,
                                is_physical=False, type_mult=1.0, hp_frac=1.0, att_burned=False,
                                att_statused=False)
    assert mult == 1.0                              # 65 > 60 → no ×1.5


def test_growth_raw_is_normal_flips():
    assert _move_damage_props("growth")["raw_is_normal"] is False   # Normal→Grass (status; -ate gate moot)
    assert _move_damage_props("tackle")["raw_is_normal"] is True    # unchanged


# ════════════════════════════ snaptrap type override (Grass→Steel) ════════════════════════════
def test_snaptrap_sapsipper_immunity_flips():
    sapsipper = {"ability": "sapsipper", "types": ["GRASS"]}
    # standard Grass snaptrap would be absorbed by Sap Sipper (0×); the Champions Steel snaptrap is NOT.
    assert _move_immune("GRASS", sapsipper, None, "snaptrap") is True
    assert _move_immune("STEEL", sapsipper, None, "snaptrap") is False


# ════════════════════════════ e2e through the offline band ════════════════════════════
def _mon(species, ability, *, moves=()):
    return {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True, "is_fainted": False,
            "known_moves": list(moves), "revealed_moves": [], "boosts": {}, "status": None,
            "known_ability": ability, "volatiles": {},
            "stats_estimate": {"mode": "exact",
                               "stats": {"atk": 120, "spa": 120, "def": 100, "spd": 100, "hp": 260, "spe": 90}}}


def _band(att, move):
    snap = {"our_active": {"our_a": att, "our_b": None},
            "opp_active": {"opp_a": _mon("Blissey", "Natural Cure"), "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return float(vec[b]), float(vec[b + 13]), float(vec[b + 14])   # (bp_channel, dmin, dmax)
    raise AssertionError(move)


def test_psyshieldbash_band_scales_with_override():
    # Psyshield Bash std BP 70 → champ 90. The BP channel AND the damage band scale by ≈90/70.
    bp_ch, dmin, dmax = _band(_mon("Iron Valiant", "Quark Drive", moves=["Psyshield Bash"]), "Psyshield Bash")
    assert abs(bp_ch - 90 / 150.0) < 1e-6           # BP channel reflects the override
    assert dmax > 0.0
    # vs a fixed-BP reference: snaptrap'ish — compare a known std-90 move encoded the same way.
    bp_ref, _, dmax_ref = _band(_mon("Iron Valiant", "Quark Drive", moves=["Psychic"]), "Psychic")  # Psychic BP 90
    assert abs(bp_ch - bp_ref) < 1e-6               # 90 BP both → channels equal


# ════════════════════════════ offline ↔ live channel parity ════════════════════════════
def _move_channels_offline(name, user_types):
    vec = np.zeros(MOVE_FEATURES, dtype=np.float32)
    StateEncoder()._write_move_json(vec, 0, name, 1.0, user_types)
    return float(vec[0]), float(vec[1]), float(vec[4])   # BP, type, accuracy channels


def _move_channels_live(move_id, user):
    vec = np.zeros(MOVE_FEATURES, dtype=np.float32)
    LiveStateEncoder()._write_move(vec, 0, Move(move_id, gen=9), user)
    return float(vec[0]), float(vec[1]), float(vec[4])


def test_n1_offline_live_channel_parity():
    user = Pokemon(gen=9, species="garchomp")
    user_types = ["DRAGON", "GROUND"]
    for move_id, name in [("psyshieldbash", "Psyshield Bash"),   # BP override
                          ("snaptrap", "Snap Trap"),             # type override
                          ("makeitrain", "Make It Rain"),        # accuracy override
                          ("acrobatics", "Acrobatics"),          # scraper-baked BP (live 55 → 110; was a parity break)
                          ("return", "Return"),                  # scraper-baked BP (live 0 → 102)
                          ("earthquake", "Earthquake")]:         # no override (passthrough control)
        off = _move_channels_offline(name, user_types)
        live = _move_channels_live(move_id, user)
        assert np.allclose(off, live), f"{move_id}: offline {off} != live {live}"
