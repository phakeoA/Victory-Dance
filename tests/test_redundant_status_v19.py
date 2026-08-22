"""Regression lock for Option 1c (layout v19) — the per-TARGET REDUNDANT-STATUS move bits.

A PURE status move (Will-O-Wisp / Thunder Wave / Toxic / Spore / Glare …) is WASTED on a target that
already carries a major status or is type/ability immune. DAMAGING moves with a status RIDER (Zap Cannon,
Scald, Nuzzle, Discharge) must NEVER flag. A pure feature — never masks an action (statusing a predicted
switch-in / setup stays learnable). Offline↔live parity is covered by tests/test_encoder_parity.py.
"""
from __future__ import annotations

from v_dance.encoders.battle_mechanics import move_redundant_status, _pure_status_move
from v_dance.encoders.encoder_layout import MOVE_FEATURES, POKEMON_FEATURES, get_state_dim, get_state_layout_version
from v_dance.encoders.action_codec import _MOVE_BLOCK_REL
from v_dance.encoders.state_encoder import StateEncoder


def test_layout_v19():
    assert MOVE_FEATURES == 59             # v18's 57 + 2 per-move redundant-status bits
    assert POKEMON_FEATURES == 413         # +2 × 4 moves
    assert get_state_dim() == 5057         # +2 × 4 moves × 12 mon slots = +96 over v18's 4961
    assert get_state_layout_version() == 19


def test_pure_status_move_detection():
    # category 'Status' + a top-level major-status field → detected
    assert _pure_status_move("willowisp")[0] == "brn"
    assert _pure_status_move("thunderwave")[0] == "par"
    assert _pure_status_move("toxic")[0] == "tox"
    assert _pure_status_move("spore") == ("slp", True)        # powder
    # damaging moves carry status only as a SECONDARY (category dmg) → NOT pure status moves
    for dmg in ("zapcannon", "scald", "nuzzle", "discharge", "sludgebomb"):
        assert _pure_status_move(dmg)[0] is None, dmg


def _D(types=(), status=None, ability=None):
    return {"types": list(types), "status": status, "ability": ability}


def test_redundant_status_logic():
    # already-statused (a mon holds only ONE major status) → any new pure status move is wasted
    assert move_redundant_status("willowisp", _D(["Water"], "brn")) == 1.0
    assert move_redundant_status("willowisp", _D(["Water"], "par")) == 1.0   # ANY status, not just burn
    assert move_redundant_status("willowisp", _D(["Water"])) == 0.0          # healthy → usable
    assert move_redundant_status("willowisp", None) == 0.0                   # empty slot
    # type immunity
    assert move_redundant_status("willowisp", _D(["Fire"])) == 1.0           # Fire can't burn
    assert move_redundant_status("toxic", _D(["Steel"])) == 1.0              # Steel can't be poisoned
    assert move_redundant_status("toxic", _D(["Poison"])) == 1.0
    assert move_redundant_status("spore", _D(["Grass"])) == 1.0              # powder vs Grass
    assert move_redundant_status("thunderwave", _D(["Ground"])) == 1.0       # Electric move vs Ground
    assert move_redundant_status("thunderwave", _D(["Electric"])) == 1.0     # Electric immune to para
    assert move_redundant_status("glare", _D(["Ground"])) == 0.0             # Glare (Normal) DOES para Ground
    # ability immunity
    assert move_redundant_status("willowisp", _D(["Water"], None, "waterveil")) == 1.0
    assert move_redundant_status("thunderwave", _D(["Water"], None, "limber")) == 1.0
    assert move_redundant_status("willowisp", _D(["Normal"], None, "comatose")) == 1.0   # blanket
    # ⚠ DAMAGING moves with a status rider must NEVER flag — the damage justifies them
    assert move_redundant_status("zapcannon", _D(["Water"], "par")) == 0.0
    assert move_redundant_status("scald", _D(["Fire"], "brn")) == 0.0
    assert move_redundant_status("nuzzle", _D(["Water"], "par")) == 0.0


def _mon(species, moves, status=None):
    return {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True,
            "is_fainted": False, "known_moves": list(moves), "revealed_moves": [], "boosts": {}, "status": status}


def test_encoding_per_target_bits():
    """The 2 per-move bits (move-0, vs opp_a / opp_b) flip against the SPECIFIC target."""
    sa, sb = _MOVE_BLOCK_REL + 26, _MOVE_BLOCK_REL + 27

    def enc(opp_a_status, opp_b_status):
        snap = {"our_active": {"our_a": _mon("Grimmsnarl", ("Will-O-Wisp", "Spirit Break")), "our_b": None},
                "opp_active": {"opp_a": _mon("Rotom-Wash", ("Hydro Pump",), opp_a_status),
                               "opp_b": _mon("Sylveon", ("Moonblast",), opp_b_status)},
                "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
        return StateEncoder().encode_snapshot(snap, turn=3)

    v = enc("brn", None)
    assert (v[sa], v[sb]) == (1.0, 0.0)        # redundant vs the BURNED opp_a only
    v = enc(None, "brn")
    assert (v[sa], v[sb]) == (0.0, 1.0)        # redundant vs the BURNED opp_b only
    v = enc(None, None)
    assert (v[sa], v[sb]) == (0.0, 0.0)        # both healthy → fully usable
