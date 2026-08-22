"""v11 Phase C.2 — expose dropped per-mon volatiles as encoder channels (audit #13 + gap-hunt).

volatile_flags previously dropped confusion / Leech Seed / Perish-count (+ Salt Cure / Curse / Nightmare).
C.2 adds 3 channels, all PURE FUNCTIONS of the volatile-id set (so offline parser ids and live poke-env
effect ids give byte-identical output): residual_damage (grouped HP-clock flag), confused, perish_norm.
NO parser change, NO re-export (raw ids already captured); layout v13 / STATE_DIM 4553.
"""
from __future__ import annotations

import numpy as np

from v_dance.parser.vod_parser.battle_models import volatile_flags, _RESIDUAL_VOL
from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
from v_dance.encoders.state_encoder import (
    StateEncoder, POKEMON_FEATURES, VOLATILE_FEATURES, get_state_dim, get_state_layout_version,
)


# ── volatile_flags derivations ────────────────────────────────────────────────
def test_residual_damage_group():
    for vid in ("leechseed", "saltcure", "curse", "nightmare"):
        assert volatile_flags({vid})["residual_damage"] is True, vid
    assert volatile_flags({"aquaring"})["residual_damage"] is False   # heal, opposite sign — NOT in group
    assert volatile_flags(set())["residual_damage"] is False
    assert _RESIDUAL_VOL == {"leechseed", "saltcure", "curse", "nightmare"}


def test_confused_flag():
    assert volatile_flags({"confusion"})["confused"] is True
    assert volatile_flags(set())["confused"] is False


def test_perish_norm_lowest_present():
    assert volatile_flags(set())["perish_norm"] == 0.0
    assert volatile_flags({"perish3"})["perish_norm"] == 0.25
    assert volatile_flags({"perish3", "perish2"})["perish_norm"] == 0.5          # accumulated → lowest=2
    assert volatile_flags({"perish3", "perish2", "perish1"})["perish_norm"] == 0.75
    assert volatile_flags({"perish3", "perish2", "perish1", "perish0"})["perish_norm"] == 1.0


def test_no_double_count_with_existing_channels():
    # the new ids must not flip any of the existing volatile booleans
    for vid in ("confusion", "leechseed", "saltcure", "curse", "nightmare", "perish2"):
        vf = volatile_flags({vid})
        assert not vf["move_restricted"] and not vf["trapped"] and not vf["locked_action"]


def test_offline_live_idset_parity():
    # the parity contract: same id strings → identical volatile_flags output. Use poke-env Effect names
    # (lower+strip _) to prove the live derivation matches the parser ids by construction.
    import pytest
    effect = pytest.importorskip("poke_env.battle.effect")
    Effect = effect.Effect
    eff_ids = {e.name.lower().replace("_", "") for e in
               (Effect.CONFUSION, Effect.LEECH_SEED, Effect.SALT_CURE, Effect.PERISH3, Effect.PERISH2)}
    parser_ids = {"confusion", "leechseed", "saltcure", "perish3", "perish2"}
    assert eff_ids == parser_ids, eff_ids                  # ids match exactly
    assert volatile_flags(eff_ids) == volatile_flags(parser_ids)


# ── parser captures them (no parser change needed — generic _handle_volatile) ─
HEADER = """|player|p1|A|1|
|player|p2|B|1|
|poke|p1|Gengar, L50, M|
|poke|p2|Venusaur, L50, M|
|teamsize|p1|1
|teamsize|p2|1
|start
|switch|p1a: Gengar|Gengar, L50, M|100/100
|switch|p2a: Venusaur|Venusaur, L50, M|100/100
"""


def _vol(body, turn, rel="our_a"):
    r = ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()
    for t in r["turns"]:
        if t["turn"] == turn:
            return t["state_before_actions"]["p1"]["our_active"][rel]["volatiles"]
    raise AssertionError(turn)


def test_parser_to_dict_exposes_new_channels():
    v = _vol(
        "|turn|1\n|-start|p1a: Gengar|confusion\n|-start|p1a: Gengar|move: Leech Seed\n"
        "|-start|p1a: Gengar|perish3\n|turn|2\n",
        2)
    assert v["confused"] is True
    assert v["residual_damage"] is True
    assert v["perish_norm"] == 0.25


def test_confusion_cleared_on_end():
    v = _vol(
        "|turn|1\n|-start|p1a: Gengar|confusion\n"
        "|turn|2\n|-end|p1a: Gengar|confusion\n|turn|3\n",
        3)
    assert v["confused"] is False


# ── encoder channel placement ─────────────────────────────────────────────────
def _mon(volatiles=None):
    return {
        "species": "Gengar", "base_species": "Gengar", "hp_pct": 100.0, "seen": True,
        "is_fainted": False, "known_moves": [], "revealed_moves": [], "boosts": {},
        "status": None, "known_ability": "Cursed Body", "volatiles": volatiles or {},
        "stats_estimate": {"mode": "exact",
                           "stats": {"atk": 65, "spa": 130, "def": 60, "spd": 75, "hp": 160, "spe": 110}},
    }


def _encode(volatiles=None):
    snap = {"our_active": {"our_a": _mon(volatiles), "our_b": None},
            "opp_active": {"opp_a": None, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    return StateEncoder().encode_snapshot(snap, turn=1)


def test_encoder_channels_placement():
    base = _encode({})
    for vols, expected in [({"residual_damage": True}, 1.0),
                           ({"confused": True}, 1.0),
                           ({"perish_norm": 0.5}, 0.5),
                           ({"perish_norm": 1.0}, 1.0)]:
        v = _encode(vols)
        diff = np.flatnonzero(np.abs(v - base) > 1e-6)
        assert len(diff) == 1, f"exactly one channel should change for {vols}, got {len(diff)}"
        assert float(v[diff[0]]) == expected


def test_layout():
    # C.2 added 3 volatile channels; C.2b's gap-fix added a 4th (drowsy) → v14.
    assert VOLATILE_FEATURES == 12
    assert POKEMON_FEATURES == 413       # v16 (B2b: +2 per-move hit-chance channels)
    assert get_state_dim() == 5057
    assert get_state_layout_version() == 19
