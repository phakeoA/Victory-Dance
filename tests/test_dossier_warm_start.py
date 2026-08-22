"""S1 L2b (2026-07-12): dossier → opp-snapshot warm-start (playbook spec S1).

Merge contract: STRICT fill-only-unknowns (in-battle evidence always wins);
moves union capped at 4 with reveals keeping priority; species mismatch skip;
id→display conversion (dossier stores poke-env ids, snapshots display names);
corrupted/missing dossier → no-op; never raises. Flag default OFF everywhere
(prod byte-identical) — wiring mirrors adapt_rules.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from v_dance.play import opponent_dossier
from v_dance.play.opponent_dossier import apply_dossier


def _battle(opp="Rival"):
    return SimpleNamespace(opponent_username=opp, battle_tag="battle-gen9x-1")


def _snapshot(**mon_over):
    mon = {"species": "Mawile", "known_ability": None, "known_item": None,
           "known_moves": [], **mon_over}
    return {"opp_active": {"opp_a": mon}, "opp_bench": []}


def _write_dossier(tmp_path, monkeypatch, mons):
    monkeypatch.setattr(opponent_dossier, "DOSSIER_DIR", tmp_path)
    (tmp_path / "rival.json").write_text(
        json.dumps({"opponent": "Rival", "games": [], "mons": mons}), encoding="utf-8")


_REC = {"mawile": {"species": "mawile", "ability": "intimidate", "item": "mawilite",
                   "moves": ["suckerpunch", "playrough", "ironhead", "protect", "fakeout"],
                   "times_seen": 3}}


def test_no_dossier_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(opponent_dossier, "DOSSIER_DIR", tmp_path)
    snap = _snapshot()
    before = copy.deepcopy(snap)
    assert apply_dossier(snap, _battle()) == before


def test_fills_only_unknowns_with_display_names(tmp_path, monkeypatch):
    _write_dossier(tmp_path, monkeypatch, _REC)
    snap = _snapshot()
    out = apply_dossier(snap, _battle())
    mon = out["opp_active"]["opp_a"]
    assert mon["known_ability"] == "Intimidate"            # id → display
    assert mon["known_item"] is not None
    assert len(mon["known_moves"]) == 4                    # capped at 4 of the 5 dossier moves
    assert "Sucker Punch" in mon["known_moves"]


def test_in_battle_evidence_always_wins(tmp_path, monkeypatch):
    _write_dossier(tmp_path, monkeypatch, _REC)
    snap = _snapshot(known_ability="Hyper Cutter", known_item="Focus Sash",
                     known_moves=["Play Rough", "Swords Dance", "Substitute"])
    out = apply_dossier(snap, _battle())
    mon = out["opp_active"]["opp_a"]
    assert mon["known_ability"] == "Hyper Cutter"          # never overwritten
    assert mon["known_item"] == "Focus Sash"
    assert mon["known_moves"][:3] == ["Play Rough", "Swords Dance", "Substitute"]
    assert len(mon["known_moves"]) == 4                    # ONE dossier move appended
    assert "Play Rough" not in mon["known_moves"][3:]      # deduped vs the reveal


def test_item_consumed_blocks_item_fill(tmp_path, monkeypatch):
    _write_dossier(tmp_path, monkeypatch, _REC)
    out = apply_dossier(_snapshot(item_consumed=True), _battle())
    assert out["opp_active"]["opp_a"]["known_item"] is None


def test_species_mismatch_skips(tmp_path, monkeypatch):
    _write_dossier(tmp_path, monkeypatch, _REC)
    snap = _snapshot(species="Torkoal")
    before = copy.deepcopy(snap)
    assert apply_dossier(snap, _battle()) == before


def test_corrupt_dossier_and_none_snapshot_are_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(opponent_dossier, "DOSSIER_DIR", tmp_path)
    (tmp_path / "rival.json").write_text("{broken", encoding="utf-8")
    snap = _snapshot()
    before = copy.deepcopy(snap)
    assert apply_dossier(snap, _battle()) == before        # load() degrades → no-op
    assert apply_dossier(None, _battle()) is None
    assert apply_dossier(snap, SimpleNamespace()) == before  # no opponent_username


def test_base_species_used_for_mega_formes(tmp_path, monkeypatch):
    _write_dossier(tmp_path, monkeypatch, _REC)
    snap = {"opp_active": {"opp_a": {"species": "Mawile-Mega", "base_species": "Mawile",
                                     "known_ability": None, "known_moves": []}},
            "opp_bench": []}
    out = apply_dossier(snap, _battle())
    assert out["opp_active"]["opp_a"]["known_ability"] == "Intimidate"
