"""Era-5 W1 — own-team SPECIALIST weighting for the BC and TP trainers (2026-09-01).

Frame: specialize the OWN side, keep the OPPONENT side general. Weight = 1 + λ·(fraction of OUR
six in the demonstrator's roster), read from each replay's first line; no game is dropped; the
own-team val slice (overlap ≥ 4/6) is the metric this arm is for. Locks the team parse, the
first-line overlap map (canonical replay ids, both sides), the weight composition, the slice, the
TP analog, and that the trainers accept the flags."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("poke_env")

from v_dance.training import bc_dataset as B

PASTE = """Charizard (M) @ Charizardite Y
Ability: Blaze
- Protect

Basculegion @ Choice Scarf
Ability: Adaptability
- Wave Crash

Whimsicott @ Focus Sash
Ability: Prankster
- Tailwind

Garchomp @ Roseli Berry
Ability: Rough Skin
- Earthquake

Kingambit @ Chople Berry
Ability: Defiant
- Kowtow Cleave

Incineroar @ Sitrus Berry
Ability: Intimidate
- Fake Out
"""
SIX = ["charizard", "basculegion", "whimsicott", "garchomp", "kingambit", "incineroar"]


def test_parse_team_species_from_text_file_and_pool_name(tmp_path: Path):
    assert B.parse_team_species(PASTE) == SIX
    f = tmp_path / "team.txt"
    f.write_text(PASTE, encoding="utf-8")
    assert B.parse_team_species(str(f)) == SIX
    assert B.parse_team_species("The_Big_6") == SIX       # the pool team resolves by name
    with pytest.raises(ValueError):
        B.parse_team_species("not a paste at all")


def _write_replay(folder: Path, name: str, rid: str, p1: list, p2: list) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    line = {"replay_id": rid, "players": {"our_side": "p1", "p1": {"roster": p1}, "p2": {"roster": p2}},
            "turn": 1}
    (folder / f"{name}.jsonl").write_text(json.dumps(line) + "\n" + json.dumps({"turn": 2}) + "\n",
                                          encoding="utf-8")


def test_overlap_map_reads_first_lines_for_both_sides_with_canonical_ids(tmp_path: Path):
    d = tmp_path / "Jsonl_X"
    _write_replay(d, "a", "battle-gen9x-1", ["Charizard", "Garchomp", "Kingambit", "Incineroar", "Pikachu", "Ditto"],
                  ["Snorlax", "Ditto", "Mew", "Mewtwo", "Lugia", "Ho-Oh"])
    _write_replay(d, "b", "gen9x-2__closed", ["Whimsicott", "Basculegion", "Mew", "Ditto", "Pikachu", "Snorlax"],
                  SIX)
    m = B.own_team_overlap_map([str(d)], SIX, cache_dir=None)
    assert m[("gen9x-1", "p1")] == pytest.approx(4 / 6) and m[("gen9x-1", "p2")] == 0.0
    assert m[("gen9x-2", "p1")] == pytest.approx(2 / 6) and m[("gen9x-2", "p2")] == 1.0


def test_overlap_map_cache_round_trip(tmp_path: Path):
    d = tmp_path / "Jsonl_X"
    _write_replay(d, "a", "gen9x-1", SIX, ["Ditto"] * 6)
    cache_dir = tmp_path / "artifacts"
    m1 = B.own_team_overlap_map([str(d)], SIX, cache_dir=str(cache_dir))
    assert list(cache_dir.glob(".own_team_overlap_*.json"))
    m2 = B.own_team_overlap_map([str(d)], SIX, cache_dir=str(cache_dir))
    assert m1 == m2 and m2[("gen9x-1", "p1")] == 1.0


def test_compute_own_team_weights_composes_and_normalises():
    m = {("gen9x-1", "p1"): 4 / 6, ("gen9x-1", "p2"): 0.0, ("gen9x-2", "p2"): 1.0}
    ex = [{"replay_id": "battle-gen9x-1", "perspective": "p1"},   # 4/6 overlap
          {"replay_id": "gen9x-1", "perspective": "p2"},          # 0
          {"replay_id": "gen9x-2__closed", "perspective": "p2"},  # 6/6
          {"replay_id": "gen9x-9", "perspective": "p1"}]          # unmapped → 1.0
    w, hist = B.compute_own_team_weights(ex, m, lam=3.0)
    assert w.dtype == np.float32 and abs(float(w.mean()) - 1.0) < 1e-6
    raw = np.array([1 + 3 * 4 / 6, 1.0, 4.0, 1.0])
    assert np.allclose(w, raw / raw.mean(), atol=1e-6)
    assert hist == Counter({4: 1, 0: 1, 6: 1, "unmapped": 1})
    assert B.own_slice_indices(ex, m, 4 / 6) == [0, 2]
    assert B.own_slice_indices(ex, m, 5 / 6) == [2]


def test_tp_weights_own_team_composes_with_outcome_and_refuses_a_no_op():
    from v_dance.training.train_teampreview import compute_tp_weights
    ex = [{"our_species": SIX, "won": True},
          {"our_species": ["ditto"] * 6, "won": False},
          {"our_species": SIX[:3] + ["ditto"] * 3, "won": True}]
    w = compute_tp_weights(ex, outcome_weight=True, loss_weight=0.5, own_species=SIX, own_team_weight=3.0)
    raw = np.array([4.0, 0.5, 2.5])
    assert np.allclose(w, raw / raw.mean(), atol=1e-6)
    with pytest.raises(SystemExit):
        compute_tp_weights(ex, own_species=["mew", "mewtwo"], own_team_weight=3.0)
    assert compute_tp_weights(ex) is None                   # nothing enabled → legacy path


def test_trainers_accept_the_specialist_flags():
    from v_dance.training import train_bc, train_teampreview
    a = train_bc.parse_args(["--own-team", "The_Big_6", "--own-team-weight", "2.5",
                             "--own-team-min-overlap", "5", "--select-own-slice", "--out", "x"])
    assert a.own_team == "The_Big_6" and a.own_team_weight == 2.5 and a.own_team_min_overlap == 5
    assert a.select_own_slice
    b = train_bc.parse_args(["--out", "x"])
    assert b.own_team is None and b.own_team_weight == 3.0 and not b.select_own_slice
    p = train_teampreview.parse_args if hasattr(train_teampreview, "parse_args") else None
    if p is not None:
        t = p(["--own-team", "The_Big_6", "--own-team-weight", "2", "--out", "y"])
        assert t.own_team == "The_Big_6" and t.own_team_weight == 2.0
