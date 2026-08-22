"""DS-4c stages 2+3 (2026-07-12): Bo3 set-state carrier + TP set-context input.

Contract: |uhtml|bestof| frames register (parent set, game idx); set_ctx_for aligns the
previous game's brings to the CURRENT rosters by species (game 1 / off-set → None);
the model's set_ctx_proj is zero-init ⇒ any context is an exact identity until trained;
dataset emits ctx keys ONLY when opted in (byte-compat); ctx examples are built from
bo3_set_id groups with room-number ordering.
"""
import json
from types import SimpleNamespace

import numpy as np
import torch

from v_dance.play import bo3_state
from v_dance.models.teampreview_model import TeamPreviewModel
from v_dance.training.teampreview_dataset import (
    TEAM_SIZE, TeamPreviewDataset, build_set_prev_map, examples_from_folders,
)
from v_dance.training.train_teampreview import apply_warm_start

# The EXACT frame shape the stage-1 probe captured from the pinned server.
_BESTOF_PAYLOAD = (">battle-gen9championsvgc2026regmb-815784\n"
                   "|uhtml|bestof|<h2><strong>Game 2</strong> of "
                   '<a href="/game-bestof3-gen9championsvgc2026regmb-815782">a best-of-3</a></h2>')

_ARCH = dict(vocab_size=16, feat_dim=8, emb_dim=8, hidden=16, dropout=0.1,
             use_self_attn=True, attn_heads=2)


def _player():
    return SimpleNamespace(_battles={})


def test_note_bestof_frame_parses_probe_shape():
    p = _player()
    bo3_state.note_bestof_frame(p, "battle-gen9championsvgc2026regmb-815784-suffixpw",
                                _BESTOF_PAYLOAD)
    assert p._bo3_of["battle-gen9championsvgc2026regmb-815784"] == (
        "game-bestof3-gen9championsvgc2026regmb-815782", 2)


def test_set_ctx_roundtrip_and_alignment():
    p = _player()
    parent = "game-bestof3-x-1"
    p._bo3_of = {"battle-x-10": (parent, 1), "battle-x-11": (parent, 2)}
    p._bo3_sets = {parent: {1: {"tag": "battle-x-10", "our_bring": [], "our_leads": [],
                                "opp_seen": [], "result": None}}}
    b1 = SimpleNamespace(battle_tag="battle-x-10")
    bo3_state.record_our_picks(p, b1, ["Mawile", "Torkoal", "Farigiraf", "Incineroar"])
    # game end harvests the opp mons that appeared
    b1.opponent_team = {"p1: Pelipper": SimpleNamespace(species="Pelipper", base_species=None),
                        "p1: Archaludon": SimpleNamespace(species="Archaludon", base_species=None)}
    p._battles = {"battle-x-10": b1}
    bo3_state.record_game_end(p, "battle-x-10", "human")
    # game 2 preview: rosters in a DIFFERENT order than the picks — species alignment
    b2 = SimpleNamespace(battle_tag="battle-x-11-privsuffix")
    our = ["Charizard", "Mawile", "Venusaur", "Torkoal", "Farigiraf", "Incineroar"]
    opp = ["Basculegion", "Pelipper", "Milotic", "Archaludon", "Grimmsnarl", "Ditto"]
    ctx = bo3_state.set_ctx_for(p, b2, our, opp)
    assert ctx is not None
    our_ctx, opp_ctx = ctx
    assert our_ctx[1].tolist() == [1.0, 1.0]      # Mawile brought + led (pick #1)
    assert our_ctx[3].tolist() == [1.0, 1.0]      # Torkoal brought + led (pick #2)
    assert our_ctx[4].tolist() == [1.0, 0.0]      # Farigiraf brought, NOT a lead (pick #3)
    assert our_ctx[0].tolist() == [0.0, 0.0]      # Charizard stayed home
    assert opp_ctx[1][0] == 1.0 and opp_ctx[3][0] == 1.0   # Pelipper/Archaludon appeared
    assert opp_ctx[:, 1].sum() == 0               # opp leads honestly unknown -> zeros
    # game 1 (no previous game) -> None
    assert bo3_state.set_ctx_for(p, b1, our, opp) is None
    # off-set battle -> None
    assert bo3_state.set_ctx_for(p, SimpleNamespace(battle_tag="battle-x-99"), our, opp) is None


def _write_set_games(tmp_path):
    """Two games of one Bo3 set + one off-set game, minimal first transitions."""
    def _t(rid, set_id, brought_p1, brought_p2):
        players = {"p1": {"roster": [f"M{i}" for i in range(6)], "brought": brought_p1},
                   "p2": {"roster": [f"O{i}" for i in range(6)], "brought": brought_p2}}
        d = {"replay_id": rid, "perspective": "p1", "players": players,
             "state_before_actions": {}}
        if set_id:
            d["bo3_set_id"] = set_id
        return d
    (tmp_path / "g1.jsonl").write_text(json.dumps(_t(
        "gen9x-101", "set1", ["M0", "M1", "M2", "M3"], ["O0", "O1", "O2", "O3"])),
        encoding="utf-8")
    (tmp_path / "g2.jsonl").write_text(json.dumps(_t(
        "gen9x-102", "set1", ["M2", "M3", "M4", "M5"], ["O1", "O2", "O4", "O5"])),
        encoding="utf-8")
    (tmp_path / "solo.jsonl").write_text(json.dumps(_t(
        "gen9x-200", None, ["M0", "M1", "M2", "M3"], ["O0", "O1", "O2", "O3"])),
        encoding="utf-8")


def test_build_set_prev_map_and_example_ctx(tmp_path):
    _write_set_games(tmp_path)
    files = sorted(str(p) for p in tmp_path.glob("*.jsonl"))
    prev = build_set_prev_map(files)
    assert set(prev) == {"gen9x-102"}             # only game 2 has a previous game
    assert prev["gen9x-102"]["p1"] == (["M0", "M1", "M2", "M3"], ["M0", "M1"])
    exs, stats = examples_from_folders([str(tmp_path)])
    assert stats["set_ctx_examples"] == 2         # p1 + p2 examples of game 2
    g2 = [e for e in exs if e["replay_id"] == "gen9x-102" and e["side"] == "p1"][0]
    assert g2["our_set_ctx"][0].tolist() == [1.0, 1.0]     # M0 brought+led game 1
    assert g2["our_set_ctx"][4].tolist() == [0.0, 0.0]     # M4 stayed home game 1
    assert g2["opp_set_ctx"][3].tolist() == [1.0, 0.0]     # O3 brought, not led
    solo = [e for e in exs if e["replay_id"] == "gen9x-200"][0]
    assert "our_set_ctx" not in solo              # off-set dicts byte-identical
    # dataset emits keys only when opted in
    vocab = {s: i + 1 for i, s in enumerate(sorted({s for e in exs for s in
                                                    e["our_species"] + e["opp_species"]}))}
    d0 = TeamPreviewDataset(exs, vocab)
    assert "our_set_ctx" not in d0[0]
    d1 = TeamPreviewDataset(exs, vocab, with_set_ctx=True)
    item = d1[0]
    assert item["our_set_ctx"].shape == (TEAM_SIZE, 2)


def test_model_zero_init_identity_and_learned_effect():
    torch.manual_seed(0)
    m = TeamPreviewModel(**_ARCH, use_set_ctx=True).eval()
    g = torch.Generator().manual_seed(1)
    oi = torch.randint(1, 16, (2, 6), generator=g); pi = torch.randint(1, 16, (2, 6), generator=g)
    of = torch.rand(2, 6, 8, generator=g) + 0.1; pf = torch.rand(2, 6, 8, generator=g) + 0.1
    ctx = torch.rand(2, 6, 2, generator=g)
    b0, l0 = m(oi, pi, of, pf)
    b1, l1 = m(oi, pi, of, pf, our_set_ctx=ctx, opp_set_ctx=ctx)
    assert torch.equal(b0, b1) and torch.equal(l0, l1)     # zero-init ⇒ exact identity
    with torch.no_grad():
        m.set_ctx_proj.weight.normal_(std=0.3)
    b2, _ = m(oi, pi, of, pf, our_set_ctx=ctx, opp_set_ctx=ctx)
    assert not torch.equal(b0, b2)                          # learned proj: ctx matters ...
    b3, _ = m(oi, pi, of, pf, our_set_ctx=torch.zeros(2, 6, 2),
              opp_set_ctx=torch.zeros(2, 6, 2))
    assert torch.equal(b0, b3)                              # ... but zero ctx stays identity


def test_warm_start_allows_fresh_ctx_proj():
    donor = TeamPreviewModel(**_ARCH)
    rec = TeamPreviewModel(**_ARCH, use_set_head=True, use_set_ctx=True)
    n_fresh = apply_warm_start(rec, donor.state_dict())
    assert n_fresh == 10                                    # 8 set-head + 2 set_ctx_proj tensors
    assert float(rec.set_ctx_proj.weight.detach().abs().sum()) == 0.0
