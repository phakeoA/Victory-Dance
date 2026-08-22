"""tpfeat-v7 (2026-07-02): OTS opponent-sheet conditioning for the TP net.

  * opp_mon_features grows an optional ``revealed`` overlay (open team sheets
    make the opponent's build preview-visible); revealed=None stays
    byte-identical to v6 — closed-sheet play is unchanged.
  * teampreview_dataset builds the per-battle sheet map from OTS transitions
    and (since tpfeat-v8) rides it on BOTH sides — the deciding player saw their
    own sheet at preview too.
  * model_io's lockstep guard serves v6/v7 checkpoints through the FROZEN v7
    extractor; v8 uses the current one (dims differ).
  * v8 crisp own side: OwnBuildBelief sharpens our own mons' BASE channels from
    the true build at serve (gated off for legacy checkpoints).
  * ingest --tp-only is covered in test_hf_ots_ingest.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np
import torch


@pytest.fixture()
def tiny_belief(tmp_path):
    from v_dance.parser.belief_state import BeliefState
    data = {"pokemon": {
        "Garchomp": {
            "usage_pct": 40.0,
            "moves": [{"name": "Earthquake", "pct": 90.0}],
            "items": [{"name": "Life Orb", "pct": 50.0}],
            "abilities": [{"name": "Rough Skin", "pct": 99.0}],
            "spreads": [{"nature": "Jolly", "evs": [0, 32, 0, 0, 0, 32], "pct": 100.0}],
        },
        "Torkoal": {
            "usage_pct": 20.0,
            "moves": [{"name": "Eruption", "pct": 80.0}],
            "items": [{"name": "Charcoal", "pct": 40.0}],
            "abilities": [{"name": "Drought", "pct": 95.0}],
            "spreads": [{"nature": "Quiet", "evs": [252, 0, 0, 252, 4, 0], "pct": 100.0}],
        },
    }}
    p = tmp_path / "pika.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return BeliefState(p)


def test_opp_overlay_off_is_v6_identical(tiny_belief):
    from v_dance.training.tp_features import (
        FEAT_DIM, OFF_OWNBIT, opp_mon_features, own_mon_features)
    a = opp_mon_features("garchomp", tiny_belief)
    b = own_mon_features("garchomp", tiny_belief, None)
    assert a.shape == (FEAT_DIM,) and np.array_equal(a, b)
    assert a[OFF_OWNBIT] == 0.0
    assert np.all(a[OFF_OWNBIT:] == 0.0)          # the whole overlay block stays zero


def test_opp_overlay_on_marks_revealed_build(tiny_belief):
    from v_dance.training.tp_features import (
        OFF_OWNBIT, OFF_KROLES, ROLE_TAGS, OwnKnown, opp_mon_features)
    revealed = OwnKnown(ability="Drought", moves=["Trick Room", "Eruption", "Protect"])
    f = opp_mon_features("torkoal", tiny_belief, revealed=revealed)
    base = opp_mon_features("torkoal", tiny_belief)
    assert f[OFF_OWNBIT] == 1.0                    # the open/closed regime marker
    assert f[OFF_KROLES + ROLE_TAGS.index("trick_room")] == 1.0   # hard known TR
    # the symmetric base block is untouched by the overlay
    assert np.array_equal(f[:OFF_OWNBIT], base[:OFF_OWNBIT])


# ── dataset: the OTS sheet map rides the OPP side only ───────────────────────

_P1_ROSTER = ["Torkoal", "Hatterene", "Indeedee-F", "Amoonguss", "Ursaluna", "Porygon2"]
_P2_ROSTER = ["Talonflame", "Garchomp", "Aegislash", "Rotom-Wash", "Sneasler",
              "Arcanine-Hisui"]


def _sheet_mon(species, ability, moves):
    return {"species": species, "known_ability": ability, "known_moves": moves}


def _tp_line(persp, rid, opp_bench):
    return {
        "replay_id": rid, "perspective": persp, "turn": 1, "ots": True,
        "players": {
            "p1": {"roster": _P1_ROSTER, "brought": _P1_ROSTER[:4]},
            "p2": {"roster": _P2_ROSTER, "brought": _P2_ROSTER[:4]},
        },
        "state_before_actions": {
            "our_active": {}, "our_bench": [],
            "opp_active": {}, "opp_bench": opp_bench,
        },
    }


def _write_tp_battle(folder, rid):
    # p1's view reveals p2's sheets; p2's view reveals p1's — union = all 12.
    p1_line = _tp_line("p1", rid, [
        _sheet_mon("Garchomp", "Rough Skin",
                   ["Earthquake", "Dragon Claw", "Protect", "Swords Dance"])])
    p2_line = _tp_line("p2", rid, [
        _sheet_mon("Torkoal", "Drought", ["Eruption", "Trick Room", "Protect"])])
    (folder / f"{rid}.jsonl").write_text(
        json.dumps(p1_line) + "\n" + json.dumps(p2_line), encoding="utf-8")


def test_build_examples_ots_opp_overlay(tmp_path, tiny_belief):
    from v_dance.training.teampreview_dataset import build_examples, sbda_feature_fn
    from v_dance.training.tp_features import OFF_OWNBIT
    corpus = tmp_path / "tp_corpus"
    corpus.mkdir()
    _write_tp_battle(corpus, "tp-syn-1")

    feat_fn = sbda_feature_fn(tiny_belief)
    exs, stats = build_examples([str(corpus / "tp-syn-1.jsonl")], feat_fn=feat_fn)
    assert len(exs) == 2 and stats["ots_sheet_replays"] == 1
    by_side = {e["side"]: e for e in exs}

    # p1's OPPONENT is p2 — the revealed Garchomp rides the overlay bit
    p1 = by_side["p1"]
    gar = p1["opp_species"].index("garchomp")
    assert p1["opp_feat"][gar, OFF_OWNBIT] == 1.0
    # v8: the OWN side rides its sheet too (the decider SAW their own build at preview)
    tor_own = p1["our_species"].index("torkoal")
    assert p1["our_feat"][tor_own, OFF_OWNBIT] == 1.0
    assert stats["ots_own_overlay_mons"] == 2      # torkoal (p1 row) + garchomp (p2 row)
    # p2's OPPONENT is p1 — the revealed Torkoal rides the overlay bit
    p2 = by_side["p2"]
    tor = p2["opp_species"].index("torkoal")
    assert p2["opp_feat"][tor, OFF_OWNBIT] == 1.0

    # the LEGACY recipe ignores sheets entirely (no supports_known)
    exs_legacy, _ = build_examples([str(corpus / "tp-syn-1.jsonl")])
    assert exs_legacy[0]["opp_feat"].shape[1] == 46


def test_closed_sheet_battle_has_no_overlay(tmp_path, tiny_belief):
    from v_dance.training.teampreview_dataset import build_examples, sbda_feature_fn
    from v_dance.training.tp_features import OFF_OWNBIT
    corpus = tmp_path / "tp_closed"
    corpus.mkdir()
    line = _tp_line("p1", "tp-syn-2", [])
    line["ots"] = False
    (corpus / "tp-syn-2.jsonl").write_text(json.dumps(line), encoding="utf-8")
    exs, stats = build_examples([str(corpus / "tp-syn-2.jsonl")],
                                feat_fn=sbda_feature_fn(tiny_belief))
    assert exs and stats["ots_sheet_replays"] == 0
    for e in exs:
        assert np.all(e["opp_feat"][:, OFF_OWNBIT] == 0.0)


# ── model_io: v6/v7 checkpoints stay loadable under v8 code (frozen extractor) ─

def _tp_ckpt(tmp_path, schema, feat_dim=None):
    from v_dance.models.teampreview_model import TeamPreviewModel
    if feat_dim is None:
        if schema in ("tpfeat-v6", "tpfeat-v7", "tpfeat-v5"):
            from v_dance.training.tp_features_v7 import FEAT_DIM as feat_dim   # frozen dims
        else:
            from v_dance.training.tp_features import FEAT_DIM as feat_dim
    torch.manual_seed(3)
    m = TeamPreviewModel(vocab_size=8, feat_dim=feat_dim, emb_dim=8, hidden=16,
                         dropout=0.0)
    cfg = {"vocab_size": 8, "feat_dim": feat_dim, "emb_dim": 8, "hidden": 16,
           "dropout": 0.0, "feature_schema": schema}
    p = tmp_path / f"tp_{schema}.pt"
    torch.save({"model_state": m.state_dict(), "config": cfg,
                "vocab": {"garchomp": 1}}, p)
    return p


def test_load_team_chooser_accepts_v6_v7_and_v8(tmp_path):
    from v_dance.play.model_io import load_team_chooser
    from v_dance.training.tp_features import FEATURE_SCHEMA_VERSION
    for schema in ("tpfeat-v6", "tpfeat-v7", FEATURE_SCHEMA_VERSION):
        model, vocab, cfg = load_team_chooser(str(_tp_ckpt(tmp_path, schema)))
        assert cfg["feature_schema"] == schema and vocab
    with pytest.raises(ValueError, match="lockstep"):
        load_team_chooser(str(_tp_ckpt(tmp_path, "tpfeat-v5")))          # unknown schema
    with pytest.raises(ValueError, match="lockstep"):
        # a v7 ckpt claiming v8 dims (or vice versa) must fail the dim check
        from v_dance.training.tp_features import FEAT_DIM as V8_DIM
        load_team_chooser(str(_tp_ckpt(tmp_path, "tpfeat-v7", feat_dim=V8_DIM)))


# ── v8 crisp own side: OwnBuildBelief + team_order schema gating ──────────────

def test_own_build_belief_crisp_and_delegating(tiny_belief):
    from v_dance.play.model_io import OwnBuildBelief
    w = OwnBuildBelief(tiny_belief, {
        "garchomp": {"ability": "roughskin", "item": "choicescarf",
                     "moves": ["earthquake", "protect"]}})
    assert w.ability_distribution("garchomp") == [{"name": "roughskin", "p": 1.0}]
    assert w.item_distribution("garchomp") == [{"name": "choicescarf", "p": 1.0}]
    assert [m["p"] for m in w.move_distribution("garchomp")] == [1.0, 1.0]
    assert w.ability_distribution("torkoal")[0]["name"] == "Drought"     # delegate
    assert w.usage("garchomp") == tiny_belief.usage("garchomp")          # delegate
    # OOV own mon (any USER team): known via the build even if the belief never saw it
    assert OwnBuildBelief(tiny_belief, {"weirdmon": {"ability": "drought"}}).known("weirdmon")


def test_own_build_two_mega_smear_resolves_to_crisp_sun(tiny_belief):
    from v_dance.parser.vod_parser.pokedex import get_pokedex
    dex = get_pokedex()
    if dex is None or not dex.mega_formes_for("Charizard"):
        pytest.skip("pokedex with mega formes unavailable")
    from v_dance.play.model_io import OwnBuildBelief
    import v_dance.training.tp_features as T
    # tiny_belief does NOT know Charizard -> raw belief gives NO tags at all
    raw = T.own_mon_features("charizard", tiny_belief)
    assert raw[T.OFF_WSETS + T.WEATHERS.index("sun")] == 0.0
    # our OWN Charizard holds the Y stone -> Drought locked -> sun tag CRISP 1.0
    w = OwnBuildBelief(tiny_belief, {"charizard": {
        "ability": "blaze", "item": "charizarditey", "moves": ["heatwave", "protect"]}})
    f = T.own_mon_features("charizard", w)
    assert f[T.OFF_WSETS + T.WEATHERS.index("sun")] == 1.0
    assert f[T.OFF_GK + T.GIMMICK_KINDS.index("mega")] == 1.0
    assert f[T.OFF_ITEMS:T.OFF_ITEMS + len(T.ITEM_TAGS)].sum() == 0.0   # stones aren't ITEM_TAGS


def test_team_order_own_build_ignored_for_legacy_schema(tmp_path, tiny_belief):
    from v_dance.play.model_io import load_team_chooser, team_order
    from v_dance.training.tp_features import FEATURE_SCHEMA_VERSION
    ours = ["garchomp", "torkoal", "a", "b", "c", "d"]
    opps = ["torkoal", "garchomp", "e", "f", "g", "h"]
    build = {"garchomp": {"ability": "roughskin", "item": "choicescarf",
                          "moves": ["earthquake"]}}
    # v7 ckpt: own_build must be a no-op (byte-identical serve through the frozen extractor)
    m7, v7, c7 = load_team_chooser(str(_tp_ckpt(tmp_path, "tpfeat-v7")))
    base = team_order(m7, v7, c7, ours, opps, 4, belief=tiny_belief)
    with_build = team_order(m7, v7, c7, ours, opps, 4, belief=tiny_belief, own_build=build)
    assert base == with_build
    # v8 ckpt: own_build is consumed and still yields a valid order
    m8, v8, c8 = load_team_chooser(str(_tp_ckpt(tmp_path, FEATURE_SCHEMA_VERSION)))
    order = team_order(m8, v8, c8, ours, opps, 4, belief=tiny_belief, own_build=build)
    assert len(order) == 4 and len(set(order)) == 4 and all(0 <= i < 6 for i in order)
