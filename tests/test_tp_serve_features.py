"""15b-io.1 — the team-preview SERVE path switches to the shared SBDA feature
extractor (tp_features.own/opp_mon_features) IN LOCKSTEP with the dataset, behind
the checkpoint's feature_schema, with a fail-loud dim guard so a schema drift can
never silently zero-pad the synergy channels.  The legacy 46-dim path is untouched.

Covers:
  * uses_tp_features() schema detection
  * _pack_side legacy (dex-only) vs SBDA (belief-derived synergy tags) recipes
  * the lockstep guard (wrong feat_dim raises) at both _pack_side and load time
  * train==serve parity: serve _pack_side row == dataset opp_mon_features
  * the OWN overlay bit + teammate-bias threading through a real SBDA model
  * the player resolving a belief for an SBDA chooser and NOT for a legacy one
"""
from __future__ import annotations

import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("poke_env")

import torch  # noqa: E402

import v_dance.play.model_io as M  # noqa: E402
from v_dance.parser.vod_parser.pokedex import norm_species  # noqa: E402
from v_dance.training import tp_features as TPF  # noqa: E402
from v_dance.models.teampreview_model import build_model  # noqa: E402


# ── a tiny belief stub: every mon "knows" Sand Stream + Earthquake (usage 50) ──
class _SandBelief:
    """Reports the same build for every species — enough to light up the weather-
    setter + ground-spread + usage channels, so a test can prove the serve path
    actually runs the belief-derived synergy features rather than zeros."""
    def known(self, s):                     # noqa: D401
        return True

    def ability_distribution(self, s, top_k=4):
        return [{"name": "Sand Stream", "p": 1.0}]

    def move_distribution(self, s, top_k=8):
        return [{"name": "Earthquake", "p": 1.0}]

    def usage(self, s):
        return 50.0

    def teammates(self, s, top_k=16):
        return []


# ── uses_tp_features ──────────────────────────────────────────────────────────
def test_uses_tp_features_detects_schema():
    assert M.uses_tp_features({"feature_schema": "tpfeat-v6"}) is True
    assert M.uses_tp_features({"feature_schema": "tpfeat-v5"}) is True
    assert M.uses_tp_features({}) is False                 # legacy: no stamp
    assert M.uses_tp_features({"lead_k": 2}) is False
    assert M.uses_tp_features(None) is False
    assert M.uses_tp_features({"feature_schema": "something-else"}) is False


# ── _pack_side legacy path is byte-identical to mon_dex_features ──────────────
def test_pack_side_legacy_is_dex_features():
    from v_dance.training.teampreview_dataset import mon_dex_features, MON_FEAT_DIM
    species = ["Tyranitar", "Charizard"]
    vocab = {norm_species("Tyranitar"): 3}
    idx, feat = M._pack_side(species, vocab, MON_FEAT_DIM)   # use_tp_features defaults False
    assert feat.shape == (6, MON_FEAT_DIM)
    assert idx[0] == 3 and idx[1] == 0                      # unseen → PAD 0
    np.testing.assert_array_equal(feat[0], mon_dex_features(norm_species("Tyranitar")))
    np.testing.assert_array_equal(feat[2], np.zeros(MON_FEAT_DIM, np.float32))  # pad slot


# ── _pack_side SBDA path runs the belief-derived synergy channels (not zeros) ──
def test_pack_side_sbda_fills_belief_tags():
    b = _SandBelief()
    idx, feat = M._pack_side(["Tyranitar"], {}, TPF.FEAT_DIM,
                             belief=b, use_tp_features=True)
    assert feat.shape == (6, TPF.FEAT_DIM)
    # Sand Stream → the 'sand' weather-setter channel (axis 0) is lit
    assert feat[0, TPF.OFF_WSETS + TPF.WEATHERS.index("sand")] == pytest.approx(1.0)
    # Earthquake → the Ground spread-move channel is lit
    assert feat[0, TPF.OFF_SPREAD + TPF._TYPE_IDX[TPF._canon_type("Ground")]] == pytest.approx(1.0)
    assert feat[0, TPF.OFF_USAGE] == pytest.approx(0.5)     # usage 50 / 100
    assert feat[0, TPF.OFF_OWNBIT] == 0.0                   # no overlay on the symmetric base


def test_pack_side_sbda_own_overlay_sets_bit():
    b = _SandBelief()
    known = {norm_species("Tyranitar"): TPF.OwnKnown(ability="Sand Stream",
                                                     moves=("Earthquake",))}
    _idx, feat = M._pack_side(["Tyranitar"], {}, TPF.FEAT_DIM,
                              belief=b, own_known=known, use_tp_features=True)
    assert feat[0, TPF.OFF_OWNBIT] == pytest.approx(1.0)    # has_own_detail flipped on


def test_pack_side_null_belief_degrades_to_typing_only_not_crash():
    # No belief → the null floor: typing-grounded base, no ability/move/usage tags.
    _idx, feat = M._pack_side(["Tyranitar"], {}, TPF.FEAT_DIM, use_tp_features=True)
    assert feat.shape == (6, TPF.FEAT_DIM)
    assert feat[0, TPF.OFF_WSETS + TPF.WEATHERS.index("sand")] == 0.0  # no belief → no setter tag
    assert feat[0, TPF.OFF_HASDATA] == 0.0
    assert feat[0].any()                                   # but typing/dex channels are present


# ── the lockstep dim guard ────────────────────────────────────────────────────
def test_pack_side_sbda_wrong_feat_dim_raises():
    with pytest.raises(ValueError, match="lockstep"):
        M._pack_side(["Tyranitar"], {}, TPF.FEAT_DIM + 7,
                     belief=_SandBelief(), use_tp_features=True)


# ── train==serve parity: serve row == the dataset's extractor output ──────────
def test_serve_features_match_dataset_extractor():
    b = _SandBelief()
    species = ["Tyranitar", "Charizard", "Garchomp"]
    _idx, feat = M._pack_side(species, {}, TPF.FEAT_DIM, belief=b, use_tp_features=True)
    for i, sp in enumerate(species):
        expected = TPF.opp_mon_features(norm_species(sp), b)
        np.testing.assert_array_equal(feat[i], expected)   # serve recipe IS the extractor


# ── load_team_chooser lockstep guard ─────────────────────────────────────────
def _save_tp_ckpt(tmp_path, feat_dim, schema, name="tp.pt",
                  use_self_attn=True, use_teammate_bias=True):
    model = build_model(vocab_size=20, feat_dim=feat_dim, hidden=16, dropout=0.0,
                        use_self_attn=use_self_attn, use_teammate_bias=use_teammate_bias)
    cfg = {"vocab_size": 20, "feat_dim": feat_dim, "emb_dim": 32, "hidden": 16,
           "dropout": 0.0, "use_self_attn": use_self_attn,
           "use_teammate_bias": use_teammate_bias, "lead_k": 2, "bring_k": 4}
    if schema is not None:
        cfg["feature_schema"] = schema
    p = tmp_path / name
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "vocab": {"tyranitar": 1, "charizard": 2}}, p)
    return p


def test_load_team_chooser_rejects_schema_dim_drift(tmp_path):
    # schema says SBDA but feat_dim is wrong → fail loud at LOAD.
    p = _save_tp_ckpt(tmp_path, feat_dim=TPF.FEAT_DIM + 9,
                      schema=TPF.FEATURE_SCHEMA_VERSION, name="drift.pt")
    with pytest.raises(ValueError, match="lockstep"):
        M.load_team_chooser(p)


def test_load_team_chooser_rejects_stale_schema_version(tmp_path):
    p = _save_tp_ckpt(tmp_path, feat_dim=TPF.FEAT_DIM, schema="tpfeat-v0", name="stale.pt")
    with pytest.raises(ValueError, match="lockstep"):
        M.load_team_chooser(p)


def test_load_team_chooser_accepts_matched_sbda_checkpoint(tmp_path):
    p = _save_tp_ckpt(tmp_path, feat_dim=TPF.FEAT_DIM,
                      schema=TPF.FEATURE_SCHEMA_VERSION, name="ok.pt")
    model, vocab, cfg = M.load_team_chooser(p)
    assert M.uses_tp_features(cfg) is True
    assert model.use_self_attn and model.use_teammate_bias
    assert cfg["feat_dim"] == TPF.FEAT_DIM


# ── team_order end-to-end on a real SBDA model (teammate-bias path exercised) ──
def test_team_order_runs_sbda_model_with_belief():
    model = build_model(vocab_size=20, feat_dim=TPF.FEAT_DIM, hidden=16,
                        use_self_attn=True, use_teammate_bias=True)
    cfg = {"feat_dim": TPF.FEAT_DIM, "feature_schema": TPF.FEATURE_SCHEMA_VERSION,
           "lead_k": 2, "bring_k": 4}
    our = ["Tyranitar", "Charizard", "Garchomp", "Amoonguss", "Incineroar", "Rillaboom"]
    opp = ["Miraidon", "Ironhands", "Chiyu", "Landorus", "Ogerpon", "Ironbundle"]
    vocab = {norm_species(s): i + 1 for i, s in enumerate(our + opp)}
    order = M.team_order(model, vocab, cfg, our, opp, n=4, belief=_SandBelief())
    assert len(order) == 4 and len(set(order)) == 4 and all(0 <= i < 6 for i in order)


# ── the player resolves a belief ONLY for an SBDA chooser ─────────────────────
def _player(cfg):
    import v_dance.play.player as P
    obj = P.VGCPlayer.__new__(P.VGCPlayer)
    obj._team_chooser = object()
    obj._tc_vocab = {}
    obj._tc_cfg = cfg
    obj._device = "cpu"
    obj._tp_source = Counter()
    return obj, P


def _team(*species):
    return [types.SimpleNamespace(species=s) for s in species]


def _battle():
    return types.SimpleNamespace(
        teampreview_opponent_team=[types.SimpleNamespace(species=s)
                                   for s in ("Charizard", "Whimsicott")],
        battle_tag="b1")


def test_player_passes_belief_for_sbda_chooser(monkeypatch):
    p, P = _player({"feature_schema": "tpfeat-v6", "lead_k": 2})
    sentinel = object()
    monkeypatch.setattr("v_dance.play.vgc_base._default_belief", lambda: sentinel)
    seen = {}

    def capture(*a, **k):
        seen["belief"] = k.get("belief")
        return [0, 1, 2, 3]

    monkeypatch.setattr(P._M, "team_order", capture)
    p._choose_team_order(_battle(), _team("a", "b", "c", "d", "e", "f"), 4)
    assert seen["belief"] is sentinel


def test_player_no_belief_for_legacy_chooser(monkeypatch):
    p, P = _player({"lead_k": 2})       # no feature_schema → legacy
    seen = {}

    def capture(*a, **k):
        seen["belief"] = k.get("belief", "MISSING")
        return [0, 1, 2, 3]

    monkeypatch.setattr(P._M, "team_order", capture)
    p._choose_team_order(_battle(), _team("a", "b", "c", "d", "e", "f"), 4)
    assert seen["belief"] is None
