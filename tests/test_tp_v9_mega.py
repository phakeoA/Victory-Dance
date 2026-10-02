"""tpfeat-v9: the team picker sees the MEGA FORME a stone makes, and the M-C species (mega audit gap 3, 2026-10-02).

The served picker (tpfeat-v8) scored every stone holder as its BASE forme — Golisopod Bug/Water, Garchomp-Z
Ground-typed at Spe 102 — and had no species row for Salamence / Golisopod / Rillaboom / Indeedee. v9 adds a
stone-share-weighted mega-forme block (+ known-stone twins), keeps every v8 channel's VALUE, and serves a v8
checkpoint through ``schema_columns("tpfeat-v8")``; ``tp_surgery_v9`` transplants the served net exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from v_dance.training import tp_features as T  # noqa: E402
from v_dance.training import tp_surgery_v9 as S  # noqa: E402


class _Belief:
    """In-memory belief with ITEM shares (the stone channel's input)."""
    def __init__(self, items=None, ability=(), usage=10.0):
        self._items = items or {}
        self._ab = [{"name": n, "p": p} for n, p in ability]
        self._u = usage

    def known(self, s): return True
    def ability_distribution(self, s, top_k=4): return self._ab[:top_k]
    def move_distribution(self, s, top_k=12): return []
    def usage(self, s): return self._u
    def item_distribution(self, s, top_k=6):
        return [{"name": n, "p": p} for n, p in self._items.get(s, [])][:top_k]


def _types(f, off):
    return {T.TYPE_NAMES[i] if hasattr(T, "TYPE_NAMES") else i: round(float(f[off + i]), 3)
            for i in np.nonzero(f[off:off + T.NUM_TYPES])[0]}


def _ti(name):
    from v_dance.training.teampreview_dataset import _TYPE_IDX
    return _TYPE_IDX[name]


def test_golisopod_reads_bug_steel_after_its_stone():
    f = T.opp_mon_features("golisopod", _Belief({"golisopod": [("Golisopite", 0.98)]}))
    assert f[T.OFF_MTYPE + _ti("BUG")] == pytest.approx(0.98)
    assert f[T.OFF_MTYPE + _ti("STEEL")] == pytest.approx(0.98)
    assert f[T.OFF_MTYPE + _ti("WATER")] == 0.0
    # the base (pre-mega) block is untouched: still Bug/Water
    assert f[T.OFF_DEX + _ti("BUG")] + f[T.OFF_DEX + T.NUM_TYPES + _ti("BUG")] >= 1.0
    # Bug/Steel resists Psychic / is weak to Fire only — the signed def-eff of the forme, share-weighted
    assert f[T.OFF_MDEFEFF + _ti("FIRE")] > 0 and f[T.OFF_MDEFEFF + _ti("ELECTRIC")] == pytest.approx(0.0)
    assert f[T.OFF_MIMMUNE + _ti("POISON")] == pytest.approx(0.98)        # Steel typing: Poison 0x


def test_garchomp_z_and_regular_stone_weigh_their_own_formes():
    b = _Belief({"garchomp": [("Garchompite Z", 0.45), ("Garchompite", 0.02)]})
    f = T.opp_mon_features("garchomp", b)
    assert f[T.OFF_MTYPE + _ti("DRAGON")] == pytest.approx(0.47, abs=1e-5)
    assert f[T.OFF_MTYPE + _ti("GROUND")] == pytest.approx(0.02, abs=1e-5)   # only the regular mega keeps Ground
    assert f[T.OFF_MIMMUNE + _ti("GROUND")] == pytest.approx(0.45, abs=1e-5)  # Levitate (Z)
    assert f[T.OFF_MIMMUNE + _ti("ELECTRIC")] == pytest.approx(0.02, abs=1e-5)
    assert f[T.OFF_GK + T.GIMMICK_KINDS.index("mega")] == pytest.approx(0.47, abs=1e-5)


def test_one_stone_counts_once_for_meowstic():
    f = T.opp_mon_features("meowsticf", _Belief({"meowsticf": [("Meowsticite", 0.6)]}))
    assert f[T.OFF_GK + T.GIMMICK_KINDS.index("mega")] <= 0.6 + 1e-6      # not 1.2 (M-Mega + F-Mega)
    assert float(f[T.OFF_MTYPE:T.OFF_MTYPE + T.NUM_TYPES].max()) <= 0.6 + 1e-6
    assert T.mega_forme_for_stone("meowsticf", "Meowsticite")[0].lower().replace("-", "") == "meowsticfmega"
    assert T.mega_forme_for_stone("meowstic", "Meowsticite")[0].lower().replace("-", "") == "meowsticmmega"


def test_no_stone_leaves_the_mega_block_zero():
    f = T.opp_mon_features("incineroar", _Belief({"incineroar": [("Sitrus Berry", 0.5)]}))
    assert float(np.abs(f[T.OFF_MTYPE:T.BASE_DIM]).sum()) == 0.0


def test_known_stone_rides_the_overlay_twins_and_closed_sheets_stay_zero():
    b = _Belief({"golisopod": [("Golisopite", 0.98)]})
    closed = T.opp_mon_features("golisopod", b)
    assert float(np.abs(closed[T.OFF_OWNBIT:]).sum()) == 0.0
    sheet = T.opp_mon_features("golisopod", b, revealed=T.OwnKnown(ability="Emergency Exit", item="Golisopite"))
    assert sheet[T.OFF_KMTYPE + _ti("STEEL")] == 1.0
    assert sheet[T.OFF_KMSTATS + 1] > closed[T.OFF_DEX + 2 * T.NUM_TYPES + 1]   # mega Atk 150 > base 125
    assert np.array_equal(sheet[:T.OFF_OWNBIT], closed[:T.OFF_OWNBIT])        # base untouched by the overlay


def test_v8_view_is_the_v9_vector_minus_the_mega_blocks():
    b = _Belief({"golisopod": [("Golisopite", 0.98)]})
    f = T.own_mon_features("golisopod", b, T.OwnKnown(item="Golisopite"))
    v8 = T.view_for_schema(f, "tpfeat-v8")
    assert v8.shape == (285,)
    assert np.array_equal(v8, np.concatenate([f[:T.OFF_MTYPE], f[T.BASE_DIM:T.OFF_KMTYPE]]))
    assert T.view_for_schema(f, T.FEATURE_SCHEMA_VERSION) is f
    with pytest.raises(ValueError):
        T.schema_columns("tpfeat-v5")


# ── surgery: exact forward parity on a tiny v8 donor (set head + Bo3 ctx + teammate bias) ──
def _tiny_v8(vocab_size=10):
    from v_dance.models.teampreview_model import build_model
    cfg = {"vocab_size": vocab_size, "feat_dim": 285, "emb_dim": 8, "hidden": 16, "dropout": 0.0,
           "use_self_attn": True, "use_cross_attn": True, "attn_heads": 2, "use_teammate_bias": True,
           "use_set_head": True, "use_set_ctx": True, "feature_schema": "tpfeat-v8"}
    torch.manual_seed(7)
    m = build_model(vocab_size, 285, 8, 16, 0.0, use_self_attn=True, use_cross_attn=True, attn_heads=2,
                    use_teammate_bias=True, use_set_head=True, use_set_ctx=True)
    with torch.no_grad():                         # PAD row stays zero (the trainer's invariant)
        m.emb.weight[0].zero_()
    return cfg, m.state_dict()


def test_surgery_is_exact_and_grows_the_vocab():
    cfg, st = _tiny_v8()
    vocab = {f"sp{i}": i for i in range(1, 10)}
    v2, added = S.grow_vocab(vocab, ["salamence", "golisopod", "sp3"])
    assert added == ["golisopod", "salamence"] and v2["sp3"] == 3 and v2["golisopod"] == 10
    st2, cfg2 = S.upgrade_state(st, cfg, len(added))
    assert cfg2["feat_dim"] == T.FEAT_DIM and cfg2["feature_schema"] == T.FEATURE_SCHEMA_VERSION
    assert cfg2["vocab_size"] == 12 and st2["emb.weight"].shape[0] == 12
    assert float(st2["emb.weight"][10:].abs().sum()) == 0.0
    assert S.parity_check(cfg, st, cfg2, st2) < 1e-5


def test_surgery_refuses_a_non_v8_donor():
    cfg, st = _tiny_v8()
    with pytest.raises(ValueError):
        S.upgrade_state(st, dict(cfg, feature_schema="tpfeat-v7"), 0)


# ── serve: a v8 checkpoint packs 285-wide rows, a v9 one 418-wide ─────────────
def _save(tmp_path, schema, feat_dim):
    from v_dance.models.teampreview_model import TeamPreviewModel
    m = TeamPreviewModel(vocab_size=8, feat_dim=feat_dim, emb_dim=8, hidden=16, dropout=0.0)
    p = tmp_path / f"{schema}.pt"
    torch.save({"model_state": m.state_dict(), "vocab": {"golisopod": 1},
                "config": {"vocab_size": 8, "feat_dim": feat_dim, "emb_dim": 8, "hidden": 16,
                           "dropout": 0.0, "feature_schema": schema}}, p)
    return p


def test_model_io_serves_v8_and_v9(tmp_path):
    from v_dance.play.model_io import _pack_side, load_team_chooser
    b = _Belief({"golisopod": [("Golisopite", 0.98)]})
    for schema, dim in (("tpfeat-v8", 285), (T.FEATURE_SCHEMA_VERSION, T.FEAT_DIM)):
        _m, vocab, cfg = load_team_chooser(str(_save(tmp_path, schema, dim)))
        idx, feat = _pack_side(["golisopod"], vocab, cfg["feat_dim"], belief=b, use_tp_features=True,
                               tp_schema=cfg["feature_schema"])
        assert feat.shape == (6, dim) and idx[0] == 1
    with pytest.raises(ValueError, match="lockstep"):
        load_team_chooser(str(_save(tmp_path, "tpfeat-v8", T.FEAT_DIM)))     # v8 stamp, v9 width
