"""tpfeat-v7 -> v8 warm-start surgery for the TP net (2026-07-23).

The v8 schema (FEAT_DIM 253 -> 285) only widens the FIRST Linear (mon_mlp.0 reads
emb32 + feat). Every other tensor is shape-identical, so instead of a from-scratch
run (RED ruler: set exact 0.264 vs the n3_rerun lineage's 0.379) we transplant the
donor and remap mon_mlp.0's feature columns segment-by-segment:

  * SHARED segments (dex/weather/terrain/roles-prefix/spread/immune/.../overlay
    twins) -> columns copied to their new offsets. The canon-casing fix changed
    those channels' VALUES, but the donor's weights on them are exactly the
    inheritance we want (Charizard's now-lit sun tag hits trained sun weights).
  * NEW v8 channels (intim_*, prio_block, weather_negate, sleep, phys_share,
    exp_speed, items, wide_guard/quick_guard/trapping + K-twins) -> zero-init
    (inert until the fine-tune lights them up).

Validated by a forward-parity check: a v7 feature vector remapped into the v8
layout must produce IDENTICAL logits through the surgery net.

Usage: .venv/Scripts/python.exe -m v_dance.training.tp_surgery
(moved from scratch/tp_v8_warmstart_surgery.py in the 2026-09-10 refactor, Phase 1)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]           # v_dance/training/ -> repo root
assert (REPO / "pyproject.toml").is_file(), REPO

import v_dance.training.tp_features as V8          # noqa: E402
import v_dance.training.tp_features_v7 as V7       # noqa: E402

DONOR = REPO / "ai_train_scripts/teamPreview_model/checkpoints_set_n3_rerun/teampreview_sbda_last.pt"
OUT_DIR = REPO / "ai_train_scripts/teamPreview_model/checkpoints_set_v8_surgery"
EMB = 32   # donor emb_dim — asserted against its config below


def feature_index_map():
    """[(old_idx, new_idx)] for every v7 channel's position in the v8 layout."""
    _T, _W, _TR, _O, _G = V7.NUM_TYPES, 4, 4, 2, 4
    R7 = len(V7.ROLE_TAGS)                       # 8 — v8 keeps them as a prefix (asserted)
    assert V8.ROLE_TAGS[:R7] == V7.ROLE_TAGS
    assert V8.GIMMICK_KINDS == V7.GIMMICK_KINDS and V8.ORDER_FLAGS == V7.ORDER_FLAGS
    segs = [  # (v7 offset, v8 offset, length)
        (V7.OFF_DEX, V8.OFF_DEX, V7.MON_FEAT_DIM),
        (V7.OFF_WSETS, V8.OFF_WSETS, _W), (V7.OFF_WABUSE, V8.OFF_WABUSE, _W),
        (V7.OFF_TSETS, V8.OFF_TSETS, _TR), (V7.OFF_TABUSE, V8.OFF_TABUSE, _TR),
        (V7.OFF_ROLES, V8.OFF_ROLES, R7),
        (V7.OFF_SPREAD, V8.OFF_SPREAD, _T), (V7.OFF_IMMUNE, V8.OFF_IMMUNE, _T),
        (V7.OFF_REVERSER, V8.OFF_REVERSER, 1), (V7.OFF_DEBUFF, V8.OFF_DEBUFF, 1),
        (V7.OFF_ORDER, V8.OFF_ORDER, _O),
        (V7.OFF_DEFEFF, V8.OFF_DEFEFF, _T),
        (V7.OFF_HASDATA, V8.OFF_HASDATA, 1), (V7.OFF_USAGE, V8.OFF_USAGE, 1),
        (V7.OFF_GK, V8.OFF_GK, _G), (V7.OFF_TERA, V8.OFF_TERA, _T),
        (V7.OFF_OWNBIT, V8.OFF_OWNBIT, 1),
        (V7.OFF_KWSETS, V8.OFF_KWSETS, _W), (V7.OFF_KWABUSE, V8.OFF_KWABUSE, _W),
        (V7.OFF_KTSETS, V8.OFF_KTSETS, _TR), (V7.OFF_KTABUSE, V8.OFF_KTABUSE, _TR),
        (V7.OFF_KROLES, V8.OFF_KROLES, R7),
        (V7.OFF_KSPREAD, V8.OFF_KSPREAD, _T), (V7.OFF_KIMMUNE, V8.OFF_KIMMUNE, _T),
        (V7.OFF_KREVERSER, V8.OFF_KREVERSER, 1), (V7.OFF_KDEBUFF, V8.OFF_KDEBUFF, 1),
        (V7.OFF_KORDER, V8.OFF_KORDER, _O),
        (V7.OFF_KGK, V8.OFF_KGK, _G), (V7.OFF_KTERA, V8.OFF_KTERA, _T),
    ]
    pairs = [(o + i, n + i) for o, n, ln in segs for i in range(ln)]
    olds = [p[0] for p in pairs]
    assert len(set(olds)) == V7.FEAT_DIM == len(olds), "v7 must be covered exactly once"
    assert len({p[1] for p in pairs}) == len(pairs) <= V8.FEAT_DIM
    return pairs


def remap_feature_vec(f7: np.ndarray, pairs) -> np.ndarray:
    f8 = np.zeros(V8.FEAT_DIM, dtype=np.float32)
    for o, n in pairs:
        f8[n] = f7[o]
    return f8


def main():
    ckpt = torch.load(DONOR, map_location="cpu", weights_only=False)
    cfg, state, vocab = dict(ckpt["config"]), ckpt["model_state"], ckpt["vocab"]
    assert cfg["feature_schema"] == "tpfeat-v7" and cfg["feat_dim"] == V7.FEAT_DIM
    assert cfg.get("emb_dim", 32) == EMB
    pairs = feature_index_map()

    w = state["mon_mlp.0.weight"]                    # (hidden, EMB + 253)
    assert w.shape[1] == EMB + V7.FEAT_DIM
    w_new = torch.zeros(w.shape[0], EMB + V8.FEAT_DIM, dtype=w.dtype)
    w_new[:, :EMB] = w[:, :EMB]                      # species-embedding columns
    for o, n in pairs:
        w_new[:, EMB + n] = w[:, EMB + o]
    state = dict(state)
    state["mon_mlp.0.weight"] = w_new

    cfg["feat_dim"] = V8.FEAT_DIM
    cfg["feature_schema"] = V8.FEATURE_SCHEMA_VERSION

    # ── forward-parity check: v7 features remapped through the surgery net must
    # reproduce the donor's logits exactly ─────────────────────────────────────
    from v_dance.models.teampreview_model import build_model
    def _mk(feat_dim, st):
        m = build_model(cfg["vocab_size"], feat_dim, cfg.get("emb_dim", 32),
                        cfg.get("hidden", 128), cfg.get("dropout", 0.0),
                        use_self_attn=cfg.get("use_self_attn", False),
                        use_cross_attn=cfg.get("use_cross_attn", False),
                        attn_heads=cfg.get("attn_heads", 4),
                        use_teammate_bias=cfg.get("use_teammate_bias", False),
                        use_set_head=cfg.get("use_set_head", False),
                        use_set_ctx=cfg.get("use_set_ctx", False))
        m.load_state_dict(st)
        return m.eval()
    donor = _mk(V7.FEAT_DIM, ckpt["model_state"])
    patient = _mk(V8.FEAT_DIM, state)
    rng = np.random.default_rng(0)
    f7 = rng.random((2, 6, V7.FEAT_DIM)).astype(np.float32)
    f8 = np.stack([[remap_feature_vec(v, pairs) for v in b] for b in f7])
    idx = torch.randint(1, cfg["vocab_size"], (2, 6), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        b7, l7 = donor(idx, idx, torch.tensor(f7), torch.tensor(f7))
        b8, l8 = patient(idx, idx, torch.tensor(f8), torch.tensor(f8))
    assert torch.allclose(b7, b8, atol=1e-5) and torch.allclose(l7, l8, atol=1e-5), \
        "forward parity FAILED — column remap is wrong"
    print("[surgery] forward parity OK (max diff "
          f"{(b7 - b8).abs().max().item():.2e} / {(l7 - l8).abs().max().item():.2e})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "teampreview_sbda.pt"
    torch.save({"model_state": state, "config": cfg, "vocab": vocab}, out)
    print(f"[surgery] donor {DONOR.name} (v7/{V7.FEAT_DIM}) -> {out} "
          f"(v8/{V8.FEAT_DIM}); new channels zero-init: "
          f"{V8.FEAT_DIM - V7.FEAT_DIM} columns")


if __name__ == "__main__":
    main()
