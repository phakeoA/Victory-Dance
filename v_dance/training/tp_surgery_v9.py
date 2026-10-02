"""tpfeat-v8 -> v9 + species-vocab growth surgery for the TP net (2026-10-02, mega audit gap 3).

The served picker (``checkpoints_set_ots_ctx``, tpfeat-v8, vocab 219) has two blind spots the mega audit
measured: (1) it scores every mega-stone holder by its BASE forme (Golisopod = Bug/Water, Garchomp-Z =
Ground-typed, Spe 102) and (2) it has NO species entry for the M-C newcomers — Rillaboom, Salamence,
Indeedee(-F), Golisopod, Pawmot, Baxcalibur, … — which all collapse onto the zero PAD row (blanking Sneasler's
row halves its bring rate). A from-scratch run cannot reach the lineage (0.264 vs 0.379 set-exact), so this
transplants the donor instead:

  * ``emb.weight`` grows by one ZERO row per new species, donor ids 1..N KEPT (re-running build_vocab would
    re-sort and shift every id). In the donor those species already hit row 0 (zero), so this is exact.
  * ``mon_mlp.0.weight`` (emb | features): the embedding columns copy over; every v8 feature column lands on
    its v9 position (``tp_features.schema_columns("tpfeat-v8")``); the 133 new MEGA-FORME columns start at ZERO.
  * Nothing else changes shape.

Forward parity is CHECKED, not assumed: random v9 features through the patient vs their v8 column view
through the donor — forward() AND score_subsets() (set head) with teammate affinity and Bo3 set context, at
atol 1e-5. Then fine-tune it (the command is printed at the end).

Usage:
  .venv/Scripts/python.exe -m v_dance.training.tp_surgery_v9 \\
      --donor ai_train_scripts/teamPreview_model/checkpoints_set_ots_ctx/teampreview_sbda.pt \\
      --out-dir ai_train_scripts/teamPreview_model/checkpoints_set_v9_surgery
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]

import v_dance.training.tp_features as TF          # noqa: E402

_DEFAULT_DONOR = "ai_train_scripts/teamPreview_model/checkpoints_set_ots_ctx/teampreview_sbda.pt"
_DEFAULT_CORPUS = ("data/vods/Prepared_training_data/Regulation_MC/Jsonl_TypeB",
                   "data/vods/Prepared_training_data/Regulation_MC/Jsonl_TypeC")
_DEFAULT_ROSTER = "data/champions_roster_regmc.json"


def grow_vocab(vocab: Dict[str, int], new_species: Iterable[str]) -> Tuple[Dict[str, int], List[str]]:
    """Donor ids kept; species not in the vocab appended in sorted order. Returns (vocab, added)."""
    out = dict(vocab)
    nxt = max(out.values(), default=0) + 1
    added = []
    for sp in sorted({s for s in new_species if s and s not in out}):
        out[sp] = nxt
        nxt += 1
        added.append(sp)
    return out, added


def upgrade_state(state: dict, cfg: dict, n_new_species: int) -> Tuple[dict, dict]:
    """The donor's (state, config) reshaped for tpfeat-v9 + ``n_new_species`` zero embedding rows."""
    if cfg.get("feature_schema") != "tpfeat-v8":
        raise ValueError(f"donor schema {cfg.get('feature_schema')!r} — this surgery takes tpfeat-v8")
    cols = TF.schema_columns("tpfeat-v8")
    emb_dim = int(cfg.get("emb_dim", 32))
    w = state["mon_mlp.0.weight"]
    if w.shape[1] != emb_dim + len(cols) or int(cfg["feat_dim"]) != len(cols):
        raise ValueError(f"donor mon_mlp.0 {tuple(w.shape)} / feat_dim {cfg['feat_dim']} != emb {emb_dim} + "
                         f"v8 {len(cols)}")
    w_new = torch.zeros(w.shape[0], emb_dim + TF.FEAT_DIM, dtype=w.dtype)
    w_new[:, :emb_dim] = w[:, :emb_dim]
    w_new[:, emb_dim + torch.as_tensor(cols)] = w[:, emb_dim:]
    emb = state["emb.weight"]
    emb_new = torch.cat([emb, torch.zeros(n_new_species, emb.shape[1], dtype=emb.dtype)], dim=0)
    st = dict(state)
    st["mon_mlp.0.weight"] = w_new
    st["emb.weight"] = emb_new
    c = dict(cfg)
    c["feat_dim"] = TF.FEAT_DIM
    c["feature_schema"] = TF.FEATURE_SCHEMA_VERSION
    c["vocab_size"] = int(emb_new.shape[0])
    c["surgery"] = {"from_schema": "tpfeat-v8", "from_vocab_size": int(emb.shape[0]),
                    "added_species": int(n_new_species), "note": "tp_surgery_v9 (mega audit gap 3, 2026-10-02)"}
    return st, c


def _build(cfg: dict, state: dict):
    from v_dance.models.teampreview_model import build_model
    m = build_model(cfg["vocab_size"], cfg["feat_dim"], cfg.get("emb_dim", 32), cfg.get("hidden", 128),
                    cfg.get("dropout", 0.0), use_self_attn=cfg.get("use_self_attn", False),
                    use_cross_attn=cfg.get("use_cross_attn", False), attn_heads=cfg.get("attn_heads", 4),
                    use_teammate_bias=cfg.get("use_teammate_bias", False),
                    use_set_head=cfg.get("use_set_head", False), use_set_ctx=cfg.get("use_set_ctx", False))
    m.load_state_dict(state)
    return m.eval()


def parity_check(donor_cfg: dict, donor_state: dict, cfg: dict, state: dict, seed: int = 0) -> float:
    """Max |Δ| between the donor on v8 views and the patient on full v9 vectors (raises over 1e-5)."""
    donor, patient = _build(donor_cfg, donor_state), _build(cfg, state)
    cols = torch.as_tensor(TF.schema_columns("tpfeat-v8"))
    g = torch.Generator().manual_seed(seed)
    B = 3
    f9o = torch.rand(B, 6, TF.FEAT_DIM, generator=g)
    f9p = torch.rand(B, 6, TF.FEAT_DIM, generator=g)
    # donor-known ids only on the donor side; the patient also sees NEW ids, whose rows are zero = donor's PAD
    n_old = int(donor_cfg["vocab_size"])
    oi = torch.randint(1, n_old, (B, 6), generator=g)
    pi = torch.randint(1, int(cfg["vocab_size"]), (B, 6), generator=g)
    pi_donor = torch.where(pi >= n_old, torch.zeros_like(pi), pi)
    kw_p, kw_d = {}, {}
    if cfg.get("use_teammate_bias"):
        aff = torch.rand(B, 6, 6, generator=g)
        kw_p["our_affinity"] = kw_d["our_affinity"] = aff
    if cfg.get("use_set_ctx"):
        ctx = (torch.rand(B, 6, 2, generator=g) > 0.5).float()
        kw_p["our_set_ctx"] = kw_d["our_set_ctx"] = ctx
        kw_p["opp_set_ctx"] = kw_d["opp_set_ctx"] = ctx.flip(1)
    worst = 0.0
    with torch.no_grad():
        bp, lp = patient(oi, pi, f9o, f9p, **kw_p)
        bd, ld = donor(oi, pi_donor, f9o[..., cols], f9p[..., cols], **kw_d)
        worst = max(worst, (bp - bd).abs().max().item(), (lp - ld).abs().max().item())
        if cfg.get("use_set_head"):
            sp, _, _ = patient.score_subsets(oi, pi, f9o, f9p, **kw_p)
            sd, _, _ = donor.score_subsets(oi, pi_donor, f9o[..., cols], f9p[..., cols], **kw_d)
            worst = max(worst, (sp - sd).abs().max().item())
    if worst > 1e-5:
        raise AssertionError(f"forward parity FAILED (max |Δ| {worst:.2e}) — the column/vocab remap is wrong")
    return worst


def corpus_species(folders: Sequence[str]) -> set:
    """Every species id in the (our + opp) preview rosters of the jsonl corpora (cheap legacy features)."""
    from v_dance.training.teampreview_dataset import examples_from_folders
    out: set = set()
    for d in folders:
        p = Path(d) if Path(d).is_absolute() else REPO / d
        if not p.exists():
            print(f"[surgery] WARNING corpus folder missing: {p}", file=sys.stderr)
            continue
        exs, _ = examples_from_folders([str(p)])
        for e in exs:
            out.update(e["our_species"])
            out.update(e["opp_species"])
    return out


def roster_species(path: str) -> set:
    """The format roster's species ids, MEGA formes excluded (the picker sees pre-mega species only)."""
    p = Path(path) if Path(path).is_absolute() else REPO / path
    if not p.exists():
        return set()
    from v_dance.dex.pokedex import get_pokedex, norm_species
    dex = get_pokedex()
    ids = json.loads(p.read_text(encoding="utf-8"))
    return {norm_species(s) for s in ids if not (dex and dex.is_mega_forme(s))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--donor", default=_DEFAULT_DONOR, help="a tpfeat-v8 TP checkpoint (default: the served one)")
    ap.add_argument("--out-dir", default="ai_train_scripts/teamPreview_model/checkpoints_set_v9_surgery")
    ap.add_argument("--corpus", nargs="*", default=list(_DEFAULT_CORPUS),
                    help="jsonl folders whose preview species join the vocab (default: Reg M-C Type B + C)")
    ap.add_argument("--roster", default=_DEFAULT_ROSTER, help="format roster json (its non-mega ids join too)")
    args = ap.parse_args(argv)

    donor_path = Path(args.donor) if Path(args.donor).is_absolute() else REPO / args.donor
    ck = torch.load(donor_path, map_location="cpu", weights_only=False)
    dcfg, dstate, dvocab = dict(ck["config"]), ck["model_state"], dict(ck["vocab"])
    new = corpus_species(args.corpus) | roster_species(args.roster)
    vocab, added = grow_vocab(dvocab, new)
    state, cfg = upgrade_state(dstate, dcfg, len(added))
    worst = parity_check(dcfg, dstate, cfg, state)
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "teampreview_sbda.pt"
    torch.save({"model_state": state, "config": cfg, "vocab": vocab}, out)
    print(f"[surgery] forward parity OK (max |Δ| {worst:.2e}, forward + set head)")
    print(f"[surgery] {donor_path.name}: tpfeat-v8/{dcfg['feat_dim']} vocab {len(dvocab)} -> {out}")
    print(f"[surgery]   tpfeat-v9/{cfg['feat_dim']} (+{cfg['feat_dim'] - dcfg['feat_dim']} zero mega-forme "
          f"columns), vocab {len(vocab)} (+{len(added)} zero rows)")
    print(f"[surgery]   added: {', '.join(added)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
