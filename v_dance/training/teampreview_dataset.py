"""
Team-preview ("bring") dataset for Victory-Dance (VGC Reg M-A, doubles).

In VGC you see both teams' SIX Pokémon at team preview and must choose FOUR to
bring (with the first two on the field as your leads).  This is a one-shot,
matchup-level decision with a completely different shape from the per-turn
in-battle policy (see ../BC_model), so it gets its own pipeline.

The labels already live in every exported transition's ``players`` block — no
re-export is needed:

    players.{side}.roster   : all 6 (team-preview order)
    players.{side}.brought  : the mons that actually appeared (first two = leads)

Team-preview info is constant across a replay, so we read ONE line per file and
emit TWO examples (p1 POV and p2 POV).  Each example:

    our_species / opp_species : 6 normalised species ids each (the matchup input)
    our_feat / opp_feat       : per-mon dex features (types + base stats)
    bring  : multi-hot(6)  — which of our 6 were brought
    lead   : multi-hot(6)  — which of our 6 led (the first two brought)
    valid_bring : 1.0 only when a CLEAN 4-bring was observed

IMPORTANT label caveat (verified on the corpus): ``brought`` records mons that
*appeared*, a lower bound on what was actually brought.  Leads (the first two)
are revealed turn 1 and are ALWAYS reliable; the full 4-set is reliable only
when 4 distinct mons appeared (~80% of games).  So the lead target is trained on
every example, the bring-4 target only on ``valid_bring`` ones.
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Bootstrap: locate data/scripts by walking up (folder-depth independent) ───
def _find_scripts_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        cand = parent / "data" / "scripts"
        if cand.is_dir():
            return cand
    raise RuntimeError(f"could not locate data/scripts above {__file__}")

_SCRIPTS_DIR = _find_scripts_dir()
from v_dance.parser.vod_parser.pokedex import get_pokedex, norm_species  # noqa: E402
from v_dance.parser.belief_state import dex_base_stats, STAT_ORDER         # noqa: E402
from v_dance.encoders.state_encoder import TYPE_NAMES                        # noqa: E402

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    _HAS_TORCH = False

TEAM_SIZE = 6
BRING_K = 4
LEAD_K = 2
PAD_IDX = 0  # reserved vocab slot for padding / unseen species

# ── Per-mon dex feature vector (types one-hot + base stats) ───────────────────
# NOTE (#A, 2026-06-17): a belief-most-likely-ability multi-hot block was trialled
# here, but a clean same-recipe A/B showed NO improvement (mean-exact 0.221 with vs
# 0.228 without) — the learned species embedding already captures a species' typical
# ability, so the explicit block was redundant.  Reverted; see the memory note.
_TYPE_IDX = {n: i for i, n in enumerate(TYPE_NAMES)}
NUM_TYPES = len(TYPE_NAMES)
MON_FEAT_DIM = NUM_TYPES * 2 + 6
_NON_ALNUM = re.compile(r"[^A-Z0-9_]")


def _canon_type(name: Optional[str]) -> str:
    if not name:
        return ""
    return _NON_ALNUM.sub("", str(name).upper().replace(" ", "_"))


def mon_dex_features(species: Optional[str]) -> np.ndarray:
    """Fixed (MON_FEAT_DIM,) vector: type1 one-hot | type2 one-hot | base/255."""
    feat = np.zeros(MON_FEAT_DIM, dtype=np.float32)
    if not species:
        return feat
    dex = get_pokedex()
    entry = dex.entry(species) if dex else None
    types = [_canon_type(t) for t in ((entry or {}).get("types") or [])]
    if types and types[0] in _TYPE_IDX:
        feat[_TYPE_IDX[types[0]]] = 1.0
    if len(types) > 1 and types[1] in _TYPE_IDX:
        feat[NUM_TYPES + _TYPE_IDX[types[1]]] = 1.0
    base = dex_base_stats(species) or {}
    for j, k in enumerate(STAT_ORDER):
        feat[NUM_TYPES * 2 + j] = (base.get(k, 0) or 0) / 255.0
    return feat


# ── SBDA feature recipe (15b-data.1) — the SAME extractor model_io serves ─────
def sbda_feature_fn(belief) -> Callable[[Optional[str]], np.ndarray]:
    """Closure ``species -> (FEAT_DIM,)`` for the SBDA TP recipe — IDENTICAL to the
    serve-side ``model_io._pack_side`` (``tp_features.opp_mon_features`` with the OWN
    overlay OFF: a Type-B replay carries no sharp own build, so ``has_own_detail`` is 0
    on both sides — exactly what a BC-pretrained net serves).  This shared CALL — not
    merely a shared dim — is the train==serve lockstep the 15b plumbing exists to
    guarantee.  Imported lazily to avoid a tp_features <-> teampreview_dataset cycle."""
    from v_dance.training.tp_features import opp_mon_features
    return lambda species: opp_mon_features(norm_species(species), belief)


def feature_recipe(mode: str = "legacy", belief=None):
    """``(feat_fn, feat_dim, feature_schema)`` for a TP feature recipe — the single place
    the train side chooses its per-mon features, mirroring ``model_io.uses_tp_features``.

    ``mode='legacy'`` -> dex-only (``MON_FEAT_DIM``=46, schema ``None``) — the original net.
    ``mode='sbda'``   -> the shared ``tp_features`` extractor (FEAT_DIM, schema ``tpfeat-vN``);
    requires a Pikalytics ``belief``.  The trainer (15b-train.1) stamps ``feat_dim`` +
    ``feature_schema`` into the checkpoint config so ``model_io`` selects the matching
    SERVE recipe and the load-time lockstep guard can reject a schema drift."""
    if mode == "sbda":
        from v_dance.training.tp_features import FEAT_DIM, FEATURE_SCHEMA_VERSION
        if belief is None:
            raise ValueError("the 'sbda' feature recipe requires a BeliefState (Pikalytics prior)")
        return sbda_feature_fn(belief), FEAT_DIM, FEATURE_SCHEMA_VERSION
    if mode == "legacy":
        return mon_dex_features, MON_FEAT_DIM, None
    raise ValueError(f"unknown TP feature mode {mode!r} (expected 'legacy' or 'sbda')")


# ══════════════════════════════════════════════════════════════════════════════
# Raw example extraction (torch-free)
# ══════════════════════════════════════════════════════════════════════════════
def iter_jsonl_files(folder: str, recursive: bool = True) -> List[str]:
    folder = str(folder)
    pattern = os.path.join(folder, "**", "*.jsonl") if recursive else os.path.join(folder, "*.jsonl")
    return sorted(glob.glob(pattern, recursive=recursive))


def _first_transition(path: str) -> Optional[dict]:
    """Team-preview is constant per replay → the first non-empty line suffices."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def _label_indices(roster_norm: List[str], picks: Sequence[str]) -> Tuple[List[int], int]:
    """Map brought species to their roster positions (by normalised species).

    Returns (ordered roster indices for the picks, n_unmatched)."""
    idxs: List[int] = []
    unmatched = 0
    used: set = set()
    for sp in picks:
        want = norm_species(sp)
        found = None
        for i, rsp in enumerate(roster_norm):
            if i not in used and rsp == want:
                found = i
                break
        if found is None:
            unmatched += 1
        else:
            used.add(found)
            idxs.append(found)
    return idxs, unmatched


def _example_for_side(t: dict, side: str, opp: str,
                      stats: Optional[Counter],
                      feat_fn: Callable = mon_dex_features) -> Optional[dict]:
    players = t.get("players") or {}
    me = players.get(side) or {}
    them = players.get(opp) or {}
    roster = me.get("roster") or []
    opp_roster = them.get("roster") or []
    brought = me.get("brought") or []
    if len(roster) != TEAM_SIZE or len(opp_roster) != TEAM_SIZE or len(brought) < LEAD_K:
        if stats is not None:
            stats["skipped_incomplete"] += 1
        return None

    roster_norm = [norm_species(s) for s in roster]
    brought_idx, unmatched = _label_indices(roster_norm, brought)
    if stats is not None and unmatched:
        stats["unmatched_picks"] += unmatched

    bring = np.zeros(TEAM_SIZE, dtype=np.float32)
    for i in brought_idx:
        bring[i] = 1.0
    # audit: derive the lead from the FIRST LEAD_K brought picks matched INDEPENDENTLY — NOT brought_idx[:K],
    # which is the COMPACTED full-pick list (unmatched picks dropped). If brought[0] fails species matching,
    # brought_idx[:K] would point at picks #2/#3 → the lead head learns the WRONG pair. Mark the lead VALID
    # only when the first LEAD_K picks all matched in order; train_teampreview masks an invalid lead loss.
    lead_idx, lead_unmatched = _label_indices(roster_norm, list(brought)[:LEAD_K])
    lead = np.zeros(TEAM_SIZE, dtype=np.float32)
    for i in lead_idx:
        lead[i] = 1.0
    valid_lead = 1.0 if (lead_unmatched == 0 and len(lead_idx) == LEAD_K) else 0.0

    valid_bring = 1.0 if (len(brought) == BRING_K and int(bring.sum()) == BRING_K) else 0.0
    if stats is not None:
        stats["examples"] += 1
        stats["valid_bring"] += int(valid_bring)
        stats["valid_lead"] += int(valid_lead)

    return {
        "our_species": roster_norm,
        "opp_species": [norm_species(s) for s in opp_roster],
        "our_feat": np.stack([feat_fn(s) for s in roster]),
        "opp_feat": np.stack([feat_fn(s) for s in opp_roster]),
        "bring": bring,
        "lead": lead,
        "valid_bring": valid_bring,
        "valid_lead": valid_lead,
        "replay_id": t.get("replay_id"),
        "side": side,
    }


def build_examples(
    files: Sequence[str],
    limit_files: Optional[int] = None,
    sides: Sequence[str] = ("p1", "p2"),
    feat_fn: Callable = mon_dex_features,
) -> Tuple[List[dict], Counter]:
    """Two examples (p1, p2) per replay file.  ``feat_fn`` chooses the per-mon feature
    recipe (default legacy dex-only; pass ``sbda_feature_fn(belief)`` for the SBDA net)."""
    stats: Counter = Counter()
    examples: List[dict] = []
    if limit_files is not None:
        files = list(files)[:limit_files]
    for fp in files:
        stats["files"] += 1
        try:
            t = _first_transition(fp)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            stats["bad_files"] += 1
            print(f"[teampreview_dataset] WARNING skipping {fp}: {exc}", file=sys.stderr)
            continue
        if not t:
            continue
        for side, opp in (("p1", "p2"), ("p2", "p1")):
            if side not in sides:
                continue
            ex = _example_for_side(t, side, opp, stats, feat_fn)
            if ex is not None:
                examples.append(ex)
    stats["replays"] = len({e["replay_id"] for e in examples})
    return examples, stats


def examples_from_folders(
    folders: Sequence[str],
    recursive: bool = True,
    limit_files: Optional[int] = None,
    feat_fn: Callable = mon_dex_features,
) -> Tuple[List[dict], Counter]:
    files: List[str] = []
    seen: set = set()
    for folder in folders:
        for f in iter_jsonl_files(folder, recursive=recursive):
            key = os.path.abspath(f)
            if key not in seen:
                seen.add(key)
                files.append(f)
    return build_examples(files, limit_files=limit_files, feat_fn=feat_fn)


# ── Vocabulary ────────────────────────────────────────────────────────────────
def build_vocab(examples: Sequence[dict]) -> Dict[str, int]:
    """Map every species seen (our + opp rosters) to an id; 0 reserved (PAD)."""
    species = set()
    for ex in examples:
        species.update(ex["our_species"])
        species.update(ex["opp_species"])
    vocab = {sp: i + 1 for i, sp in enumerate(sorted(species))}
    return vocab


def split_by_replay(examples: Sequence[dict], val_frac: float = 0.1,
                    seed: int = 0) -> Tuple[List[dict], List[dict]]:
    """Split by replay_id so a game's p1 and p2 examples stay on one side."""
    ids = sorted({e["replay_id"] for e in examples})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_frac))
    n_val = min(max(n_val, 1 if len(ids) > 1 else 0), len(ids))
    val_ids = set(ids[:n_val])
    train = [e for e in examples if e["replay_id"] not in val_ids]
    val = [e for e in examples if e["replay_id"] in val_ids]
    return train, val


# ══════════════════════════════════════════════════════════════════════════════
# Torch Dataset
# ══════════════════════════════════════════════════════════════════════════════
class TeamPreviewDataset(Dataset):
    """Tensor-backed dataset; each item is a dict of fixed-shape tensors.

        our_idx / opp_idx : int64 (6,)        species vocab ids
        our_feat/opp_feat : float32 (6, F)    dex features
        bring / lead      : float32 (6,)      multi-hot targets
        valid_bring       : float32 ()        1.0 if the 4-bring is complete
    """

    def __init__(self, examples: Sequence[dict], vocab: Dict[str, int],
                 feat_dim: Optional[int] = None,
                 affinity_fn: Optional[Callable] = None):
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("torch is required for TeamPreviewDataset")
        n = len(examples)
        # feat_dim follows the example recipe (46 legacy dex / FEAT_DIM SBDA), inferred
        # from the data unless pinned, so one Dataset class serves both nets.
        if feat_dim is None:
            feat_dim = int(examples[0]["our_feat"].shape[1]) if examples else MON_FEAT_DIM
        self.feat_dim = feat_dim
        # 15b-train.1: when the SBDA net uses the teammate-bias, the TRAIN affinity matrix must be the
        # SAME teammate_affinity_matrix(our_species, belief) serve feeds model.forward — precompute it
        # per example so run_epoch can pass it (train==serve for the bias too). None -> no bias input.
        self.use_affinity = affinity_fn is not None
        self.our_idx = np.zeros((n, TEAM_SIZE), dtype=np.int64)
        self.opp_idx = np.zeros((n, TEAM_SIZE), dtype=np.int64)
        self.our_feat = np.zeros((n, TEAM_SIZE, feat_dim), dtype=np.float32)
        self.opp_feat = np.zeros((n, TEAM_SIZE, feat_dim), dtype=np.float32)
        if self.use_affinity:
            self.our_affinity = np.zeros((n, TEAM_SIZE, TEAM_SIZE), dtype=np.float32)
        self.bring = np.zeros((n, TEAM_SIZE), dtype=np.float32)
        self.lead = np.zeros((n, TEAM_SIZE), dtype=np.float32)
        self.valid_bring = np.zeros((n,), dtype=np.float32)
        self.valid_lead = np.zeros((n,), dtype=np.float32)     # audit: mask a species-match-shifted lead
        self.replay_ids: List[str] = []

        for i, ex in enumerate(examples):
            self.our_idx[i] = [vocab.get(s, PAD_IDX) for s in ex["our_species"]]
            self.opp_idx[i] = [vocab.get(s, PAD_IDX) for s in ex["opp_species"]]
            self.our_feat[i] = ex["our_feat"]
            self.opp_feat[i] = ex["opp_feat"]
            self.bring[i] = ex["bring"]
            self.lead[i] = ex["lead"]
            self.valid_bring[i] = ex["valid_bring"]
            self.valid_lead[i] = ex.get("valid_lead", 1.0)     # tolerate older example dicts
            self.replay_ids.append(ex["replay_id"])
            if self.use_affinity:
                self.our_affinity[i] = affinity_fn(ex["our_species"])

        self.t_our_idx = torch.from_numpy(self.our_idx)
        self.t_opp_idx = torch.from_numpy(self.opp_idx)
        self.t_our_feat = torch.from_numpy(self.our_feat)
        self.t_opp_feat = torch.from_numpy(self.opp_feat)
        self.t_bring = torch.from_numpy(self.bring)
        self.t_lead = torch.from_numpy(self.lead)
        self.t_valid = torch.from_numpy(self.valid_bring)
        self.t_valid_lead = torch.from_numpy(self.valid_lead)
        if self.use_affinity:
            self.t_our_affinity = torch.from_numpy(self.our_affinity)

    def __len__(self) -> int:
        return self.t_our_idx.shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {
            "our_idx": self.t_our_idx[idx],
            "opp_idx": self.t_opp_idx[idx],
            "our_feat": self.t_our_feat[idx],
            "opp_feat": self.t_opp_feat[idx],
            "bring": self.t_bring[idx],
            "lead": self.t_lead[idx],
            "valid_bring": self.t_valid[idx],
            "valid_lead": self.t_valid_lead[idx],
        }
        if self.use_affinity:
            item["our_affinity"] = self.t_our_affinity[idx]
        return item


def print_stats(stats: Counter) -> None:
    order = ["files", "replays", "examples", "valid_bring", "valid_lead",
             "skipped_incomplete", "unmatched_picks", "bad_files"]
    print("── Team-preview dataset stats ───────────────")
    for k in order:
        if k in stats:
            print(f"  {k:20s}: {stats[k]}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect a team-preview JSONL folder")
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    examples, stats = examples_from_folders(args.folders, limit_files=args.limit_files)
    print_stats(stats)
    vocab = build_vocab(examples)
    print(f"vocab size (species): {len(vocab)}")
    train, val = split_by_replay(examples, val_frac=args.val_frac)
    print(f"split: {len(train)} train / {len(val)} val examples "
          f"({len({e['replay_id'] for e in train})} / "
          f"{len({e['replay_id'] for e in val})} replays)")
