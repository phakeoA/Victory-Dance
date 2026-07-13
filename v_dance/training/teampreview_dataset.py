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
def sbda_feature_fn(belief) -> Callable[..., np.ndarray]:
    """Closure ``(species, known=None) -> (FEAT_DIM,)`` for the SBDA TP recipe —
    IDENTICAL to the serve-side ``model_io._pack_side``.  ``known=None`` is the
    symmetric belief-only base (a Type-B/closed-sheet mon — ``has_own_detail`` 0,
    exactly what a BC-pretrained net serves); ``known`` (an ``OwnKnown``) rides
    the overlay — the tpfeat-v7 OTS pathway where a REVEALED sheet is
    preview-visible on either side.  This shared CALL — not merely a shared dim —
    is the train==serve lockstep the 15b plumbing exists to guarantee.  Imported
    lazily to avoid a tp_features <-> teampreview_dataset cycle."""
    from v_dance.training.tp_features import own_mon_features

    def _fn(species: Optional[str], known=None) -> np.ndarray:
        return own_mon_features(norm_species(species), belief, known)

    _fn.supports_known = True          # _example_for_side gates the sheet map on this
    return _fn


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


def _first_transitions(path: str) -> Dict[str, dict]:
    """First transition PER PERSPECTIVE.  Preview labels are constant, but the
    OTS sheet map needs BOTH sides' views: one perspective's own containers hold
    only the 4 brought mons — the other perspective's opp stubs carry its full 6."""
    out: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            p = t.get("perspective") or "p1"
            if p not in out:
                out[p] = t
            if len(out) >= 2:
                break
    return out


def _revealed_of(mon: dict):
    """A mon dict's OTS-revealed build as an ``OwnKnown`` (None when nothing is
    revealed — a closed-sheet stub contributes no overlay).  Turn-1 mons are
    never mega'd yet, so ``known_ability`` is the sheet's base ability; the item
    has no channel in the TP feature schema and is skipped."""
    if not isinstance(mon, dict):
        return None
    ab = mon.get("known_ability") or mon.get("pre_mega_ability")
    mv = [m for m in (mon.get("known_moves") or []) if m]
    tera = mon.get("known_tera_type")
    if not ab and not mv and not tera:
        return None
    from v_dance.training.tp_features import OwnKnown
    return OwnKnown(ability=ab, moves=mv, tera=tera, will_mega=False)


def _ots_sheet_map(firsts: Dict[str, dict]) -> Dict[str, "object"]:
    """{norm_species: OwnKnown} across ALL 12 preview mons of one OTS battle.

    Every container of every perspective's first transition contributes (the
    opp containers carry the full 6-mon preview stubs; own containers the 4
    brought with the same sheet stamps — post the 2026-07-02 sheet-authoritative
    fix both views agree, so first-seen wins)."""
    out: Dict[str, object] = {}
    for t in firsts.values():
        snap = t.get("state_before_actions") or {}
        for key in ("opp_active", "opp_bench", "our_active", "our_bench"):
            cont = snap.get(key) or {}
            mons = cont.values() if isinstance(cont, dict) else cont
            for mon in mons:
                if not isinstance(mon, dict) or not mon.get("species"):
                    continue
                r = _revealed_of(mon)
                if r is not None:
                    out.setdefault(
                        norm_species(mon.get("base_species") or mon["species"]), r)
    return out


def _rid_order(rid: str):
    """Sort key for games within a Bo3 set: numeric room suffix when present."""
    tail = str(rid).rsplit("-", 1)[-1]
    return (0, int(tail)) if tail.isdigit() else (1, str(rid))


def build_set_prev_map(files: Sequence[str]) -> Dict[str, dict]:
    """DS-4c stage 3: {rid: {"p1": (brought, leads), "p2": (...)}} — each Bo3 game's
    PREVIOUS game's brought/leads per side, from a cheap first-line pre-pass over the
    corpus (``bo3_set_id`` groups games; room-number order = play order; leads = the
    first LEAD_K brought, the always-reliable part of the label). Games without a set
    (or game 1 of a set) simply don't appear."""
    by_set: Dict[str, List[Tuple[str, dict]]] = {}
    for fp in files:
        try:
            t = _first_transition(fp)
        except (OSError, json.JSONDecodeError):
            continue
        if not t or not t.get("bo3_set_id"):
            continue
        by_set.setdefault(str(t["bo3_set_id"]), []).append((str(t.get("replay_id")), t))
    out: Dict[str, dict] = {}
    for games in by_set.values():
        games.sort(key=lambda g: _rid_order(g[0]))
        for (rid, _t), (_prid, pt) in zip(games[1:], games[:-1]):
            prev = {}
            for pid in ("p1", "p2"):
                brought = list(((pt.get("players") or {}).get(pid) or {}).get("brought") or [])
                prev[pid] = (brought, brought[:LEAD_K])
            out[rid] = prev
    return out


def _set_ctx_vec(roster_norm: List[str], brought: Sequence[str],
                 leads: Sequence[str]) -> np.ndarray:
    """(TEAM_SIZE, 2) [brought_last, led_last] aligned to the roster by norm species."""
    b = {norm_species(s) for s in brought if s}
    ld = {norm_species(s) for s in leads if s}
    v = np.zeros((TEAM_SIZE, 2), dtype=np.float32)
    for i, ns in enumerate(roster_norm[:TEAM_SIZE]):
        if ns in b:
            v[i, 0] = 1.0
        if ns in ld:
            v[i, 1] = 1.0
    return v


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
                      feat_fn: Callable = mon_dex_features,
                      sheet_map: Optional[dict] = None,
                      set_prev: Optional[dict] = None) -> Optional[dict]:
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

    # tpfeat-v7 OTS: the OPPONENT's revealed sheets ride the overlay when open
    # team sheets made them preview-visible.  OWN side stays symmetric base
    # (overlay off) — parity with the prod serve path, whose own_known is None.
    use_sheet = bool(sheet_map) and getattr(feat_fn, "supports_known", False)

    def _opp_f(s):
        if use_sheet:
            return feat_fn(s, sheet_map.get(norm_species(s)))
        return feat_fn(s)

    if use_sheet and stats is not None:
        stats["ots_opp_overlay_mons"] += sum(
            1 for s in opp_roster if sheet_map.get(norm_species(s)) is not None)

    ex = {
        "our_species": roster_norm,
        "opp_species": [norm_species(s) for s in opp_roster],
        "our_feat": np.stack([feat_fn(s) for s in roster]),
        "opp_feat": np.stack([_opp_f(s) for s in opp_roster]),
        "bring": bring,
        "lead": lead,
        "valid_bring": valid_bring,
        "valid_lead": valid_lead,
        "replay_id": t.get("replay_id"),
        "side": side,
    }
    # DS-4c stage 3: Bo3 previous-game context (this game = game 2/3 of a set).
    # Keys only present on set games — off-set examples stay byte-identical dicts.
    if set_prev:
        ob, ol = set_prev.get(side) or ([], [])
        pb, pl = set_prev.get(opp) or ([], [])
        ex["our_set_ctx"] = _set_ctx_vec(roster_norm, ob, ol)
        ex["opp_set_ctx"] = _set_ctx_vec(ex["opp_species"], pb, pl)
        if stats is not None:
            stats["set_ctx_examples"] += 1
    return ex


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
    set_prev_map = build_set_prev_map(files)     # DS-4c: {rid: prev-game brought/leads}
    for fp in files:
        stats["files"] += 1
        try:
            firsts = _first_transitions(fp)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            stats["bad_files"] += 1
            print(f"[teampreview_dataset] WARNING skipping {fp}: {exc}", file=sys.stderr)
            continue
        if not firsts:
            continue
        t = firsts.get("p1") or next(iter(firsts.values()))
        # tpfeat-v7: OTS battles expose both sides' sheets at preview — build the
        # revealed map once per file (only when the recipe can consume it).
        sheet_map = None
        if t.get("ots") and getattr(feat_fn, "supports_known", False):
            sheet_map = _ots_sheet_map(firsts)
            if sheet_map:
                stats["ots_sheet_replays"] += 1
        set_prev = set_prev_map.get(str(t.get("replay_id")))
        for side, opp in (("p1", "p2"), ("p2", "p1")):
            if side not in sides:
                continue
            ex = _example_for_side(t, side, opp, stats, feat_fn, sheet_map=sheet_map,
                                   set_prev=set_prev)
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
                 affinity_fn: Optional[Callable] = None,
                 subset_mask_p: float = 0.0, subset_mask_k: int = 0, aug_seed: int = 0,
                 with_set_ctx: bool = False):
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("torch is required for TeamPreviewDataset")
        # Subset-mask augmentation (2026-07-10, TP tier-2 — docs/teampreview_bring_lead_design.md
        # addendum): with prob ``subset_mask_p`` an item masks K∈{1,2} random roster slots
        # (idx 0 + zero feats = the pad convention) so PARTIAL rosters are in-distribution and
        # model_io's joint subset re-decode becomes valid. Brought and unbrought slots are masked
        # ALIKE (masking only left-behind mons would teach the "4 mons left ⇒ bring all" count
        # shortcut); masked slots are EXCLUDED from the BCE via the returned ``slot_mask`` (they
        # are absent, not choices). Labels are otherwise unchanged. TRAIN-only (val keeps p=0);
        # only full 6-mon rosters are augmented. ⚠ train-metric noise on augmented rows is
        # expected (a masked brought slot still carries label 1) — checkpoint selection uses the
        # clean val split, so it is monitoring noise only.
        # ``subset_mask_k``: 0 = draw K∈{1,2} per item (the first aug run); 2 = mask EXACTLY two
        # slots, making every augmented item precisely the joint decode's 4-mon context (gate
        # round 2 — the {1,2} dose left true 4-mon contexts in only ~p/2 of items and the joint
        # decode recovered +6.3pp but stayed −10.9pp vs greedy).
        self.subset_mask_p = float(subset_mask_p)
        self.subset_mask_k = int(subset_mask_k)
        self._aug_rng = np.random.default_rng(aug_seed) if self.subset_mask_p > 0 else None
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
        # DS-4c stage 3: Bo3 previous-game context tensors — emitted ONLY when the
        # consumer opts in (with_set_ctx), so every existing batch stays byte-identical.
        # Off-set examples carry zeros (== "no context", the zero-init identity).
        self.with_set_ctx = bool(with_set_ctx)
        if self.with_set_ctx:
            self.our_set_ctx = np.zeros((n, TEAM_SIZE, 2), dtype=np.float32)
            self.opp_set_ctx = np.zeros((n, TEAM_SIZE, 2), dtype=np.float32)
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
            if self.with_set_ctx:
                if ex.get("our_set_ctx") is not None:
                    self.our_set_ctx[i] = ex["our_set_ctx"]
                if ex.get("opp_set_ctx") is not None:
                    self.opp_set_ctx[i] = ex["opp_set_ctx"]

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
        if self.with_set_ctx:
            self.t_our_set_ctx = torch.from_numpy(self.our_set_ctx)
            self.t_opp_set_ctx = torch.from_numpy(self.opp_set_ctx)

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
            "slot_mask": torch.ones(TEAM_SIZE),   # per-slot loss weights (aug masks zero some)
        }
        if self.use_affinity:
            item["our_affinity"] = self.t_our_affinity[idx]
        if self.with_set_ctx:
            item["our_set_ctx"] = self.t_our_set_ctx[idx]
            item["opp_set_ctx"] = self.t_opp_set_ctx[idx]
        # Subset-mask augmentation (train-only; see __init__). Fresh draw per access → different
        # masks per epoch. Only full 6-mon rosters are touched; the model's own pad handling
        # (feat-zero test → attention-key + context-mean exclusion) does the rest in forward.
        if self._aug_rng is not None and self._aug_rng.random() < self.subset_mask_p:
            valid_n = int((self.our_feat[idx] != 0).any(axis=-1).sum())
            if valid_n == TEAM_SIZE:
                k = self.subset_mask_k or int(self._aug_rng.integers(1, 3))   # fixed K or K∈{1,2}
                slots = self._aug_rng.choice(TEAM_SIZE, size=k, replace=False)
                our_idx = item["our_idx"].clone()
                our_feat = item["our_feat"].clone()
                slot_mask = item["slot_mask"].clone()
                our_idx[slots] = 0                                     # pad convention: idx 0 +
                our_feat[slots] = 0.0                                  # all-zero feature row
                slot_mask[slots] = 0.0                                 # absent ≠ a choice → no BCE
                item.update(our_idx=our_idx, our_feat=our_feat, slot_mask=slot_mask)
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
