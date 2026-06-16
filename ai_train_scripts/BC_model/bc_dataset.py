"""
Behaviour-Cloning dataset for Victory-Dance (VGC Reg M-A, doubles).

Loads a folder of per-replay JSONL exports (one transition per line, the
output of ``bulk_parse_replays.py`` / the team-builder ``/export``) and turns
each transition into supervised targets for the two-head BC policy:

    X      = StateEncoder().encode_snapshot(t["state_before_actions"], turn)
    head   = "our_a" / "our_b"   (the two active slots we control)
    target = that slot's chosen action_index (0-15, the FROZEN codec)
    mask   = t["action_mask"][slot]  (decision-time legality, 16 entries)

Per-transition, ONE example carries BOTH heads with a per-head validity flag,
because that is exactly how the model is used at inference: one board state →
predict both slots' actions at once.  A head is dropped (marked invalid) when:

  * the slot did not act this turn (fainted / empty active slot), or
  * its action_index is null (action not expressible in the frozen codec), or
  * its mask row is missing / all-zero, or
  * the target is illegal under its own mask (a rare annotation edge case —
    masked cross-entropy would otherwise produce inf loss).

A transition with no valid head is skipped entirely (and its state is never
encoded, so this is cheap).

Slot semantics: ``our_actions`` may list a slot TWICE in one turn — the
turn-start choice followed by a mid-turn forced replacement (a faint).  Only
the FIRST entry per slot is a turn-start decision aligned with
``state_before_actions`` / ``action_mask``; later same-slot entries are
dropped.

The train/val split is BY replay_id, so no two transitions from the same game
land on opposite sides of the split (prevents trivial leakage).
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── Bootstrap: locate data/scripts by walking up (folder-depth independent) ───
def _find_scripts_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        cand = parent / "data" / "scripts"
        if cand.is_dir():
            return cand
    raise RuntimeError(f"could not locate data/scripts above {__file__}")

_SCRIPTS_DIR = _find_scripts_dir()
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from state_encoder import (  # noqa: E402  (import after sys.path bootstrap)
    StateEncoder,
    ACTIONS_PER_SLOT,
    get_state_dim,
    get_gimmick_dim,
    NUM_MOVES,
    OPP_HEADS,
    permute_move_slots,
    permute_action_index,
    permute_action_mask_row,
)

GIMMICK_DIM = get_gimmick_dim()

# The two active slots we output policies for; index in this tuple == head index
HEADS: Tuple[str, str] = ("our_a", "our_b")

try:  # torch is only needed for the Dataset wrapper, not for raw example loading
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch always present in the GPU venv
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    _HAS_TORCH = False


# ══════════════════════════════════════════════════════════════════════════════
# Raw example extraction (torch-free)
# ══════════════════════════════════════════════════════════════════════════════
def iter_jsonl_files(folder: str, recursive: bool = True) -> List[str]:
    """Sorted list of .jsonl files under ``folder``."""
    folder = str(folder)
    pattern = os.path.join(folder, "**", "*.jsonl") if recursive else os.path.join(folder, "*.jsonl")
    return sorted(glob.glob(pattern, recursive=recursive))


def _first_action_per_slot(our_actions: Sequence[dict]) -> Dict[str, dict]:
    """First ``our_actions`` entry for each slot (the turn-start decision)."""
    first: Dict[str, dict] = {}
    for act in our_actions or []:
        slot = act.get("slot")
        if slot is not None and slot not in first:
            first[slot] = act
    return first


def _our_player_meta(t: dict) -> Tuple[Optional[float], float, Optional[bool]]:
    """Per-transition demonstrator metadata for OUR side (the side we imitate).

    Returns ``(rating_before, rating_delta, won)`` where:
      * ``rating_before`` is our player's pre-game ladder rating (float, or None
        if absent/malformed),
      * ``rating_delta``  is our rating change for the game (float, 0.0 if absent),
      * ``won``           is True/False if the game ``winner`` can be matched to our
        username, else None (tie / unknown).

    The "our" side is identified by ``players.our_side`` (a 'p1'/'p2' key into
    ``players``).  These fields are already present in every JSONL transition; the
    BC loader simply never read them before (TIER-1 #1 — imitate STRONGER players).
    """
    players = t.get("players") or {}
    our_side = players.get("our_side")
    me = players.get(our_side) if isinstance(our_side, str) else None
    if not isinstance(me, dict):
        return None, 0.0, None
    raw_rating = me.get("rating_before")
    rating = float(raw_rating) if isinstance(raw_rating, (int, float)) else None
    raw_delta = me.get("rating_delta")
    delta = float(raw_delta) if isinstance(raw_delta, (int, float)) else 0.0
    winner = t.get("winner")
    uname = me.get("username")
    won: Optional[bool] = None
    if winner is not None and uname is not None:
        won = bool(winner == uname)
    return rating, delta, won


def transition_to_example(
    t: dict,
    encoder: StateEncoder,
    stats: Optional[Counter] = None,
    with_opp: bool = False,
) -> Optional[dict]:
    """
    Convert one transition dict into a BC example, or None if no head is usable.

    Returns ``{"x", "targets", "masks", "replay_id"}`` where ``targets`` maps a
    valid head name → action_index and ``masks`` maps it → np.float32 (16,).
    """
    mask_all = t.get("action_mask") or {}
    gimmick_mask_all = t.get("gimmick_mask") or {}
    first = _first_action_per_slot(t.get("our_actions") or [])

    # Count the dropped same-slot (forced-replacement) entries for visibility.
    if stats is not None:
        extra = len(t.get("our_actions") or []) - len(first)
        if extra > 0:
            stats["dropped_forced_replacement"] += extra

    targets: Dict[str, int] = {}
    masks: Dict[str, np.ndarray] = {}
    # Gimmick (mega) labels, paired per head with the action label.  A head gets
    # a gimmick target only when its action is valid AND the transition carries a
    # legal gimmick label for it — older JSONL (pre-gimmick export) has neither,
    # so the gimmick head simply receives no signal until a re-export.
    gimmick_targets: Dict[str, int] = {}
    gimmick_masks: Dict[str, np.ndarray] = {}
    for head in HEADS:
        act = first.get(head)
        if act is None:
            continue  # slot did not act (fainted / empty)
        if stats is not None:
            stats["slot_decisions"] += 1
        ai = act.get("action_index")
        if ai is None:
            if stats is not None:
                stats["skipped_null_index"] += 1
            continue
        row = mask_all.get(head)
        if not row or sum(row) == 0:
            if stats is not None:
                stats["skipped_no_mask"] += 1
            continue
        if ai < 0 or ai >= len(row) or row[ai] != 1:
            if stats is not None:
                stats["skipped_illegal_target"] += 1
            continue
        targets[head] = int(ai)
        masks[head] = np.asarray(row, dtype=np.float32)
        if stats is not None:
            stats["usable_examples"] += 1

        # Paired gimmick label: only when present, in range, and mask-legal.
        gi = act.get("gimmick_index")
        grow = gimmick_mask_all.get(head)
        if (gi is not None and grow and len(grow) == GIMMICK_DIM
                and 0 <= gi < GIMMICK_DIM and grow[gi] == 1):
            gimmick_targets[head] = int(gi)
            gimmick_masks[head] = np.asarray(grow, dtype=np.float32)
            if stats is not None:
                stats["usable_gimmick_examples"] += 1
                if gi == 1:
                    stats["gimmick_positives"] += 1

    if not targets:
        return None

    x = encoder.encode_snapshot(
        t.get("state_before_actions") or {}, turn=t.get("turn") or 0
    )
    rating, rating_delta, won = _our_player_meta(t)
    ex = {
        "x": x,
        "targets": targets,
        "masks": masks,
        "gimmick_targets": gimmick_targets,
        "gimmick_masks": gimmick_masks,
        "replay_id": t.get("replay_id"),
        # Demonstrator metadata (TIER-1 #1): used to filter/weight by skill.
        "rating": rating,
        "rating_delta": rating_delta,
        "won": won,
        # Game-phase metadata (per-situation eval / calibration diagnostic).
        "turn": t.get("turn"),
        "decision_type": t.get("decision_type"),
    }

    # Auxiliary opponent-head labels (task #9): the opponent's action index +
    # legality mask per opp slot, computed from the SAME stored snapshot via the
    # opp-perspective codec — no re-export needed.  Only attached when requested.
    if with_opp:
        from state_encoder import annotate_opp_actions, OPP_HEADS  # noqa: E402
        annotate_opp_actions(t)
        opp_mask_all = t.get("opp_action_mask") or {}
        opp_first = _first_action_per_slot(t.get("opp_actions_actual") or [])
        opp_targets: Dict[str, int] = {}
        opp_masks: Dict[str, np.ndarray] = {}
        for head in OPP_HEADS:
            act = opp_first.get(head)
            if act is None:
                continue
            ai = act.get("opp_action_index")
            row = opp_mask_all.get(head)
            if ai is None or not row or sum(row) == 0:
                continue
            if ai < 0 or ai >= len(row) or row[ai] != 1:
                continue
            opp_targets[head] = int(ai)
            opp_masks[head] = np.asarray(row, dtype=np.float32)
            if stats is not None:
                stats["usable_opp_examples"] += 1
        ex["opp_targets"] = opp_targets
        ex["opp_masks"] = opp_masks
    return ex


def build_examples(
    files: Sequence[str],
    encoder: Optional[StateEncoder] = None,
    limit_transitions: Optional[int] = None,
    limit_files: Optional[int] = None,
    with_opp: bool = False,
) -> Tuple[List[dict], Counter]:
    """
    Walk JSONL ``files`` and return ``(examples, stats)``.

    ``limit_files`` / ``limit_transitions`` cap the work for smoke runs.
    ``with_opp`` also extracts the auxiliary opponent-head labels (task #9).
    """
    encoder = encoder or StateEncoder()
    stats: Counter = Counter()
    examples: List[dict] = []

    if limit_files is not None:
        files = list(files)[:limit_files]

    for fp in files:
        stats["files"] += 1
        try:
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    stats["transitions"] += 1
                    t = json.loads(line)
                    ex = transition_to_example(t, encoder, stats, with_opp=with_opp)
                    if ex is not None:
                        examples.append(ex)
                    if limit_transitions is not None and stats["transitions"] >= limit_transitions:
                        break
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            stats["bad_files"] += 1
            print(f"[bc_dataset] WARNING: skipping {fp}: {exc}", file=sys.stderr)
        if limit_transitions is not None and stats["transitions"] >= limit_transitions:
            break

    stats["replays"] = len({e["replay_id"] for e in examples})
    return examples, stats


def examples_from_folders(
    folders: Sequence[str],
    encoder: Optional[StateEncoder] = None,
    recursive: bool = True,
    limit_transitions: Optional[int] = None,
    limit_files: Optional[int] = None,
    with_opp: bool = False,
) -> Tuple[List[dict], Counter]:
    """Convenience: gather examples from one or more folders of JSONL.

    Missing/empty folders contribute nothing; files are de-duplicated by
    absolute path so overlapping folders (or --type-a re-listing a default
    folder) are never double-counted.
    """
    files: List[str] = []
    seen: set = set()
    for folder in folders:
        for f in iter_jsonl_files(folder, recursive=recursive):
            key = os.path.abspath(f)
            if key not in seen:
                seen.add(key)
                files.append(f)
    return build_examples(
        files,
        encoder=encoder,
        limit_transitions=limit_transitions,
        limit_files=limit_files,
        with_opp=with_opp,
    )


def split_by_replay(
    examples: Sequence[dict],
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[List[dict], List[dict]]:
    """
    Split examples into (train, val) BY replay_id so no game spans the split.

    Deterministic given ``seed``.  Guarantees at least one replay in val when
    there is more than one replay available.
    """
    ids = sorted({e["replay_id"] for e in examples})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_frac))
    n_val = min(max(n_val, 1 if len(ids) > 1 else 0), len(ids))
    val_ids = set(ids[:n_val])
    train = [e for e in examples if e["replay_id"] not in val_ids]
    val = [e for e in examples if e["replay_id"] in val_ids]
    return train, val


# ── Demonstrator skill filtering / weighting (TIER-1 #1) ──────────────────────
def filter_by_rating(examples: Sequence[dict], rating_min: Optional[float]) -> List[dict]:
    """Drop examples whose OUR-side ``rating`` is below ``rating_min``.

    ``rating_min`` None → no filtering (returns a list copy).  Examples with an
    UNKNOWN rating (None) are KEPT (never silently dropped) — the threshold can
    only exclude a demonstrator it can actually score.  Intended for TRAIN only
    (val stays a true, unfiltered metric)."""
    if rating_min is None:
        return list(examples)
    return [e for e in examples
            if e.get("rating") is None or e.get("rating") >= rating_min]


def compute_sample_weights(
    examples: Sequence[dict],
    rating_weight: bool = False,
    outcome_weight: bool = False,
    rating_weight_floor: float = 0.25,
    loss_weight: float = 0.5,
) -> np.ndarray:
    """Per-example loss weight (np.float32, length len(examples)) for the action
    head, MEAN-NORMALISED to 1.0 so the overall loss scale (which divides by the
    valid-decision COUNT) is preserved — the weights only re-distribute emphasis.

    Components (multiplicative, each optional, all OFF → all-ones):
      * ``rating_weight``  — weight ∝ the demonstrator's rating PERCENTILE among
        the supplied examples, mapped to ``[rating_weight_floor, 1.0]`` (top
        players ~1.0, bottom ~floor; unknown rating → median 0.5).  Imitates
        STRONGER players without hard-dropping the rest.
      * ``outcome_weight`` — won game → 1.0, lost → ``loss_weight``, unknown → 1.0.
        Up-weights decisions from games our side actually WON.

    With both OFF the returned vector is exactly ones (the default training path
    is then byte-identical to the unweighted loss)."""
    n = len(examples)
    w = np.ones(n, dtype=np.float64)
    if n == 0:
        return w.astype(np.float32)

    if rating_weight:
        ratings = np.array(
            [e.get("rating") if e.get("rating") is not None else np.nan for e in examples],
            dtype=np.float64,
        )
        known = ~np.isnan(ratings)
        pct = np.full(n, 0.5, dtype=np.float64)   # unknown rating → median
        if int(known.sum()) > 1:
            idx = np.where(known)[0]
            order = idx[np.argsort(ratings[idx], kind="mergesort")]
            ranks = np.linspace(0.0, 1.0, order.size)
            pct[order] = ranks
        rw = rating_weight_floor + (1.0 - rating_weight_floor) * pct
        w *= rw

    if outcome_weight:
        ow = np.array(
            [1.0 if e.get("won") is True
             else (loss_weight if e.get("won") is False else 1.0)
             for e in examples],
            dtype=np.float64,
        )
        w *= ow

    mean = w.mean()
    if mean > 0:
        w = w / mean
    return w.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Torch Dataset wrapper
# ══════════════════════════════════════════════════════════════════════════════
class BCDataset(Dataset):
    """
    Tensor-backed dataset over BC examples.

    Each item is a dict of fixed-shape tensors (default collate stacks them):

        x             : float32 (STATE_DIM,)
        target        : int64   (2,)        action_index per head, -1 invalid
        mask          : float32 (2, 16)     action legality per head
        valid         : float32 (2,)        1.0 where the head has an action target
        gimmick_target: int64   (2,)        gimmick index per head, -1 invalid
        gimmick_mask  : float32 (2, G)      gimmick legality per head
        gimmick_valid : float32 (2,)        1.0 where the head has a gimmick target
    """

    def __init__(self, examples: Sequence[dict], augment_move_order: bool = False,
                 aug_seed: int = 0, with_opp: bool = False,
                 weights: Optional[Sequence[float]] = None):
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("torch is required for BCDataset")
        # Train-only move-slot permutation augmentation (task #22): when True,
        # each fetch independently permutes every own active mon's 4 move blocks
        # AND remaps that head's action target + mask, so the net learns move
        # FEATURES rather than slot POSITION.  Leave False for val (true metrics).
        self.augment_move_order = bool(augment_move_order)
        # Auxiliary opponent-head targets/masks (task #9) when the examples carry
        # them.  Move-order augmentation does NOT touch the opp move blocks (it
        # only permutes our slots), so the opp fields pass through unchanged.
        self.with_opp = bool(with_opp)
        self._rng = np.random.RandomState(aug_seed)
        n = len(examples)
        state_dim = get_state_dim()
        self.X = np.zeros((n, state_dim), dtype=np.float32)
        self.target = np.full((n, len(HEADS)), -1, dtype=np.int64)
        self.mask = np.zeros((n, len(HEADS), ACTIONS_PER_SLOT), dtype=np.float32)
        self.valid = np.zeros((n, len(HEADS)), dtype=np.float32)
        self.gimmick_target = np.full((n, len(HEADS)), -1, dtype=np.int64)
        self.gimmick_mask = np.zeros((n, len(HEADS), GIMMICK_DIM), dtype=np.float32)
        self.gimmick_valid = np.zeros((n, len(HEADS)), dtype=np.float32)
        # Scalar value target (win=1.0 / loss=0.0) + validity (#2 value head).  The
        # label is the game OUTCOME from our perspective, back-filled to EVERY turn
        # (the value head learns expected win-prob by averaging over games — the
        # standard AlphaZero MC return); unknown outcome → value_valid 0.
        self.value_target = np.zeros((n,), dtype=np.float32)
        self.value_valid = np.zeros((n,), dtype=np.float32)
        # Per-example action-loss weight (TIER-1 #1).  Default all-ones → no effect
        # (the trainer only applies it when a weighting flag is set).
        self.weight = np.ones((n,), dtype=np.float32)
        if weights is not None:
            wv = np.asarray(weights, dtype=np.float32).ravel()
            if wv.shape[0] != n:
                raise ValueError(
                    f"weights length {wv.shape[0]} != number of examples {n}")
            self.weight = wv
        if self.with_opp:
            self.opp_target = np.full((n, len(OPP_HEADS)), -1, dtype=np.int64)
            self.opp_mask = np.zeros((n, len(OPP_HEADS), ACTIONS_PER_SLOT), dtype=np.float32)
            self.opp_valid = np.zeros((n, len(OPP_HEADS)), dtype=np.float32)
        self.replay_ids: List[str] = []

        for i, ex in enumerate(examples):
            self.X[i] = ex["x"]
            self.replay_ids.append(ex["replay_id"])
            won = ex.get("won")
            if won is not None:
                self.value_target[i] = 1.0 if won else 0.0
                self.value_valid[i] = 1.0
            g_targets = ex.get("gimmick_targets") or {}
            g_masks = ex.get("gimmick_masks") or {}
            for h_idx, head in enumerate(HEADS):
                if head in ex["targets"]:
                    self.target[i, h_idx] = ex["targets"][head]
                    self.mask[i, h_idx] = ex["masks"][head]
                    self.valid[i, h_idx] = 1.0
                if head in g_targets:
                    self.gimmick_target[i, h_idx] = g_targets[head]
                    self.gimmick_mask[i, h_idx] = g_masks[head]
                    self.gimmick_valid[i, h_idx] = 1.0
            if self.with_opp:
                o_targets = ex.get("opp_targets") or {}
                o_masks = ex.get("opp_masks") or {}
                for o_idx, ohead in enumerate(OPP_HEADS):
                    if ohead in o_targets:
                        self.opp_target[i, o_idx] = o_targets[ohead]
                        self.opp_mask[i, o_idx] = o_masks[ohead]
                        self.opp_valid[i, o_idx] = 1.0

        # Pre-convert to tensors once (dataset fits comfortably in RAM).
        self.X_t = torch.from_numpy(self.X)
        self.target_t = torch.from_numpy(self.target)
        self.mask_t = torch.from_numpy(self.mask)
        self.valid_t = torch.from_numpy(self.valid)
        self.gimmick_target_t = torch.from_numpy(self.gimmick_target)
        self.gimmick_mask_t = torch.from_numpy(self.gimmick_mask)
        self.gimmick_valid_t = torch.from_numpy(self.gimmick_valid)
        self.value_target_t = torch.from_numpy(self.value_target)
        self.value_valid_t = torch.from_numpy(self.value_valid)
        self.weight_t = torch.from_numpy(self.weight)
        if self.with_opp:
            self.opp_target_t = torch.from_numpy(self.opp_target)
            self.opp_mask_t = torch.from_numpy(self.opp_mask)
            self.opp_valid_t = torch.from_numpy(self.opp_valid)

    def __len__(self) -> int:
        return self.X_t.shape[0]

    def _opp_fields(self, idx: int) -> dict:
        """Opponent aux targets/masks for this item (empty unless with_opp).  The
        move-order augmentation only permutes OUR move blocks, so opp fields are
        identical on both fetch paths."""
        if not self.with_opp:
            return {}
        return {
            "opp_target": self.opp_target_t[idx],
            "opp_mask": self.opp_mask_t[idx],
            "opp_valid": self.opp_valid_t[idx],
        }

    def __getitem__(self, idx: int) -> dict:
        if not self.augment_move_order:
            return {
                "x": self.X_t[idx],
                "target": self.target_t[idx],
                "mask": self.mask_t[idx],
                "valid": self.valid_t[idx],
                "gimmick_target": self.gimmick_target_t[idx],
                "gimmick_mask": self.gimmick_mask_t[idx],
                "gimmick_valid": self.gimmick_valid_t[idx],
                "value_target": self.value_target_t[idx],
                "value_valid": self.value_valid_t[idx],
                "weight": self.weight_t[idx],
                **self._opp_fields(idx),
            }

        # Augmented fetch: permute each own active slot's move blocks + remap the
        # matching action label/mask.  Work on numpy copies so the stored tensors
        # stay pristine.  Gimmick (per-slot, not per-move) is unaffected.
        x = self.X[idx].copy()
        target = self.target[idx].copy()
        mask = self.mask[idx].copy()
        for h_idx in range(len(HEADS)):
            perm = self._rng.permutation(NUM_MOVES).tolist()
            permute_move_slots(x, h_idx, perm)
            if self.valid[idx, h_idx] > 0.5:
                target[h_idx] = permute_action_index(int(target[h_idx]), perm)
                mask[h_idx] = np.asarray(
                    permute_action_mask_row(mask[h_idx].tolist(), perm), dtype=np.float32)
        return {
            "x": torch.from_numpy(x),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "valid": self.valid_t[idx],
            "gimmick_target": self.gimmick_target_t[idx],
            "gimmick_mask": self.gimmick_mask_t[idx],
            "gimmick_valid": self.gimmick_valid_t[idx],
            "value_target": self.value_target_t[idx],
            "value_valid": self.value_valid_t[idx],
            "weight": self.weight_t[idx],
            **self._opp_fields(idx),
        }


# ══════════════════════════════════════════════════════════════════════════════
def print_stats(stats: Counter) -> None:
    """Pretty-print a build_examples stats Counter."""
    order = [
        "files", "transitions", "replays", "slot_decisions", "usable_examples",
        "skipped_null_index", "skipped_no_mask", "skipped_illegal_target",
        "dropped_forced_replacement", "bad_files",
    ]
    print("── BC dataset stats ─────────────────────────")
    for k in order:
        if k in stats:
            print(f"  {k:28s}: {stats[k]}")
    extra = set(stats) - set(order)
    for k in sorted(extra):
        print(f"  {k:28s}: {stats[k]}")


if __name__ == "__main__":  # quick CLI sanity check / stats dump
    import argparse

    ap = argparse.ArgumentParser(description="Inspect a BC JSONL folder")
    ap.add_argument("folders", nargs="+", help="folder(s) of per-replay .jsonl")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples, stats = examples_from_folders(args.folders, limit_files=args.limit_files)
    print_stats(stats)
    train, val = split_by_replay(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"split: {len(train)} train / {len(val)} val examples "
          f"({len({e['replay_id'] for e in train})} / "
          f"{len({e['replay_id'] for e in val})} replays)")
    tgt_hist = Counter()
    for e in examples:
        for v in e["targets"].values():
            tgt_hist[v] += 1
    print("target histogram:", dict(sorted(tgt_hist.items())))
