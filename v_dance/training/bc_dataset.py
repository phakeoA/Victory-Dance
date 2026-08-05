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
from v_dance.encoders.state_encoder import (  # noqa: E402  (import after sys.path bootstrap)
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
    # 2026-07-24 futility batch: stored rows were stamped at EXPORT time and predate the
    # futility rules — subtract ONLY the futility bits (never a full mask recompute: the
    # stored row is the export-era legality truth; re-deriving everything would also
    # re-apply rules the snapshot can't always back). Training thus learns under the same
    # sharpened mask serve uses WITHOUT a corpus re-export; results are baked into the
    # encoded cache (encoded_cache._CACHE_SCHEMA bumped for exactly this). Any failure
    # falls open to the stored row.
    snap = t.get("state_before_actions")
    if snap and mask_all:
        try:
            from v_dance.encoders.action_codec import (
                _futile_buckets_offline, move_slots_for_mon)
            our_active = snap.get("our_active") or {}
            merged = {}
            for h, row in mask_all.items():
                mon = our_active.get(h)
                if not mon or not row:
                    merged[h] = row
                    continue
                row2 = list(row)
                for m_idx, (name, _c) in enumerate(move_slots_for_mon(mon)):
                    for b in _futile_buckets_offline(mon, snap, name):
                        idx = m_idx * 3 + b
                        if idx < len(row2):
                            row2[idx] = 0
                merged[h] = row2
            mask_all = merged
        except Exception:
            mask_all = t.get("action_mask") or {}
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
                orig = (t.get("action_mask") or {}).get(head) or []
                if 0 <= ai < len(orig) and orig[ai] == 1:
                    # LOUD (dont-defer-gaps): the futility refresh masked a HUMAN label —
                    # deliberate (don't imitate provably-wasted clicks), but visible.
                    stats["skipped_futile_target"] += 1
                else:
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
        elif stats is not None and grow:
            # #27: SURFACE a stale-export gap. gimmick_mask is BAKED into the JSONL at parse time, so a
            # file exported before GIMMICK_DIM grew (the v11 2->3 tera bump) stores len-2 rows -> the
            # strict-equality guard silently drops EVERY gimmick label (incl. the 'none' negatives) for
            # that file. A loud counter beats silent supervision loss (dont-defer-gaps).
            if len(grow) != GIMMICK_DIM:
                stats["skipped_gimmick_dim_mismatch"] += 1
            elif gi is not None:
                # #4: correct dim but the chosen gimmick is mask-illegal (the gimmick analogue of
                # skipped_illegal_target). Dropping the label is correct; count it so it isn't silent.
                # (gi is None = a legitimate no-gimmick decision — NOT a skip, so it's not counted.)
                stats["skipped_gimmick_illegal"] += 1

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
        # Trajectory key half (Phase 1b sequence BC): one replay yields TWO
        # independent trajectories, one per perspective — never mix them.
        "perspective": t.get("perspective"),
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
        from v_dance.encoders.state_encoder import annotate_opp_actions, OPP_HEADS  # noqa: E402
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

    import time as _time
    files = list(files)
    if limit_files is not None:
        files = files[:limit_files]

    _n_files = len(files)
    _t0 = _time.time()
    for _fi, fp in enumerate(files, 1):
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
        # Encode heartbeat: build_examples is the ~35-min SILENT gap before a first-launch
        # train/ruler (cache-MISS path). A periodic line reassures the long encode is live.
        # Gated on a large folder so small val sets / tests stay silent (no output change there).
        if _n_files >= 4000 and (_fi % 2000 == 0 or _fi == _n_files):
            _dt = max(_time.time() - _t0, 1e-6)
            print(f"[bc_dataset] encoding {_fi}/{_n_files} files "
                  f"({stats['transitions']} transitions, {_fi/_dt:.0f} files/s)", flush=True)
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


def canonical_rid(rid: str) -> str:
    """Canonical replay-id for CROSS-SOURCE duplicate detection (M8, 2026-07-11).

    The same battle can appear as ``gen9…`` (TypeB/HF exports), ``battle-gen9…``
    (Type_C live-recorded replays) or ``gen9…__closed`` (a closed-strip re-ingest
    twin) — raw string comparison misses all of those collisions. Strips the
    ``battle-`` prefix and the ``__closed`` suffix for COMPARISON ONLY (storage,
    filenames and stored replay_id fields are untouched)."""
    r = str(rid)
    if r.startswith("battle-"):
        r = r[len("battle-"):]
    if r.endswith("__closed"):
        r = r[: -len("__closed")]
    return r


def split_with_reference(
    extra_examples: Sequence[dict],
    ref_examples: Sequence[dict],
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[List[dict], List[dict]]:
    """Split with the val set PINNED to a fixed reference corpus.

    ``ref_examples`` (e.g. the anchor's original folders) is split exactly like
    :func:`split_by_replay` — same seed + val_frac reproduce the SAME val
    replays as a run trained on the reference corpus alone, so val metrics stay
    comparable across data-expansion runs.  ``extra_examples`` only ever add
    TRAIN data: any of them sharing a replay_id with the reference corpus is
    dropped (no val leakage, no double-weighted duplicate games).

    Returns ``(train, val)``.
    """
    ref_train, val = split_by_replay(ref_examples, val_frac=val_frac, seed=seed)
    # M8: compare CANONICAL ids so a "battle-" or "__closed" twin of a val
    # replay can never slip into train (raw-string comparison missed those).
    ref_rids = {canonical_rid(e["replay_id"]) for e in ref_examples}
    extra = [e for e in extra_examples if canonical_rid(e["replay_id"]) not in ref_rids]
    return list(ref_train) + extra, val


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


def compute_closed_copy_weights(
    examples: Sequence[dict],
    lam_closed: float,
) -> np.ndarray:
    """Era-4 arm 2a (design §2a): per-decision normalization of the HF open/closed
    twin ingest.

    Arm-B-style corpora contain every HF tournament game TWICE — the open parse
    (``rid``) and the closed-strip re-ingest (``rid__closed``) — both at full
    weight, i.e. a verified 2× upweight of tournament modal lines.  This weight
    splits each PAIR so it totals ONE decision: the closed member gets
    ``lam_closed``, the open member ``1 − lam_closed`` (``lam_closed=1.0`` =
    closed-only; ``0.5`` = equal halves).  Replays with no twin present in the
    dataset keep raw weight 1.0 — the flag only touches actual pairs.

    Pairing is REPLAY-level on :func:`canonical_rid` (robust to the ``battle-``
    prefix and to small decision-count drift between the two parses).  Returns a
    MEAN-NORMALISED vector (compute_sample_weights' invariant); compose with
    other weight vectors by multiplying and re-normalising.
    """
    if not (0.0 <= lam_closed <= 1.0):
        raise ValueError(f"lam_closed must be in [0, 1], got {lam_closed}")
    n = len(examples)
    w = np.ones(n, dtype=np.float64)
    if n == 0:
        return w.astype(np.float32)

    open_ids, closed_ids = set(), set()
    for e in examples:
        rid = str(e["replay_id"])
        (closed_ids if rid.endswith("__closed") else open_ids).add(canonical_rid(rid))
    paired = open_ids & closed_ids

    for i, e in enumerate(examples):
        rid = str(e["replay_id"])
        if canonical_rid(rid) in paired:
            w[i] = lam_closed if rid.endswith("__closed") else 1.0 - lam_closed

    mean = w.mean()
    if mean > 0:
        w = w / mean
    return w.astype(np.float32)


def compute_advantage_weights(
    examples: Sequence[dict],
    value_ckpt: str,
    mode: str = "exp",
    beta: float = 1.6,
    w_min: float = 0.2,
    w_max: float = 5.0,
    batch_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Phase-2 offline advantage weights (Metamon's "Exp" / the binary filter).

    Per decision, ``A = G − V(s)`` where ``G`` is the game outcome (1 won /
    0 lost — already on every example as ``won``) and ``V(s)`` is the TRAINED
    value head's sigmoid win-prob from ``value_ckpt`` — plain MC advantage, no
    bootstrapping, computed once here with one batched no-grad value pass.
    Shifts BC from "imitate everyone" to "imitate what beat expectation"
    without ever leaving the data manifold (the OOD search-leaf lesson).

      * ``mode="exp"``:    ``w = clip(exp(beta·A), w_min, w_max)``
      * ``mode="filter"``: ``w = 1[A > 0]`` (keep only better-than-expected)

    Unknown-outcome examples keep raw weight 1.0 (advantage needs a label).
    The vector is MEAN-NORMALISED to 1.0 (compute_sample_weights' invariant) so
    the loss scale is preserved; COMPOSE with other weight vectors by
    multiplying and re-normalising.

    ⚠ Calibration (P0.4 ruler 2026-07-02): current value heads' val Brier is
    ~0.26–0.30, barely under the 0.25 coin-flip line.  Advantage only has to
    RANK decisions within a game — that survives weak calibration — but keep
    ``beta`` conservative and the clip tight until a stronger V lands.
    """
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("torch is required for compute_advantage_weights")
    if mode not in ("exp", "filter"):
        raise ValueError(f"unknown advantage mode: {mode!r} (want 'exp' or 'filter')")
    n = len(examples)
    w = np.ones(n, dtype=np.float64)
    if n == 0:
        return w.astype(np.float32)

    from v_dance.play.model_io import load_bc_policy
    model, _heads = load_bc_policy(value_ckpt, device=device)
    model.eval()
    if not bool(getattr(model, "_value_trained", False)):
        raise ValueError(
            f"advantage weighting needs a checkpoint with a TRAINED value head: "
            f"{value_ckpt} is stamped value_trained=False (advantages from an "
            f"untrained V would bake pure noise into every loss weight)")

    labelled = np.array([i for i, e in enumerate(examples)
                         if e.get("won") is not None], dtype=np.int64)
    if labelled.size == 0:
        return w.astype(np.float32)   # no outcome labels anywhere → all-ones
    G = np.array([1.0 if examples[i]["won"] else 0.0 for i in labelled],
                 dtype=np.float64)
    V = np.empty(labelled.size, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, int(labelled.size), int(batch_size)):
            idx = labelled[s:s + batch_size]
            xb = torch.from_numpy(np.stack(
                [np.asarray(examples[i]["x"], dtype=np.float32) for i in idx]))
            _a, _g, value = model(xb.to(device))
            V[s:s + idx.size] = torch.sigmoid(value).double().cpu().numpy()

    A = G - V
    if mode == "exp":
        w[labelled] = np.clip(np.exp(beta * A), w_min, w_max)
    else:
        w[labelled] = (A > 0).astype(np.float64)
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
                 weights: Optional[Sequence[float]] = None,
                 sequence_len: int = 1,
                 archetype_ids: Optional[Sequence[int]] = None,
                 lazy_x: bool = False):
        if not _HAS_TORCH:  # pragma: no cover
            raise RuntimeError("torch is required for BCDataset")
        # Step B low-RAM mode (train_bc --mmap-cache): keep each example's x as
        # a reference (typically a read-only row view of the folder's mmap'd
        # cache X) instead of materializing the (N, STATE_DIM) matrix here —
        # ~26 GB for the full MA-Bo3 corpus.  __getitem__ copies ONE row per
        # fetch.  Sequence mode gathers history frames through the in-RAM
        # matrix, so it is unsupported (memory-seq is a closed null anyway).
        self.lazy_x = bool(lazy_x)
        if self.lazy_x and int(sequence_len) > 1:
            raise ValueError(
                f"lazy_x does not support sequence_len > 1 (got {sequence_len}): "
                "the sequence path gathers history frames from the in-RAM X matrix")
        # Phase 1b sequence BC: with sequence_len > 1 each item ALSO carries
        # ``x_seq`` (T, STATE_DIM) — the last ``sequence_len`` decision states
        # of the SAME (replay_id, perspective) trajectory ending at this item —
        # plus ``frame_padding_mask`` (T,) bool, True on the LEFT-padded zero
        # frames of early-game items.  Targets stay the item's own (the LAST
        # frame is supervised).  Never crosses a replay/perspective boundary by
        # construction.  sequence_len == 1 keeps every item byte-identical to
        # the stateless dataset (no x_seq key at all).
        self.sequence_len = int(sequence_len)
        if self.sequence_len < 1:
            raise ValueError(f"sequence_len must be >= 1, got {sequence_len}")
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
        self._n = n
        state_dim = get_state_dim()
        # M6 loader-workers (2026-07-11): a spawned DataLoader worker cannot inherit
        # memmap row views (pickling them materializes the FULL X — ~28 GB on the big
        # corpus), so lazy mode ALSO records each row's (cache file, row) from the
        # ``x_src`` stamps encoded_cache.load_cache writes; a worker re-opens the
        # cache by path on first fetch (see __getstate__/_x_np). In-process fetches
        # keep using the inherited views — byte-identical to the pre-M6 path.
        self._x_open: Dict[int, np.ndarray] = {}    # process-local mmap handles
        self._x_files: List[str] = []
        self._x_src: Optional[np.ndarray] = None    # (N, 2) int64 [file_id, row]
        if self.lazy_x:
            self.X = None
            self._x_rows: Optional[List[np.ndarray]] = [ex["x"] for ex in examples]
            fid: Dict[str, int] = {}
            src = np.full((len(examples), 2), -1, dtype=np.int64)
            for i, ex in enumerate(examples):
                s = ex.get("x_src")
                if not s:
                    src = None                      # non-cache examples: no worker path
                    break
                j = fid.setdefault(str(s[0]), len(fid))
                if j == len(self._x_files):
                    self._x_files.append(str(s[0]))
                src[i, 0], src[i, 1] = j, int(s[1])
            self._x_src = src
        else:
            self.X = np.zeros((n, state_dim), dtype=np.float32)
            self._x_rows = None
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
        # Phase-2 z: per-example own-team archetype id (team_archetypes join).
        # None (default) → the "archetype" item key is not emitted at all, so
        # the batch shape stays byte-identical for every non-z training run.
        self.archetype: Optional[np.ndarray] = None
        if archetype_ids is not None:
            av = np.asarray(archetype_ids, dtype=np.int64).ravel()
            if av.shape[0] != n:
                raise ValueError(
                    f"archetype_ids length {av.shape[0]} != number of examples {n}")
            self.archetype = av
        if self.with_opp:
            self.opp_target = np.full((n, len(OPP_HEADS)), -1, dtype=np.int64)
            self.opp_mask = np.zeros((n, len(OPP_HEADS), ACTIONS_PER_SLOT), dtype=np.float32)
            self.opp_valid = np.zeros((n, len(OPP_HEADS)), dtype=np.float32)
        self.replay_ids: List[str] = []
        # Per-example trajectory bookkeeping (sequence mode): examples arrive in
        # file order = temporal order within each (replay_id, perspective), so
        # grouping by key while preserving order reconstructs each trajectory.
        self._traj_of: List[Tuple[str, Optional[str]]] = []
        self._traj_indices: Dict[Tuple[str, Optional[str]], List[int]] = {}
        self._pos_in_traj = np.zeros(n, dtype=np.int64)

        for i, ex in enumerate(examples):
            if not self.lazy_x:
                self.X[i] = ex["x"]
            self.replay_ids.append(ex["replay_id"])
            tkey = (ex["replay_id"], ex.get("perspective"))
            traj = self._traj_indices.setdefault(tkey, [])
            self._pos_in_traj[i] = len(traj)
            traj.append(i)
            self._traj_of.append(tkey)
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
        # Lazy mode: no X matrix — rows are fetched (and copied) per item.
        self.X_t = torch.from_numpy(self.X) if not self.lazy_x else None
        self.target_t = torch.from_numpy(self.target)
        self.mask_t = torch.from_numpy(self.mask)
        self.valid_t = torch.from_numpy(self.valid)
        self.gimmick_target_t = torch.from_numpy(self.gimmick_target)
        self.gimmick_mask_t = torch.from_numpy(self.gimmick_mask)
        self.gimmick_valid_t = torch.from_numpy(self.gimmick_valid)
        self.value_target_t = torch.from_numpy(self.value_target)
        self.value_valid_t = torch.from_numpy(self.value_valid)
        self.weight_t = torch.from_numpy(self.weight)
        self.archetype_t = (torch.from_numpy(self.archetype)
                            if self.archetype is not None else None)
        if self.with_opp:
            self.opp_target_t = torch.from_numpy(self.opp_target)
            self.opp_mask_t = torch.from_numpy(self.opp_mask)
            self.opp_valid_t = torch.from_numpy(self.opp_valid)

    def __len__(self) -> int:
        return self._n

    @property
    def workers_safe(self) -> bool:
        """True when this dataset can be shipped to DataLoader worker processes:
        non-lazy (tensors ride torch shared memory) or lazy WITH x_src cache
        stamps (workers re-open the encoded caches by path). A lazy dataset from
        non-cache examples (--limit-* / --no-cache runs) is main-process-only."""
        return (not self.lazy_x) or self._x_src is not None

    def _x_np(self, idx: int) -> np.ndarray:
        """One example's x as a fresh writable float32 row (lazy mode only:
        copies the — possibly mmap'd, read-only — stored row). In a DataLoader
        worker the inherited views are gone (see __getstate__): the row is read
        from a per-process re-open of the encoded cache instead."""
        if self._x_rows is not None:
            return np.array(self._x_rows[idx], dtype=np.float32)
        j, row = int(self._x_src[idx, 0]), int(self._x_src[idx, 1])
        arr = self._x_open.get(j)
        if arr is None:
            arr = np.load(self._x_files[j], mmap_mode="r")
            self._x_open[j] = arr
        return np.array(arr[row], dtype=np.float32)

    # ── M6 loader-workers: worker-view pickling ────────────────────────────────
    # What crosses to a spawned worker: the torch tensors (shared memory — no
    # copy) + the small config fields + _x_files/_x_src. What must NOT cross:
    # memmap row views (materialize X), the numpy twins of the tensors (would
    # copy; rebuilt as tensor views), and the per-example python bookkeeping the
    # fetch path never touches when sequence_len == 1 (lazy forbids >1 anyway).
    _NUMPY_TWINS = ("X", "target", "mask", "valid", "gimmick_target",
                    "gimmick_mask", "gimmick_valid", "value_target",
                    "value_valid", "weight", "archetype",
                    "opp_target", "opp_mask", "opp_valid")

    def __getstate__(self):
        if self.lazy_x and self._x_src is None:
            raise RuntimeError(
                "BCDataset(lazy_x=True) built from non-cache examples carries no x_src "
                "stamps, so DataLoader workers cannot re-open its X rows — run without "
                "--loader-workers (smoke --limit-*/--no-cache runs bypass the encoded cache).")
        st = self.__dict__.copy()
        st["_x_rows"] = None
        st["_x_open"] = {}
        for k in self._NUMPY_TWINS:
            st.pop(k, None)
        if self.sequence_len <= 1:      # fetch path never reads these when T == 1
            st["replay_ids"] = []
            st["_traj_of"] = []
            st["_traj_indices"] = {}
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        # Rebuild the numpy twins as views of the (shared-memory) tensors.
        self.X = self.X_t.numpy() if self.X_t is not None else None
        self.target = self.target_t.numpy()
        self.mask = self.mask_t.numpy()
        self.valid = self.valid_t.numpy()
        self.gimmick_target = self.gimmick_target_t.numpy()
        self.gimmick_mask = self.gimmick_mask_t.numpy()
        self.gimmick_valid = self.gimmick_valid_t.numpy()
        self.value_target = self.value_target_t.numpy()
        self.value_valid = self.value_valid_t.numpy()
        self.weight = self.weight_t.numpy()
        self.archetype = (self.archetype_t.numpy()
                          if self.archetype_t is not None else None)
        if self.with_opp:
            self.opp_target = self.opp_target_t.numpy()
            self.opp_mask = self.opp_mask_t.numpy()
            self.opp_valid = self.opp_valid_t.numpy()

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

    def _z_fields(self, idx: int) -> dict:
        """Phase-2 archetype id (empty unless archetype_ids were supplied)."""
        if self.archetype_t is None:
            return {}
        return {"archetype": self.archetype_t[idx]}

    def _seq_fields(self, idx: int, x_last: "torch.Tensor") -> dict:
        """``x_seq`` (T, D) + ``frame_padding_mask`` (T,) in sequence mode.

        History frames come from the STORED (never-augmented) X — only the
        supervised LAST frame is ``x_last`` (the possibly-augmented fetch), so
        the action labels always match the frame they were remapped for.
        Early-game items are LEFT-padded with zero frames + a True mask (the
        model's memory attention ignores them)."""
        T = self.sequence_len
        if T <= 1:
            return {}
        traj = self._traj_indices[self._traj_of[idx]]
        pos = int(self._pos_in_traj[idx])
        hist = traj[max(0, pos - T + 1): pos]          # frames strictly before idx
        n_pad = T - 1 - len(hist)
        x_seq = torch.zeros(T, self.X_t.shape[1], dtype=torch.float32)
        if hist:
            x_seq[n_pad: T - 1] = self.X_t[hist]
        x_seq[T - 1] = x_last
        fpm = torch.zeros(T, dtype=torch.bool)
        if n_pad:
            fpm[:n_pad] = True
        return {"x_seq": x_seq, "frame_padding_mask": fpm}

    def __getitem__(self, idx: int) -> dict:
        if not self.augment_move_order:
            x_t = (torch.from_numpy(self._x_np(idx)) if self.lazy_x
                   else self.X_t[idx])
            return {
                "x": x_t,
                **self._seq_fields(idx, x_t),
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
                **self._z_fields(idx),
            }

        # Augmented fetch: permute each own active slot's move blocks + remap the
        # matching action label/mask.  Work on numpy copies so the stored tensors
        # stay pristine.  Gimmick (per-slot, not per-move) is unaffected.
        x = self._x_np(idx) if self.lazy_x else self.X[idx].copy()
        target = self.target[idx].copy()
        mask = self.mask[idx].copy()
        for h_idx in range(len(HEADS)):
            perm = self._rng.permutation(NUM_MOVES).tolist()
            permute_move_slots(x, h_idx, perm)
            if self.valid[idx, h_idx] > 0.5:
                target[h_idx] = permute_action_index(int(target[h_idx]), perm)
                mask[h_idx] = np.asarray(
                    permute_action_mask_row(mask[h_idx].tolist(), perm), dtype=np.float32)
        x_aug = torch.from_numpy(x)
        return {
            "x": x_aug,
            **self._seq_fields(idx, x_aug),
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
            **self._z_fields(idx),
        }


def bc_worker_init(worker_id: int) -> None:  # pragma: no cover - exercised via DataLoader
    """DataLoader ``worker_init_fn`` (M6): give each worker's move-order
    augmentation RNG an independent, run-deterministic stream. Without this,
    every worker would inherit the SAME pickled RandomState and draw identical
    permutation sequences. ``info.seed`` is torch's per-worker seed (derived
    from the run's base seed), so runs stay reproducible at a fixed worker count."""
    info = torch.utils.data.get_worker_info()
    if info is not None and getattr(info.dataset, "_rng", None) is not None:
        info.dataset._rng = np.random.RandomState(info.seed % (2 ** 32))


# ══════════════════════════════════════════════════════════════════════════════
def print_stats(stats: Counter) -> None:
    """Pretty-print a build_examples stats Counter."""
    order = [
        "files", "transitions", "replays", "slot_decisions", "usable_examples",
        "skipped_null_index", "skipped_no_mask", "skipped_illegal_target",
        "skipped_gimmick_dim_mismatch", "skipped_gimmick_illegal",
        "dropped_forced_replacement", "bad_files",
    ]
    # cp1252 consoles (non-interactive bash on Windows) can't encode the box-drawing chars —
    # a cosmetic header must never kill a multi-hour training launch (bit 2x on 2026-07-05/06).
    try:
        print("── BC dataset stats ─────────────────────────")
    except UnicodeEncodeError:
        print("-- BC dataset stats -------------------------")
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
