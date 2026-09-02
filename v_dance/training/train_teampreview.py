"""
Team-preview ("bring") trainer for Victory-Dance (VGC Reg M-A).

Behaviour-cloning of the team-preview decision from parsed replays:

  * input  : both teams' 6-mon rosters (the matchup)
  * targets: which 4 we brought (bring head) + which 2 led (lead head)
  * loss   : per-mon binary cross-entropy; the bring head is trained ONLY on
             examples with a complete observed 4-bring (valid_bring), the lead
             head on every example (leads are always revealed turn 1)
  * metrics: val exact-set match and top-k overlap for both heads
  * output : checkpoints the best model (by mean exact-match) to --out

Run under the GPU venv:

    .venv\\Scripts\\python.exe ai_train_scripts\\teamPreview_model\\train_teampreview.py \\
        --data data\\vods\\Prepared_training_data\\Regulation_MA\\Jsonl_TypeB --epochs 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
from v_dance.training.teampreview_dataset import (  # noqa: E402
    BRING_K,
    LEAD_K,
    MON_FEAT_DIM,
    TEAM_SIZE,
    TeamPreviewDataset,
    build_vocab,
    examples_from_folders,
    feature_recipe,
    print_stats,
    split_by_replay,
)
from v_dance.models.teampreview_model import build_model  # noqa: E402
from v_dance.training.resumable import (  # noqa: E402  (2026-07-19 shared resume machinery)
    RESUME_NAME,
    config_fingerprint,
    load_resume_state,
    restore_rng,
    save_resume_state,
)

# Contrastive set head (2026-07-11, docs/tp_contrastive_set_head_design.md): the canonical
# subset order shared by the loss, the metrics and model.score_subsets' default — all C(6,4)
# bring-subsets in itertools.combinations order, so a target index is well-defined everywhere.
SET_SUBSETS = tuple(combinations(range(TEAM_SIZE), BRING_K))
_SET_TARGET = {s: i for i, s in enumerate(SET_SUBSETS)}


def apply_warm_start(model, state: dict) -> int:
    """Load a donor checkpoint's weights into a set-head model. Every backbone key must
    load; ONLY fresh set-head keys may be missing (they keep their zero-init) — anything
    else is a config drift and fails loud. Returns the number of fresh keys."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith(("set_pair_mlp.", "set_global_mlp.",
                                                   "set_ctx_proj."))]
    if bad or unexpected:
        raise SystemExit(
            f"[train_teampreview] warm-start key mismatch — the donor checkpoint does not fit "
            f"this architecture: missing(non-set)={bad} unexpected={list(unexpected)}")
    return len(missing)


def _set_targets(batch) -> torch.Tensor:
    """(B,) long: index into SET_SUBSETS of the human 4-set, or -1 when the row cannot
    supervise the set head (incomplete observed bring, partial/aug-masked roster)."""
    bring = batch["bring"]
    valid = batch["valid_bring"]
    full = (batch["our_feat"].abs().sum(dim=-1) > 0).sum(dim=-1) == TEAM_SIZE   # (B,)
    slot_m = batch.get("slot_mask")
    tgt = torch.full((bring.shape[0],), -1, dtype=torch.long)
    for r in range(bring.shape[0]):
        if float(valid[r]) < 0.5 or not bool(full[r]):
            continue
        if slot_m is not None and float(slot_m[r].sum()) < TEAM_SIZE:
            continue                                     # aug-masked roster: members absent
        members = tuple(torch.nonzero(bring[r] > 0.5, as_tuple=False).flatten().tolist())
        if len(members) == BRING_K:
            tgt[r] = _SET_TARGET[members]
    return tgt


def _topk_set_metrics(logits: torch.Tensor, target: torch.Tensor, k: int,
                      sample_mask: Optional[torch.Tensor] = None):
    """Return (n_exact, sum_overlap, n) for 'predict the k-set' over a batch.

    ``sample_mask`` (bool, (B,)) restricts to valid rows (e.g. valid_bring)."""
    B = logits.shape[0]
    pred = logits.topk(k, dim=1).indices                        # (B, k)
    n = 0
    n_exact = 0
    sum_overlap = 0
    for i in range(B):
        if sample_mask is not None and not bool(sample_mask[i]):
            continue
        n += 1
        ps = set(pred[i].tolist())
        ts = set(torch.nonzero(target[i], as_tuple=False).flatten().tolist())
        ov = len(ps & ts)
        sum_overlap += ov
        if ps == ts:
            n_exact += 1
    return n_exact, sum_overlap, n


def _norm_uname(name) -> str:
    """Showdown username normalization for matching (display names vary in case/spacing)."""
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def compute_tp_weights(examples: Sequence[dict],
                       outcome_weight: bool = False, loss_weight: float = 0.5,
                       own_folders: Optional[Sequence[str]] = None,
                       own_boost: float = 15.0,
                       own_username: Optional[str] = None,
                       own_species: Optional[Sequence[str]] = None,
                       own_team_weight: float = 0.0) -> Optional[np.ndarray]:
    """TP-N3 per-example loss weights (docs/tp_n3_outcome_finetune_design.md).

    Multiplicative: WON examples from an ``--own-folders`` folder ×``own_boost`` (lost/
    unknown own rows stay unboosted — see the won-only comment below); with
    ``outcome_weight``, lost games ×``loss_weight`` (won/unknown ×1.0). The vector is
    MEAN-NORMALISED to 1.0 (bc_dataset.compute_sample_weights' invariant) so weights only
    redistribute emphasis. Returns None when nothing is enabled — the dataset then emits
    no "weight" key and run_epoch keeps the exact legacy loss path (byte-identical).

    ``own_username`` (N3 re-run, 2026-07-21): the TP extractor mines BOTH perspectives of
    every replay, so a bare folder match boosts OPPONENT-perspective wins too — opponents
    beating us got imitated ×15 at n=171 (the parking root cause). When set, only rows
    whose perspective username matches are ``own``.
    """
    own_set = {s for s in (own_species or ()) if s}
    if not outcome_weight and not own_folders and not own_set:
        return None
    own = tuple(os.path.normcase(os.path.abspath(f)) for f in (own_folders or ()))
    uname_want = _norm_uname(own_username) if own_username else None
    w = np.ones(len(examples), dtype=np.float64)
    n_own = n_boost = n_lost = n_opp_perspective = 0
    for i, ex in enumerate(examples):
        src = os.path.normcase(os.path.abspath(str(ex.get("source_file") or "")))
        is_own = bool(own) and src.startswith(own)
        if is_own and uname_want is not None and _norm_uname(ex.get("username")) != uname_want:
            n_opp_perspective += 1
            is_own = False
        if is_own:
            n_own += 1
        # Won-ONLY boost (N3 iter-3): BC has no negative gradient — boosting a LOST own
        # game amplifies the mistake (iter-2 probe: own losses at ×boost×loss_weight = 7.5×
        # corpus tripled the 0W–5L Zard brings). Lost/unknown own rows stay unboosted.
        if is_own and ex.get("won") is True:
            w[i] *= float(own_boost)
            n_boost += 1
        if outcome_weight and ex.get("won") is False:
            w[i] *= float(loss_weight)
            n_lost += 1
    if own and n_own == 0:
        # A typo'd folder silently weighting nothing is exactly the failure mode we refuse
        # to ship (a flag that LOOKS on but does nothing) — fail loud instead.
        raise SystemExit(f"[train_teampreview] --own-folders matched 0 examples: {own_folders} "
                         f"(folder in --data? --own-username '{own_username}' spelled like the "
                         f"in-replay name? {n_opp_perspective} rows matched the folder but not "
                         f"the username)")
    # Era-5 W1 (2026-09-01): own-TEAM weighting — 1 + λ·(fraction of OUR six in the example's
    # roster); every TP example carries ``our_species`` so no corpus walk is needed. Composes
    # multiplicatively with the folder/outcome weights above.
    if own_set:
        from collections import Counter as _Counter
        hist = _Counter()
        for i, ex in enumerate(examples):
            ov = len(set(ex.get("our_species") or ()) & own_set) / float(len(own_set))
            hist[int(round(ov * 6))] += 1
            w[i] *= 1.0 + float(own_team_weight) * ov
        if sum(hist.get(k, 0) for k in range(1, 7)) == 0:
            raise SystemExit(f"[train_teampreview] --own-team ({', '.join(sorted(own_set))}) overlaps "
                             f"NO example — wrong team / paste? Refusing to run a no-op.")
        print(f"[train_teampreview] own-team weighting λ={own_team_weight:g} — overlap histogram "
              f"(mons of 6): " + ", ".join(f"{k}:{hist.get(k, 0)}" for k in range(7)))
    mean = w.mean()
    if mean > 0:
        w = w / mean
    uname_note = (f", {n_opp_perspective} opp-perspective rows excluded by --own-username"
                  if uname_want is not None else "")
    print(f"[train_teampreview] TP-N3 weights: {n_own} own-folder examples "
          f"({n_boost} WON → ×{own_boost:g}; lost/unknown unboosted){uname_note}, "
          f"{n_lost} lost-game examples (×{loss_weight:g}), mean-normalised over {len(examples)}")
    return w.astype(np.float32)


def run_epoch(model, loader, device, optimizer=None, set_weight: float = 1.0,
              progress_secs: float = 600.0, progress_label: str = "") -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    use_set = bool(getattr(model, "use_set_head", False))

    tot_loss = 0.0
    n_batches = 0
    bring_exact = bring_overlap = bring_n = 0
    lead_exact = lead_overlap = lead_n = 0
    set_exact = set_n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    _n_total = len(loader)
    _t0 = _last_beat = time.time()
    with ctx:
        for _bi, batch in enumerate(loader, 1):
            # Intra-epoch heartbeat (2026-07-19, USER): time-based, ON by default —
            # TP epochs are usually short so this rarely fires, but a future big-corpus
            # TP run is never a silent wait. Print-only.
            _now = time.time()
            if train and progress_secs and (_now - _last_beat) >= progress_secs:
                _last_beat = _now
                _dt = max(_now - _t0, 1e-6)
                _eta = (_n_total - _bi) / max(_bi / _dt, 1e-9) / 60.0
                print(f"    {time.strftime('%H:%M:%S')} [{progress_label} batch "
                      f"{_bi}/{_n_total} ({100.0 * _bi / _n_total:.1f}%)] "
                      f"epoch ETA {_eta:.1f} min", flush=True)
            our_idx = batch["our_idx"].to(device)
            opp_idx = batch["opp_idx"].to(device)
            our_feat = batch["our_feat"].to(device)
            opp_feat = batch["opp_feat"].to(device)
            bring = batch["bring"].to(device)
            lead = batch["lead"].to(device)
            valid = batch["valid_bring"].to(device)
            valid_lead = batch["valid_lead"].to(device)
            # 15b-train.1: the teammate-bias prior (None unless the SBDA dataset precomputed it).
            our_aff = batch.get("our_affinity")
            if our_aff is not None:
                our_aff = our_aff.to(device)
            # DS-4c stage 3: Bo3 previous-game context (None unless the dataset emitted it
            # AND the model consumes it — zero rows are the zero-init identity either way).
            ctx_kw = {}
            if getattr(model, "use_set_ctx", False) and batch.get("our_set_ctx") is not None:
                ctx_kw = {"our_set_ctx": batch["our_set_ctx"].to(device),
                          "opp_set_ctx": batch["opp_set_ctx"].to(device)}

            if use_set:
                # ONE trunk forward yields subset scores AND the marginal logits.
                set_scores, bring_logits, lead_logits = model.score_subsets(
                    our_idx, opp_idx, our_feat, opp_feat, our_aff, subsets=SET_SUBSETS,
                    **ctx_kw)
            else:
                bring_logits, lead_logits = model(our_idx, opp_idx, our_feat, opp_feat,
                                                  our_aff, **ctx_kw)

            # Per-slot loss weights (subset-mask aug, 2026-07-10): an aug-masked slot is ABSENT,
            # not a choice — exclude it from both heads' BCE. Unaugmented rows carry an all-ones
            # mask (sum/count == the old plain mean, numerically identical); batches without the
            # key (older callers) fall back to mean(dim=1) unchanged.
            slot_m = batch.get("slot_mask")
            if slot_m is not None:
                slot_m = slot_m.to(device)

            def _slot_mean(bce):                                     # (B,6) -> (B,)
                if slot_m is None:
                    return bce.mean(dim=1)
                return (bce * slot_m).sum(dim=1) / slot_m.sum(dim=1).clamp_min(1.0)

            # TP-N3: optional per-row loss weight (mean-normalised over the split; the count
            # denominators below stay UNweighted, so weights only redistribute emphasis —
            # bc_dataset.compute_sample_weights' invariant). Batches without the key run the
            # exact legacy expressions — byte-identical by construction.
            w = batch.get("weight")
            if w is not None:
                w = w.to(device)

            # lead head: only valid_lead rows (audit: a species-match-shifted lead is masked out, same as
            # valid_bring masks the bring head). On a clean corpus (all valid) this == the old plain mean.
            lead_bce = _slot_mean(F.binary_cross_entropy_with_logits(
                lead_logits, lead, reduction="none"))                # (B,)
            lead_num = (lead_bce * valid_lead) if w is None else (lead_bce * valid_lead * w)
            lead_loss = lead_num.sum() / valid_lead.sum().clamp_min(1.0)
            bring_bce = _slot_mean(F.binary_cross_entropy_with_logits(
                bring_logits, bring, reduction="none"))              # (B,)
            denom = valid.sum().clamp_min(1.0)
            bring_num = (bring_bce * valid) if w is None else (bring_bce * valid * w)
            bring_loss = bring_num.sum() / denom
            loss = lead_loss + bring_loss

            if use_set:
                # Listwise contrastive: 15-way CE, target = the human 4-set, ALL other
                # subsets of the same roster are the negatives (no sampling — C(6,4)=15).
                # Rows that cannot supervise the set (incomplete bring / partial roster)
                # are masked out of this term only; they still train the BCEs above.
                tgt = _set_targets(batch)
                mask = tgt >= 0
                if bool(mask.any()):
                    if w is None:
                        set_loss = F.cross_entropy(set_scores[mask.to(device)],
                                                   tgt[mask].to(device))
                    else:
                        # TP-N3 weighted rows; count denominator (weights are mean-normalised).
                        md = mask.to(device)
                        ce = F.cross_entropy(set_scores[md], tgt[mask].to(device),
                                             reduction="none")
                        set_loss = (ce * w[md]).sum() / md.sum().clamp_min(1).to(ce.dtype)
                    loss = loss + set_weight * set_loss
                    pred = set_scores.argmax(dim=1).cpu()
                    set_exact += int((pred[mask] == tgt[mask]).sum())
                    set_n += int(mask.sum())

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tot_loss += float(loss.item())
            n_batches += 1

            be, bo, bn = _topk_set_metrics(bring_logits, bring, BRING_K, valid > 0.5)
            # audit: mirror the loss's valid_lead mask in the METRIC too — else an invalid (species-shifted)
            # lead row is a guaranteed exact-miss that dilutes lead_exact + the checkpoint-selection score.
            le, lo, ln = _topk_set_metrics(lead_logits, lead, LEAD_K, valid_lead > 0.5)
            bring_exact += be; bring_overlap += bo; bring_n += bn
            lead_exact += le; lead_overlap += lo; lead_n += ln

    bn = max(bring_n, 1); ln = max(lead_n, 1)
    return {
        "loss": tot_loss / max(n_batches, 1),
        "bring_exact": bring_exact / bn,
        "bring_overlap": bring_overlap / (bn * BRING_K),
        "lead_exact": lead_exact / ln,
        "lead_overlap": lead_overlap / (ln * LEAD_K),
        "bring_n": bring_n,
        "lead_n": lead_n,
        "set_exact": set_exact / max(set_n, 1),
        "set_n": set_n,
    }


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[train_teampreview] WARNING: cuda unavailable -> cpu")
        device = "cpu"

    folders = list(args.data)
    if args.type_a:
        folders.extend(args.type_a)

    # Set-head warm start (2026-07-11): adopt the donor's vocab + architecture stamps so
    # every backbone weight fits exactly; only the fresh (zero-inited) set-head keys differ.
    # Loaded BEFORE the feature-recipe block because the adopted attn flags feed it.
    warm_ckpt = None
    if args.warm_start:
        warm_ckpt = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if not (isinstance(warm_ckpt, dict) and "model_state" in warm_ckpt):
            raise SystemExit(f"[train_teampreview] --warm-start {args.warm_start} is not a "
                             f"dict checkpoint")
        wcfg = warm_ckpt.get("config", {})
        if wcfg.get("features", "legacy") != args.features:
            raise SystemExit(f"[train_teampreview] --warm-start features="
                             f"{wcfg.get('features')!r} != --features {args.features!r}")
        args.emb_dim = wcfg.get("emb_dim", args.emb_dim)
        args.hidden = wcfg.get("hidden", args.hidden)
        args.dropout = wcfg.get("dropout", args.dropout)
        args.attn_heads = wcfg.get("attn_heads", args.attn_heads)
        args.self_attn = wcfg.get("use_self_attn", args.self_attn)
        args.cross_attn = wcfg.get("use_cross_attn", args.cross_attn)
        args.teammate_bias = wcfg.get("use_teammate_bias", args.teammate_bias)
        print(f"[train_teampreview] warm-start donor {args.warm_start}: adopted arch stamps "
              f"(emb={args.emb_dim} hidden={args.hidden} self_attn={args.self_attn} "
              f"cross_attn={args.cross_attn} teammate_bias={args.teammate_bias}) + vocab "
              f"({len(warm_ckpt.get('vocab', {}))} species; new-corpus species -> PAD)")

    # 15b-train.1: choose the per-mon feature recipe + (for the teammate-bias) the affinity provider.
    # 'legacy' = the original 46-dim dex net (no belief, no schema -> byte-identical). 'sbda' = the
    # shared tp_features extractor (FEAT_DIM, stamped feature_schema) — needs the SAME Pikalytics belief
    # the serve path resolves, so train==serve. teammate-bias only acts inside self-attention.
    belief = None
    affinity_fn = None
    if args.features == "sbda":
        from v_dance.parser.belief_state import BeliefState
        from v_dance.formats import pikalytics_path_for, default_format
        fmt = args.format or default_format()
        belief_path = Path(args.belief) if args.belief else pikalytics_path_for(fmt)
        if not (belief_path and Path(belief_path).exists()):
            raise SystemExit(f"[train_teampreview] --features sbda needs a Pikalytics belief; "
                             f"missing for {fmt}: {belief_path}")
        belief = BeliefState(belief_path)
        if args.teammate_bias and not args.self_attn:
            print("[train_teampreview] --teammate-bias requires self-attention (the bias acts inside "
                  "it) -> enabling --self-attn")
            args.self_attn = True
        if args.teammate_bias:
            from v_dance.training.tp_features import teammate_affinity_matrix
            affinity_fn = lambda sp: teammate_affinity_matrix(sp, belief, n=TEAM_SIZE)  # noqa: E731
        print(f"[train_teampreview] features=sbda belief={Path(belief_path).name} "
              f"self_attn={args.self_attn} cross_attn={args.cross_attn} "
              f"teammate_bias={args.teammate_bias}")
    feat_fn, feat_dim, feature_schema = feature_recipe(args.features, belief)
    if warm_ckpt is not None and int(warm_ckpt.get("config", {}).get("feat_dim", 0)) != feat_dim:
        raise SystemExit(
            f"[train_teampreview] --warm-start feat_dim="
            f"{warm_ckpt.get('config', {}).get('feat_dim')} is out of lockstep with the current "
            f"extractor ({feat_dim}) — the donor predates a tp_features schema change.")

    print(f"[train_teampreview] loading from: {folders}")
    t0 = time.time()
    examples, stats = examples_from_folders(folders, limit_files=args.limit_files, feat_fn=feat_fn)
    print_stats(stats)
    print(f"[train_teampreview] {len(examples)} examples in {time.time()-t0:.1f}s")
    if not examples:
        raise SystemExit("[train_teampreview] no examples found")

    # Warm start adopts the donor vocab (backbone emb rows must keep their meaning);
    # species only in the new corpus map to PAD 0 — the exact serve OOV behavior.
    vocab = warm_ckpt["vocab"] if warm_ckpt is not None else build_vocab(examples)
    train_ex, val_ex = split_by_replay(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"[train_teampreview] vocab={len(vocab)} species | "
          f"split -> {len(train_ex)} train / {len(val_ex)} val "
          f"({len({e['replay_id'] for e in train_ex})} / "
          f"{len({e['replay_id'] for e in val_ex})} replays)")

    # TP-N3: outcome/own-folder weights on the TRAIN split only — val metrics stay
    # unweighted (comparable across runs). None when the flags are off.
    own_species = None
    if getattr(args, "own_team", None):
        from v_dance.training.bc_dataset import parse_team_species
        own_species = parse_team_species(args.own_team)
        print(f"[train_teampreview] own team: {', '.join(own_species)} "
              f"(λ={args.own_team_weight:g})")
    tp_weights = compute_tp_weights(
        train_ex, outcome_weight=args.outcome_weight, loss_weight=args.loss_weight,
        own_folders=args.own_folders, own_boost=args.own_boost,
        own_username=args.own_username,
        own_species=own_species, own_team_weight=getattr(args, "own_team_weight", 0.0))

    # TP-N3 re-run: checkpoint selection on an OWN-GAMES val slice (bot-perspective rows of
    # the own folders inside the standard val split). Global val stays the printed no-forgetting
    # metric; ONLY best-ckpt selection switches — iter-1's failure was corpus-val selection
    # saving an epoch the treatment never shaped.
    own_val_loader = None
    if args.select_own_val:
        if not args.own_folders:
            raise SystemExit("[train_teampreview] --select-own-val requires --own-folders")
        _own = tuple(os.path.normcase(os.path.abspath(f)) for f in args.own_folders)
        _want = _norm_uname(args.own_username) if args.own_username else None
        own_val_ex = [e for e in val_ex
                      if os.path.normcase(os.path.abspath(str(e.get("source_file") or "")))
                      .startswith(_own)
                      and (_want is None or _norm_uname(e.get("username")) == _want)]
        if len(own_val_ex) < 25:
            raise SystemExit(f"[train_teampreview] --select-own-val: only {len(own_val_ex)} "
                             f"own val examples (<25) — selection would be noise. Grow the "
                             f"corpus or raise --val-frac.")
        print(f"[train_teampreview] --select-own-val: best-ckpt selection on "
              f"{len(own_val_ex)} own-games val examples (global val stays the printed metric)")
        own_val_loader = DataLoader(
            TeamPreviewDataset(own_val_ex, vocab, feat_dim=feat_dim, affinity_fn=affinity_fn,
                               with_set_ctx=args.set_ctx),
            batch_size=args.batch_size, shuffle=False)

    # Subset-mask aug is TRAIN-only — the val split stays clean (checkpoint selection + the
    # tp_val_report gates must measure the un-augmented task).
    train_loader = DataLoader(
        TeamPreviewDataset(train_ex, vocab, feat_dim=feat_dim, affinity_fn=affinity_fn,
                           subset_mask_p=args.subset_mask_aug, subset_mask_k=args.subset_mask_k,
                           aug_seed=args.seed, with_set_ctx=args.set_ctx,
                           weights=tp_weights),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TeamPreviewDataset(val_ex, vocab, feat_dim=feat_dim, affinity_fn=affinity_fn,
                           with_set_ctx=args.set_ctx),
        batch_size=args.batch_size, shuffle=False)
    if args.set_ctx:
        n_ctx = sum(1 for e in train_ex if e.get("our_set_ctx") is not None)
        print(f"[train_teampreview] Bo3 set-context: ON — {n_ctx}/{len(train_ex)} train "
              f"examples carry previous-game context (zeros elsewhere = identity)")

    model = build_model(
        vocab_size=len(vocab) + 1,   # +1 for the reserved PAD id 0
        feat_dim=feat_dim,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        device=device,
        use_self_attn=args.self_attn,
        use_cross_attn=args.cross_attn,
        attn_heads=args.attn_heads,
        use_teammate_bias=args.teammate_bias,
        use_set_head=args.set_head,
        use_set_ctx=args.set_ctx,
    )
    if warm_ckpt is not None:
        n_fresh = apply_warm_start(model, warm_ckpt["model_state"])
        print(f"[train_teampreview] warm-started backbone ({n_fresh} fresh set-head keys keep "
              f"their zero-init -> set decode == donor greedy at epoch 0)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # <type>_<variant> scheme: SBDA vs legacy base, by the --features mode (avoids the old generic best.pt)
    ckpt_path = out_dir / ("teampreview_sbda.pt" if args.features == "sbda" else "teampreview_base.pt")
    config = {
        "vocab_size": len(vocab) + 1,
        "feat_dim": feat_dim,
        "emb_dim": args.emb_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "bring_k": BRING_K,
        "lead_k": LEAD_K,
        "lr": args.lr,
        "data": folders,
        "subset_mask_aug": args.subset_mask_aug,   # tier-2 stamp: joint decode valid iff > 0
        "subset_mask_k": args.subset_mask_k,
        "patience": args.patience,
        # 15b-train.1: architecture + feature-recipe stamps so model_io.load_team_chooser rebuilds the
        # exact net and uses_tp_features/the lockstep guard select the matching SERVE recipe. The
        # feature_schema is stamped ONLY for SBDA (legacy carries none -> uses_tp_features False).
        "use_self_attn": bool(args.self_attn),
        "use_cross_attn": bool(args.cross_attn),
        "attn_heads": args.attn_heads,
        "use_teammate_bias": bool(args.teammate_bias),
        "use_set_head": bool(args.set_head),   # model_io dispatches the set decode on this stamp
        "use_set_ctx": bool(args.set_ctx),     # DS-4c: serve feeds Bo3 prev-game context when stamped
        # M5 guard (2026-07-13): only a run that ACTUALLY trained on non-zero opp OTS overlays may
        # receive them at serve — schema=tpfeat-v7 does not imply it (checkpoints_set is v7, closed).
        "ots_overlay_trained": bool(args.ots_overlay),
        "set_weight": args.set_weight,
        "warm_start": str(args.warm_start) if args.warm_start else None,
        "features": args.features,
        # TP-N3 provenance stamps (docs/tp_n3_outcome_finetune_design.md).
        "outcome_weight": bool(args.outcome_weight),
        "loss_weight": args.loss_weight,
        "own_folders": list(args.own_folders) if args.own_folders else None,
        "own_boost": args.own_boost,
        # N3 re-run provenance (2026-07-21). New config keys shift the resume fingerprint —
        # fine: no interrupted TP runs are pending, and era-chain runs start fresh anyway.
        "own_username": args.own_username,
        "select_own_val": bool(args.select_own_val),
    }
    if feature_schema:
        config["feature_schema"] = feature_schema

    best = -1.0
    epochs_no_improve = 0
    history: List[dict] = []
    start_epoch = 1
    resume_path = out_dir / RESUME_NAME
    config_fp = config_fingerprint(config)
    if getattr(args, "resume", False):
        rs = load_resume_state(resume_path, config_fp, device=device,
                               label="train_teampreview")
        # TP integrity guards beyond the config fingerprint: the model's embedding rows
        # are keyed by the vocab, and both are derived from the CORPUS — if the data
        # changed since the interrupted run, resuming would silently mis-index species.
        ex_ = rs.get("extra") or {}
        if ex_.get("vocab") != vocab:
            raise SystemExit("[train_teampreview] --resume REFUSED: the rebuilt species "
                             "vocab differs from the interrupted run's (the corpus "
                             "changed). Start fresh without --resume.")
        if ex_.get("n_examples") != len(examples):
            raise SystemExit(f"[train_teampreview] --resume REFUSED: example count "
                             f"changed ({ex_.get('n_examples')} -> {len(examples)}) — "
                             f"the corpus changed. Start fresh without --resume.")
        model.load_state_dict(rs["model_state"])
        optimizer.load_state_dict(rs["optimizer_state"])
        best = float(rs["best"])
        epochs_no_improve = int(rs["epochs_no_improve"])
        history = list(rs.get("history") or [])
        restore_rng(rs["rng"])
        start_epoch = int(rs["epoch"]) + 1
        print(f"[train_teampreview] RESUMED from {resume_path.name}: completed epoch "
              f"{rs['epoch']}, best mean-exact {best:.4f}, continuing at epoch "
              f"{start_epoch}/{args.epochs} (RNG streams restored).")
        if start_epoch > args.epochs:
            print("[train_teampreview] resume: epoch budget already exhausted — raise "
                  "--epochs to extend the run.")
    print(f"[train_teampreview] resumable: state -> {resume_path.name} after every "
          f"epoch (interrupt any time; continue with --resume)")
    for epoch in range(start_epoch, args.epochs + 1):
        _ep_t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer, set_weight=args.set_weight,
                       progress_secs=args.progress_secs, progress_label=f"e{epoch}")
        va = run_epoch(model, val_loader, device, optimizer=None, set_weight=args.set_weight)
        history.append({"epoch": epoch, "train": tr, "val": va})
        set_col = f"set exact {va['set_exact']:.3f} | " if args.set_head else ""
        print(
            f"{time.strftime('%H:%M:%S')} epoch {epoch:3d} "
            f"[{(time.time() - _ep_t0) / 60.0:.1f} min] | train loss {tr['loss']:.4f} | "
            f"val loss {va['loss']:.4f} {set_col}"
            f"lead exact {va['lead_exact']:.3f} ovlp {va['lead_overlap']:.3f} | "
            f"bring exact {va['bring_exact']:.3f} ovlp {va['bring_overlap']:.3f}"
        )
        # Set-head runs are selected on the SERVE metric (the set decode picks the bring);
        # marginal-only runs keep the historical bring_exact score. --select-own-val swaps
        # the SELECTION basis to the own-games slice (global va stays printed + in history).
        sel = va
        if own_val_loader is not None:
            sel = run_epoch(model, own_val_loader, device, optimizer=None,
                            set_weight=args.set_weight)
            print(f"           own-val: lead {sel['lead_exact']:.3f} "
                  f"set {sel.get('set_exact', float('nan')):.3f} "
                  f"bring {sel['bring_exact']:.3f}")
        score = 0.5 * (sel["lead_exact"] + (sel["set_exact"] if args.set_head
                                            else sel["bring_exact"]))
        if score > best:
            best = score
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "config": config,
                        "vocab": vocab, "epoch": epoch, "val_metrics": va}, ckpt_path)
            print(f"           ↑ new best (mean exact {best:.3f}) -> {ckpt_path}")
        else:
            epochs_no_improve += 1
            if args.patience and epochs_no_improve >= args.patience:
                print(f"[train_teampreview] early stop at epoch {epoch} "
                      f"(no val mean-exact gain for {args.patience} epochs)")
                break
        # Epoch-boundary resume state (atomic; early-stop ``break`` above skips it — an
        # early-stopped run is COMPLETE, not paused). extra = TP integrity guards.
        save_resume_state(resume_path, epoch=epoch, model=model, optimizer=optimizer,
                          best=best, epochs_no_improve=epochs_no_improve,
                          history=history, config_fp=config_fp,
                          extra={"vocab": vocab, "n_examples": len(examples)})

    if args.save_last:
        last_path = ckpt_path.with_name(ckpt_path.stem + "_last.pt")
        torch.save({"model_state": model.state_dict(),
                    "config": {**config, "saved_epoch": "last"},
                    "vocab": vocab, "epoch": epoch, "val_metrics": va}, last_path)
        print(f"[train_teampreview] --save-last: final-epoch (ep{epoch}) weights -> {last_path}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[train_teampreview] done. best mean-exact = {best:.3f}. ckpt: {ckpt_path}")
    return {"best": best, "history": history, "checkpoint": str(ckpt_path), "vocab_size": len(vocab)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the team-preview (bring) model")
    _prep = Path("data") / "vods" / "Prepared_training_data" / "Regulation_MA"
    default_data = [str(_prep / f"Jsonl_Type{t}") for t in ("A", "B", "C", "D")]
    ap.add_argument("--data", nargs="+", default=default_data,
                    help="folder(s) of per-replay .jsonl (default: all VOD types "
                         "A-D; missing/empty folders are skipped, files deduped)")
    ap.add_argument("--type-a", nargs="+", default=None,
                    help="extra folder(s) to append (deduped). Type A is already in "
                         "the default --data; use only for ad-hoc extra data.")
    # 15b-train.1: feature recipe + SBDA architecture (defaults reproduce the original legacy net)
    ap.add_argument("--features", choices=["legacy", "sbda"], default="legacy",
                    help="per-mon feature recipe: 'legacy' 46-dim dex (default) or 'sbda' tp_features")
    ap.add_argument("--self-attn", action="store_true",
                    help="SBDA: self-attention over our 6 (makes the synergy tags interact)")
    ap.add_argument("--cross-attn", action="store_true",
                    help="SBDA: our mons cross-attend to the opponent's 6")
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--teammate-bias", action="store_true",
                    help="SBDA: Pikalytics co-occurrence prior as a self-attn bias (implies --self-attn)")
    ap.add_argument("--format", default=None,
                    help="format whose Pikalytics belief feeds SBDA features (default: active format)")
    ap.add_argument("--belief", default=None,
                    help="explicit Pikalytics json for SBDA features (default: --format's resolved file)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--subset-mask-aug", type=float, default=0.0,
                    help="TP tier-2 (2026-07-10): prob of masking roster slots per TRAIN item "
                         "so partial rosters are in-distribution (enables model_io's joint bring "
                         "decode; gate = tp_val_report --joint-ab). 0 = off (byte-identical).")
    ap.add_argument("--subset-mask-k", type=int, default=0,
                    help="slots to mask per augmented item: 0 = random K in {1,2}; 2 = exactly "
                         "two (every augmented item = the joint decode's 4-mon context).")
    ap.add_argument("--set-head", action="store_true",
                    help="contrastive set-scoring head (2026-07-11 design): score complete "
                         "4-subsets as units; listwise 15-way CE vs the human set. The ckpt "
                         "stamp switches model_io to the set decode (TP_SET_HEAD kill-switch).")
    ap.add_argument("--set-weight", type=float, default=1.0,
                    help="weight of the set CE in the joint loss (lead_bce + bring_bce + λ·set_ce)")
    ap.add_argument("--warm-start", default=None,
                    help="donor TP checkpoint: adopt its vocab + arch stamps, load every "
                         "backbone weight; only fresh set-head keys stay zero-inited (so the "
                         "run STARTS at the donor's serve behavior).")
    ap.add_argument("--set-ctx", action="store_true",
                    help="DS-4c stage 3: consume Bo3 previous-game [brought, led] per mon as "
                         "a zero-init side input (bo3_set_id groups the corpus; game-1/off-set "
                         "rows carry zeros = identity). Serve feeds it via bo3_state.")
    ap.add_argument("--ots-overlay", action="store_true",
                    help="CERTIFY that this run's data exercises non-zero opponent OTS overlays "
                         "(open-team-sheet reveals). Stamps config.ots_overlay_trained=True so the "
                         "M5 serve guard (player.ots_opp_known) will feed VD_TP_OTS_OVERLAY input "
                         "to THIS ckpt only. Omit for closed-sheet data (e.g. Type A/B) — the "
                         "tpfeat-v7 schema alone does NOT certify overlay training.")
    ap.add_argument("--progress-secs", type=float, default=600.0,
                    help="TIME-based intra-epoch heartbeat every N seconds of a TRAIN "
                         "epoch (default 600; 0 = off).")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from <out>/resume_state.pt "
                         "(written after every completed epoch; model+optimizer+counters+"
                         "RNG restored; refuses config-fingerprint / vocab / example-count "
                         "mismatches). Also extends a finished run when --epochs is "
                         "raised. Pass the SAME flags as the original run.")
    ap.add_argument("--save-last", action="store_true",
                    help="TP-N3: additionally save the FINAL-epoch weights to *_last.pt. "
                         "Best-val selection fights a weighted objective (it saved epoch 1 "
                         "in the first N3 run) — this exposes the trained-through model.")
    ap.add_argument("--outcome-weight", action="store_true",
                    help="TP-N3: weight each TRAIN example by game outcome — won ×1.0, "
                         "lost ×--loss-weight, unknown ×1.0 (mean-normalised).")
    ap.add_argument("--loss-weight", type=float, default=0.5,
                    help="TP-N3: multiplier for lost-game examples under --outcome-weight.")
    ap.add_argument("--own-folders", nargs="+", default=None,
                    help="TP-N3: folders (e.g. Type_C own games) whose examples get "
                         "×--own-boost weight. Must also appear in --data. Fails loud "
                         "if it matches 0 examples.")
    ap.add_argument("--own-boost", type=float, default=15.0,
                    help="TP-N3: weight multiplier for WON --own-folders examples "
                         "(lost/unknown own rows stay unboosted).")
    ap.add_argument("--own-username", default=None,
                    help="TP-N3 re-run: restrict the own-boost to rows whose PERSPECTIVE "
                         "username matches (normalized). Without this, opponent-perspective "
                         "wins in own folders get boosted too — the n=171 parking root cause.")
    ap.add_argument("--own-team", default=None,
                    help="era-5 W1 specialist: a team (paste file / pool name / text); every TRAIN "
                         "example is weighted 1 + --own-team-weight × (fraction of this team's six "
                         "in the example's own roster). Composes with the flags above.")
    ap.add_argument("--own-team-weight", type=float, default=3.0,
                    help="λ for --own-team (a full-overlap example counts (1+λ)×).")
    ap.add_argument("--select-own-val", action="store_true",
                    help="TP-N3 re-run: best-checkpoint selection on the own-games val slice "
                         "(requires --own-folders; composes with --own-username). Global val "
                         "stays the printed/no-forgetting metric.")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after N epochs with no val mean-exact gain (0=off)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-files", type=int, default=None)
    # T4.1 + rename-wiring: default each --features mode to the dir its CONSUMER reads, so a default retrain is
    # never an orphan. sbda -> checkpoints/ (served by model_io.DEFAULT_TP_CHECKPOINT as teampreview_sbda.pt);
    # legacy -> checkpoints_pre_sbda/ (read by scratch/tp_headtohead_eval.py TP_LEGACY as teampreview_base.pt).
    # (Was a single checkpoints/ default, so a legacy retrain's teampreview_base.pt landed in a dir nothing read.)
    ap.add_argument("--out", default=None,
                    help="checkpoint output dir (default: checkpoints/ for --features sbda, "
                         "checkpoints_pre_sbda/ for legacy — each mode's consumer dir)")
    args = ap.parse_args(argv)
    if args.out is None:
        _tp = _HERE.parents[1] / "ai_train_scripts" / "teamPreview_model"
        args.out = str(_tp / ("checkpoints" if args.features == "sbda" else "checkpoints_pre_sbda"))
    return args


if __name__ == "__main__":
    train(parse_args())
