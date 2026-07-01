"""
Level C / B1 — belief-weighted depth-1 expectimax search (offline core).
========================================================================

Plans over the gate-validated ``white_box_sim`` forward model: for each of our top-K joint actions, average the
value-head win-prob of the successor over the opponent's top-M actions (``opp_action_prior``) and top-S stat-belief
scenarios, and pick the argmax. This is the BELIEF-WEIGHTED search the B0d gate cleared (it hedges KO odds via
probabilities rather than committing to a point-estimate KO). Design: ``docs/levelC_B1_search_design.md``.

Pure, poke-env-free, dict-only. ``expectimax`` is the testable core (takes pre-generated candidates + a value
function); ``search_with_model`` wires the model heads + encoder to it. Serve integration (behind a default-OFF
flag) is B1d — NOT here, so prod is untouched.

v1 scope: depth-1; mega follows the policy gimmick head (not searched); opp STAT-spread marginalisation (the
Sash/Sturdy item axis from the probe is a v1.1 leaf refinement). All knobs in ``SearchConfig``.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from v_dance.encoders.action_codec import (
    ACTIONS_PER_SLOT, BENCH_SLOTS, index_to_action, move_slots_for_mon,
    build_action_mask, build_gimmick_mask, GIMMICK_MEGA,
)
from v_dance.encoders.white_box_sim import white_box_sim
from v_dance.parser.belief_state import dex_base_stats, calc_full_stats
from v_dance.parser.vod_parser.pokedex import norm_species


class SearchConfig:
    """Search budget knobs (start small — RTX 3070 Ti / live serve time budget)."""
    def __init__(self, k_our: int = 6, k_our_slot: int = 4, m_opp: int = 4, m_opp_slot: int = 3,
                 s_belief: int = 3, use_mega: bool = True):
        self.k_our = k_our              # kept our JOINT actions
        self.k_our_slot = k_our_slot    # per-slot our action candidates before the joint product
        self.m_opp = m_opp              # kept opp JOINT actions
        self.m_opp_slot = m_opp_slot    # per-slot opp action candidates
        self.s_belief = s_belief        # opp stat-belief scenarios marginalised
        self.use_mega = use_mega


# ── action adapter: codec index → white_box_sim action (per perspective) ───────
def _ally_slot(slot_key: str) -> str:
    return {"our_a": "our_b", "our_b": "our_a", "opp_a": "opp_b", "opp_b": "opp_a"}.get(slot_key, slot_key)


def _bench_index(snap: dict, side: str, bench_slot: int) -> Optional[int]:
    """Map a codec ``bench_slot`` (index into the LIVING bench, encoder order) → the raw index in
    ``snap[side+'_bench']`` that ``white_box_sim.switch_in`` pops. None if unresolvable (e.g. unrevealed opp)."""
    raw = snap.get(f"{side}_bench") or []
    living = [m for m in raw if m and not m.get("is_fainted")][:BENCH_SLOTS]
    if not (0 <= bench_slot < len(living)):
        return None
    target = living[bench_slot]
    for i, m in enumerate(raw):
        if m is target:
            return i
    return None


def decode_action(idx: int, slot_key: str, snap: dict, *, mega: bool = False) -> Optional[dict]:
    """Decode a 0–15 policy action index for ``slot_key`` into a ``white_box_sim`` action, resolving the move
    NAME (from the actor's move slots), the absolute TARGET (bucket → foe/ally, perspective-aware), and the
    switch BENCH INDEX. Returns None if the index can't be resolved on this snapshot (caller drops it)."""
    side = "our" if str(slot_key).startswith("our") else "opp"
    act = index_to_action(idx)
    if act["kind"] == "switch":
        bi = _bench_index(snap, side, act["bench_slot"])
        return None if bi is None else {"kind": "switch", "bench_index": bi}
    actor = (snap.get(f"{side}_active") or {}).get(slot_key)
    moves = move_slots_for_mon(actor)
    ms = act["move_slot"]
    if ms >= len(moves) or not moves[ms][0]:
        return None
    foes = ("opp_a", "opp_b") if side == "our" else ("our_a", "our_b")
    bucket = act["target_bucket"]
    target = _ally_slot(slot_key) if bucket == 2 else foes[bucket]
    out = {"kind": "move", "move": moves[ms][0], "target": target}
    if mega:
        out["mega"] = True
    return out


# ── masked-softmax top-k ───────────────────────────────────────────────────────
def _masked_softmax(logits, mask) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64).ravel()
    n = z.shape[0]
    m = np.asarray(list(mask), dtype=bool)
    if m.shape[0] < n:
        m = np.concatenate([m, np.zeros(n - m.shape[0], dtype=bool)])
    out = np.zeros(n, dtype=np.float64)
    idx = np.nonzero(m[:n])[0]
    if idx.size == 0:
        return out
    zz = z[idx] - z[idx].max()
    p = np.exp(zz)
    out[idx] = p / p.sum()
    return out


def _topk(probs: np.ndarray, k: int) -> list:
    """Indices of the top-k positive-probability entries (desc)."""
    nz = [i for i in range(len(probs)) if probs[i] > 0]
    nz.sort(key=lambda i: probs[i], reverse=True)
    return nz[:k]


# ── candidate generation ───────────────────────────────────────────────────────
def _slot_candidates(logits, mask, k) -> list:
    p = _masked_softmax(logits, mask)
    return [(i, float(p[i])) for i in _topk(p, k)]


def _joint(side_a, side_b, ka, kb, slot_a, slot_b, snap, *, mega_slot=None) -> list:
    """Cartesian product of two slots' (index, prob) candidates → kept joints
    ``[(label, {slot:wbs_action}, prior)]``, pruned to top-(ka·kb→k) by prob product; unresolved actions dropped.
    ``mega_slot`` (slot key) gets mega=True applied to its MOVE action (the policy's gimmick decision)."""
    out = []
    if not side_a and side_b:
        # SYMMETRIC single-slot case: slot A is empty (fainted / absent) but slot B is the sole active mon.
        # Without this, the outer loop over the empty side_a never runs and _joint returns [] — which for the
        # OPPONENT collapses opp_candidates to the no-op joint (opp modelled as doing NOTHING) and lets
        # expectimax evaluate our moves against a passive foe in exactly the 1-mon endgames that decide games
        # (audit 2026-06-30). Mirror the `if not side_b` branch below for slot B.
        for (ib, pb) in side_b:
            ab = decode_action(ib, slot_b, snap, mega=(mega_slot == slot_b))
            if ab is not None:
                out.append((f"{slot_b}:{ib}", {slot_b: ab}, pb))
        out.sort(key=lambda r: r[2], reverse=True)
        return out
    for (ia, pa) in side_a:
        aa = decode_action(ia, slot_a, snap, mega=(mega_slot == slot_a))
        if aa is None:
            continue
        if not side_b:                          # only one active slot (forced / single mon)
            out.append((f"{slot_a}:{ia}", {slot_a: aa}, pa))
            continue
        for (ib, pb) in side_b:
            ab = decode_action(ib, slot_b, snap, mega=(mega_slot == slot_b))
            if ab is None:
                continue
            out.append((f"{slot_a}:{ia}|{slot_b}:{ib}", {slot_a: aa, slot_b: ab}, pa * pb))
    out.sort(key=lambda r: r[2], reverse=True)
    return out


def our_candidates(snap: dict, our_logits, gimmick_logits, cfg: SearchConfig) -> list:
    """Top-K our joint actions. ``our_logits`` = (l_our_a, l_our_b); ``gimmick_logits`` = (g_a, g_b) or None.
    Mega is applied (decoupled from the search) to the slot whose gimmick head argmaxes MEGA under the mask."""
    amask = build_action_mask(snap)
    a = _slot_candidates(our_logits[0], amask.get("our_a") or [], cfg.k_our_slot)
    has_b = (snap.get("our_active") or {}).get("our_b") is not None
    b = _slot_candidates(our_logits[1], amask.get("our_b") or [], cfg.k_our_slot) if has_b else []
    mega_slot = None
    if cfg.use_mega and gimmick_logits is not None:
        gmask = build_gimmick_mask(snap)
        for sk, gl in (("our_a", gimmick_logits[0]), ("our_b", gimmick_logits[1])):
            gm = gmask.get(sk) or []
            if GIMMICK_MEGA < len(gm) and gm[GIMMICK_MEGA] and int(np.argmax(_masked_softmax(gl, gm))) == GIMMICK_MEGA:
                mega_slot = sk          # at most one mega per team; first qualifying slot
                break
    return _joint(a, b, cfg.k_our_slot, cfg.k_our_slot, "our_a", "our_b", snap, mega_slot=mega_slot)[: cfg.k_our]


def opp_candidates(snap: dict, opp_prior: dict, cfg: SearchConfig) -> list:
    """Top-M opp joint actions from ``opp_action_prior`` ({"opp_a":[A],"opp_b":[A]}), renormalised. Returns
    ``[({slot:wbs_action}, p_opp)]`` summing to 1 (or a single empty no-op joint if the opp has no legal action)."""
    if not opp_prior:
        return [({}, 1.0)]
    pa = np.asarray(opp_prior["opp_a"] if opp_prior.get("opp_a") is not None else [], dtype=float)
    a = [(i, float(pa[i])) for i in _topk(pa, cfg.m_opp_slot)]
    has_b = (snap.get("opp_active") or {}).get("opp_b") is not None
    pb = np.asarray(opp_prior["opp_b"] if opp_prior.get("opp_b") is not None else [], dtype=float)
    b = ([(i, float(pb[i])) for i in _topk(pb, cfg.m_opp_slot)] if has_b else [])
    joints = _joint(a, b, cfg.m_opp_slot, cfg.m_opp_slot, "opp_a", "opp_b", snap)[: cfg.m_opp]
    if not joints:
        return [({}, 1.0)]
    tot = sum(pr for _l, _a, pr in joints) or 1.0
    return [(a, pr / tot) for _l, a, pr in joints]


def _spread_full_stats(mon: dict, k: int):
    spreads = ((mon.get("belief") or {}).get("spreads")) or []
    if k >= len(spreads):
        return None, 0.0
    s = spreads[k]
    # base stats from the CURRENT forme (species) first — an already-mega'd opp carries species=
    # 'charizardmegay' / base_species='charizard'; belief_block's expected_stats use the MEGA base stats
    # (stats_species=species), so the scenario must too, else it injects BASE-forme stats and understates a
    # mega's offense/defense/KO odds by ~25% (audit 2026-07-01, mirrors belief_block).
    base = dex_base_stats(mon.get("species") or mon.get("base_species"))
    if not base or not s.get("evs_actual") or not s.get("nature"):
        return None, 0.0
    try:
        return calc_full_stats(base, s["evs_actual"], s["nature"]), float(s.get("p") or 0.0)
    except Exception:
        return None, 0.0


def belief_scenarios(snap: dict, cfg: SearchConfig) -> list:
    """Top-S opp stat-belief scenarios as ``[(apply, weight)]`` where ``apply(snap) -> restore_callable`` swaps
    every opp mon's stats to its k-th spread (in place, restored after the step). Weight = the mean spread-k prob
    across opp mons (a shared scenario weight). Falls back to a single modal no-op scenario."""
    opp = [m for m in list((snap.get("opp_active") or {}).values()) + (snap.get("opp_bench") or []) if m]
    have = any(((m.get("belief") or {}).get("spreads")) for m in opp)
    if not have or cfg.s_belief <= 1:
        return [(lambda s: (lambda: None), 1.0)]
    scen = []
    for k in range(cfg.s_belief):
        ws = [p for (_fs, p) in (_spread_full_stats(m, k) for m in opp) if p > 0]
        if not ws:
            continue
        scen.append((k, sum(ws) / len(ws)))
    if not scen:
        return [(lambda s: (lambda: None), 1.0)]
    tot = sum(w for _k, w in scen) or 1.0

    def _make(k):
        def apply(s):
            saved = []
            for m in [mm for mm in list((s.get("opp_active") or {}).values()) + (s.get("opp_bench") or []) if mm]:
                fs, _p = _spread_full_stats(m, k)
                if fs is not None:
                    saved.append((m, m.get("stats_estimate")))
                    # mode MUST be a known encoder mode ("distribution") — an unknown mode encodes the
                    # stats-known confidence bit as 0.0 instead of 0.5 (audit 2026-06-30).
                    m["stats_estimate"] = {"mode": "distribution", "stats": fs}

            def restore():
                for m, old in saved:
                    m["stats_estimate"] = old
            return restore
        return apply
    return [(_make(k), w / tot) for k, w in scen]


# ── the expectimax core (pure; testable with synthetic candidates) ─────────────
def expectimax(snap: dict, our_cands: list, opp_cands: list, scenarios: list,
               value_of: Callable[[dict], float]) -> list:
    """Depth-1 belief-weighted expectimax. ``our_cands``=[(label,{slot:act},prior)], ``opp_cands``=[({slot:act},
    p_opp)] (Σp=1), ``scenarios``=[(apply,w)] (Σw=1), ``value_of``: successor-snap → win-prob. Returns the our
    candidates ranked by Q desc: ``[(label, {slot:act}, Q)]`` (best first)."""
    ranked = []
    for (label, our_a, _prior) in our_cands:
        q = 0.0
        for (opp_o, p_o) in opp_cands:
            if p_o <= 0:
                continue
            joint = dict(our_a)
            joint.update(opp_o)
            for (apply, w) in scenarios:
                if w <= 0:
                    continue
                restore = apply(snap)
                try:
                    nxt, _log = white_box_sim(snap, joint)
                finally:
                    restore()
                q += p_o * w * float(value_of(nxt))
        ranked.append((label, our_a, q))
    ranked.sort(key=lambda r: r[2], reverse=True)
    return ranked


def expectimax_batched(snap: dict, our_cands: list, opp_cands: list, scenarios: list,
                       value_batch: Callable[[list], list]) -> list:
    """Same result as ``expectimax`` but evaluates ALL leaves in ONE batched value call (GPU-friendly): enumerate
    every (our × opp × scenario) successor, value them together, then aggregate Q per our action. ``value_batch``:
    list[successor-snap] → list[win-prob]. Returns ``[(label, action_dict, Q)]`` best-first."""
    leaves = []          # (our_i, weight, successor_snap)
    for ui, (_label, our_a, _prior) in enumerate(our_cands):
        for (opp_o, p_o) in opp_cands:
            if p_o <= 0:
                continue
            joint = dict(our_a)
            joint.update(opp_o)
            for (apply, w) in scenarios:
                if w <= 0:
                    continue
                restore = apply(snap)
                try:
                    nxt, _log = white_box_sim(snap, joint)
                finally:
                    restore()
                leaves.append((ui, p_o * w, nxt))
    vals = value_batch([lf[2] for lf in leaves]) if leaves else []
    q = [0.0] * len(our_cands)
    for (ui, wgt, _snap), v in zip(leaves, vals):
        q[ui] += wgt * float(v)
    ranked = [(our_cands[i][0], our_cands[i][1], q[i]) for i in range(len(our_cands))]
    ranked.sort(key=lambda r: r[2], reverse=True)
    return ranked


def search_with_model(snap: dict, model, encoder, *, cfg: SearchConfig = None, device: str = "cpu",
                      turn: int = 0) -> tuple:
    """Generate candidates from the model heads + value head and run ``expectimax``. Returns
    ``(best_label, best_action_dict, ranked)``. ``encoder`` provides ``encode_snapshot``; the leaf value is the
    sigmoid value head. Falls back to ``(None, None, [])`` if the policy yields no resolvable our action."""
    from v_dance.play import model_io as M
    cfg = cfg or SearchConfig()
    if not M.value_trained(model):           # no value head → the search has no leaf signal → let caller fall back
        return None, None, []
    state_vec = encoder.encode_snapshot(snap, turn)
    our_logits = M.head_logits(model, ("our_a", "our_b"), state_vec, device)
    # only search over mega if the gimmick head was actually trained (else don't trust it)
    gim = (M.gimmick_logits(model, ("our_a", "our_b"), state_vec, device)
           if (cfg.use_mega and M.gimmick_trained(model)) else None)
    our_c = our_candidates(snap, our_logits, gim, cfg)
    if not our_c:
        return None, None, []
    opp_prior = M.opp_action_prior(model, state_vec, snap, device=device)
    opp_c = opp_candidates(snap, opp_prior, cfg)
    scen = belief_scenarios(snap, cfg)

    def value_batch(snaps: list) -> list:
        if not snaps:
            return []
        vecs = [encoder.encode_snapshot(s, turn + 1) for s in snaps]
        vals = M.value_logit_batch(model, vecs, device)
        return [0.5] * len(snaps) if vals is None else list(vals)

    ranked = expectimax_batched(snap, our_c, opp_c, scen, value_batch)
    if not ranked:
        return None, None, []
    best_label, best_action, _q = ranked[0]
    return best_label, best_action, ranked


def enrich_snapshot(snap: dict, belief, *, top_k: int = 5) -> dict:
    """Attach a Pikalytics ``belief`` block + ``stats_estimate`` to EVERY mon (both sides, active ∪ bench) of a
    RAW live snapshot, mirroring the export's per-mon ``fill_blanks`` enrichment. WITHOUT this the live snapshot
    (from ``reconstruct_full_for_decision`` → ``_snapshot_state``) has no stats → ``white_box_sim`` deals ~zero
    damage and ``belief_scenarios`` no-op, so the search degenerates to ≈policy-argmax (audit 2026-06-30). Mutates
    in place (the snapshot is a fresh per-decision parse) and returns it; no-op on already-exact mons / no belief."""
    if belief is None or not snap:
        return snap
    from v_dance.parser.belief_state import _enrich_mon_belief
    warned: set = set()
    for side in ("our", "opp"):
        mons = list((snap.get(f"{side}_active") or {}).values()) + (snap.get(f"{side}_bench") or [])
        for m in mons:
            if not m:
                continue
            est = m.get("stats_estimate") or {}
            if est.get("mode") == "exact" and est.get("stats"):
                continue                         # already exact (team-sheet) — leave it
            try:
                _enrich_mon_belief(m, belief, top_k, [], warned)
            except Exception:
                pass                             # a mon with no Pikalytics data stays unenriched (skipped in search)
    return snap


def parse_label(label: str) -> dict:
    """Decode a candidate label ('our_a:0|our_b:6') → {slot_key: action_index} for the serve path."""
    out = {}
    for part in (label or "").split("|"):
        if ":" in part:
            slot, idx = part.rsplit(":", 1)
            try:
                out[slot] = int(idx)
            except ValueError:
                pass
    return out
