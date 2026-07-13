"""M4 — generative, scored team builder (DS-M4 / docs/execution_plan_4a_m4.md PART 2).

Belief-driven, not neural: rosters grow by beam search over the blend's teammate
co-occurrence × usage (B1); sets fill from the belief's top ability/item/moves/spread (B2);
legality is Showdown's own validator (B3, node subprocess — no server); scoring reuses the
deployed pieces (B4): archetype coherence (k10_full distance), the TP set-head's decision
margin vs a meta sample, and the 4b cluster-vs-cluster matchup prior when the matrix sidecar
exists (regenerate via scratch/seed_router_priors.py; absent → None, surfaced not silent).

The rollout scorer is DEFERRED v2 — see the marker in score_team.  # TODO(M4-v2)
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

log = logging.getLogger(__name__)
_REPO = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = _REPO / "ai_train_scripts" / "BC_model" / "team_archetypes_k10_full.npz"
MATRIX_PATH = _REPO / "data" / "router_matrix.json"     # written by seed_router_priors.py
_STATS = ("hp", "atk", "def", "spa", "spd", "spe")
_STAT_LABEL = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}


# ── B1: roster beam ───────────────────────────────────────────────────────────
def _affinity_maps(belief, species: Sequence[str]) -> Dict[str, Dict[str, float]]:
    from v_dance.parser.vod_parser.pokedex import norm_species
    return {s: {norm_species(t["name"]): float(t["p"]) for t in belief.teammates(s, top_k=24)}
            for s in species}


def generate_rosters(seed_core: Sequence[str], n: int, belief,
                     beam_mult: int = 3) -> List[List[str]]:
    """``n`` six-species rosters grown from ``seed_core`` (1-2 species) by beam search.

    Extension score for candidate c on partial P: mean directional co-occurrence
    affinity(P, c) × (1 + usage(c)/100) — established pairings first, popularity as the
    tiebreak. Deterministic (pure argsort, no RNG)."""
    from v_dance.parser.vod_parser.pokedex import norm_species
    core = [s for s in seed_core if s and belief.known(s)]
    if not core:
        raise ValueError(f"seed core {list(seed_core)!r} has no belief-known species")
    pool = [sp for sp, _u in belief.usage_ranking() if belief.known(sp)]
    aff = _affinity_maps(belief, pool)
    usage = {sp: float(belief.usage(sp)) for sp in pool}
    norm_of = {sp: norm_species(sp) for sp in pool}

    def _score(partial: List[str], cand: str) -> float:
        a = [aff.get(m, {}).get(norm_of[cand], 0.0) + aff.get(cand, {}).get(norm_of[m], 0.0)
             for m in partial]
        return (sum(a) / (2 * len(a))) * (1.0 + usage.get(cand, 0.0) / 100.0)

    beams: List[tuple] = [(0.0, list(core))]
    width = max(n * beam_mult, 6)
    while any(len(p) < 6 for _s, p in beams):
        nxt: List[tuple] = []
        seen = set()
        for s, partial in beams:
            if len(partial) >= 6:
                nxt.append((s, partial))
                continue
            have = {norm_of.get(m, m) for m in partial}
            scored = [( _score(partial, c), c) for c in pool if norm_of[c] not in have]
            for cs, c in sorted(scored, reverse=True)[:width]:
                team = partial + [c]
                key = frozenset(norm_of.get(m, m) for m in team)
                if key not in seen:
                    seen.add(key)
                    nxt.append((s + cs, team))
        beams = sorted(nxt, reverse=True, key=lambda t: t[0])[:width]
    return [p for _s, p in beams[:n]]


# ── B2: set filler + paste writer ─────────────────────────────────────────────
def fill_sets(roster: Sequence[str], belief) -> List[dict]:
    """Per-species most-likely set from the belief (team_sheet mon-dict shape)."""
    from v_dance.parser.vod_parser.pokedex import norm_species
    mons = []
    used_items: set = set()
    for sp in roster:
        ab = belief.ability_distribution(sp, top_k=1)
        # VGC Item Clause: one of each item per team — fall down each species'
        # belief item list past already-taken items (validator-proven, Life Orb ×2).
        it = [i for i in belief.item_distribution(sp, top_k=6)
              if norm_species(i["name"]) not in used_items][:1]
        if it:
            used_items.add(norm_species(it[0]["name"]))
        moves, seen = [], set()
        for m in belief.move_distribution(sp, top_k=12):
            nm = norm_species(m["name"])
            if nm and nm not in seen and m["name"].strip("- "):
                seen.add(nm)
                moves.append(m["name"])
            if len(moves) == 4:
                break
        spread = (belief.spread_distribution(sp, top_k=1) or
                  [{"nature": "Serious", "evs": [0, 0, 0, 0, 0, 0]}])[0]
        # ⚠ Champions uses the 0-32 STAT-POINT budget (total 66) — the paste takes the
        # BUCKET scale (spread["evs"]), NOT classic 0-252 evs_actual (validator-proven).
        evs = {k: v for k, v in zip(_STATS, spread["evs"]) if v}
        mons.append({"species": sp, "item": (it[0]["name"] if it else None),
                     "ability": (ab[0]["name"] if ab else None), "level": 50,
                     "nature": spread.get("nature"), "evs": evs, "moves": moves})
    return mons


def to_paste(mons: Sequence[dict]) -> str:
    """Showdown-export paste (round-trips through team_sheet.parse_showdown_team)."""
    blocks = []
    for m in mons:
        head = m["species"] + (f" @ {m['item']}" if m.get("item") else "")
        lines = [head]
        if m.get("ability"):
            lines.append(f"Ability: {m['ability']}")
        lines.append(f"Level: {m.get('level', 50)}")
        evs = m.get("evs") or {}
        if evs:
            lines.append("EVs: " + " / ".join(f"{v} {_STAT_LABEL[k]}"
                                              for k, v in evs.items() if v))
        if m.get("nature"):
            lines.append(f"{m['nature']} Nature")
        lines.extend(f"- {mv}" for mv in (m.get("moves") or []))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


# ── B4: scorers ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _artifact():
    from v_dance.datatools.team_archetypes import load_artifact
    return load_artifact(str(ARTIFACT_PATH))


_MATRIX_CACHE: dict = {"mtime": None, "data": None}


def _matrix() -> Optional[dict]:
    """The cluster-vs-cluster matchup sidecar, reloaded when seed_router_priors.py
    rewrites it (mtime-keyed — a long-running server picks up a regen without restart)."""
    try:
        mt = MATRIX_PATH.stat().st_mtime
    except OSError:
        return None
    if _MATRIX_CACHE["mtime"] != mt:
        try:
            _MATRIX_CACHE["data"] = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
            _MATRIX_CACHE["mtime"] = mt
        except (OSError, json.JSONDecodeError):
            _MATRIX_CACHE["data"] = None
    return _MATRIX_CACHE["data"]


def sample_opp_rosters(k: int = 20) -> List[List[dict]]:
    """Meta sample: k rosters from the largest team-keys cache (corpus rosters —
    species-only mon dicts, enough for both scorers). Deterministic: evenly-strided."""
    caches = sorted(_REPO.glob("artifacts/.team_keys_*.json"),
                    key=lambda p: p.stat().st_size, reverse=True)
    if not caches:
        return []
    data = json.loads(caches[0].read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else list(data.values())
    rosters = []
    for row in rows:
        team = row.get("team") if isinstance(row, dict) else row
        if isinstance(team, (list, tuple)) and len(team) >= 6 and all(
                isinstance(s, str) for s in list(team)[:6]):
            rosters.append([{"species": s} for s in list(team)[:6]])
    if not rosters:
        return []
    step = max(1, len(rosters) // k)
    return rosters[::step][:k]


def _tp_margin(model, vocab, cfg, our_species, opp_species, belief) -> float:
    """The set head's decision margin (top subset score − median) — a cheap 'does the
    TP net see a plan vs this matchup' proxy. Higher = a more decisive bring."""
    import numpy as np
    import torch
    from v_dance.play.model_io import _pack_side, uses_tp_features
    from v_dance.training.train_teampreview import SET_SUBSETS
    use_tp = uses_tp_features(cfg)
    oi, of = _pack_side(our_species, vocab, cfg["feat_dim"], belief=belief,
                        use_tp_features=use_tp)
    pi, pf = _pack_side(opp_species, vocab, cfg["feat_dim"], belief=belief,
                        use_tp_features=use_tp)
    aff_t = None
    if use_tp and getattr(model, "use_teammate_bias", False):
        from v_dance.training.tp_features import teammate_affinity_matrix
        aff_t = torch.as_tensor(teammate_affinity_matrix(our_species, belief, n=6)[None])
    with torch.no_grad():
        scores, _b, _l = model.score_subsets(
            torch.as_tensor([oi]), torch.as_tensor([pi]),
            torch.as_tensor(of[None]), torch.as_tensor(pf[None]),
            aff_t, subsets=SET_SUBSETS)
    s = np.asarray(scores[0])
    return float(s.max() - np.median(s))


def score_team(mons: Sequence[dict], belief, opp_rosters=None,
               tp_bundle=None) -> dict:
    """Three scores per DS-M4: archetype (zid, dist — LOWER dist = more coherent),
    tp_confidence (mean set-head margin vs the meta sample), matchup_prior (cluster-vs-
    cluster mean win-rate; None until the matrix sidecar is regenerated — surfaced)."""
    from v_dance.datatools.team_archetypes import assign_team_sheet
    zid, dist = assign_team_sheet(list(mons), _artifact())
    out = {"archetype": {"z": int(zid), "dist": round(float(dist), 2)},
           "tp_confidence": None, "matchup_prior": None}
    opp_rosters = sample_opp_rosters() if opp_rosters is None else opp_rosters
    if tp_bundle is None:
        from v_dance.play.model_io import DEFAULT_TP_CHECKPOINT, load_team_chooser
        tp_bundle = load_team_chooser(str(DEFAULT_TP_CHECKPOINT))
    model, vocab, cfg = tp_bundle
    if getattr(model, "use_set_head", False) and opp_rosters:
        our_species = [m["species"] for m in mons]
        margins = []
        for opp in opp_rosters:
            try:
                margins.append(_tp_margin(model, vocab, cfg, our_species,
                                          [o["species"] for o in opp], belief))
            except Exception:                       # noqa: BLE001 — one bad roster ≠ no score
                log.debug("tp margin failed for one opp roster", exc_info=True)
        if margins:
            out["tp_confidence"] = round(sum(margins) / len(margins), 4)
    mat = _matrix()
    if mat and opp_rosters:
        wrs = []
        for opp in opp_rosters:
            try:
                oz, _d = assign_team_sheet(list(opp), _artifact())
                cell = (mat.get(str(zid)) or {}).get(str(int(oz)))
                if cell:
                    wrs.append(float(cell[0]))
            except Exception:                       # noqa: BLE001
                pass
        if wrs:
            out["matchup_prior"] = round(sum(wrs) / len(wrs), 4)
    elif mat is None:
        out["matchup_note"] = ("no data/router_matrix.json — regenerate via "
                               "scratch/seed_router_priors.py")
    # TODO(M4-v2): rollout scorer — N quick self-play games per candidate on the
    # multi-server harness (DS-M4 deferred; do not build without a new gate).
    return out


# ── pipeline (B1→B4) ─────────────────────────────────────────────────────────
def generate_teams(seed_core: Sequence[str], n: int, belief, *,
                   validate: bool = True, fmt: Optional[str] = None,
                   score: bool = True) -> dict:
    """The endpoint pipeline. Returns {"teams": [...], "dropped_illegal": int}."""
    rosters = generate_rosters(seed_core, max(n * 2, n + 2), belief)
    out, dropped = [], 0
    tp_bundle = None
    opp = sample_opp_rosters() if score else []
    for roster in rosters:
        if len(out) >= n:
            break
        mons = fill_sets(roster, belief)
        paste = to_paste(mons)
        if validate:
            from v_dance.datatools.validate_teams import validate as _validate
            ok, reasons = _validate(paste, fmt=fmt)
            if not ok:
                dropped += 1
                log.info("generated roster dropped (illegal): %s — %s", roster, reasons[:2])
                continue
        row = {"roster": list(roster), "paste": paste}
        if score:
            if tp_bundle is None:
                from v_dance.play.model_io import DEFAULT_TP_CHECKPOINT, load_team_chooser
                tp_bundle = load_team_chooser(str(DEFAULT_TP_CHECKPOINT))
            row["scores"] = score_team(mons, belief, opp_rosters=opp, tp_bundle=tp_bundle)
        out.append(row)
    return {"teams": out, "dropped_illegal": dropped}
