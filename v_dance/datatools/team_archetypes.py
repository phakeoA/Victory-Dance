"""
team_archetypes.py — Phase-2 §2: discrete own-team strategy archetypes (z v1)
=============================================================================
Clusters every corpus TEAM (the 6-mon preview roster) into k discrete
archetypes and persists the centroid table as an artifact.  The archetype id
becomes the latent-strategy conditioning ``z`` for the battle net: ``z`` = OUR
game plan (a function of OUR OWN team, constant per battle, known at team
preview) — deliberately DISJOINT from the match-memory core, which encodes
what THIS opponent has revealed (docs/latent_z_offline_weighting_phase2_design.md §1).

Features per team (NO species one-hots — the mechanic-encoder philosophy, so
a NOVEL sandbox team lands on its nearest archetype):

  * per-mon mechanic features: ``StateEncoder._write_pokemon_json`` in a
    NEUTRAL context (bench, no enemies, no field) — the exact per-mon block
    the battle net consumes — mean- AND max-pooled over the roster,
  * aggregate play statistics from that team's games: switch rate, Protect
    rate, Fake Out rate, Trick Room rate, mean game length.

A NOVEL team has no play history, so serve-time assignment measures distance
on the mon-feature slice only (``assign(..., mon_only=True)``) — the play
stats sharpen the offline clustering without blocking the sandbox entry point.

Inspectability is the point of v1: run the CLI and EYEBALL the cluster
report (is Trick Room separated from tailwind offense?) BEFORE any training
consumes the ids — a bad clustering is visible here, unlike a collapsed
continuous latent.

Usage
-----
    python -m v_dance.datatools.team_archetypes \
        --data data/vods/Prepared_training_data/Regulation_MA/Jsonl_TypeA ... \
        --k 10 --seed 0 \
        --out ai_train_scripts/BC_model/team_archetypes_k10.npz

The artifact (.npz + sidecar .json) stores centroids, the standardisation
(mu/sd), the per-(replay_id, perspective) assignment map the trainer joins
on, and per-cluster summaries for the report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from v_dance.encoders.state_encoder import POKEMON_FEATURES, StateEncoder
from v_dance.parser.vod_parser.pokedex import norm_species
from v_dance.training.bc_dataset import iter_jsonl_files

# Aggregate play statistics appended after the pooled mon features.
PLAY_STATS = ("switch_rate", "protect_rate", "fakeout_rate",
              "trickroom_rate", "mean_turns")
PLAY_STATS_DIM = len(PLAY_STATS)
# Team feature layout: [mean-pool P | max-pool P | play stats].
MON_FEATURE_DIM = 2 * POKEMON_FEATURES
TEAM_FEATURE_DIM = MON_FEATURE_DIM + PLAY_STATS_DIM

_ARTIFACT_SCHEMA = 1


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

def mon_feature_vector(encoder: StateEncoder, mon: dict) -> np.ndarray:
    """One mon's mechanic-feature block in a NEUTRAL context (bench slot, no
    enemies / field / side state) — team identity, not battle state."""
    vec = np.zeros(POKEMON_FEATURES, dtype=np.float32)
    if mon:
        encoder._write_pokemon_json(vec, 0, mon, is_active=False)
    return vec


def team_feature_vector(mon_vecs: Sequence[np.ndarray],
                        play_stats: Sequence[float]) -> np.ndarray:
    """[mean-pool | max-pool | play stats] over the roster's mon vectors."""
    if len(mon_vecs):
        M = np.stack(mon_vecs)
        pooled = np.concatenate([M.mean(axis=0), M.max(axis=0)])
    else:
        pooled = np.zeros(MON_FEATURE_DIM, dtype=np.float32)
    ps = np.asarray(list(play_stats), dtype=np.float32)
    if ps.shape != (PLAY_STATS_DIM,):
        raise ValueError(f"play stats must have {PLAY_STATS_DIM} entries, got {ps.shape}")
    return np.concatenate([pooled, ps]).astype(np.float32)


def _team_key(species: Sequence[str]) -> Tuple[str, ...]:
    """Order-free roster identity key (norm'd species, sorted)."""
    return tuple(sorted(norm_species(s) for s in species if s))


def _roster_and_mons(first_turn: dict, opp_view: Optional[dict]) -> Dict[str, dict]:
    """{norm_species: best mon dict} for one side's 6-mon preview roster.

    Own actives + bench give the 4 BROUGHT mons with the richest knowledge
    (retrofit/OTS movesets); the OPPOSITE perspective's opp side carries the
    full preview roster including never-entered stubs — union covers all 6.
    """
    out: Dict[str, dict] = {}
    snap = first_turn.get("state_before_actions") or {}
    for mon in (list((snap.get("our_active") or {}).values())
                + list(snap.get("our_bench") or [])):
        if isinstance(mon, dict) and mon.get("species"):
            out.setdefault(norm_species(mon["species"]), mon)
    if opp_view:
        osnap = opp_view.get("state_before_actions") or {}
        for mon in (list((osnap.get("opp_active") or {}).values())
                    + list(osnap.get("opp_bench") or [])):
            if isinstance(mon, dict) and mon.get("species"):
                out.setdefault(norm_species(mon["species"]), mon)
    return out


def _play_stats_of(transitions: Sequence[dict]) -> Tuple[np.ndarray, int]:
    """(play-stat vector, n_action_entries) over ONE (replay, perspective)."""
    n_act = n_switch = n_protect = n_fakeout = n_tr = 0
    total_turns = 0
    for t in transitions:
        total_turns = max(total_turns, int(t.get("total_turns") or t.get("turn") or 0))
        for act in t.get("our_actions") or []:
            n_act += 1
            if act.get("action") == "switch":
                n_switch += 1
            if act.get("is_protect"):
                n_protect += 1
            mv = norm_species(act.get("move") or "")
            if mv == "fakeout":
                n_fakeout += 1
            elif mv == "trickroom":
                n_tr += 1
    denom = max(n_act, 1)
    return (np.array([n_switch / denom, n_protect / denom, n_fakeout / denom,
                      n_tr / denom, float(total_turns)], dtype=np.float64),
            n_act)


def collect_team_records(folders: Sequence[str],
                         limit_files: Optional[int] = None) -> dict:
    """Stream the JSONL corpus → one record per distinct TEAM (roster key).

    Returns ``{team_key: {"feats": (TEAM_FEATURE_DIM,) float32,
                          "species": [display names], "n_games": int,
                          "sightings": [(replay_id, perspective), ...]}}``.
    Play stats are averaged over every game the team appears in; mon features
    come from the first sighting (same roster ⇒ same sheet-level identity).
    """
    encoder = StateEncoder()
    teams: dict = {}
    pending_stats: Dict[Tuple[str, ...], List[np.ndarray]] = defaultdict(list)
    files = [f for folder in folders for f in iter_jsonl_files(str(folder))]
    if limit_files is not None:
        files = files[:limit_files]
    t0 = time.time()
    for fi, path in enumerate(files, 1):
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            trans = [json.loads(ln) for ln in lines if ln.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        by_persp: Dict[str, List[dict]] = defaultdict(list)
        for t in trans:
            p = t.get("perspective")
            if p:
                by_persp[p].append(t)
        for persp, ts in by_persp.items():
            first = ts[0]
            rid = first.get("replay_id") or Path(path).stem
            opp_persp = "p2" if persp == "p1" else "p1"
            opp_first = (by_persp.get(opp_persp) or [None])[0]
            mons = _roster_and_mons(first, opp_first)
            if not mons:
                continue
            key = _team_key(list(mons))
            stats, _ = _play_stats_of(ts)
            rec = teams.get(key)
            if rec is None:
                mon_vecs = [mon_feature_vector(encoder, m) for m in mons.values()]
                rec = teams[key] = {
                    "mon_pool": np.concatenate([
                        np.stack(mon_vecs).mean(axis=0),
                        np.stack(mon_vecs).max(axis=0)]).astype(np.float32),
                    "species": sorted({(m.get("species") or "?") for m in mons.values()}),
                    "n_games": 0,
                    "sightings": [],
                }
            rec["n_games"] += 1
            rec["sightings"].append((rid, persp))
            pending_stats[key].append(stats)
        if fi % 2000 == 0 or fi == len(files):
            print(f"  [teams] {fi}/{len(files)} files, {len(teams)} distinct teams "
                  f"({time.time() - t0:.0f}s)")
    # Finalise: mean play stats per team → the full feature vector.
    for key, rec in teams.items():
        ps = np.stack(pending_stats[key]).mean(axis=0)
        rec["feats"] = np.concatenate([rec.pop("mon_pool"), ps]).astype(np.float32)
    return teams


def team_keys_for_folders(folders: Sequence[str], cache_dir: str = "artifacts",
                          limit_files: Optional[int] = None):
    """(team_key_set, {(replay_id, perspective): key_str}) for a folder list — the G2
    held-out-team join (bc_val_report --held-out-slice). ``key_str`` = "|".join of the
    order-free roster key. JSON-cached per sorted folder list (first 80k-file walk is
    ~10-30 min; afterwards instant). ⚠ the cache keys on the folder LIST only — delete
    ``artifacts/.team_keys_*.json`` after growing a corpus folder."""
    import hashlib
    folders = sorted(str(f) for f in folders)
    cache = None
    if cache_dir and limit_files is None:
        h = hashlib.md5("\n".join(folders).encode()).hexdigest()[:12]
        cache = Path(cache_dir) / f".team_keys_{h}.json"
        if cache.is_file():
            try:
                d = json.loads(cache.read_text(encoding="utf-8"))
                if d.get("folders") == folders:
                    sight = {tuple(k.split("::", 1)): v for k, v in d["sightings"].items()}
                    return set(d["keys"]), sight
            except (json.JSONDecodeError, KeyError, OSError):
                pass                                    # unreadable cache → rebuild
    recs = collect_team_records(folders, limit_files=limit_files)
    keys, sight = set(), {}
    for key, rec in recs.items():
        ks = "|".join(key)
        keys.add(ks)
        for rid, persp in rec["sightings"]:
            sight[(rid, persp)] = ks
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(
            {"folders": folders, "keys": sorted(keys),
             "sightings": {f"{r}::{p}": k for (r, p), k in sight.items()}}),
            encoding="utf-8")
    return keys, sight


# ══════════════════════════════════════════════════════════════════════════════
# k-means (numpy, seeded, restart-best — no sklearn dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _kmeans_once(X: np.ndarray, k: int, rng: np.random.RandomState,
                 iters: int) -> Tuple[np.ndarray, np.ndarray, float]:
    n = X.shape[0]
    # k-means++ init
    centroids = np.empty((k, X.shape[1]), dtype=np.float64)
    centroids[0] = X[rng.randint(n)]
    d2 = ((X - centroids[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        total = float(d2.sum())
        if total <= 0.0:                           # all points identical/covered
            centroids[c] = X[rng.randint(n)]
            continue
        centroids[c] = X[rng.choice(n, p=d2 / total)]
        d2 = np.minimum(d2, ((X - centroids[c]) ** 2).sum(axis=1))
    labels = np.zeros(n, dtype=np.int64)
    for it in range(iters):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if it > 0 and (new_labels == labels).all():
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                centroids[c] = X[m].mean(axis=0)
            else:                                  # dead cluster → farthest point
                centroids[c] = X[dists.min(axis=1).argmax()]
    inertia = float(((X - centroids[labels]) ** 2).sum())
    return centroids, labels, inertia


def kmeans(X: np.ndarray, k: int, seed: int = 0, n_init: int = 10,
           iters: int = 100) -> Tuple[np.ndarray, np.ndarray, float]:
    """Seeded k-means++ with ``n_init`` restarts; returns the lowest-inertia
    (centroids (k,D), labels (N,), inertia)."""
    if X.shape[0] < k:
        raise ValueError(f"need at least k={k} teams, got {X.shape[0]}")
    rng = np.random.RandomState(seed)
    best: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    for _ in range(n_init):
        c, l, inertia = _kmeans_once(X, k, rng, iters)
        if best is None or inertia < best[2]:
            best = (c, l, inertia)
    return best


# ══════════════════════════════════════════════════════════════════════════════
# Artifact build / load / assign
# ══════════════════════════════════════════════════════════════════════════════

def build_archetypes(folders: Sequence[str], k: int = 10, seed: int = 0,
                     out: Optional[str] = None,
                     limit_files: Optional[int] = None) -> dict:
    """Cluster the corpus teams → the archetype artifact (optionally saved)."""
    teams = collect_team_records(folders, limit_files=limit_files)
    if not teams:
        raise SystemExit("[team_archetypes] no teams found in the supplied folders")
    keys = sorted(teams)                              # deterministic order
    X = np.stack([teams[key]["feats"] for key in keys]).astype(np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0                               # constant dims stay put
    Xs = (X - mu) / sd
    centroids, labels, inertia = kmeans(Xs, k=k, seed=seed)

    assignments: Dict[str, int] = {}
    clusters: List[dict] = [
        {"archetype_id": c, "n_teams": 0, "n_games": 0,
         "species": Counter(), "stats_sum": np.zeros(PLAY_STATS_DIM)}
        for c in range(k)]
    for key, label in zip(keys, labels):
        rec, cl = teams[key], clusters[int(label)]
        cl["n_teams"] += 1
        cl["n_games"] += rec["n_games"]
        cl["species"].update(rec["species"])
        cl["stats_sum"] += rec["feats"][-PLAY_STATS_DIM:]
        for rid, persp in rec["sightings"]:
            assignments[f"{rid}::{persp}"] = int(label)

    summaries = []
    for cl in clusters:
        nt = max(cl["n_teams"], 1)
        summaries.append({
            "archetype_id": cl["archetype_id"],
            "n_teams": cl["n_teams"], "n_games": cl["n_games"],
            "top_species": cl["species"].most_common(8),
            "play_stats": {name: round(float(v) / nt, 4)
                           for name, v in zip(PLAY_STATS, cl["stats_sum"])},
        })

    artifact = {
        "schema": _ARTIFACT_SCHEMA, "k": int(k), "seed": int(seed),
        "feature_dim": int(TEAM_FEATURE_DIM),
        "mon_feature_dim": int(MON_FEATURE_DIM),
        "pokemon_features": int(POKEMON_FEATURES),
        "play_stats": list(PLAY_STATS),
        "inertia": float(inertia),
        "n_teams": len(keys),
        "centroids": centroids,                      # STANDARDISED space
        "mu": mu, "sd": sd,
        "assignments": assignments,
        "clusters": summaries,
    }
    if out:
        save_artifact(artifact, out)
    return artifact


def save_artifact(artifact: dict, path: str) -> None:
    """.npz for the arrays + a .json sidecar for the readable parts."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, centroids=artifact["centroids"], mu=artifact["mu"],
             sd=artifact["sd"])
    meta = {key: v for key, v in artifact.items()
            if key not in ("centroids", "mu", "sd")}
    p.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load_artifact(path: str) -> dict:
    p = Path(path)
    arrays = np.load(p if p.suffix == ".npz" else p.with_suffix(".npz"))
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    meta.update({"centroids": arrays["centroids"], "mu": arrays["mu"],
                 "sd": arrays["sd"]})
    return meta


def assign(features: np.ndarray, artifact: dict, mon_only: bool = False) -> Tuple[int, float]:
    """Nearest-centroid archetype for a raw team-feature vector.

    ``mon_only=True`` measures distance on the pooled-mon-feature slice only —
    the serve/sandbox path where a NOVEL team has no play history (pass any
    values in the stats slots; they are ignored).  Returns
    ``(archetype_id, distance)`` — distance doubles as the off-meta-ness
    diagnostic (far from every centroid = novel strategy shape)."""
    x = (np.asarray(features, dtype=np.float64) - artifact["mu"]) / artifact["sd"]
    C = artifact["centroids"]
    if mon_only:
        d = int(artifact["mon_feature_dim"])
        x, C = x[:d], C[:, :d]
    dists = np.sqrt(((C - x) ** 2).sum(axis=1))
    a = int(dists.argmin())
    return a, float(dists[a])


def artifact_slim(artifact: dict) -> dict:
    """The checkpoint-embeddable slice of an artifact: exactly what serve-time
    nearest-centroid assignment needs (train_bc stamps this into the ckpt as
    ``z_artifact`` so serve has no sidecar-file dependency)."""
    return {
        "k": int(artifact["k"]),
        "mon_feature_dim": int(artifact["mon_feature_dim"]),
        "centroids": np.asarray(artifact["centroids"], dtype=np.float64),
        "mu": np.asarray(artifact["mu"], dtype=np.float64),
        "sd": np.asarray(artifact["sd"], dtype=np.float64),
    }


def sheet_mon_to_parser_mon(mon: dict) -> dict:
    """Adapt a team-sheet mon (team_sheet.parse_showdown_team /
    parse_packed_team shape, or a poke-env-derived {species/item/ability/
    moves/nature} dict) to the parser mon-dict keys ``mon_feature_vector``
    reads — the serve-side twin of the corpus extraction."""
    moves = [m for m in (mon.get("moves") or []) if m]
    evs = mon.get("evs") or {}
    return {
        "species": mon.get("species"),
        "known_item": mon.get("item"),
        "known_ability": mon.get("ability"),
        "known_moves": list(moves)[:4],
        "nature": mon.get("nature"),
        "ev_spread": ({s: int(v) for s, v in evs.items()} if evs else None),
    }


def assign_team_sheet(sheet_mons: Sequence[dict], artifact_like: dict,
                      encoder: Optional[StateEncoder] = None) -> Tuple[int, float]:
    """Serve/sandbox entry point: a TEAM SHEET (≤6 mons) → (archetype_id,
    distance).  Distance is measured on the mon-feature slice only (a novel
    team has no play history) and doubles as the off-meta-ness diagnostic.
    ``artifact_like`` is a full artifact or the ckpt-embedded ``z_artifact``."""
    enc = encoder or StateEncoder()
    vecs = [mon_feature_vector(enc, sheet_mon_to_parser_mon(m))
            for m in sheet_mons if m and m.get("species")]
    feats = team_feature_vector(vecs, [0.0] * PLAY_STATS_DIM)
    return assign(feats, artifact_like, mon_only=True)


def load_archetype_assignments(path: str) -> Dict[Tuple[str, str], int]:
    """{(replay_id, perspective): archetype_id} join map for the trainer."""
    meta = load_artifact(path)
    out: Dict[Tuple[str, str], int] = {}
    for key, a in (meta.get("assignments") or {}).items():
        rid, _, persp = key.rpartition("::")
        out[(rid, persp)] = int(a)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser(
        description="Cluster corpus teams into discrete strategy archetypes (Phase-2 z v1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True,
                    help="JSONL corpus folder(s) to cluster")
    ap.add_argument("--k", type=int, default=10, help="number of archetypes (8-16 sensible)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-files", type=int, default=None, help="cap files (smoke runs)")
    ap.add_argument("--out", required=True,
                    help="artifact path (.npz; a .json sidecar is written next to it)")
    args = ap.parse_args(argv)

    art = build_archetypes(args.data, k=args.k, seed=args.seed,
                           out=args.out, limit_files=args.limit_files)
    print(f"\n[team_archetypes] {art['n_teams']} distinct teams → k={art['k']} "
          f"archetypes (inertia {art['inertia']:.1f}); artifact: {args.out}")
    for cl in art["clusters"]:
        stats = cl["play_stats"]
        top = ", ".join(f"{s}×{n}" for s, n in cl["top_species"][:6])
        print(f"  z={cl['archetype_id']:>2}  teams={cl['n_teams']:>4} games={cl['n_games']:>5}  "
              f"TR={stats['trickroom_rate']:.3f} switch={stats['switch_rate']:.3f} "
              f"protect={stats['protect_rate']:.3f} turns={stats['mean_turns']:.1f}")
        print(f"        {top}")
    print("\nEYEBALL CHECK: distinct plans (e.g. Trick Room vs offense) should land in "
          "different clusters before any training consumes these ids (design §2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
