"""Seed the archetype MATCHUP MATRIX (DS-4b 2026-07-12; trimmed 2026-10-04) — DATA-DRIVEN, not hand-set.

Builds the archetype-vs-archetype WIN MATRIX from the corpus (k10_full assignments joined to per-replay outcomes
via the encoded-cache meta arrays — no re-parse) and writes ``data/router_matrix.json``:
matrix[a][b] = [win rate of cluster a vs cluster b, games] (cells under MIN_GAMES omitted). The team builder's
matchup prior (``team_generator.score_team``) reads it for GENERATED teams.

(Was ``seed_router_priors``: it also wrote ``data/router_priors.json`` for the 4b between-game team router — a
closed experiment, removed with the router in cleanup pass 2, 2026-10-04.)

Run:  PYTHONUTF8=1 .venv/Scripts/python.exe -X utf8 -m v_dance.datatools.seed_matchup_matrix
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]          # v_dance/datatools/ -> repo root
assert (_REPO / "pyproject.toml").is_file(), _REPO

ARTIFACT = _REPO / "ai_train_scripts" / "BC_model" / "team_archetypes_k10_full.npz"
OUT = _REPO / "data" / "router_matrix.json"          # the file name the team builder reads (kept as is)
MIN_GAMES = 30          # matrix cells thinner than this are omitted
PREP = _REPO / "data" / "vods" / "Prepared_training_data"
FOLDERS = [PREP / reg / sub
           for reg in ("Regulation_MA", "Regulation_MA_Bo3", "Regulation_MB", "Regulation_MB_Bo3")
           for sub in ("Jsonl_TypeA", "Jsonl_TypeB", "Jsonl_HF_OTS", "Jsonl_TypeC")]


def outcomes_from_caches() -> dict:
    """{(rid, persp): won_bool} from every folder's encoded-cache meta (one load, no parse)."""
    from v_dance.training.encoded_cache import _cache_dir, folder_fingerprint
    out = {}
    for folder in FOLDERS:
        if not folder.is_dir():
            continue
        d = _cache_dir(str(folder)) / folder_fingerprint(str(folder))
        if not (d / "DONE").exists():
            print(f"  (no cache for {folder.name} under {folder.parent.name} — skipped)")
            continue
        m = np.load(d / "meta.npz")
        rid, persp, won = m["rid"], m["persp"], m["won"]
        for i in range(len(rid)):
            if won[i] >= 0:
                out[(str(rid[i]), str(persp[i]) or None)] = bool(won[i])
    return out


def main() -> int:
    from v_dance.datatools.team_archetypes import load_archetype_assignments, load_artifact

    art = load_artifact(str(ARTIFACT))
    k = int(art["k"])
    asg = load_archetype_assignments(str(ARTIFACT))
    print(f"[seed] assignments: {len(asg)} across k={k}")

    print("[seed] loading outcomes from encoded caches …")
    won = outcomes_from_caches()
    print(f"[seed] outcomes: {len(won)} (rid, persp) rows")

    # Directed win matrix: wins[a][b] / n[a][b] = games a-side won vs b-side.
    wins = np.zeros((k, k)); n = np.zeros((k, k))
    rids = defaultdict(dict)
    for (rid, persp), z in asg.items():
        rids[rid][persp] = z
    joined = 0
    for rid, sides in rids.items():
        if len(sides) != 2:
            continue
        (pa, za), (pb, zb) = sorted(sides.items())
        for me, zm, zt in ((pa, za, zb), (pb, zb, za)):
            w = won.get((rid, me))
            if w is None:
                continue
            wins[zm][zt] += int(w); n[zm][zt] += 1
            joined += 1
    print(f"[seed] joined directed samples: {joined}")

    matrix = {str(i): {str(j): [round(float(wins[i][j] / n[i][j]), 4), int(n[i][j])]
                       for j in range(k) if n[i][j] >= MIN_GAMES}
              for i in range(k)}
    OUT.write_text(json.dumps(matrix, indent=1), encoding="utf-8")
    print(f"[seed] wrote {OUT} (cluster-vs-cluster win matrix)")

    print("\n  win-rate matrix (row=our cluster, col=opp cluster; '.'=n<%d):" % MIN_GAMES)
    print("        " + "".join(f"z{j:<7d}" for j in range(k)))
    for i in range(k):
        cells = "".join((f"{wins[i][j]/n[i][j]:.2f}" + f"({int(n[i][j])})").ljust(8)
                        if n[i][j] >= MIN_GAMES else ".".ljust(8) for j in range(k))
        print(f"  z{i:<5d} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
