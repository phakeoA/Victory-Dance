"""DS-4b priors seeding (2026-07-12) — DATA-DRIVEN, not hand-set.

Builds the archetype-vs-archetype WIN MATRIX from the corpus (k10_full assignments
joined to per-replay outcomes via the encoded-cache meta arrays — no re-parse), maps
every team in the active pool to its cluster (assign_team_sheet on the paste), and
writes ``data/router_priors.json``: priors[opp_z][team] = winrate(cluster(team) vs z).

Run:  PYTHONUTF8=1 .venv/Scripts/python.exe -X utf8 -m v_dance.datatools.seed_router_priors
(moved from scratch/seed_router_priors.py in the 2026-09-10 refactor, Phase 1)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]          # v_dance/datatools/ -> repo root
assert (_REPO / "pyproject.toml").is_file(), _REPO

ARTIFACT = _REPO / "ai_train_scripts" / "BC_model" / "team_archetypes_k10_full.npz"
OUT = _REPO / "data" / "router_priors.json"
MIN_GAMES = 30          # matrix cells thinner than this contribute no prior
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
    from v_dance.datatools.team_archetypes import (
        load_archetype_assignments, load_artifact, assign_team_sheet)
    from v_dance.dex.team_sheet import parse_showdown_team

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

    # Pool teams → clusters.
    pool_dir = _REPO / "teams" / "Champions" / "M-B"
    team_z = {}
    for f in sorted(pool_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            mons = parse_showdown_team(f.read_text(encoding="utf-8"))
            if len(mons) < 6:
                print(f"  ! {f.name}: only {len(mons)} mons parsed — skipped")
                continue
            z, dist = assign_team_sheet(mons, art)
            team_z[f.name] = (int(z), float(dist))
        except Exception as exc:                        # noqa: BLE001
            print(f"  ! {f.name}: {exc!r} — skipped")
    zc = Counter(z for z, _ in team_z.values())
    print(f"[seed] pool teams assigned: {len(team_z)} (cluster spread {dict(zc)})")

    # priors[opp_z][team] = winrate of the team's cluster vs opp_z (thin cells skipped).
    priors: dict = {}
    for oz in range(k):
        row = {}
        for team, (tz, _d) in team_z.items():
            if n[tz][oz] >= MIN_GAMES:
                row[team] = round(float(wins[tz][oz] / n[tz][oz]), 4)
        if row:
            priors[str(oz)] = row
    doc = ("DS-4b priors — DATA-SEEDED 2026-07-12 from the k10_full archetype-vs-archetype "
           f"win matrix ({joined} directed corpus samples, min cell n={MIN_GAMES}); "
           "priors[opp_z][team] = winrate(cluster(team) vs opp_z). Regenerate: "
           "python -m v_dance.datatools.seed_router_priors. Edit freely — the router treats this as opinion.")
    OUT.write_text(json.dumps({"_doc": doc, **priors}, indent=1), encoding="utf-8")
    print(f"[seed] wrote {OUT} ({len(priors)}/{k} opponent clusters seeded)")

    # M4 (B4): the RAW cluster-vs-cluster matrix sidecar — team_generator.score_team's
    # matchup_prior for GENERATED teams (which have no priors-file row). Cells under
    # MIN_GAMES are omitted (same thinness rule as the priors).
    matrix = {str(i): {str(j): [round(float(wins[i][j] / n[i][j]), 4), int(n[i][j])]
                       for j in range(k) if n[i][j] >= MIN_GAMES}
              for i in range(k)}
    mx_path = _REPO / "data" / "router_matrix.json"
    mx_path.write_text(json.dumps(matrix, indent=1), encoding="utf-8")
    print(f"[seed] wrote {mx_path} (cluster-vs-cluster win matrix)")

    # Receipt: matrix + pool mapping.
    print("\n  win-rate matrix (row=our cluster, col=opp cluster; '.'=n<%d):" % MIN_GAMES)
    hdr = "        " + "".join(f"z{j:<7d}" for j in range(k))
    print(hdr)
    for i in range(k):
        cells = "".join((f"{wins[i][j]/n[i][j]:.2f}" + f"({int(n[i][j])})").ljust(8)
                        if n[i][j] >= MIN_GAMES else ".".ljust(8) for j in range(k))
        print(f"  z{i:<5d} {cells}")
    print("\n  pool → cluster (distance):")
    for t, (z, d) in sorted(team_z.items(), key=lambda kv: kv[1][0]):
        print(f"    z{z}  {t:<24s} d={d:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
