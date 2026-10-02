"""Freeze per-format MEGA-STONE shares into ``data/mega_stone_shares.json`` (layout v20 mega preview, 2026-10-02).

The v20 mega-preview channels need, for a mon that has not mega-evolved yet, the chance it holds each of its
species' stones. The exported training snapshots never carry that (``belief_block`` strips stones before the
top-5 cut), and the live belief is re-scraped during a regulation — so both encoders read ONE frozen table,
keyed by regulation (a Reg M-B Garchomp never held Garchompite Z; a Reg M-C one does 45 % of the time):

    {"version": 1, "sources": {reg: path}, "formats": {reg: {species_id: {stone_id: p}}}}

``p`` = the stone's raw usage share (pct/100, not renormalised). Only stones that make a mega forme OF THAT
species are kept. The table's content hash is part of the encoded-cache fingerprint, so regenerating it
re-encodes the training caches (``v_dance.encoders.mega_preview.TABLE_FINGERPRINT``).

Regenerate (only when a regulation's belief is re-pinned — a new table changes what the net reads):
    .venv/Scripts/python.exe -m v_dance.datatools.mega_stone_shares
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "mega_stone_shares.json"

# The belief each corpus was EXPORTED with (transitions' belief_fill.pikalytics_source) — so the training
# rows of a regulation read that regulation's stone mix.
SOURCES = {
    "regma": "data/pikalytics_regma.json",
    "regmb": "data/pikalytics_regmb_blend_2026-07-11.json",
    "regmc": "data/belief_regmc.json",
}


def _canon(s) -> str:
    return "".join(c for c in str(s or "").lower() if c.isalnum())


def build_table(path: str) -> dict:
    from v_dance.dex.pokedex import get_pokedex, norm_species
    from v_dance.parser.belief_state import BeliefState, is_mega_stone
    dex = get_pokedex()
    b = BeliefState(REPO / path)
    out: dict = {}
    for name in b.all_pokemon():
        if dex.is_mega_forme(name):
            continue
        formes = dex.mega_formes_for(name)
        if not formes:
            continue
        own_stones = {_canon((dex.entry(f["forme"]) or {}).get("requiredItem")) for f in formes} - {""}
        shares = {}
        for it in b.item_distribution(name, top_k=64):
            st = _canon(it["name"])
            if is_mega_stone(it["name"]) and st in own_stones and float(it["p"]) > 0:
                shares[st] = max(shares.get(st, 0.0), round(float(it["p"]), 4))
        if shares:
            out[norm_species(name)] = dict(sorted(shares.items()))
    # Second pass (review 10-02): some scrapes (Pikalytics Reg M-A) file stone usage under the MEGA FORME entries
    # (Charizard-Mega-Y 32 %, 100 % Charizardite Y) and list no stone on the base. There the share of a stone =
    # usage(its mega forme) / (usage(the pre-mega forme) + Σ usage(that forme's megas)). Base-entry data wins.
    from v_dance.encoders.mega_preview import mega_reachable
    usage = {norm_species(n): float((b._data.get(n) or {}).get("usage_pct") or 0.0) for n in b.all_pokemon()}
    by_pre: dict = {}
    for name in b.all_pokemon():
        if not dex.is_mega_forme(name):
            continue
        fe = dex.entry(name) or {}
        bo = fe.get("battleOnly")
        pre = norm_species((bo[0] if isinstance(bo, (list, tuple)) else bo) or fe.get("baseSpecies") or "")
        st = _canon(fe.get("requiredItem"))
        u = usage.get(norm_species(name), 0.0)
        if pre and st and u > 0 and mega_reachable(pre, fe):
            by_pre.setdefault(pre, {})[st] = by_pre.setdefault(pre, {}).get(st, 0.0) + u
    for pre, stones in by_pre.items():
        if pre in out:
            continue
        tot = usage.get(pre, 0.0) + sum(stones.values())
        if tot > 0:
            out[pre] = dict(sorted((st, round(u / tot, 4)) for st, u in stones.items()))
    return dict(sorted(out.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    table = {"version": 1, "sources": dict(SOURCES),
             "formats": {reg: build_table(p) for reg, p in SOURCES.items()}}
    out = Path(args.out)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    for reg, t in table["formats"].items():
        print(f"[mega-stones] {reg}: {len(t)} species with stone shares")
    print(f"[mega-stones] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
