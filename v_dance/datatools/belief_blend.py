"""
belief_blend.py
===============
Era-retrain step 0 (USER GO, 2026-07-10): blend the scraped Pikalytics prior with the OBSERVED
ladder meta (``observed_meta_<reg>.json`` — what the bot ACTUALLY faces at its rating band) into
one BeliefState-schema JSON that the whole era uses: the Type_C ingest, the corpus re-export, and
(at deploy) the live serve belief.

Blend rule (per species, matched via ``norm_species`` — observed uses normalised ids, Pikalytics
display names):
  * per-species weight  w = n_s / (n_s + k)   where n_s = observed sightings
    (``usage_pct/100 × n_games``) and ``k`` is the pseudo-count (default 20) — a species seen 20+
    times leans on reality, a species seen once stays ~at the prior.
  * ``moves`` / ``items`` / ``abilities``: entry-wise convex blend over the UNION
    (missing side = 0), output keeps the Pikalytics display name when it exists.
  * ``usage_pct``: same blend; a null Pikalytics usage counts as 0 (the loader already treats
    None as 0.0 — blending strictly ADDS information).
  * ``spreads`` / ``natures`` / ``teammates``: Pikalytics pass-through (spreads/natures are not
    observable from replays; observed teammates use pct where Pikalytics uses rank — mixing the
    two would corrupt the sort the loader does).
  * species only in OBSERVED are added as-is (the aggregator already emits BeliefState-schema).

Usage (defaults target the active reg's files)::

    python -m v_dance.datatools.belief_blend                       # writes the dated blend file
    python -m v_dance.datatools.belief_blend --k 20 --out data/pikalytics_regmb_blend_X.json

⚠ The output is NOT installed as the live serve belief — the deployed net was trained on the
OLD belief and must keep matching inputs. Steps 1-2 pass the blend via ``--pikalytics``; the
deploy step copies it over ``pikalytics_<reg>.json`` AND re-pins the parity test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from v_dance.formats import reg_token, DEFAULT_FORMAT
from v_dance.parser.vod_parser.pokedex import norm_species

_REPO = Path(__file__).resolve().parents[2]
_BLEND_BLOCKS = ("moves", "items", "abilities")   # entry-lists of {"name": …, "pct": …}


def _blend_block(pika: list, obs: list, w: float) -> list:
    """Convex blend of two {"name","pct"} lists over the union of normalised names.
    Pikalytics display names win; entries keep any EXTRA Pikalytics fields untouched."""
    out: dict = {}
    for e in pika or []:
        nid = norm_species(e.get("name") or "")
        if nid:
            out[nid] = dict(e)
            out[nid]["pct"] = (1.0 - w) * float(e.get("pct") or 0.0)
    for e in obs or []:
        nid = norm_species(e.get("name") or "")
        if not nid:
            continue
        if nid in out:
            out[nid]["pct"] += w * float(e.get("pct") or 0.0)
        else:
            out[nid] = {"name": e.get("name"), "pct": w * float(e.get("pct") or 0.0)}
    entries = sorted(out.values(), key=lambda e: -e["pct"])
    for e in entries:
        e["pct"] = round(e["pct"], 2)
    return entries


def blend(pika: dict, observed: dict, k: float = 20.0) -> dict:
    """Blend full Pikalytics + observed-meta docs → a new Pikalytics-schema doc (+ provenance)."""
    n_games = float(observed.get("n_games") or 0)
    obs_pk = observed.get("pokemon") or {}
    obs_by_id = {norm_species(sp): (sp, entry) for sp, entry in obs_pk.items()}

    out_pokemon: dict = {}
    stats = {"blended": 0, "passthrough": 0, "added": 0}
    for sp, entry in (pika.get("pokemon") or {}).items():
        hit = obs_by_id.pop(norm_species(sp), None)
        if hit is None:
            out_pokemon[sp] = entry
            stats["passthrough"] += 1
            continue
        _osp, oe = hit
        n_s = (float(oe.get("usage_pct") or 0.0) / 100.0) * n_games
        w = n_s / (n_s + k) if k > 0 else 1.0
        ne = dict(entry)
        for block in _BLEND_BLOCKS:
            ne[block] = _blend_block(entry.get(block), oe.get(block), w)
        ne["usage_pct"] = round((1.0 - w) * float(entry.get("usage_pct") or 0.0)
                                + w * float(oe.get("usage_pct") or 0.0), 2)
        ne["blend_w"] = round(w, 3)                # provenance (loader ignores unknown keys)
        out_pokemon[sp] = ne
        stats["blended"] += 1
    for _nid, (osp, oe) in obs_by_id.items():      # observed-only species (prior had nothing)
        out_pokemon[osp] = oe
        stats["added"] += 1

    doc = {k2: v for k2, v in pika.items() if k2 != "pokemon"}
    doc["pokemon"] = out_pokemon
    doc["blend_provenance"] = {
        "blended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "observed_n_games": n_games, "k": k,
        "pikalytics_scraped_at": pika.get("scraped_at"),
        "observed_scraped_at": observed.get("scraped_at"),
        "counts": stats,
    }
    return doc


def main() -> None:
    reg = reg_token(DEFAULT_FORMAT) or "regmb"
    today = time.strftime("%Y-%m-%d", time.gmtime())
    ap = argparse.ArgumentParser(description="Blend Pikalytics prior × observed ladder meta.")
    ap.add_argument("--pikalytics", default=str(_REPO / "data" / f"pikalytics_{reg}.json"))
    ap.add_argument("--observed", default=str(_REPO / "data" / f"observed_meta_{reg}.json"))
    ap.add_argument("--out", default=str(_REPO / "data" / f"pikalytics_{reg}_blend_{today}.json"))
    ap.add_argument("--k", type=float, default=20.0,
                    help="pseudo-count: species weight w = n/(n+k) (default 20).")
    args = ap.parse_args()

    pika = json.loads(Path(args.pikalytics).read_text(encoding="utf-8"))
    observed = json.loads(Path(args.observed).read_text(encoding="utf-8"))
    doc = blend(pika, observed, k=args.k)
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    st = doc["blend_provenance"]["counts"]
    print(f"[belief_blend] {args.pikalytics}  ×  {args.observed}  (n_games={observed.get('n_games')})")
    print(f"    blended {st['blended']} species · {st['passthrough']} prior-only pass-through · "
          f"{st['added']} observed-only added   (k={args.k})")
    top = sorted(((e.get('blend_w') or 0, sp) for sp, e in doc['pokemon'].items()
                  if isinstance(e, dict) and e.get('blend_w')), reverse=True)[:8]
    print("    heaviest observed influence: " + ", ".join(f"{sp} w={w}" for w, sp in top))
    print(f"    → {args.out}")
    print("    ⚠ NOT installed as the live serve belief — pass it via --pikalytics to the era's "
          "ingest/re-export; the deploy step installs + re-pins.")


if __name__ == "__main__":
    main()
