"""Phase-4b v1 — BPR-lite between-game team routing (DS-4b, docs/backlog_audit_design_2026-07-11.md).

Harness-level TEAM-CHOICE policy vs a KNOWN opponent (challenge-accept path — the ladder
queue can't route, the opponent is unknown until matched). Policy v1: if we LOST the last
game vs this opponent, pick the pool team with the best prior vs their revealed archetype
(``team_archetypes.assign_team_sheet`` over the dossier's mon records); otherwise no
opinion. NEVER a net input (within-game z-conditioning is a closed null) and NEVER raises —
routing must not break play (the dossier-capture contract).

Default OFF: the ``VD_ROUTE_TEAMS=1`` env flag gates the wiring in
``play_vs_human_browser._pick_ai_team`` (precedence: pin > router > Teambuilder > random).

Priors: ``data/router_priors.json`` = ``{"<archetype_id>": {"<team_name>": score}}``.
⚠ Ships as an EMPTY template: the old TR→z4/Rain→z8 id mapping belonged to the SUPERSEDED
k10 artifact — ids must be re-derived from ``team_archetypes_k10_full.npz`` (inspect cluster
exemplars) before seeding. Unseeded ⇒ the router always answers "no opinion".
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[2]
PRIORS_PATH = _REPO / "data" / "router_priors.json"
ARTIFACT_PATH = _REPO / "ai_train_scripts" / "BC_model" / "team_archetypes_k10_full.npz"

# Serve gate — read at import; tests/panel may set the module attr directly.
ROUTE_TEAMS = os.environ.get("VD_ROUTE_TEAMS", "") == "1"


@lru_cache(maxsize=1)
def _artifact():
    from v_dance.datatools.team_archetypes import load_artifact
    return load_artifact(str(ARTIFACT_PATH))


def _assign_archetype(mons: List[dict]) -> Optional[int]:
    """Dossier mon records → archetype id via the serve-side sheet assigner.
    Isolated so tests can monkeypatch it (building a real artifact is heavy)."""
    from v_dance.datatools.team_archetypes import assign_team_sheet
    sheet = [{"species": m.get("species"), "item": m.get("item"),
              "ability": m.get("ability"), "moves": m.get("moves") or []}
             for m in mons if m.get("species")]
    if not sheet:
        return None
    zid, _dist = assign_team_sheet(sheet, _artifact())
    return int(zid)


def _priors(path: Optional[Path]) -> dict:
    p = Path(path) if path else PRIORS_PATH
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def route(opponent: str, pool: List[str],
          priors_path: Optional[Path] = None) -> Tuple[Optional[str], str]:
    """(team_or_None, reason). None = no opinion → the caller falls through to its
    normal pick (Teambuilder-open / random). Never raises."""
    try:
        from v_dance.play import opponent_dossier
        d = opponent_dossier.load(opponent)
        games = d.get("games") or []
        if not games:
            return None, "no dossier games"
        last = games[-1]
        if last.get("result") != "human":            # bench convention: "human" = we LOST
            return None, "won/drew last game — keep course"
        lost_team = last.get("our_team")
        # Assign the archetype from the LAST game's revealed team when the dossier records it
        # (new format) — falls back to the all-time union for pre-2026-07-13 dossiers.
        all_mons = d.get("mons") or {}
        last_revealed = set(last.get("revealed") or [])
        mons = ([all_mons[k] for k in last_revealed if k in all_mons] if last_revealed
                else list(all_mons.values()))
        zid = _assign_archetype(mons)
        if zid is None:
            return None, "no revealed mons to assign an archetype"
        entry = _priors(priors_path).get(str(zid)) or {}
        scored = {t: float(s) for t, s in entry.items()
                  if t in pool and isinstance(s, (int, float))}
        if not scored:
            return None, f"no prior for archetype z{zid} (priors unseeded?)"
        # Prefer NOT repeating the team we just lost with, unless it is the only candidate.
        candidates = {t: s for t, s in scored.items() if t != lost_team} or scored
        best = max(candidates, key=candidates.get)
        return best, (f"lost last game vs {opponent} with {lost_team!r}; "
                      f"their archetype z{zid} → best prior {best!r} ({candidates[best]:.2f})")
    except Exception as exc:                          # noqa: BLE001 — never break play
        return None, f"router error ({exc}) — no opinion"
