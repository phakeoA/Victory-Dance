"""DS-4c stage 2 (2026-07-12) — Bo3 set-state carrier.

Stage-1 facts (docs/backlog_audit_design_2026-07-11.md): every Bo3 game room announces
``|uhtml|bestof|…Game N</strong> of <a href="/game-bestof3-…">`` — game index + parent set
room, parseable per battle; teams lock per set; teampreview runs PER GAME. This module keeps
per-set state on the PLAYER instance (the ``_ots_sheets`` pattern — consumer-populated dicts,
``getattr`` reads, never raises) so game-2/3 team previews can see what both sides brought in
the previous game (the stage-3 ``set_ctx`` TP input) and the panel/logs can show set progress.

Stores on the player:
  ``_bo3_of``   : {battle_base_tag: (parent_id, game_idx)}
  ``_bo3_sets`` : {parent_id: {game_idx: {"tag", "our_bring", "our_leads", "opp_seen", "result"}}}
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_BESTOF_RE = re.compile(
    r"\|uhtml\|bestof\|.*?Game (\d+)</strong> of <a href=\"/(game-bestof3-[a-z0-9-]+)\"")


def _base(tag: str) -> str:
    parts = (tag or "").lstrip(">").split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else (tag or "")


def note_bestof_frame(player, tag: str, payload: str) -> None:
    """Register a battle room as game N of its parent set (from the |uhtml|bestof| frame)."""
    try:
        m = _BESTOF_RE.search(payload)
        if not m:
            return
        idx, parent = int(m.group(1)), m.group(2)
        player.__dict__.setdefault("_bo3_of", {})[_base(tag)] = (parent, idx)
        games = player.__dict__.setdefault("_bo3_sets", {}).setdefault(parent, {})
        games.setdefault(idx, {"tag": _base(tag), "our_bring": [], "our_leads": [],
                               "opp_seen": [], "result": None})
        log.info("Bo3 [%s]: game %d of set %s", _base(tag), idx, parent)
    except Exception:                                     # noqa: BLE001 — never break the feed
        log.debug("bo3 note_bestof_frame failed", exc_info=True)


def record_our_picks(player, battle, picks_species: List[str]) -> None:
    """Called from _choose_team_order after the picks resolve (leads-first order)."""
    try:
        ref = (getattr(player, "_bo3_of", None) or {}).get(_base(getattr(battle, "battle_tag", "")))
        if not ref:
            return
        parent, idx = ref
        g = player._bo3_sets[parent][idx]
        g["our_bring"] = list(picks_species)
        g["our_leads"] = list(picks_species)[:2]
    except Exception:                                     # noqa: BLE001
        log.debug("bo3 record_our_picks failed", exc_info=True)


def record_game_end(player, tag: str, result: Optional[str]) -> None:
    """Harvest the finished game: opp mons that APPEARED (the training 'brought' lower
    bound) + the result. Consumer calls this on the |win|/|tie| frame."""
    try:
        ref = (getattr(player, "_bo3_of", None) or {}).get(_base(tag))
        if not ref:
            return
        parent, idx = ref
        b = getattr(player, "_battles", {}).get(_base(tag)) \
            or getattr(player, "_battles", {}).get((tag or "").lstrip(">"))
        opp_seen: List[str] = []
        if b is not None:
            for mon in (getattr(b, "opponent_team", None) or {}).values():
                sp = getattr(mon, "base_species", None) or getattr(mon, "species", None)
                if sp:
                    opp_seen.append(str(sp))
        g = player._bo3_sets[parent][idx]
        g["opp_seen"] = opp_seen
        g["result"] = result
        log.info("Bo3 set %s: game %d ended (%s) — opp showed %s",
                 parent, idx, result, opp_seen)
    except Exception:                                     # noqa: BLE001
        log.debug("bo3 record_game_end failed", exc_info=True)


def set_ctx_for(player, battle, our_species: List[str], opp_species: List[str]
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(our_ctx, opp_ctx) each (6, 2) float32 [brought_last, led_last] aligned to the
    CURRENT preview rosters by norm_species — from the previous game of this battle's set.
    None when the battle is game 1 / not in a set / nothing recorded (→ the caller serves
    exactly the set-blind path)."""
    try:
        ref = (getattr(player, "_bo3_of", None) or {}).get(_base(getattr(battle, "battle_tag", "")))
        if not ref:
            return None
        parent, idx = ref
        prev = (getattr(player, "_bo3_sets", {}).get(parent) or {}).get(idx - 1)
        if not prev or (not prev.get("our_bring") and not prev.get("opp_seen")):
            return None
        from v_dance.parser.vod_parser.pokedex import norm_species

        def _ctx(roster: List[str], brought: List[str], leads: List[str]) -> np.ndarray:
            b = {norm_species(s) for s in brought if s}
            ld = {norm_species(s) for s in leads if s}
            out = np.zeros((6, 2), dtype=np.float32)
            for i, sp in enumerate(list(roster)[:6]):
                ns = norm_species(sp or "")
                if ns in b:
                    out[i, 0] = 1.0
                if ns in ld:
                    out[i, 1] = 1.0
            return out

        our = _ctx(our_species, prev.get("our_bring") or [], prev.get("our_leads") or [])
        # Opp leads unknown from appearances alone → led_last stays 0 for them (honest zero).
        opp = _ctx(opp_species, prev.get("opp_seen") or [], [])
        return our, opp
    except Exception:                                     # noqa: BLE001
        log.debug("bo3 set_ctx_for failed", exc_info=True)
        return None
