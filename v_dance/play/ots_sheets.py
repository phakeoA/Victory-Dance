"""Open team sheets at serve time (USER 2026-09-03: "a toggle where the bot accepts open team sheets and
processes that data to help the neural networks think if the enemy also accepts").

The server offers open team sheets at team preview (rule ``openteamsheets``); when BOTH players accept it
emits ``|showteam|p1|<packed>`` + ``|showteam|p2|<packed>`` — every non-stat fact about both teams (moves,
item, ability, nature, tera type; EVs stripped). The consumer (``play_vs_human_browser``) captures those
frames into ``player._ots_sheets[room_base_tag] = {"p1": [mon...], "p2": [...]}``; this module hands the
OPPONENT's sheet to the battle net's opponent snapshot, stamped SHEET-AUTHORITATIVE with the very routine
the training corpus used for its OTS games (``transitions._inject_known_stats(sheet_authoritative=True)``:
known moves = the sheet's four, a sheet with no item = a confirmed itemless mon, nature kept, EVs stay
belief-estimated). The dossier then fills nothing and the match belief narrows around ground truth — the
same tier order the nets were trained on. Closed games: no sheet → every function is a no-op.

The TEAM-PREVIEW net keeps its closed-sheet input (``player.ots_opp_known`` stays behind
``VD_TP_OTS_OVERLAY`` + an OTS-trained TP checkpoint — the deployed one never saw sheet overlays).
"""
from __future__ import annotations

import logging
from typing import List, Optional

log = logging.getLogger(__name__)


def room_base_tag(tag: str) -> str:
    """``battle-<fmt>-<num>[-privsuffix]`` -> ``battle-<fmt>-<num>`` (the capture's key)."""
    parts = (tag or "").lstrip(">").split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else (tag or "")


def opp_sheet_mons(player, battle) -> Optional[List[dict]]:
    """The OPPONENT's revealed sheet for this battle (parsed packed mons), or None (closed game /
    nothing captured / role unknown). Never raises."""
    try:
        sheets = getattr(player, "_ots_sheets", None) or {}
        sides = sheets.get(room_base_tag(getattr(battle, "battle_tag", "") or "")) or {}
        role = getattr(battle, "player_role", None)
        if role not in ("p1", "p2"):
            return None
        mons = sides.get("p2" if role == "p1" else "p1") or None
        return list(mons) if mons else None
    except Exception:                                    # noqa: BLE001 — never break a turn
        return None


def stamp_ots_sheets(opp_snapshot: Optional[dict], mons: Optional[List[dict]]) -> int:
    """Stamp the opponent's sheet into the live opponent snapshot IN PLACE (``opp_active`` /
    ``opp_bench`` mons, norm-species keyed; exact or transformed mons skipped — same rules as the
    offline ``_apply_ots_knowledge``). Returns the number of mons stamped."""
    if not opp_snapshot or not mons:
        return 0
    from v_dance.parser.vod_parser.pokedex import norm_species
    from v_dance.parser.vod_parser.team_sheet import packed_team_to_known_side
    from v_dance.parser.vod_parser.transitions import _inject_known_stats
    side = {norm_species(k): v for k, v in packed_team_to_known_side(mons).items()}
    n = 0
    for key in ("opp_active", "opp_bench"):
        cont = opp_snapshot.get(key) or {}
        for mon in (cont.values() if isinstance(cont, dict) else cont):
            if not isinstance(mon, dict) or mon.get("exact") or mon.get("is_transformed"):
                continue
            base = mon.get("base_species") or mon.get("species") or ""
            inj = side.get(norm_species(base)) or side.get(norm_species(mon.get("species") or "")) or {}
            if inj:
                _inject_known_stats(mon, inj, sheet_authoritative=True)
                n += 1
    return n


def apply_ots_sheets(opp_snapshot: Optional[dict], mons: Optional[List[dict]]) -> Optional[dict]:
    """``stamp_ots_sheets`` returning the snapshot (the tier-order call site shape)."""
    stamp_ots_sheets(opp_snapshot, mons)
    return opp_snapshot


def ots_known(player, tag: str) -> bool:
    """Both sheets captured for ``tag`` (the server only emits them when both players accepted)."""
    try:
        sides = (getattr(player, "_ots_sheets", None) or {}).get(room_base_tag(tag)) or {}
        return bool(sides.get("p1") and sides.get("p2"))
    except Exception:                                    # noqa: BLE001
        return False
