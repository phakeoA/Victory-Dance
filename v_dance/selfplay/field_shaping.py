"""REWARD v2 — the FIELD POTENTIAL Φ and the per-game GUARDS (2026-10-04; memory 14 'REWARD v2: DESIGN DECIDED').

USER: "rework our reward / punishment system … better than +1 win -1 losing while not being harmful to pokemon
playstyles … make sure that winning and losing is overall the biggest punishment and reward" → the design (grill-me):
teach WEATHER and TERRAIN control through a potential whose CHANGE the trainer pays (v_dance/rl/reward.shaped_rewards).

    Φ(s) = 0.1 · w + 0.1 · t

w = the OWNER of the weather that is up (+1 ours, −1 theirs, 0 neutral or none), t = the same for the terrain.
OWNERSHIP = WHO CAN SET IT, read once per game from the two team sheets — privileged in self-play (the chunk's own team
files), which is fine: Φ only shapes the reward, it is never a network input. A kind only our sheet can set is ours, a
kind only theirs can set is theirs, a kind both (or neither) can set is neutral: a mirror is neutral everywhere, and
Psychic Terrain against another Psychic Surge pays nothing. A sheet sets a kind through
  · its ABILITY (Drizzle, Drought, Sand Stream, Snow Warning, Orichalcum Pulse, the primals, Sand Spit; Psychic /
    Grassy / Electric / Misty Surge, Hadron Engine, Seed Sower),
  · its MEGA STONE (the mega's ability — Tyranitarite → Sand Stream, Charizardite Y → Drought, Raichunite X → Electric
    Surge, Froslassite → Snow Warning; dex-derived, ``field_control_report.ITEM_*``),
  · a SETTING MOVE (Rain Dance, Sunny Day, Sandstorm, Snowscape, Hail, Chilly Reception; Psychic / Grassy / Electric /
    Misty Terrain).
The sheets are the full six (the decided "team sheets"): a kind the opponent's six can set stays contested even when
that mon was left at home — never a wrong-signed credit, only a missed one.

Paid as F = γ·Φ(s') − Φ(s) with Φ(terminal) = 0, the shaping telescopes to −Φ(s₀) over a game: re-setting the same
weather earns nothing, the best policy is unchanged (it only speeds up learning what already wins), and a whole game of
field control is worth at most 0.2 next to a result of ±1. The battle net's first decision already sees the leads'
weather / terrain, so the picker (no shaping) is never paid for Φ(s₀).

The per-game GUARDS (``game_guard``) are logged, never rewarded: our own faints in won games (are sacrifices still
made?), Protect-family and Trick Room uses — the playstyles the loss margin must not bend.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import numpy as np

from v_dance.rl.reward import FIELD_STRENGTH, REWARD_V2, is_v2, terminal_reward

DOMAINS = ("weather", "terrain")
# setting MOVES (field_control_report reads abilities + mega stones only)
MOVE_WEATHER = {"raindance": "rain", "sunnyday": "sun", "sandstorm": "sand", "snowscape": "snow", "hail": "snow",
                "chillyreception": "snow"}
MOVE_TERRAIN = {"psychicterrain": "psychic", "grassyterrain": "grassy", "electricterrain": "electric",
                "mistyterrain": "misty"}
# setters the field report's ability table lacks: Sand Spit (sand when the holder is hit), Delta Stream (strong winds)
EXTRA_ABILITY_WEATHER = {"sandspit": "sand", "deltastream": "wind"}
PROTECT_MOVES = frozenset({"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap",
                           "burningbulwark", "obstruct", "maxguard"})


def _id(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


# ── ownership (once per game) ────────────────────────────────────────────────
def sheet_kinds(paste: str) -> Dict[str, frozenset]:
    """The weather and terrain kinds a team sheet (a Showdown paste) CAN set: ``{"weather": …, "terrain": …}``."""
    from v_dance.dex.team_sheet import parse_showdown_team
    from v_dance.eval.field_control_report import ABILITY_TERRAIN, ABILITY_WEATHER, ITEM_TERRAIN, ITEM_WEATHER
    weather, terrain = set(), set()
    for mon in parse_showdown_team(paste or ""):
        ab, item = _id(mon.get("ability")), _id(mon.get("item"))
        for table, out in ((ABILITY_WEATHER, weather), (EXTRA_ABILITY_WEATHER, weather), (ABILITY_TERRAIN, terrain)):
            if ab in table:
                out.add(table[ab])
        if item in ITEM_WEATHER:                       # a mega stone: the mega's ability (Drought, Sand Stream …)
            weather.add(ITEM_WEATHER[item])
        if item in ITEM_TERRAIN:
            terrain.add(ITEM_TERRAIN[item])
        for mv in mon.get("moves") or ():
            m = _id(mv)
            if m in MOVE_WEATHER:
                weather.add(MOVE_WEATHER[m])
            if m in MOVE_TERRAIN:
                terrain.add(MOVE_TERRAIN[m])
    return {"weather": frozenset(weather), "terrain": frozenset(terrain)}


def owners(own_paste: str, opp_paste: str) -> Dict[str, Dict[str, int]]:
    """``{"weather": {kind: +1 | -1}, "terrain": {…}}`` from THIS side's view: +1 = only our sheet can set the kind,
    -1 = only theirs can; a kind both (or neither) can set is absent = neutral (0)."""
    ours, theirs = sheet_kinds(own_paste), sheet_kinds(opp_paste)
    out: Dict[str, Dict[str, int]] = {}
    for dom in DOMAINS:
        d = {k: 1 for k in ours[dom] - theirs[dom]}
        d.update({k: -1 for k in theirs[dom] - ours[dom]})
        out[dom] = d
    return out


def flip(own: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """The same ownership from the OTHER side's view."""
    return {dom: {k: -v for k, v in (own.get(dom) or {}).items()} for dom in DOMAINS}


def describe(own: Dict[str, Dict[str, int]]) -> str:
    def side(sign):
        ks = [k for dom in DOMAINS for k, v in sorted((own.get(dom) or {}).items()) if v == sign]
        return ", ".join(ks) or "-"
    return f"ours: {side(1)} | theirs: {side(-1)}"


# ── the potential (every decision) ───────────────────────────────────────────
def _owner(table: Dict[str, int], kinds: Iterable[str]) -> int:
    return max(-1, min(1, sum(int(table.get(k, 0)) for k in kinds)))


def phi_from_kinds(weather_kinds: Iterable[str], terrain_kinds: Iterable[str], own: Dict[str, Dict[str, int]]) -> float:
    """Φ = FIELD_STRENGTH × (owner of the weather that is up + owner of the terrain that is up)."""
    return float(FIELD_STRENGTH * _owner(own.get("weather") or {}, weather_kinds)
                 + FIELD_STRENGTH * _owner(own.get("terrain") or {}, terrain_kinds))


def phi(battle, own: Dict[str, Dict[str, int]]) -> float:
    """Φ of a live poke-env battle (its ``weather`` / ``fields``) for the side that owns ``own``."""
    from v_dance.selfplay.mega_hold import current_terrain_kinds, current_weather_kinds
    return phi_from_kinds(current_weather_kinds(battle), current_terrain_kinds(battle), own)


# ── end of game: the margin's input + the logged-only guards ─────────────────
def fainted_count(team) -> int:
    mons = team.values() if isinstance(team, dict) else (team or ())
    return int(sum(1 for m in mons if getattr(m, "fainted", False)))


def _move_uses(lines: Iterable[str], side: Optional[str], moves: frozenset) -> int:
    if not side:
        return 0
    n = 0
    for ln in lines:
        p = ln.split("|")
        if len(p) > 3 and p[1] == "move" and p[2].startswith(side) and _id(p[3]) in moves:
            n += 1
    return n


def game_guard(battle) -> dict:
    """The logged-only per-game readouts from OUR side: our faints, theirs, Protect-family and Trick Room uses."""
    from v_dance.selfplay.replay_html import battle_replay_lines
    try:
        lines = battle_replay_lines(battle)
    except Exception:
        lines = []
    side = getattr(battle, "player_role", None)
    return {"own_fainted": fainted_count(getattr(battle, "team", None)),
            "opp_fainted": fainted_count(getattr(battle, "opponent_team", None)),
            "protect": _move_uses(lines, side, PROTECT_MOVES),
            "trick_room": _move_uses(lines, side, frozenset({"trickroom"}))}


# ── the per-generation readout ───────────────────────────────────────────────
def summarize_trajectories(trajs, field_kappa: float = 1.0) -> dict:
    """Totals for the generation's reward-v2 line (and the preflight's wiring check)."""
    v2 = [t for t in (trajs or ()) if is_v2(getattr(t, "meta", None))]
    s = {"trajectories": len(trajs or ()), "v2": len(v2), "kappa": round(float(field_kappa), 4),
         "steps": 0, "steps_phi": 0, "ours": 0, "theirs": 0, "gains": 0, "losses_field": 0,
         "games_won": 0, "games_lost": 0, "lost_margin_recorded": 0, "margin_sum": 0.0, "opp_ko_on_loss": 0,
         "own_faints_in_wins": 0, "guard_games": 0, "guard_wins": 0, "protect": 0, "trick_room": 0}
    for t in v2:
        m = t.meta
        ph = [x.phi for x in t.transitions]
        s["steps"] += len(ph)
        known = [float(p) for p in ph if p is not None]
        s["steps_phi"] += len(known)
        s["ours"] += sum(1 for p in known if p > 1e-9)
        s["theirs"] += sum(1 for p in known if p < -1e-9)
        d = np.diff(np.asarray(known, dtype=np.float64)) if len(known) > 1 else np.zeros(0)
        s["gains"] += int(np.sum(d > 1e-9))
        s["losses_field"] += int(np.sum(d < -1e-9))
        if m.won is True:
            s["games_won"] += 1
        elif m.won is False:
            s["games_lost"] += 1
            if m.opp_fainted is not None:
                s["lost_margin_recorded"] += 1
                s["opp_ko_on_loss"] += int(m.opp_fainted)
                r = terminal_reward(m)
                s["margin_sum"] += (r + 1.0) if r is not None else 0.0
        g = getattr(m, "guard", None)
        if g:
            s["guard_games"] += 1
            s["protect"] += int(g.get("protect", 0))
            s["trick_room"] += int(g.get("trick_room", 0))
            if m.won is True:
                s["guard_wins"] += 1
                s["own_faints_in_wins"] += int(g.get("own_fainted", 0))
    s["margin_sum"] = round(s["margin_sum"], 4)
    return s


def format_stats(s: dict) -> str:
    def pct(a, b):
        return f"{100.0 * a / b:.0f}%" if b else "n/a"

    def per(a, b, nd=1):
        return f"{a / b:.{nd}f}" if b else "n/a"
    n_v2 = s.get("v2", 0)
    lost = s.get("lost_margin_recorded", 0)
    return (f"reward v2: κ {s.get('kappa', 1.0):.2f} · {n_v2}/{s.get('trajectories', 0)} trajectories · losses "
            f"{s.get('games_lost', 0)}: margin +{per(s.get('margin_sum', 0.0), lost, 3)} avg (their KOs "
            f"{per(s.get('opp_ko_on_loss', 0), lost)}/4) · field: ours {pct(s.get('ours', 0), s.get('steps_phi', 0))} / "
            f"theirs {pct(s.get('theirs', 0), s.get('steps_phi', 0))} of decisions, gained {per(s.get('gains', 0), n_v2)}"
            f" / lost {per(s.get('losses_field', 0), n_v2)} per game · guards: our faints in wins "
            f"{per(s.get('own_faints_in_wins', 0), s.get('guard_wins', 0))}, Protect "
            f"{per(s.get('protect', 0), s.get('guard_games', 0))}/game, Trick Room "
            f"{per(s.get('trick_room', 0), s.get('guard_games', 0), 2)}/game")


__all__ = ["DOMAINS", "REWARD_V2", "sheet_kinds", "owners", "flip", "describe", "phi_from_kinds", "phi",
           "fainted_count", "game_guard", "summarize_trajectories", "format_stats"]
