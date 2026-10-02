"""DRILLS — reusable, targeted self-play (2026-10-02, USER: "make sure this drill league is reusable for situations we
would want otherwise as well, such as discovering some weak point like the team picker undervaluing Salamence").

A drill bundles three pluggable parts, all optional except the pool:

  · POOL        — which opponent teams to draw, and how often (the league's pool shape: team paths, duplicates =
                  weights) — e.g. the teams whose weather / terrain fights ours, or the teams that carry Rillaboom;
  · PRESSURE    — a logit bias for the OPPONENT model players that makes them play the situation hard (e.g. switch
                  their weather / terrain setter back in when the field is not theirs) — without it, opponents from
                  the bot's own family never contest the skill either;
  · SCOREBOARD  — per-game metrics parsed from the battle log, aggregated per generation next to the win rate, so a
                  run shows whether the SKILL moved, not only the result.

Drills are looked up by NAME (``get_drill``) so the spawn-based workers receive a string, never a closure. A drill
NAME may carry arguments after a colon: ``field``, ``focus:opp=rillaboom,mon=salamence``.

Built-in drills:
  · ``field`` — weather + terrain control (the 2026-10-02 report: vs Grassy 38.7 %, reclaim 2.9 %; vs rain 41.3 %).
  · ``focus`` — a species-centred drill: over-sample opponents carrying ``opp=`` species, score how often we bring /
    lead ``mon=`` and our win rate with and without it (the Salamence-vs-Rillaboom question, 2026-10-01).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from v_dance.selfplay.drill_pool import DrillPool, DrillTeam, _read_paste, build_drill_pool, cap_weights, pool_weights


def _id(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


@dataclass
class Drill:
    name: str
    describe: str
    build_pool: Callable[[List[str], str], DrillPool]
    score_game: Callable[[str, set, dict], Optional[dict]]        # (log text, our ids, ctx) -> per-game metrics
    aggregate: Callable[[List[dict]], Dict[str, object]]           # per-game metrics -> the generation's scoreboard
    pressure: Optional[str] = None                                 # opponent bias name (field_fight.BIASES), or None
    ctx: Dict[str, object] = field(default_factory=dict)          # what score_game needs (e.g. our setters)
    refine: Optional[Callable[[dict, dict], dict]] = None          # (per-game metrics, collection row) -> metrics


# ── shared log helpers ───────────────────────────────────────────────────────
def _our_side(text: str, our_ids: set) -> Optional[str]:
    for ln in text.splitlines():
        m = re.match(r"\|player\|(p[12])\|([^|]*)\|", ln)
        if m and _id(m.group(2)) in our_ids:
            return m.group(1)
    return None


def _won(text: str, our_ids: set) -> Optional[bool]:
    m = re.search(r"^\|win\|([^<|\n]+)", text, re.M)
    if not m:
        return None
    return _id(m.group(1)) in our_ids


def _brought_and_led(text: str, side: str) -> Tuple[List[str], List[str]]:
    brought, led, started = [], [], False
    for ln in text.splitlines():
        if ln.startswith("|turn|1"):
            started = True
        p = ln.split("|")
        if len(p) > 3 and p[1] in ("switch", "drag") and p[2].startswith(side):
            sp = _id(p[3].split(",")[0].split("-")[0])
            if sp not in brought:
                brought.append(sp)
            if not started and sp not in led:
                led.append(sp)
    return brought, led


def _move_uses(text: str, side: str, move_id: str) -> int:
    n = 0
    for ln in text.splitlines():
        p = ln.split("|")
        if len(p) > 3 and p[1] == "move" and p[2].startswith(side) and _id(p[3]) == move_id:
            n += 1
    return n


def _rate(num, den) -> Optional[float]:
    return round(num / den, 3) if den else None


# ── the FIELD drill: weather + terrain control ───────────────────────────────
def _field_score(text: str, our_ids: set, ctx: dict) -> Optional[dict]:
    from v_dance.eval.field_control_report import parse_replay
    g = parse_replay(text, our_ids, ctx["ours"])
    if g is None:
        return None
    side = _our_side(text, our_ids)
    g["trick"] = _move_uses(text, side, "trick") + _move_uses(text, side, "switcheroo") if side else 0
    return g


def _field_aggregate(games: List[dict]) -> Dict[str, object]:
    out: Dict[str, object] = {"games": len(games),
                              "win": _rate(sum(g["won"] for g in games), len(games)),
                              "trick_per_game": _rate(sum(g.get("trick", 0) for g in games), len(games))}
    for d in ("weather", "terrain"):
        fights = [g for g in games if g["contested"][d]]
        turns = sum(g["turns"] for g in fights)
        ev = [e for g in fights for e in g["events"] if e["domain"] == d and e["setter_alive"] and e["setter_brought"]]
        out[f"{d}_fights"] = len(fights)
        out[f"{d}_fight_win"] = _rate(sum(g["won"] for g in fights), len(fights))
        out[f"{d}_share"] = _rate(sum(g["held"][d] for g in fights), turns)
        out[f"{d}_reclaim"] = _rate(sum(1 for e in ev if e["reclaimed"]), len(ev))
        out[f"{d}_setter_used"] = _rate(sum(1 for g in fights if g["setter_entered"][d]), len(fights))
    return out


def _field_drill(args: Dict[str, str]) -> Drill:
    holder: Dict[str, object] = {}

    def pool(team_pool: List[str], own_team: str) -> DrillPool:
        dp = build_drill_pool(team_pool, own_team, neutral_frac=float(args.get("neutral", 0.2)))
        holder["ours"] = dp.ours
        return dp

    drill = Drill(name="field",
                  describe="weather + terrain control: opponents whose weather / terrain fights ours, weighted to the "
                           "ladder mix; opponents switch their setter back in when the field is not theirs",
                  build_pool=pool, score_game=_field_score, aggregate=_field_aggregate,
                  pressure=None if args.get("pressure") == "off" else "field")
    drill.ctx = holder                                              # filled by build_pool (our setters)
    return drill


# ── the FOCUS drill: one opponent species / one of ours (e.g. Salamence vs Rillaboom) ────────────────────────
def _focus_drill(args: Dict[str, str]) -> Drill:
    opp = {_id(s) for s in str(args.get("opp", "")).split("+") if s}
    mon = _id(args.get("mon", ""))
    share = float(args.get("share", 0.7))
    if not opp and not mon:
        raise ValueError("focus drill needs opp=<species>[+<species>…] and/or mon=<our species>")

    def pool(team_pool: List[str], own_team: str) -> DrillPool:
        from v_dance.eval.field_control_report import team_setters
        hits, rest = [], []
        own_key = Path(own_team).name
        from v_dance.selfplay.drill_pool import unique_by_name
        for p in unique_by_name(team_pool, exclude=own_team):     # review: one entry per team NAME, like the league
            from v_dance.eval.field_control_report import paste_species
            species = set(paste_species(_read_paste(p)))
            (hits if (not opp or species & opp) else rest).append(p)
        if not hits:
            raise ValueError(f"focus drill: no pool team carries {sorted(opp)}")
        h_share = share if rest else 1.0
        raw = {p: h_share / len(hits) for p in hits}
        raw.update({p: (1.0 - h_share) / len(rest) for p in rest})
        weights = cap_weights(raw, float(args.get("cap", 0.06)))
        pool_ = [p for p, w in weights.items() for _ in range(max(1, int(round(w * 200))))]
        return DrillPool(pool=pool_, ours=team_setters(_read_paste(own_team)),
                         teams=[DrillTeam(p, ()) for p in hits], weights=weights,
                         shares={f"opp:{'+'.join(sorted(opp)) or 'any'}": round(sum(weights.get(p, 0) for p in hits), 6),
                                 "other": round(sum(weights.get(p, 0) for p in rest), 6)})

    def score(text: str, our_ids: set, ctx: dict) -> Optional[dict]:
        side = _our_side(text, our_ids)
        won = _won(text, our_ids)
        if side is None or won is None:
            return None
        brought, led = _brought_and_led(text, side)
        other = "p2" if side == "p1" else "p1"
        opp_seen = set(_brought_and_led(text, other)[0])
        return {"won": won, "brought": mon in brought if mon else None, "led": mon in led if mon else None,
                "opp_hit": bool(opp & opp_seen) if opp else True}

    def aggregate(games: List[dict]) -> Dict[str, object]:
        hit = [g for g in games if g["opp_hit"]]
        out: Dict[str, object] = {"games": len(games), "win": _rate(sum(g["won"] for g in games), len(games)),
                                  "opp_seen_games": len(hit), "opp_seen_win": _rate(sum(g["won"] for g in hit), len(hit))}
        if mon:
            b = [g for g in hit if g["brought"]]
            nb = [g for g in hit if not g["brought"]]
            out.update({f"{mon}_brought": _rate(len(b), len(hit)),
                        f"{mon}_led": _rate(sum(1 for g in hit if g["led"]), len(hit)),
                        f"win_with_{mon}": _rate(sum(g["won"] for g in b), len(b)),
                        f"win_without_{mon}": _rate(sum(g["won"] for g in nb), len(nb))})
        return out

    def refine(g: dict, row: dict) -> dict:
        # review F6: 'brought' = the TEAM PREVIEW's pick (a back-line mon that never switched in is still brought);
        # the log can only say whether it ENTERED. Rows without the pick keep the log-based value.
        if mon and row.get("tp_brought") is not None:
            g["entered"] = g.get("brought")
            g["brought"] = mon in (row.get("tp_brought") or [])
            if row.get("tp_led") is not None:
                g["led"] = mon in (row.get("tp_led") or [])
        return g

    return Drill(name="focus", describe=f"focus: opponents carrying {sorted(opp) or 'any'}; our {mon or '—'} "
                                        f"brought / led / win-with",
                 build_pool=pool, score_game=score, aggregate=aggregate, pressure=None, refine=refine)


_REGISTRY: Dict[str, Callable[[Dict[str, str]], Drill]] = {"field": _field_drill, "focus": _focus_drill}


def parse_spec(spec: str) -> Tuple[str, Dict[str, str]]:
    """``focus:opp=rillaboom,mon=salamence`` -> ("focus", {"opp": "rillaboom", "mon": "salamence"})."""
    name, _, rest = (spec or "").partition(":")
    args = {}
    for part in filter(None, rest.split(",")):
        k, _, v = part.partition("=")
        args[k.strip()] = v.strip()
    return name.strip(), args


def get_drill(spec: str) -> Drill:
    name, args = parse_spec(spec)
    if name not in _REGISTRY:
        raise ValueError(f"unknown drill {name!r} — known: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name](args)


def known_drills() -> List[str]:
    return sorted(_REGISTRY)


def scoreboard(drill: Drill, texts: List[str], our_ids: set) -> Dict[str, object]:
    """The generation's scoreboard from its battle logs (unreadable / undecided games are skipped and counted)."""
    games, skipped = [], 0
    for t in texts:
        try:
            g = drill.score_game(t, our_ids, drill.ctx)
        except Exception:
            g = None
        if g is None:
            skipped += 1
        else:
            games.append(g)
    out = drill.aggregate(games) if games else {"games": 0}
    out["skipped"] = skipped
    return out


# ── the launch-time setup + the per-generation scoreboard (MAIN process only — a Drill does not pickle, and a drill
#    re-resolved inside a worker has an EMPTY ctx: every game would be a silently skipped KeyError) ──────────────
@dataclass
class DrillSetup:
    drill: Drill
    pool: DrillPool
    opp_weights: Dict[str, float]               # {exact team_pool entry: weight} → own_team_matchups
    target_keys: frozenset                      # team_key()s of the target teams (the scoreboard's "targets" split)
    pressure: Optional[str]                     # opponent bias NAME (field_fight.BIASES) or None
    bias: float


def setup_drill(spec: str, *, own_team_path: str, team_pool: List[str], bias: float = 2.5) -> DrillSetup:
    import math
    from v_dance.eval.gauntlet import team_key
    bias = float(bias)
    if not math.isfinite(bias) or bias < 0:
        raise ValueError(f"--drill-bias must be a finite number ≥ 0 (got {bias})")
    drill = get_drill(spec)
    dp = drill.build_pool(list(team_pool), own_team_path)       # fills drill.ctx (our setters) for the field drill
    w = pool_weights(dp, team_pool)
    if not w:
        raise ValueError(f"drill {spec!r}: no pool team got a weight — wrong --own-team or an empty pool?")
    return DrillSetup(drill=drill, pool=dp, opp_weights=w,
                      target_keys=frozenset(team_key(t.path) for t in dp.teams),
                      pressure=(drill.pressure if bias > 0 else None), bias=bias)


def gen_scoreboard(setup: DrillSetup, rows: List[dict], pressure_stats: Optional[dict] = None) -> Dict[str, object]:
    """One generation's DRILL scoreboard from its collection games (rows from ``mp_collect._drill_rows``: the battle
    log, our player name, the opponent kind / team, mirror + fallback flags). Mirror games (our team vs itself) and
    fallback games (a forfeit / crash) are counted and left out; everything else is scored by the drill."""
    from v_dance.eval.gauntlet import team_key
    drill = setup.drill
    scored, skipped, mirror, fallback = [], 0, 0, 0
    for r in rows or []:
        if r.get("fallback"):
            fallback += 1
            continue
        if r.get("mirror"):
            mirror += 1
            continue
        try:
            g = drill.score_game(r.get("text") or "", {_id(r.get("our"))}, drill.ctx)
            if g is not None and drill.refine is not None:
                g = drill.refine(g, r)
        except Exception:
            g = None
        if g is None:
            skipped += 1
            continue
        scored.append((r, g))

    def agg(pairs):
        gs = [g for _, g in pairs]
        return drill.aggregate(gs) if gs else {"games": 0}

    out: Dict[str, object] = {"drill": drill.name, **agg(scored), "skipped": skipped, "mirror": mirror,
                              "fallback": fallback}
    if setup.target_keys:
        out["targets"] = agg([(r, g) for r, g in scored if team_key(r.get("team_b") or "") in setup.target_keys])
    kinds = sorted({r.get("kind") or "?" for r, _ in scored})
    out["by_kind"] = {k: agg([(r, g) for r, g in scored if (r.get("kind") or "?") == k]) for k in kinds}
    out["pressure"] = {"games": sum(1 for r, _ in scored if r.get("pressure")), **dict(pressure_stats or {})}
    return out


def format_gen_line(sb: Dict[str, object]) -> str:
    """The generation report's DRILL line: the top-level numbers, then the target teams and the pressure counts."""
    def f(k, v):
        if v is None:
            return "—"
        if isinstance(v, float) and "per_game" not in k:
            return f"{100 * v:.0f}%"
        return f"{v:.2f}" if isinstance(v, float) else str(v)
    flat = [f"{k} {f(k, v)}" for k, v in sb.items() if not isinstance(v, dict) and k != "drill"]
    line = f"DRILL {sb.get('drill', '?')}: " + " · ".join(flat)
    t = sb.get("targets")
    if isinstance(t, dict) and t.get("games"):
        line += " | targets: " + " · ".join(f"{k} {f(k, v)}" for k, v in t.items() if not isinstance(v, dict))
    bk = sb.get("by_kind")
    if isinstance(bk, dict) and len(bk) > 1:
        def _kind(k, d):
            parts = [f"{d.get('games', 0)}g"]
            for key, lab in (("win", "win"), ("terrain_share", "T"), ("weather_share", "W")):
                if d.get(key) is not None:
                    parts.append(f"{lab} {100 * d[key]:.0f}%")
            return f"{k} " + " ".join(parts)
        line += " | by opponent: " + " · ".join(_kind(k, d) for k, d in bk.items() if isinstance(d, dict))
    p = sb.get("pressure")
    if isinstance(p, dict) and (p.get("games") or p.get("fired")):
        line += f" | pressure: {p.get('games', 0)} g, fired {p.get('fired', 0)}, taken {p.get('taken', 0)}"
    return line


def format_scoreboard(drill: Drill, sb: Dict[str, object]) -> str:
    def f(k, v):
        if v is None:
            return "—"
        if isinstance(v, float) and "per_game" not in k:
            return f"{100 * v:.0f}%"                          # rates
        return f"{v:.2f}" if isinstance(v, float) else str(v)
    return f"DRILL {drill.name}: " + " · ".join(f"{k} {f(k, v)}" for k, v in sb.items())
