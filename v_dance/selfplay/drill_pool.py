"""The DRILL team pool — opponents whose weather / terrain FIGHTS ours (2026-10-02, USER: "have it fight for terrain
control and also for weather control — kill two birds with one stone").

Evidence (``-m v_dance.eval.field_control_report``, Baltimore_Sand_Psy, 2,168 ladder games): a different terrain in
33 % of games (Grassy 678 → we win 38.7 %, reclaim 2.9 %), a different weather in 31 % (rain 356 → 41.3 %). The
self-play pool draws teams uniformly, so a fight shows up only as often as the pool happens to hold such teams. This
module turns the regular pool into a drill pool:

  · CONFLICT teams = a pool team that sets a weather or terrain KIND our team does not (read off the pastes'
    abilities + weather items with ``field_control_report.team_setters`` — the same rule the report uses);
  · each conflict KIND gets a share ∝ how often we meet it on the ladder (``LADDER_MIX``, the 10-02 report), split
    evenly over the teams that set it (a rain + Grassy team collects both);
  · ``neutral_frac`` of the draws stay ordinary teams, so the bot keeps meeting the rest of the meta;
  · no ONE team exceeds ``max_team_share`` (water-filled: the excess is spread over the rest).

The league consumes ``weights`` (``{pool entry: weight}`` → ``gauntlet.own_team_matchups``), via ``pool_weights``:
the league DE-DUPLICATES the pool by team file name, so duplicates in a list are NOT weights there. ``pool`` (a list
with duplicates) is kept for a quick look and the tests. Kinds the pool cannot field (no team sets them — Electric
terrain on 10-02) are reported, never silently dropped.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]

# (domain, kind) -> Baltimore ladder games against it, field_control_report 2026-10-02 (2,168 games since 09-30)
LADDER_MIX: Dict[Tuple[str, str], int] = {
    ("terrain", "grassy"): 678, ("weather", "rain"): 356, ("weather", "sun"): 195,
    ("weather", "snow"): 138, ("terrain", "electric"): 43, ("terrain", "misty"): 1,
}
DEFAULT_SLOTS = 200            # copies the conflict share is spread over (resolution of ``pool``)


@dataclass
class DrillTeam:
    path: str
    kinds: Tuple[Tuple[str, str], ...]          # the (domain, kind) pairs it sets that we do NOT


@dataclass
class DrillPool:
    pool: List[str]                             # team paths, duplicates ≈ weights (a quick look; NOT the league input)
    ours: Dict[str, Dict[str, str]]             # our team's {domain: {kind: setter}}
    teams: List[DrillTeam] = field(default_factory=list)       # the TARGET teams (conflict / focus hits)
    shares: Dict[str, float] = field(default_factory=dict)      # "domain:kind" / "neutral" -> share of draws
    missing: List[str] = field(default_factory=list)            # ladder kinds no pool team sets
    weights: Dict[str, float] = field(default_factory=dict)     # normalised per-team draw weights (the league input)
    no_ladder_data: List[str] = field(default_factory=list)     # conflict kinds the ladder mix has no count for
    unreadable: List[str] = field(default_factory=list)         # pool entries whose paste could not be read

    def summary(self) -> str:
        n_teams = len(self.weights) or len(set(self.pool))
        parts = [f"{k} {100 * s:.0f}%" for k, s in sorted(self.shares.items(), key=lambda kv: -kv[1])]
        miss = f" · NO TEAM for: {', '.join(self.missing)}" if self.missing else ""
        nld = (f" · no ladder data (mean weight) for: {', '.join(self.no_ladder_data)}"
               if self.no_ladder_data else "")
        bad = f" · UNREADABLE: {', '.join(Path(p).name for p in self.unreadable)}" if self.unreadable else ""
        return (f"drill pool: {n_teams} teams ({len(self.teams)} target teams) — " + " · ".join(parts)
                + miss + nld + bad)


def _read_paste(path: str) -> str:
    """A pool entry's paste: a path (absolute / repo-relative), else a bare team NAME resolved the way the launch and
    the workers resolve it (review F2: a ``--train-teams <name>`` entry was silently dropped)."""
    p = Path(path)
    if not p.is_absolute():
        p = _REPO / p
    if not p.exists():
        try:
            import v_dance.play.run_local_battle as R
            p = Path(R.resolve_team_path(str(path)))
        except (SystemExit, Exception):
            raise OSError(f"team not found: {path}")
    if p.is_dir():
        return "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in sorted(p.rglob("*")) if f.is_file())
    return p.read_text(encoding="utf-8", errors="replace")


def classify(paths: Iterable[str], ours: Dict[str, Dict[str, str]], reader=_read_paste,
             unreadable: Optional[List[str]] = None) -> List[DrillTeam]:
    """Every pool team with the (domain, kind) pairs it sets that OUR team does not. Entries that cannot be read go
    to ``unreadable`` (reported by the launch, never silently dropped)."""
    from v_dance.eval.field_control_report import team_setters
    out = []
    for path in paths:
        try:
            theirs = team_setters(reader(path))
        except OSError:
            if unreadable is not None:
                unreadable.append(path)
            continue
        kinds = tuple((d, k) for d in ("weather", "terrain") for k in sorted(theirs[d]) if k not in ours[d])
        out.append(DrillTeam(path, kinds))
    return out


def unique_by_name(team_pool: Iterable[str], exclude: str = "") -> List[str]:
    """The pool de-duplicated by team FILE NAME, first entry kept — exactly what the league keeps
    (``gauntlet.own_team_matchups``). Review: a team in two reg folders (M-B + M-C The_Big_6_v2) was weighed TWICE."""
    seen, out = {Path(exclude).name.lower()} if exclude else set(), []
    for p in team_pool:
        k = Path(str(p)).name.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def cap_weights(w: Dict[str, float], cap: float) -> Dict[str, float]:
    """Normalise, then water-fill: no entry above ``cap``, the excess spread over the rest in proportion."""
    tot = sum(v for v in w.values() if v > 0)
    if tot <= 0:
        return {}
    base = {k: v / tot for k, v in w.items() if v > 0}
    if cap * len(base) < 1.0:                   # infeasible: every team at the cap still sums below 1 → plain normalise
        return base
    capped: set = set()
    while True:                                 # exact: pin the over-cap set, share the rest by the ORIGINAL weights
        free = {k: v for k, v in base.items() if k not in capped}
        room = 1.0 - cap * len(capped)
        ftot = sum(free.values())
        out = {k: cap for k in capped}
        out.update({k: room * v / ftot for k, v in free.items()} if ftot > 0 else {})
        newly = {k for k in free if out[k] > cap}
        if not newly:
            return out
        capped |= newly


def build_drill_pool(team_pool: List[str], own_team: str, *, mix: Optional[Dict[Tuple[str, str], int]] = None,
                     neutral_frac: float = 0.2, slots: int = DEFAULT_SLOTS, max_team_share: float = 0.06,
                     reader=_read_paste) -> DrillPool:
    """The weighted drill pool for ``own_team`` (a path or pool entry) out of ``team_pool``. ``max_team_share``
    caps any ONE team's share of the draws — the M-C pool's only snow team would otherwise be 9 % of every game
    (memorising one team is not learning snow)."""
    from v_dance.eval.field_control_report import team_setters
    mix = dict(LADDER_MIX if mix is None else mix)
    ours = team_setters(reader(own_team))
    candidates = unique_by_name(team_pool, exclude=own_team)      # never our own team; one entry per team name
    unreadable: List[str] = []
    teams = classify(candidates, ours, reader=reader, unreadable=unreadable)
    conflict = [t for t in teams if t.kinds]
    neutral = [t.path for t in teams if not t.kinds]
    by_kind: Dict[Tuple[str, str], List[DrillTeam]] = {}
    for t in conflict:
        for k in t.kinds:
            by_kind.setdefault(k, []).append(t)
    missing = [f"{d}:{k}" for (d, k) in mix if (d, k) not in by_kind and k not in ours.get(d, {})]
    # review F2 (mechanics): a conflict kind the ladder mix has NO count for (LADDER_MIX is Baltimore's record — it
    # never lists Baltimore's OWN sand / Psychic Terrain) gets the MEAN kind weight, not the minimum (misty = 1 vs
    # grassy 678 made those teams effectively undrawable for any other --own-team); reported in the summary
    seen_w = [float(v) for v in mix.values() if v > 0]
    mean_w = (sum(seen_w) / len(seen_w)) if seen_w else 1.0
    no_data = sorted(f"{d}:{k}" for (d, k) in by_kind if (d, k) not in mix)
    w_kind = {k: float(mix.get(k, mean_w)) for k in by_kind}
    total = sum(w_kind.values()) or 1.0
    conflict_frac = ((1.0 - neutral_frac) if neutral else 1.0) if conflict else 0.0
    raw: Dict[str, float] = {}
    for k, ts in by_kind.items():
        per_team = conflict_frac * w_kind[k] / total / len(ts)
        for t in ts:
            raw[t.path] = raw.get(t.path, 0.0) + per_team
    for p in neutral:
        raw[p] = raw.get(p, 0.0) + (1.0 - conflict_frac) / len(neutral)
    weights = cap_weights(raw, max_team_share)
    pool: List[str] = []
    for p, w in weights.items():
        pool.extend([p] * max(1, int(round(w * slots))))
    shares: Dict[str, float] = {}
    for k, ts in by_kind.items():
        # a team setting two kinds counts toward both (its draws ARE fights in both)
        shares[f"{k[0]}:{k[1]}"] = round(sum(weights.get(t.path, 0.0) for t in ts), 6)
    if neutral:
        shares["neutral"] = round(sum(weights.get(p, 0.0) for p in neutral), 6)
    return DrillPool(pool=pool, ours=ours, teams=conflict, shares=shares, missing=missing, weights=weights,
                     no_ladder_data=no_data, unreadable=unreadable)


def pool_weights(dp: DrillPool, team_pool: Sequence[str]) -> Dict[str, float]:
    """``{exact team_pool entry: weight}`` for ``gauntlet.own_team_matchups`` — keyed by the FIRST pool entry with each
    team file name (the league keeps that one when it de-duplicates; e.g. The_Big_6_v2 is in both M-B and M-C), the
    weights of same-named entries summed. A pool without ``weights`` (a list-only drill) falls back to its duplicates."""
    from v_dance.eval.gauntlet import team_key
    first: Dict[str, str] = {}
    for t in team_pool:
        first.setdefault(team_key(t), t)
    src = dp.weights or {p: float(c) for p, c in Counter(dp.pool).items()}
    out: Dict[str, float] = {}
    for p, w in src.items():
        entry = first.get(team_key(p))
        if entry is not None and w > 0 and math.isfinite(w):
            out[entry] = out.get(entry, 0.0) + float(w)
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()} if tot > 0 else {}
