"""Opponent league for self-play (task 3c.2) — anti-collapse opponent sampling.

Instead of always playing the current policy against a fresh copy of itself (which can
collapse onto a narrow self-exploiting strategy), each game draws an opponent from a
LEAGUE (docs/ppo_reward_design.md sec 6):

  * ~50% the LATEST checkpoint (ordinary self-play),
  * ~30% a past GAUNTLET-ACCEPTED frozen snapshot, **PFSP-weighted** toward the ones the
    latest currently LOSES to (so it trains against its hardest counters), and
  * ~20% scripted gauntlet anchors (random / max_damage / heuristic), the scripted
    fraction DECAYING toward ~10% as the snapshot pool grows.

The pool, the mixture, the PFSP weighting, the anchor decay, and the persistence are all
pure (no poke-env), so the sampling logic is fully unit-tested and inspectable offline
(``python local_battle/self_play/league.py --demo``). ``make_league_opponent`` maps a
sampled spec to a live player via injected constructors — the runner (3c.3) passes the
real ``gauntlet._make_opponent`` + a model-player factory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_SCRIPTED = ("random", "max_damage", "heuristic")


@dataclass
class LeagueSnapshot:
    """A frozen past checkpoint in the pool."""
    snapshot_id: str
    path: str
    generation: int
    elo: float = 1000.0
    wins_vs_latest: int = 0     # the LATEST policy's wins against this snapshot (PFSP signal)
    games_vs_latest: int = 0
    is_champion: bool = False   # was this gen an accepted CHAMPION? (never evicted — anti-cycle memory)

    def latest_winrate(self) -> float:
        """The latest policy's win-rate vs this snapshot (0.5 prior with no games)."""
        return self.wins_vs_latest / self.games_vs_latest if self.games_vs_latest else 0.5

    def to_obj(self) -> dict:
        return {"snapshot_id": self.snapshot_id, "path": self.path,
                "generation": int(self.generation), "elo": float(self.elo),
                "wins_vs_latest": int(self.wins_vs_latest),
                "games_vs_latest": int(self.games_vs_latest),
                "is_champion": bool(self.is_champion)}

    @classmethod
    def from_obj(cls, d: dict) -> "LeagueSnapshot":
        return cls(snapshot_id=d["snapshot_id"], path=d["path"],
                   generation=int(d.get("generation", 0)), elo=float(d.get("elo", 1000.0)),
                   wins_vs_latest=int(d.get("wins_vs_latest", 0)),
                   games_vs_latest=int(d.get("games_vs_latest", 0)),
                   is_champion=bool(d.get("is_champion", False)))


@dataclass
class LeagueConfig:
    latest_frac: float = 0.5            # ordinary self-play share
    past_frac: float = 0.3             # past-snapshot share (PFSP)
    scripted_frac_max: float = 0.25    # scripted-anchor share when the pool is small
    scripted_frac_min: float = 0.10    # ... decaying to this as the pool grows
    scripted_decay_per_snapshot: float = 0.02
    pfsp_power: float = 2.0            # (1 - latest_winrate)^power: favour hard counters
    pfsp_floor: float = 0.05          # min weight so even an always-beaten snapshot recurs


@dataclass
class OpponentLeague:
    latest_path: str
    scripted: Tuple[str, ...] = DEFAULT_SCRIPTED
    snapshots: List[LeagueSnapshot] = field(default_factory=list)
    cfg: LeagueConfig = field(default_factory=LeagueConfig)

    # ── mixture (anchor decay tied to pool size) ──────────────────────────────
    def scripted_fraction(self) -> float:
        if not self.scripted:
            return 0.0
        n = len(self.snapshots)
        if n == 0:
            return self.cfg.scripted_frac_max
        return max(self.cfg.scripted_frac_min,
                   self.cfg.scripted_frac_max - self.cfg.scripted_decay_per_snapshot * n)

    def mixture(self) -> Dict[str, float]:
        """``{"latest": x, "past": y, "scripted": z}`` summing to 1. With no snapshots the
        past share is 0 (redistributed to latest); latest/past keep their 0.5:0.3 ratio
        within the non-scripted remainder."""
        s = self.scripted_fraction()
        if not self.snapshots:
            return {"latest": 1.0 - s, "past": 0.0, "scripted": s}
        lp = self.cfg.latest_frac + self.cfg.past_frac
        rem = 1.0 - s
        return {"latest": rem * self.cfg.latest_frac / lp,
                "past": rem * self.cfg.past_frac / lp, "scripted": s}

    # ── PFSP weighting (favour the snapshots the latest loses to) ─────────────
    def pfsp_weights(self) -> List[float]:
        return [max(self.cfg.pfsp_floor, (1.0 - s.latest_winrate()) ** self.cfg.pfsp_power)
                for s in self.snapshots]

    # ── sampling one opponent ─────────────────────────────────────────────────
    def sample(self, rng: np.random.Generator) -> Tuple[str, object]:
        """Draw one opponent. Returns ``("latest", latest_path)`` /
        ``("snapshot", LeagueSnapshot)`` / ``("scripted", kind)``."""
        mix = self.mixture()
        u = float(rng.random())
        if u < mix["latest"]:
            return ("latest", self.latest_path)
        if u < mix["latest"] + mix["past"] and self.snapshots:
            w = np.asarray(self.pfsp_weights(), dtype=np.float64)
            w = w / w.sum()
            i = int(rng.choice(len(self.snapshots), p=w))
            return ("snapshot", self.snapshots[i])
        if self.scripted:
            return ("scripted", self.scripted[int(rng.integers(len(self.scripted)))])
        return ("latest", self.latest_path)   # degenerate: no scripted, no snapshots

    # ── mutation (admission / promotion / result recording) ───────────────────
    def admit(self, snapshot_id: str, path: str, generation: int, elo: float = 1000.0,
              is_champion: bool = False) -> LeagueSnapshot:
        """Add a frozen checkpoint to the pool. Sec 16: admission is DECOUPLED from champion
        promotion — every COMPETENT gen is admitted (so PFSP diversity grows even while the
        champion holds), with ``is_champion`` marking the accepted champions (never evicted)."""
        snap = LeagueSnapshot(snapshot_id, path, generation, elo, is_champion=is_champion)
        self.snapshots.append(snap)
        return snap

    def prune(self, max_snapshots: int, keep_recent: int = 6) -> List[LeagueSnapshot]:
        """Bound the pool to ``max_snapshots``, DIVERSITY-aware (sec 16). ALWAYS keep champion
        snapshots (the anti-cycle memory) + the most-recent ``keep_recent``. From the remaining
        OLD non-champion snapshots, keep a generation-STRIDED spread so early-gen archetypes
        aren't all dropped (a naive 'drop oldest' would lose the most-distinct counters). Returns
        the EVICTED snapshots so the caller can delete their checkpoint files. Champions + recent
        are kept even if that exceeds the cap (a soft cap — never evict a protected snapshot).

        MUST be called only BETWEEN generations (never mid-collection): ``record_result`` /
        in-flight chunks look snapshots up by id, so removing one mid-collection would silently
        drop PFSP data."""
        n = len(self.snapshots)
        if max_snapshots <= 0 or n <= max_snapshots:
            return []
        keep = {i for i, s in enumerate(self.snapshots) if s.is_champion}     # champions
        keep |= set(range(max(0, n - keep_recent), n))                        # most-recent
        remaining = [i for i in range(n) if i not in keep]
        budget = max_snapshots - len(keep)
        if budget > 0 and remaining:                                          # strided spread
            step = len(remaining) / budget
            keep |= {remaining[min(len(remaining) - 1, int(k * step))] for k in range(budget)}
        evicted = [s for i, s in enumerate(self.snapshots) if i not in keep]
        self.snapshots = [s for i, s in enumerate(self.snapshots) if i in keep]
        return evicted

    def promote_latest(self, new_path: str, *, demote_old_as: Optional[str] = None,
                       generation: int = 0, elo: float = 1000.0) -> None:
        """Point the league at a new latest checkpoint. Optionally admit the OLD latest as
        a frozen snapshot (``demote_old_as`` = its snapshot_id); the new latest's PFSP
        records start fresh (it has not played the pool yet)."""
        if demote_old_as is not None:
            self.admit(demote_old_as, self.latest_path, generation, elo)
        self.latest_path = new_path
        for s in self.snapshots:           # new latest hasn't faced the pool yet
            s.wins_vs_latest = s.games_vs_latest = 0

    def reset_pfsp(self) -> None:
        """Zero every snapshot's latest-vs-snapshot tally.  MUST be called whenever the
        LATEST policy changes (promote / revert) — otherwise wins_vs_latest accumulates
        outcomes from MANY different past 'latest' policies and pfsp_weights()
        ((1-latest_winrate)^power) targets the hard counters of an ANCESTOR, not the
        current latest, silently breaking the anti-collapse 'train against what you lose
        to' guarantee.  (promote_latest already resets; this is for the live loop, which
        sets latest_path directly.)"""
        for s in self.snapshots:
            s.wins_vs_latest = s.games_vs_latest = 0

    def record_result(self, snapshot_id: str, latest_won: bool) -> None:
        """Record one latest-vs-snapshot outcome for PFSP weighting.

        ⚠ #24 (KNOWN GAP, fix at RL launch): the in-memory 'latest' is PPO-updated EVERY generation —
        including HOLD gens, where ``reset_pfsp`` is NOT called — so over a long frozen-champion hold these
        tallies blend outcomes from a SEQUENCE of progressively-different policies, and ``pfsp_weights()``
        then targets an ANCESTOR mixture's hard counters, not the current latest (a diluted/lagged signal).
        The fix is an EMA decay here (``games = decay*games + 1``; ``wins = decay*wins + won``, decay≈0.95)
        applied ALSO in the MP fold (mp_collect's PFSP merge). Deferred — it's a measured RL-tuning change
        that needs the league tests updated and is dormant until self-play runs."""
        for s in self.snapshots:
            if s.snapshot_id == snapshot_id:
                s.games_vs_latest += 1
                s.wins_vs_latest += 1 if latest_won else 0
                return

    # ── persistence (3c.4 resume snapshot) ────────────────────────────────────
    def to_obj(self) -> dict:
        c = self.cfg
        return {"latest_path": self.latest_path, "scripted": list(self.scripted),
                "snapshots": [s.to_obj() for s in self.snapshots],
                "cfg": {"latest_frac": c.latest_frac, "past_frac": c.past_frac,
                        "scripted_frac_max": c.scripted_frac_max,
                        "scripted_frac_min": c.scripted_frac_min,
                        "scripted_decay_per_snapshot": c.scripted_decay_per_snapshot,
                        "pfsp_power": c.pfsp_power, "pfsp_floor": c.pfsp_floor}}

    @classmethod
    def from_obj(cls, d: dict) -> "OpponentLeague":
        return cls(latest_path=d["latest_path"],
                   scripted=tuple(d.get("scripted", DEFAULT_SCRIPTED)),
                   snapshots=[LeagueSnapshot.from_obj(s) for s in d.get("snapshots", [])],
                   cfg=LeagueConfig(**d.get("cfg", {})))


def make_league_opponent(spec: Tuple[str, object], *, username: str, team,
                         make_model: Callable, make_scripted: Callable):
    """Map a sampled ``spec`` to a live opponent player via injected constructors
    (the runner, 3c.3, passes a model-player factory + ``gauntlet._make_opponent``):
      ("latest", path) / ("snapshot", snap) -> make_model(username, team, model_path)
      ("scripted", kind)                     -> make_scripted(kind, username, team)
    """
    kind, val = spec
    if kind == "latest":
        return make_model(username, team, model_path=val)
    if kind == "snapshot":
        return make_model(username, team, model_path=val.path)
    if kind == "scripted":
        return make_scripted(val, username, team)
    raise ValueError(f"unknown opponent spec kind: {kind!r}")


# ── manual demo (no server): print the sampling distribution ──────────────────
def _demo(n: int = 5000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    league = OpponentLeague(latest_path="checkpoints_attn/battle_selfplay_gen141.pt")
    # three past snapshots: latest beats gen1, even vs gen2, LOSES to gen3
    for sid, gen, wins, games in [("gen1", 1, 80, 100), ("gen2", 2, 50, 100), ("gen3", 3, 25, 100)]:
        s = league.admit(sid, f"archive/{sid}.pt", gen, elo=1000 + gen * 30)
        s.wins_vs_latest, s.games_vs_latest = wins, games

    print("== Opponent-league sampling demo (no server) ===============")
    mix = league.mixture()
    print(f"  pool: {len(league.snapshots)} snapshots + {len(league.scripted)} scripted anchors")
    print(f"  target mixture : latest {mix['latest']*100:.0f}%  past {mix['past']*100:.0f}%  "
          f"scripted {mix['scripted']*100:.0f}%")
    print(f"  PFSP weights   : " + "  ".join(
        f"{s.snapshot_id}(lat wr {s.latest_winrate()*100:.0f}%)={w:.3f}"
        for s, w in zip(league.snapshots, [x / sum(league.pfsp_weights()) for x in league.pfsp_weights()])))

    buckets: Dict[str, int] = {}
    snaps: Dict[str, int] = {}
    scr: Dict[str, int] = {}
    for _ in range(n):
        kind, val = league.sample(rng)
        buckets[kind] = buckets.get(kind, 0) + 1
        if kind == "snapshot":
            snaps[val.snapshot_id] = snaps.get(val.snapshot_id, 0) + 1
        elif kind == "scripted":
            scr[val] = scr.get(val, 0) + 1
    print(f"\n  empirical over {n} draws:")
    for k in ("latest", "snapshot", "scripted"):
        print(f"    {k:9s}: {buckets.get(k,0)/n*100:5.1f}%")
    print("  among PAST snapshots (PFSP should favour gen3, the hard counter):")
    tot = sum(snaps.values()) or 1
    for sid in ("gen1", "gen2", "gen3"):
        print(f"    {sid}: {snaps.get(sid,0)/tot*100:5.1f}%")

    print("\n  scripted-anchor decay as the pool grows (tie to pool size):")
    for size in (0, 1, 3, 8, 15):
        lg = OpponentLeague(latest_path="x")
        for i in range(size):
            lg.admit(f"g{i}", f"{i}.pt", i)
        m = lg.mixture()
        print(f"    pool={size:2d}: latest {m['latest']*100:4.0f}%  past {m['past']*100:4.0f}%  "
              f"scripted {m['scripted']*100:4.0f}%")
    print("============================================================")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Opponent league (3c.2)")
    ap.add_argument("--demo", action="store_true", help="print the sampling distribution (no server)")
    ap.add_argument("-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.demo:
        _demo(args.n, args.seed)
    else:
        ap.print_help()
