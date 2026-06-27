"""Self-play promotion gate + generation bookkeeping (pure — no torch / poke-env).

Split out of ``generation.py`` (which had grown to the whole live loop) so the GATE — the
statistical decision logic, the configs, and the persistable ``GenerationHistory`` it reads/
writes — lives in one focused, fully-offline-testable place. ``generation.py`` re-exports every
public name here, so existing ``from ...generation import promotion_gate`` imports are unchanged.

Two gates live here:
  * ``promotion_gate``     — the legacy scripted-ladder + ``prev_best`` head-to-head gate.
  * ``promotion_gate_v2``  — the FROZEN-CHAMPION ladder (sec 16, calibrated on ``gate_sim``):
    keep the champion static until the candidate clears a HIGH bar vs it (70% over >=200 mirror
    games), OR the head-to-head climb PLATEAUS at a not-losing level (the backstop); a scripted
    COLLAPSE revert is the safety net, taking precedence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SCRIPTED_OPPONENTS = ("random", "max_damage", "heuristic")


# ── statistics ────────────────────────────────────────────────────────────────
def _two_prop_se(p1: float, n1: int, p2: float, n2: int) -> float:
    return math.sqrt(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1))


def wilson_lower_bound(wins: int, games: int, z: float = 1.645) -> float:
    """Wilson score interval LOWER bound for a binomial proportion. Used for
    ``scripted_high_water`` so the competence floor tracks DEMONSTRATED strength (a conservative
    estimate that already accounts for the sample size), not a lucky point-estimate peak — the
    red-team's fix for the upward-biased ``max()`` high-water that manufactured spurious reverts.
    Returns 0.0 for no games."""
    if games <= 0:
        return 0.0
    p = wins / games
    z2 = z * z
    denom = 1.0 + z2 / games
    centre = p + z2 / (2.0 * games)
    margin = z * math.sqrt(p * (1.0 - p) / games + z2 / (4.0 * games * games))
    return max(0.0, (centre - margin) / denom)


def wilson_upper_bound(wins: int, games: int, z: float = 1.645) -> float:
    """Wilson score interval UPPER bound — the symmetric sibling of ``wilson_lower_bound``.
    Used by the HoF breadth veto (P2.2): a suspect is PROVEN losing only when its win-rate's
    UPPER CI sits below 0.50, and by the mirror-collapse revert (learner significantly worse than
    its own frozen champion). Returns 1.0 for no games (no evidence ⇒ cannot prove a loss).
    Mirrors ``wilson_lower_bound`` exactly with ``+margin`` instead of ``-margin``."""
    if games <= 0:
        return 1.0
    p = wins / games
    z2 = z * z
    denom = 1.0 + z2 / games
    centre = p + z2 / (2.0 * games)
    margin = z * math.sqrt(p * (1.0 - p) / games + z2 / (4.0 * games * games))
    return min(1.0, (centre + margin) / denom)


# ── legacy gate (scripted ladder + prev_best head-to-head) ────────────────────
@dataclass
class GateConfig:
    z: float = 1.0               # one-sided significance band (1.0 ~84%, 1.645 ~95%)
    min_delta: float = 0.0       # require at least this absolute win-rate improvement
    revert_on_regression: bool = True
    use_prev_best: bool = True   # head-to-head non-saturating bar (sec 16). False => pure
                                 # scripted gate AND the eval skips the prev_best mirror
                                 # (no head-to-head games). Toggled by --no-prev-best / the wizard.


def promotion_gate(new_wins: int, new_games: int, base_wins: int, base_games: int,
                   cfg: GateConfig = GateConfig(),
                   prevbest_wins: Optional[int] = None,
                   prevbest_games: int = 0) -> Tuple[str, dict]:
    """Decide promote / hold / revert (LEGACY gate; ``promotion_gate_v2`` is the live one now).

      * promote  if scripted improves significantly OR the candidate beats prev_best
                 head-to-head significantly (>50% by the z-band);
      * revert   only on a real scripted COLLAPSE (regression below the best);
      * hold     otherwise.
    No baseline (first generation) → auto-promote. No prev_best data → scripted-only gate."""
    if base_games <= 0:
        return "promote", {"reason": "no_baseline", "p_new": (new_wins / new_games)
                           if new_games else None}
    p_new = new_wins / new_games if new_games else 0.0
    p_base = base_wins / base_games
    delta = p_new - p_base
    se = _two_prop_se(p_new, new_games, p_base, base_games)
    lo, hi = delta - cfg.z * se, delta + cfg.z * se
    if lo > cfg.min_delta:
        scripted_verdict = "promote"
    elif cfg.revert_on_regression and hi < -cfg.min_delta:
        scripted_verdict = "revert"
    else:
        scripted_verdict = "hold"
    stats = {"p_new": p_new, "p_base": p_base, "delta": delta, "se": se,
             "ci": (lo, hi), "z": cfg.z, "scripted_verdict": scripted_verdict}

    # Non-saturating best-self anchor: did the candidate beat prev_best head-to-head? The
    # reference 0.5 is a FIXED null, so the correct one-sample SE is sqrt(0.25/n) — NOT
    # _two_prop_se(p,n,0.5,n), which spuriously inflates it up to sqrt(2) (red-team fix).
    beats_best = False
    if prevbest_wins is not None and prevbest_games > 0:
        p_h2h = prevbest_wins / prevbest_games
        se_h2h = math.sqrt(0.25 / prevbest_games)
        margin_lo = (p_h2h - 0.5) - cfg.z * se_h2h
        beats_best = margin_lo > cfg.min_delta
        stats["prevbest"] = {"p": p_h2h, "n": prevbest_games, "se": se_h2h,
                             "margin_lo": margin_lo, "beats_best": beats_best}

    # Collapse safety FIRST: a real scripted regression reverts even if the candidate
    # edges prev_best (which would be noise in a genuinely-collapsed policy).
    if scripted_verdict == "revert":
        verdict, reason = "revert", "scripted_collapse"
    elif scripted_verdict == "promote":
        verdict, reason = "promote", "scripted"
    elif beats_best:
        verdict, reason = "promote", "beats_prev_best"   # non-saturating bar
    else:
        verdict, reason = "hold", "hold"
    stats["verdict_reason"] = reason
    return verdict, stats


# ── v2 gate: frozen-champion ladder (sec 16, calibrated via gate_sim) ──────────
def is_plateau(obs_history, window: int = 5, margin: float = 0.01) -> bool:
    """True when the head-to-head climb vs the FROZEN champion has STALLED: the recent
    ``window``-gen mean observed win-rate is not meaningfully (``> margin``) above the prior
    window's. Averaging over a window is what makes this robust to per-gen mirror noise — a
    single noisy gen can't trip it. Needs >= ``2*window`` samples since the champion was frozen;
    before that returns False (still gathering evidence → keep waiting). Pure-Python (no numpy)
    so the gate stays importable for ``--dry-run``.

    NOTE: a VERY slow real climb (per-gen rise smaller than the windowed noise) is genuinely
    indistinguishable from a plateau over a short window — widen ``window`` or use more mirror
    games (gate_sim showed ~240 is the floor at threshold 0.70). Not a logic bug."""
    n = len(obs_history)
    if n < 2 * window:
        return False
    recent = sum(obs_history[-window:]) / window
    prior = sum(obs_history[-2 * window:-window]) / window
    return recent <= prior + margin


@dataclass
class GateConfigV2:
    """Frozen-champion ladder (the user's 'static until proven' design, calibrated on gate_sim).
    The champion stays FROZEN until the candidate clears a HIGH observed bar vs it, OR a PLATEAU
    backstop fires because the climb genuinely stalled. Collapse-revert is the safety net."""
    promote_threshold: float = 0.55     # observed mirror win-rate vs champion to crown a new one
                                        # (P2.1: 0.70->0.55 — promote on CONVINCING, not crushing, improvement;
                                        #  leans on the HoF breadth veto to catch lateral RPS promotes)
    promote_z: float = 1.645            # the mirror rate's lower CI must also exceed 0.5 (sig. guard)
    min_h2h_games: int = 360            # need >= this many mirror games to trust the bar (P2.1 sim: the
                                        # 0.55 bar needs ~360 to hold false-promote ~3%; was 200 for 0.70)
    floor_margin: float = 0.06          # scripted may sit this far below high-water before "collapse"
    floor_z: float = 1.645             # significance band for the collapse test
    plateau_window: int = 5            # gens of h2h history per plateau-comparison window
    plateau_margin: float = 0.01       # recent window must beat prior by > this to be "still rising"
    plateau_not_losing: float = 0.5    # backstop only re-anchors if the plateau is at/above this
    # mirror-collapse revert (P2.1): the learner eroded significantly BELOW its own frozen champion — a real
    # degradation the scripted floor can miss. Reverts to champion (reuses the scripted-collapse wiring).
    mirror_collapse_margin: float = 0.05   # revert if wilson_upper(mirror) < 0.5 - this (bar 0.45)
    mirror_collapse_z: float = 1.645       # significance band for the mirror-collapse test
    mirror_collapse_min_games: int = 360   # need >= this many mirror games so a freshly-re-anchored learner
                                           # (~50% vs the new champion) is never falsely reverted (sim: ~0%)


def promotion_gate_v2(*, scripted_wins: int, scripted_games: int,
                      high_water: Optional[float], mirror_wins: int, mirror_games: int,
                      h2h_history, have_champion: bool = True,
                      cfg: GateConfigV2 = GateConfigV2()) -> Tuple[str, dict]:
    """Decide promote / hold / revert for the v2 frozen-champion ladder.

    Priority (safety first, then proof, then the stall backstop):
      1. REVERT  — scripted COLLAPSE: even the optimistic (upper-CI) scripted rate sits below
                   ``high_water - floor_margin`` (catastrophic forgetting). Takes precedence.
      2. PROMOTE/``beat_champion`` — the candidate cleared the HIGH bar: observed mirror win-rate
                   >= ``promote_threshold`` AND its lower CI > 0.5 (decisive, not a coin-flip), with
                   >= ``min_h2h_games`` games so the read is trustworthy.
      3. PROMOTE/``plateau_reanchor`` — the head-to-head climb has PLATEAUED (``is_plateau``) at a
                   not-losing level (>= ``plateau_not_losing``): advance the champion to the current
                   not-worse policy so the run doesn't freeze forever / the rollback floor stays
                   current. (The backstop, not a timer — only fires on a genuine stall.)
      4. HOLD    — otherwise (still climbing toward the bar, or too few games to tell).

    Pure (math only). ``have_champion=False`` (gen 0) auto-promotes the first champion. ``high_water
    is None`` disables the collapse floor (no accepted champion baseline yet). ``h2h_history`` is the
    list of per-gen observed mirror win-rates since the champion was last frozen."""
    if not have_champion:
        return "promote", {"reason": "no_baseline",
                           "p_mirror": (mirror_wins / mirror_games) if mirror_games else None}

    p_scr = scripted_wins / scripted_games if scripted_games else 0.0
    se_scr = math.sqrt(p_scr * (1 - p_scr) / max(1, scripted_games))
    scr_upper = p_scr + cfg.floor_z * se_scr
    floor = (high_water - cfg.floor_margin) if high_water is not None else None
    collapsed = floor is not None and scr_upper < floor

    # T4.3 (convention, by design): mirror_games is the player's n_finished_battles = wins+losses+TIES,
    # so a TIE counts in the denominator but not as a win — p_mir is "fraction of finished games WON".
    # That is intentional for a "must DECISIVELY beat the frozen champion" bar (a draw is not a win);
    # the collapse-revert side is symmetric (a tie is not a loss via mir_upper). Ties are rare in VGC and
    # hit candidate + champion alike in symmetric self-play, so the bias is negligible — NOT changed.
    p_mir = mirror_wins / mirror_games if mirror_games else 0.0
    se_mir = math.sqrt(p_mir * (1 - p_mir) / max(1, mirror_games))
    mir_lower = p_mir - cfg.promote_z * se_mir
    enough = mirror_games >= cfg.min_h2h_games
    beats_bar = enough and p_mir >= cfg.promote_threshold and mir_lower > 0.5

    # Mirror-collapse: the learner has eroded significantly BELOW its own frozen champion (a real
    # degradation the scripted floor can miss). Wilson UPPER bound < 0.5 - margin = "provably worse
    # than the champion", needing >= mirror_collapse_min_games so a freshly re-anchored learner
    # (~50% vs the new, stronger champion) is never falsely reverted (P2.1 sim: false-revert@0.50 ~0%).
    mir_upper = wilson_upper_bound(mirror_wins, mirror_games, cfg.mirror_collapse_z) \
        if mirror_games > 0 else 1.0
    mirror_enough = mirror_games >= cfg.mirror_collapse_min_games
    mirror_collapsed = mirror_enough and mir_upper < (0.5 - cfg.mirror_collapse_margin)

    plateaued = is_plateau(h2h_history, cfg.plateau_window, cfg.plateau_margin)
    backstop = plateaued and (p_mir >= cfg.plateau_not_losing) and not beats_bar

    stats = {
        "scripted": {"p": p_scr, "upper_ci": scr_upper, "floor": floor, "collapsed": collapsed},
        "mirror": {"p": p_mir, "lower_ci": mir_lower, "upper_ci": mir_upper, "n": mirror_games,
                   "enough_games": enough, "beats_bar": beats_bar, "collapsed": mirror_collapsed},
        "plateau": {"detected": plateaued, "not_losing": p_mir >= cfg.plateau_not_losing,
                    "backstop": backstop},
    }
    # Priority: safety reverts first (scripted collapse, then mirror collapse — both restore the
    # champion), then the high bar, then the stall backstop, else hold.
    if collapsed:
        verdict, reason = "revert", "scripted_collapse"
    elif mirror_collapsed:
        verdict, reason = "revert", "mirror_collapse"
    elif beats_bar:
        verdict, reason = "promote", "beat_champion"
    elif backstop:
        verdict, reason = "promote", "plateau_reanchor"
    else:
        verdict, reason = "hold", "hold"
    stats["reason"] = reason
    return verdict, stats


# ── Hall-of-Fame anti-cycle gate (Phase 2 — breadth veto on a mirror-promote) ──
@dataclass
class HoFConfig:
    """Phase-2 Hall-of-Fame breadth veto (docs/hof_anticycle_design.md). On a v2 mirror-promote
    the candidate must ALSO not be PROVEN losing to any of its last ``n_champions`` PAST CHAMPIONS
    (accepted promotions, EXCLUDING the current champion the mirror already tests). This catches
    LINEAGE CYCLING — a candidate that beats the current champion but loses to an OLDER champion =
    the accepted lineage isn't monotonic improvement, it's an RPS cycle (the 'G5 promoted backward
    over G4' failure that motivated the v2 redesign). Past champions are the STRONG milestones and
    are spread across training eras, so the set is naturally diverse. Numbers locked in P2.1."""
    enabled: bool = True
    n_champions: int = 5             # the candidate must not-lose to its last N past champions (excl. current)
    games_per_snapshot: int = 60     # mirror games vs each sampled past champion
    min_games_per_snap: int = 40     # below this a suspect is reported but CANNOT veto (no noise-freeze)
    z: float = 1.96                  # veto significance band (P2.1: halves per-snapshot false-reject)
    min_pool: int = 2                # need >= this many PAST-CHAMPION suspects to render a veto, else SKIP
                                     # (fail-open — early in a run there are too few promotions to cycle)
    # NOTE: the HoF-standoff FORCE-VALVE was calibrated to alert-only (the auto-advance 'marginal'
    # window is empty at n=60, gate_sim.force_valve_rate), so there is no force_limit config field —
    # operator_alert fires at its own hof_standoff_limit (default 2). Don't re-add a dead knob here.
    override: bool = False           # operator escape (--hof-override): let HoF-rejected promotes through
                                     # (still EVALUATED + logged loudly) to release a frozen catastrophic standoff


def cluster_hof_suspects(snapshots, *, n: int = 5, current_champion_path=None):
    """Pick the HoF suspect set = the candidate's PAST CHAMPIONS (accepted promotions) — the
    STRONG milestones of its own lineage — EXCLUDING the current champion (the v2 mirror already
    tests it 360 games deep). Requiring the candidate to not-lose to its prior ``n`` champions
    directly catches LINEAGE CYCLING (beats the current champion but loses to an older one). Past
    champions are promoted across training time, so the set is naturally era-diverse AND strong
    (each one cleared the bar — a sharper test than a held non-champion gen). Returns the most-
    recent ``n`` past champions, NEWEST first. Pure — works on any object with ``.generation``,
    ``.is_champion``, ``.snapshot_id``, ``.path``.

    The current champion is dropped by ``current_champion_path`` when given (exact, the wiring
    knows ``history.best_path``); otherwise the highest-generation champion is treated as current.
    The gate (``hall_of_fame_gate``) only sees ``(id, wins, games)`` — never how the suspects were
    chosen — so swapping the selection (e.g. add diverse non-champions later) touches THIS fn only."""
    champs = [s for s in snapshots if getattr(s, "is_champion", False)]
    if not champs:
        return []
    champs = sorted(champs, key=lambda s: (s.generation, s.snapshot_id))
    if current_champion_path is not None:
        champs = [s for s in champs if s.path != current_champion_path]
    else:
        champs = champs[:-1]                       # drop the latest-gen champion = the current one
    return list(reversed(champs[-max(0, int(n)):]))   # the most-recent n past champions, newest first


def hall_of_fame_gate(snap_results, *, cfg: HoFConfig = HoFConfig()) -> Tuple[str, dict]:
    """The HoF breadth VETO (pure). ``snap_results`` = list of ``(snapshot_id, wins, games)`` for
    the sampled suspects. Returns ``("confirm" | "reject" | "skip", stats)``:
      * REJECT — ANY eligible (``games >= min_games_per_snap``) suspect is PROVEN losing, i.e. its
        ``wilson_upper(wins, games, z) < 0.50`` (worst-of-snapshots; no band averaging — that washes
        out a single hidden counter, the design's core finding).
      * SKIP   — fewer than ``min_pool`` ELIGIBLE suspects: fail-open (== confirm), never block a
        promote on absence of breadth evidence (a thin / champion-flooded pool).
      * CONFIRM — otherwise (no suspect proven losing).
    A suspect below the games floor is reported (for observability) but cannot veto."""
    rows = []
    eligible = 0
    worst_id, worst_upper = None, 1.0
    rejected = False
    for sid, wins, games in snap_results:
        wins, games = int(wins), int(games)
        upper = wilson_upper_bound(wins, games, cfg.z)
        lower = wilson_lower_bound(wins, games, cfg.z)
        can_veto = games >= cfg.min_games_per_snap
        eligible += int(can_veto)
        vetoed = can_veto and upper < 0.5
        rejected = rejected or vetoed
        if upper < worst_upper:
            worst_upper, worst_id = upper, sid
        rows.append({"snapshot_id": sid, "wins": wins, "games": games,
                     "wilson_lower": lower, "wilson_upper": upper,
                     "eligible": can_veto, "vetoed": vetoed})
    if eligible < cfg.min_pool:
        return "skip", {"reason": "thin_pool_skip", "eligible": eligible,
                        "min_pool": cfg.min_pool, "n_suspects": len(rows), "suspects": rows}
    verdict = "reject" if rejected else "confirm"
    return verdict, {"reason": "hof_reject" if rejected else "hof_confirm",
                     "eligible": eligible, "worst_snapshot_id": worst_id,
                     "worst_upper": worst_upper, "n_suspects": len(rows), "suspects": rows}


# ── gauntlet-result aggregation ───────────────────────────────────────────────
def aggregate_scripted(results: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Sum (wins, games) over the scripted-anchor opponents in a gauntlet result dict."""
    w = g = 0
    for name, (wins, n) in results.items():
        if name in SCRIPTED_OPPONENTS:
            w += wins
            g += n
    return w, g


def aggregate_prev_best(results: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    """(wins, games) of the candidate's head-to-head vs the prev_best/champion mirror, or
    (0, 0) when the eval did not run it (gen 0 / no accepted best yet)."""
    w, g = results.get("prev_best", (0, 0))
    return int(w), int(g)


# ── generation history (pure, persistable for 3c.4) ───────────────────────────
@dataclass
class GenerationRecord:
    generation: int
    n_trajectories: int
    scripted_wins: int
    scripted_games: int
    model_elo: Optional[float]
    verdict: str
    promoted: bool
    update_stats: dict = field(default_factory=dict)
    champion_elo: Optional[float] = None     # the champion-LINEAGE Elo as of this gen (non-saturating)
    hof: Optional[dict] = None               # the Phase-2 HoF breadth-veto result this gen (None = not run)

    def to_obj(self) -> dict:
        return {"generation": self.generation, "n_trajectories": self.n_trajectories,
                "scripted_wins": self.scripted_wins, "scripted_games": self.scripted_games,
                "model_elo": self.model_elo, "verdict": self.verdict,
                "promoted": self.promoted, "champion_elo": self.champion_elo, "hof": self.hof,
                "update_stats": {k: v for k, v in self.update_stats.items()
                                 if isinstance(v, (int, float))}}

    @classmethod
    def from_obj(cls, d: dict) -> "GenerationRecord":
        return cls(generation=int(d["generation"]), n_trajectories=int(d.get("n_trajectories", 0)),
                   scripted_wins=int(d.get("scripted_wins", 0)),
                   scripted_games=int(d.get("scripted_games", 0)),
                   model_elo=d.get("model_elo"), verdict=d.get("verdict", "hold"),
                   promoted=bool(d.get("promoted", False)),
                   champion_elo=d.get("champion_elo"), hof=d.get("hof"),
                   update_stats=d.get("update_stats", {}))


@dataclass
class GenerationHistory:
    records: List[GenerationRecord] = field(default_factory=list)
    best_path: Optional[str] = None
    best_scripted: Tuple[int, int] = (0, 0)   # (wins, games) of the accepted CHAMPION
    # v2 frozen-champion state (sec 16):
    scripted_high_water: Optional[float] = None     # Wilson lower-bound competence floor (never regresses)
    h2h_history: List[float] = field(default_factory=list)  # per-gen observed mirror win-rate since
                                                            # the champion was last frozen (plateau input)
    champion_elo: Optional[float] = None            # champion-LINEAGE Elo (rises each promote; non-saturating)
    hof_reject_streak: int = 0                       # consecutive HoF-rejected promotes (force-valve + alert)

    @property
    def generation(self) -> int:
        return len(self.records)

    def add(self, rec: GenerationRecord) -> None:
        self.records.append(rec)

    def elo_curve(self) -> List[Tuple[int, Optional[float]]]:
        return [(r.generation, r.model_elo) for r in self.records]

    def champion_generation(self) -> Optional[int]:
        """The generation of the CURRENT champion = the latest promoted gen (or None)."""
        proms = [r.generation for r in self.records if r.promoted]
        return proms[-1] if proms else None

    # ── v2 champion maintenance ───────────────────────────────────────────────
    def record_h2h(self, mirror_rate: Optional[float]) -> None:
        """Append this gen's observed mirror win-rate to the (since-frozen) head-to-head history
        used for plateau detection. No-op when the mirror didn't run (rate None)."""
        if mirror_rate is not None:
            self.h2h_history.append(float(mirror_rate))

    def advance_champion(self, path: str, scripted: Tuple[int, int],
                         mirror_rate: Optional[float] = None, base_elo: float = 1000.0) -> None:
        """Crown a new champion: point at it, RAISE the high-water floor to the conservative
        (Wilson) lower bound of its scripted rate (never regresses), RESET the head-to-head
        history (the edge starts over vs the new champion), and step the champion-LINEAGE Elo by
        the rating gain implied by the mirror win-rate (a NON-saturating progress metric — the
        scripted ``model_elo`` flatlines once the policy crushes the scripts)."""
        self.best_path = path
        self.best_scripted = scripted
        cand = wilson_lower_bound(scripted[0], scripted[1])
        self.scripted_high_water = cand if self.scripted_high_water is None \
            else max(self.scripted_high_water, cand)
        self.h2h_history = []
        if self.champion_elo is None:
            self.champion_elo = float(base_elo)                 # first champion seeds the lineage
        elif mirror_rate is not None:
            p = min(0.999, max(0.001, float(mirror_rate)))      # rating gain over the prior champion
            self.champion_elo += 400.0 * math.log10(p / (1.0 - p))

    def to_obj(self) -> dict:
        return {"records": [r.to_obj() for r in self.records], "best_path": self.best_path,
                "best_scripted": list(self.best_scripted),
                "scripted_high_water": self.scripted_high_water,
                "h2h_history": list(self.h2h_history), "champion_elo": self.champion_elo,
                "hof_reject_streak": self.hof_reject_streak}

    @classmethod
    def from_obj(cls, d: dict) -> "GenerationHistory":
        return cls(records=[GenerationRecord.from_obj(r) for r in d.get("records", [])],
                   best_path=d.get("best_path"),
                   best_scripted=tuple(d.get("best_scripted", (0, 0))),
                   scripted_high_water=d.get("scripted_high_water"),
                   h2h_history=list(d.get("h2h_history", [])),
                   champion_elo=d.get("champion_elo"),
                   hof_reject_streak=int(d.get("hof_reject_streak", 0)))


# ── generation config ─────────────────────────────────────────────────────────
@dataclass
class GenConfig:
    n_games: int = 300            # self-play games collected per generation
    warmup_updates: int = 5       # critic-only warm-up updates on the FIRST generation
    gate: GateConfig = field(default_factory=GateConfig)            # legacy gate (use_prev_best toggle)
    gate_v2: GateConfigV2 = field(default_factory=GateConfigV2)     # the live frozen-champion gate
    hof: "HoFConfig" = field(default_factory=lambda: HoFConfig())   # Phase-2 HoF breadth veto (wired in P2.4)
    league_cap: int = 20          # max league snapshots (sec 16; diversity-aware eviction)
    keep_recent: int = 6          # snapshots always kept regardless of the cap


# ── operator alert (unattended-run watchdog, sec 16) ──────────────────────────
def operator_alert(history, *, revert_limit: int = 3, stall_limit: int = 25,
                   hof_standoff_limit: int = 2) -> Optional[str]:
    """Surface a loud ALERT for an UNATTENDED run, with a 3-WAY taxonomy so a freeze is never
    ambiguous: a COLLAPSE LOOP (>= ``revert_limit`` consecutive REVERTs), a HoF STANDOFF (>=
    ``hof_standoff_limit`` consecutive HoF-rejected promotes — a lineage cycle the champion is
    frozen against), or a long generic champion STALL (no promotion in >= ``stall_limit`` gens).
    Returns the alert text or None. Pure (reads ``history``) so the live loop just prints it."""
    recs = getattr(history, "records", None) or []
    if not recs:
        return None
    streak = 0
    for r in reversed(recs):
        if r.verdict == "revert":
            streak += 1
        else:
            break
    if streak >= revert_limit:
        return (f"OPERATOR ALERT: {streak} consecutive REVERTs — collapse loop; consider "
                f"stopping / lowering the LR.")
    hof_streak = int(getattr(history, "hof_reject_streak", 0) or 0)
    if hof_streak >= hof_standoff_limit:
        return (f"OPERATOR ALERT: {hof_streak} consecutive HoF rejects — the candidate keeps beating "
                f"the champion but LOSES to a PAST champion (a lineage cycle). The champion is frozen; "
                f"re-run with --hof-override to force the advance if that's intended.")
    since = next((i for i, r in enumerate(reversed(recs)) if r.promoted), len(recs))
    if since >= stall_limit:
        return (f"OPERATOR ALERT: champion frozen {since} gens (no promotion) — check the h2h "
                f"trend: a genuine plateau (backstop should fire) or stuck?")
    return None
