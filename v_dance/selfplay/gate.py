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
    promote_threshold: float = 0.70     # observed mirror win-rate vs champion to crown a new one
    promote_z: float = 1.645            # the mirror rate's lower CI must also exceed 0.5 (sig. guard)
    min_h2h_games: int = 200            # need >= this many mirror games to trust the bar (sim floor)
    floor_margin: float = 0.06          # scripted may sit this far below high-water before "collapse"
    floor_z: float = 1.645             # significance band for the collapse test
    plateau_window: int = 5            # gens of h2h history per plateau-comparison window
    plateau_margin: float = 0.01       # recent window must beat prior by > this to be "still rising"
    plateau_not_losing: float = 0.5    # backstop only re-anchors if the plateau is at/above this


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

    p_mir = mirror_wins / mirror_games if mirror_games else 0.0
    se_mir = math.sqrt(p_mir * (1 - p_mir) / max(1, mirror_games))
    mir_lower = p_mir - cfg.promote_z * se_mir
    enough = mirror_games >= cfg.min_h2h_games
    beats_bar = enough and p_mir >= cfg.promote_threshold and mir_lower > 0.5

    plateaued = is_plateau(h2h_history, cfg.plateau_window, cfg.plateau_margin)
    backstop = plateaued and (p_mir >= cfg.plateau_not_losing) and not beats_bar

    stats = {
        "scripted": {"p": p_scr, "upper_ci": scr_upper, "floor": floor, "collapsed": collapsed},
        "mirror": {"p": p_mir, "lower_ci": mir_lower, "n": mirror_games,
                   "enough_games": enough, "beats_bar": beats_bar},
        "plateau": {"detected": plateaued, "not_losing": p_mir >= cfg.plateau_not_losing,
                    "backstop": backstop},
    }
    if collapsed:
        verdict, reason = "revert", "scripted_collapse"
    elif beats_bar:
        verdict, reason = "promote", "beat_champion"
    elif backstop:
        verdict, reason = "promote", "plateau_reanchor"
    else:
        verdict, reason = "hold", "hold"
    stats["reason"] = reason
    return verdict, stats


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

    def to_obj(self) -> dict:
        return {"generation": self.generation, "n_trajectories": self.n_trajectories,
                "scripted_wins": self.scripted_wins, "scripted_games": self.scripted_games,
                "model_elo": self.model_elo, "verdict": self.verdict,
                "promoted": self.promoted, "champion_elo": self.champion_elo,
                "update_stats": {k: v for k, v in self.update_stats.items()
                                 if isinstance(v, (int, float))}}

    @classmethod
    def from_obj(cls, d: dict) -> "GenerationRecord":
        return cls(generation=int(d["generation"]), n_trajectories=int(d.get("n_trajectories", 0)),
                   scripted_wins=int(d.get("scripted_wins", 0)),
                   scripted_games=int(d.get("scripted_games", 0)),
                   model_elo=d.get("model_elo"), verdict=d.get("verdict", "hold"),
                   promoted=bool(d.get("promoted", False)),
                   champion_elo=d.get("champion_elo"),
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
                "h2h_history": list(self.h2h_history), "champion_elo": self.champion_elo}

    @classmethod
    def from_obj(cls, d: dict) -> "GenerationHistory":
        return cls(records=[GenerationRecord.from_obj(r) for r in d.get("records", [])],
                   best_path=d.get("best_path"),
                   best_scripted=tuple(d.get("best_scripted", (0, 0))),
                   scripted_high_water=d.get("scripted_high_water"),
                   h2h_history=list(d.get("h2h_history", [])),
                   champion_elo=d.get("champion_elo"))


# ── generation config ─────────────────────────────────────────────────────────
@dataclass
class GenConfig:
    n_games: int = 300            # self-play games collected per generation
    warmup_updates: int = 5       # critic-only warm-up updates on the FIRST generation
    gate: GateConfig = field(default_factory=GateConfig)            # legacy gate (use_prev_best toggle)
    gate_v2: GateConfigV2 = field(default_factory=GateConfigV2)     # the live frozen-champion gate
    league_cap: int = 20          # max league snapshots (sec 16; diversity-aware eviction)
    keep_recent: int = 6          # snapshots always kept regardless of the cap


# ── operator alert (unattended-run watchdog, sec 16) ──────────────────────────
def operator_alert(history, *, revert_limit: int = 3, stall_limit: int = 25) -> Optional[str]:
    """Surface a loud ALERT for an UNATTENDED run: a COLLAPSE LOOP (>= ``revert_limit`` consecutive
    REVERTs) or a long champion STALL (no promotion in >= ``stall_limit`` gens). Returns the alert
    text or None. Pure (reads ``history.records``) so the live loop just prints it; tested offline."""
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
    since = next((i for i, r in enumerate(reversed(recs)) if r.promoted), len(recs))
    if since >= stall_limit:
        return (f"OPERATOR ALERT: champion frozen {since} gens (no promotion) — check the h2h "
                f"trend: a genuine plateau (backstop should fire) or stuck?")
    return None
