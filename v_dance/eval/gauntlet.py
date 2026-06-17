"""
local_battle/gauntlet.py  —  win-rate eval gauntlet for the BC model (#3)
=========================================================================
The metric that matters.  Plays the trained BC checkpoint a balanced number of
battles against a SKILL LADDER of opponents:

    random  <  max_damage  <  heuristic  (<  a previous-best checkpoint)

on a ROTATING TEAM POOL with SWAPPED sides (each team is played by both the model
and the opponent equally often), then:

  * tallies per-opponent win-rates,
  * estimates a single ``model_elo`` from the win-rates vs the fixed-anchor
    scripted opponents (order-independent, clamped so 0%/100% stay finite),
  * appends a versioned row to a JSON history file, and
  * GATES the run against the previous row (Elo / win-rate delta).

The pure scoring + persistence helpers below carry no poke-env / asyncio
dependency so they unit-test without a Showdown server; the async ``run_gauntlet``
orchestrator reuses ``run_local_battle.make_player`` / ``start_showdown``.

CLI (repo root):
    .venv\\Scripts\\python.exe local_battle/gauntlet.py --battles 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ── Path bootstrap (local_battle FIRST) ───────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]   # v_dance/eval/ -> repo root
log = logging.getLogger(__name__)

# Fixed Elo anchors for the SCRIPTED opponents (the ladder the model_elo is
# measured against).  A previous-best checkpoint is reported as a head-to-head
# win-rate, NOT folded into the anchor estimate (its rating is what we're trying
# to establish, so anchoring on it would be circular).
ANCHORS: Dict[str, float] = {
    "random": 1000.0,
    "max_damage": 1300.0,
    "heuristic": 1500.0,
}
DEFAULT_HISTORY = _REPO_ROOT / "artifacts" / "eval_results" / "gauntlet_history.json"


# ══════════════════════════════════════════════════════════════════════════════
# Pure scoring / Elo (no poke-env, no asyncio) — unit-tested
# ══════════════════════════════════════════════════════════════════════════════
def expected_score(rating: float, opp_rating: float) -> float:
    """Standard logistic Elo expectation that ``rating`` beats ``opp_rating``."""
    return 1.0 / (1.0 + 10.0 ** ((opp_rating - rating) / 400.0))


def implied_rating(wins: float, n: int, opp_rating: float) -> Optional[float]:
    """The rating that, vs a fixed ``opp_rating`` anchor, explains an observed
    win-rate over ``n`` games.  The win-rate is continuity-clamped to
    ``[1/2n, 1-1/2n]`` so 0% / 100% map to a finite, bounded rating."""
    if n <= 0:
        return None
    w = wins / n
    lo, hi = 1.0 / (2 * n), 1.0 - 1.0 / (2 * n)
    w = min(max(w, lo), hi)
    return opp_rating - 400.0 * math.log10(1.0 / w - 1.0)


def model_elo(results: Dict[str, Tuple[int, int]],
              anchors: Dict[str, float] = ANCHORS) -> Optional[float]:
    """Weighted (by game count) average of the per-anchor implied ratings.

    ``results`` maps opponent name → (wins, n).  Only opponents present in
    ``anchors`` with n>0 contribute (a previous-best mirror is excluded)."""
    num = den = 0.0
    for name, (wins, n) in results.items():
        anchor = anchors.get(name)
        if anchor is None or n <= 0:
            continue
        r = implied_rating(wins, n, anchor)
        if r is None:
            continue
        num += r * n
        den += n
    return num / den if den > 0 else None


def build_run_row(
    results: Dict[str, Tuple[int, int]],
    ckpt: str,
    run_id: str,
    timestamp: str,
    extra: Optional[dict] = None,
) -> dict:
    """Assemble the versioned history row from raw (wins, n) per opponent."""
    per_opponent = {}
    scripted_wins = scripted_n = 0
    for name, (wins, n) in results.items():
        wr = wins / n if n else None
        per_opponent[name] = {"wins": wins, "n": n, "win_rate": wr,
                              "anchor": ANCHORS.get(name)}
        if name in ANCHORS:
            scripted_wins += wins
            scripted_n += n
    row = {
        "run": run_id,
        "timestamp": timestamp,
        "ckpt": ckpt,
        "per_opponent": per_opponent,
        "scripted_win_rate": (scripted_wins / scripted_n) if scripted_n else None,
        "model_elo": model_elo(results),
    }
    if extra:
        row.update(extra)
    return row


# ── Persistence + regression gate ─────────────────────────────────────────────
def load_history(path) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def append_run(path, row: dict) -> List[dict]:
    hist = load_history(path)
    hist.append(row)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    return hist


def regression_gate(history: Sequence[dict], current: dict,
                    key: str = "model_elo", tolerance: float = 0.0) -> dict:
    """Compare ``current`` to the most recent PRIOR run in ``history`` (which must
    NOT yet contain ``current``).  ``regressed`` is True when the chosen metric
    dropped by more than ``tolerance``."""
    prior = history[-1] if history else None
    if not prior or prior.get(key) is None or current.get(key) is None:
        return {"have_baseline": False, "regressed": False, "delta": None,
                "prior": prior.get(key) if prior else None,
                "current": current.get(key)}
    delta = current[key] - prior[key]
    return {"have_baseline": True, "regressed": delta < -abs(tolerance),
            "delta": delta, "prior": prior[key], "current": current[key]}


def print_sources(sources) -> None:
    """Show how the model player's decisions were actually made.  A win-rate is
    only trustworthy if the MODEL drove most decisions; retry/default/forfeit
    fallbacks mean a mask/board desync took over.  Team-preview (``tp_*``) is
    reported separately — it's one decision per BATTLE, not per turn (#4)."""
    actions = {k: v for k, v in sources.items() if not k.startswith("tp_")}
    tp = {k[len("tp_"):]: v for k, v in sources.items() if k.startswith("tp_")}

    total = sum(actions.values())
    if total:
        model = actions.get("model", 0) + actions.get("forced_switch_model", 0)
        print("  -- decisions by source --")
        for k, v in sorted(actions.items(), key=lambda kv: -kv[1]):
            print(f"    {k:22s}: {v:6d}  ({v / total * 100:4.1f}%)")
        print(f"  MODEL-DRIVEN      : {model}/{total} = {model / total * 100:.1f}%  "
              f"(lower ⇒ more fallbacks ⇒ win-rate less trustworthy)")

    tp_total = sum(tp.values())
    if tp_total:
        tp_model = tp.get("model", 0)
        print("  -- team-preview by source (#4) --")
        for k, v in sorted(tp.items(), key=lambda kv: -kv[1]):
            print(f"    {k:22s}: {v:6d}")
        print(f"  TP NET-DRIVEN     : {tp_model}/{tp_total} = "
              f"{tp_model / tp_total * 100:.1f}%  (want 100% = the net chose every team)")
    print("============================================================")


def print_report(row: dict, gate: dict) -> None:
    print("== BC win-rate gauntlet ====================================")
    print(f"  run        : {row['run']}  ({row['timestamp']})")
    print(f"  ckpt       : {row['ckpt']}")
    print("  -- per opponent (model win-rate) --")
    for name, b in row["per_opponent"].items():
        wr = b["win_rate"]
        anchor = f" anchor {b['anchor']:.0f}" if b["anchor"] is not None else " (mirror)"
        wr_s = f"{wr*100:5.1f}%" if wr is not None else "  n/a"
        print(f"    {name:12s}: {b['wins']:3d}/{b['n']:<3d} = {wr_s}{anchor}")
    sr = row["scripted_win_rate"]
    print(f"  scripted win-rate : {sr*100:.1f}%" if sr is not None else "  scripted win-rate : n/a")
    print(f"  model Elo         : "
          f"{row['model_elo']:.0f}" if row["model_elo"] is not None else "  model Elo: n/a")
    if gate["have_baseline"]:
        arrow = "DOWN" if gate["regressed"] else "up/flat"
        print(f"  vs last run       : Elo {gate['prior']:.0f} -> {gate['current']:.0f} "
              f"(delta {gate['delta']:+.0f}) [{arrow}]")
    else:
        print("  vs last run       : (no baseline yet - this is the first run)")
    print("============================================================")


# ══════════════════════════════════════════════════════════════════════════════
# Side-balanced team rotation (pure) — unit-tested
# ══════════════════════════════════════════════════════════════════════════════
def team_matchups(team_pool: Sequence[str], n_battles: int,
                  seed: int = 0) -> List[Tuple[str, str, int]]:
    """Split ``n_battles`` across the team pool over distinct ordered matchups.

    Ordered pairs ``(A,B)`` and ``(B,A)`` are both candidates, so the model plays
    BOTH sides of every matchup — a Tyranitar-beats-Charizard style bias cancels
    rather than favouring one checkpoint.  The pair list is SHUFFLED with ``seed``
    before slicing, so when ``n_battles`` < #pairs (a big pool) the sampled subset
    is REPRESENTATIVE of the whole pool, not a biased prefix of ``team[0]``-vs-all —
    and it is REPRODUCIBLE, so two checkpoints A/B'd at the same seed face the
    IDENTICAL matchup set (the cleanest control for the matchup confound).

    Returns ``(model_team, opp_team, n)`` chunks summing to n_battles; one team
    degenerates to a mirror."""
    pool = list(team_pool) or ["team1"]
    if len(pool) == 1:
        return [(pool[0], pool[0], n_battles)]
    pairs = [(a, b) for a in pool for b in pool if a != b]
    random.Random(seed).shuffle(pairs)
    chunks: List[Tuple[str, str, int]] = []
    base, rem = divmod(n_battles, len(pairs))
    for k, (a, b) in enumerate(pairs):
        n = base + (1 if k < rem else 0)
        if n > 0:
            chunks.append((a, b, n))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Async orchestrator (reuses run_local_battle) — exercised by the live smoke
# ══════════════════════════════════════════════════════════════════════════════
def _make_opponent(kind: str, username: str, team: str, model_path=None,
                   team_chooser_path=None):
    """Construct one opponent player of the given ``kind``."""
    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    if kind == "random":
        return R.make_player(username, team, model_path=None)
    if kind in ("max_damage", "heuristic"):
        from v_dance.eval.eval_opponents import MaxDamageVGCPlayer, HeuristicVGCPlayer
        cls = MaxDamageVGCPlayer if kind == "max_damage" else HeuristicVGCPlayer
        return cls(
            replay_path=_REPO_ROOT / "artifacts" / "replay_buffer" / f"{username}.jsonl",
            account_configuration=AccountConfiguration(username, None),
            battle_format=R.BATTLE_FORMAT, team=team, max_concurrent_battles=1,
            log_level=logging.WARNING,
        )
    if kind == "prev_best":
        return R.make_player(username, team, model_path=model_path,
                             team_chooser_path=team_chooser_path)
    raise ValueError(f"unknown opponent kind: {kind}")


async def _open_spectator(player) -> None:
    """Open a browser tab spectating ``player``'s first battle (like
    run_local_battle).  Prints the URL too, so it's visible even headless."""
    import webbrowser
    import v_dance.play.run_local_battle as R
    for _ in range(150):
        await asyncio.sleep(0.1)
        battles = getattr(player, "battles", None)
        if battles:
            tag = next(iter(battles))
            url = f"http://{R.SHOWDOWN_HOST}:{R.SHOWDOWN_PORT}/{tag}"
            print(f"[gauntlet] SPECTATE: {url}", flush=True)
            log.warning("Spectate this battle: %s", url)
            try:
                webbrowser.open(url)
            except Exception:  # pragma: no cover - headless / no browser
                pass
            return
    log.warning("Spectate: no battle tag appeared to open.")


async def _play_chunk(model_player, opponent, n: int,
                      timeout: Optional[float] = None) -> Tuple[int, int]:
    """Run ``n`` battles model_player vs opponent; return (model_wins, n_finished).

    ``timeout`` (seconds, for the whole chunk) is a WATCHDOG: if a battle hangs
    (e.g. a forced-switch / illusion desync that never resolves), the chunk is
    abandoned and the gauntlet CONTINUES instead of freezing for hours.  Finished
    battles up to the hang still count."""
    won_before = model_player.n_won_battles
    fin_before = model_player.n_finished_battles
    coro = model_player.battle_against(opponent, n_battles=n)
    try:
        if timeout and timeout > 0:
            await asyncio.wait_for(coro, timeout=timeout)
        else:
            await coro
    except asyncio.TimeoutError:
        log.warning("chunk WATCHDOG fired after %.0fs (%d battles requested, %d "
                    "finished) — abandoning this chunk, continuing the gauntlet.",
                    timeout, n, model_player.n_finished_battles - fin_before)
    return (model_player.n_won_battles - won_before,
            model_player.n_finished_battles - fin_before)


async def run_gauntlet(
    opponents: Sequence[str],
    team_pool: Sequence[str],
    battles_per_opponent: int,
    ckpt: Path,
    team_chooser: Path,
    prev_best_ckpt: Optional[Path] = None,
    manage_server: bool = True,
    matchup_seed: int = 0,
    battle_timeout: Optional[float] = 90.0,
    spectate: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """Play the model vs each opponent over the rotating team pool and return
    ``{opponent_name: (model_wins, n_finished)}``."""
    import v_dance.play.run_local_battle as R
    server = R.start_showdown() if manage_server else None
    results: Dict[str, Tuple[int, int]] = {}
    source_totals: Counter = Counter()      # model vs retry/default/forfeit fallbacks
    uid = 0
    spectated = False
    try:
        for kind in opponents:
            wins = fin = 0
            for model_team_name, opp_team_name, n in team_matchups(
                    team_pool, battles_per_opponent, seed=matchup_seed):
                model_team = R.load_team(R.resolve_team_path(model_team_name))
                opp_team = R.load_team(R.resolve_team_path(opp_team_name))
                uid += 1
                model_player = R.make_player(
                    f"BC{uid}", model_team, model_path=ckpt, team_chooser_path=team_chooser)
                opp = _make_opponent(
                    kind, f"OP{kind[:4]}{uid}", opp_team,
                    model_path=prev_best_ckpt, team_chooser_path=team_chooser)
                if spectate and not spectated:
                    spectated = True
                    asyncio.ensure_future(_open_spectator(model_player))
                try:
                    chunk_timeout = (battle_timeout * n) if battle_timeout else None
                    w, f = await _play_chunk(model_player, opp, n, timeout=chunk_timeout)
                    wins += w
                    fin += f
                finally:
                    source_totals.update(getattr(model_player, "_source_counts", {}) or {})
                    for k, v in (getattr(model_player, "_tp_source", {}) or {}).items():
                        source_totals[f"tp_{k}"] += v          # team-preview tally (#4)
                    await model_player.ps_client.stop_listening()
                    await opp.ps_client.stop_listening()
                    model_player.close()
                    opp.close()
            results[kind] = (wins, fin)
            log.info("vs %s: %d/%d", kind, wins, fin)
    finally:
        if server is not None:
            R.stop_showdown(server)
    return results, source_totals


# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BC win-rate eval gauntlet (#3)")
    ap.add_argument("--battles", "-n", type=int, default=20,
                    help="battles per opponent (split across the team pool)")
    ap.add_argument("--opponents", nargs="+",
                    default=["random", "max_damage", "heuristic"],
                    help="opponent ladder (subset of random/max_damage/heuristic/prev_best)")
    ap.add_argument("--teams", nargs="+",
                    default=["team1", "WolfeGlick", "Kronomono1", "Kronomono3"],
                    help="rotating team pool (names under teams/M-A/ or paths). Use "
                         ">=4 teams: with DETERMINISTIC opponents a 2-team pool replays "
                         "the same matchups, so per-opponent win-rates collapse to a "
                         "handful of fixed outcomes instead of a real distribution.")
    ap.add_argument("--ckpt", default=str(_REPO_ROOT / "ai_train_scripts" / "BC_model"
                                          / "checkpoints" / "bc_best.pt"))
    ap.add_argument("--team-chooser", default=str(_REPO_ROOT / "ai_train_scripts"
                    / "teamPreview_model" / "checkpoints" / "teampreview_best.pt"))
    ap.add_argument("--prev-best", default=None,
                    help="checkpoint for the prev_best mirror opponent (if used)")
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    ap.add_argument("--matchup-seed", type=int, default=0,
                    help="seed for the team-matchup sampling; A/B two checkpoints at "
                         "the SAME seed so they face identical matchups (default 0)")
    ap.add_argument("--battle-timeout", type=float, default=90.0,
                    help="watchdog: max seconds PER BATTLE before a hung battle is "
                         "abandoned and the gauntlet continues (0 = no timeout; "
                         "default 90).")
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--spectate", action="store_true",
                    help="open a browser tab spectating the first battle (live view)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="DEBUG logging (incl. poke-env) for diagnosing stalls")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timestamp", default=None)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from datetime import datetime
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    if not args.verbose:                     # keep poke-env quiet unless diagnosing
        logging.getLogger("poke_env").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"[gauntlet] checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    run_id = args.run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    timestamp = args.timestamp or datetime.now().isoformat(timespec="seconds")

    results, sources = asyncio.run(run_gauntlet(
        opponents=args.opponents,
        team_pool=args.teams,
        battles_per_opponent=args.battles,
        ckpt=ckpt,
        team_chooser=Path(args.team_chooser),
        prev_best_ckpt=Path(args.prev_best) if args.prev_best else None,
        manage_server=not args.no_server,
        matchup_seed=args.matchup_seed,
        battle_timeout=args.battle_timeout,
        spectate=args.spectate,
    ))

    row = build_run_row(results, ckpt=str(ckpt), run_id=run_id, timestamp=timestamp,
                        extra={"sources": dict(sources)})
    history = load_history(args.history)            # BEFORE appending current
    gate = regression_gate(history, row)
    append_run(args.history, row)
    print_report(row, gate)
    print_sources(sources)
    return 1 if gate["regressed"] else 0


if __name__ == "__main__":
    sys.exit(main())
