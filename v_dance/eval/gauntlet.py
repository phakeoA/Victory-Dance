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
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    base, rem = divmod(n_battles, len(pairs))
    counts = {p: base for p in pairs}
    # #10: distribute the remainder in MIRROR-PAIR UNITS so (a,b) and (b,a) always get the SAME count —
    # else a shuffled prefix of ORDERED pairs could bonus (a,b) but not (b,a), leaving the team played
    # by the model more often than against (a residual side imbalance the docstring promises cancels).
    # Build the unordered pairs, shuffle reproducibly, +1 to BOTH halves of the first rem//2; an odd
    # leftover goes to one ordered pair (an unavoidable single-game imbalance) so the sum stays == n_battles.
    unordered, seen = [], set()
    for a in pool:
        for b in pool:
            if a != b and (a, b) not in seen and (b, a) not in seen:
                seen.add((a, b))
                unordered.append((a, b))
    random.Random(seed).shuffle(unordered)
    for (a, b) in unordered[: rem // 2]:
        counts[(a, b)] += 1
        counts[(b, a)] += 1
    if rem % 2:
        a, b = unordered[rem // 2]
        counts[(a, b)] += 1
    return [(a, b, n) for (a, b), n in counts.items() if n > 0]


# ==============================================================================
# W2 (2026-09-03): single-team ASYMMETRIC pairing - the OWN team on the model seat every game,
# the opponent seat drawn from the pool (docs/w2_era5b_run_design_2026-09-03.md section 2). Pure /
# torch-free; both the asyncio and the multiprocess collect + eval planners call these.
# ==============================================================================
OBSERVED_FLOOR_PCT = 3.0     # a pool team never drops below this usage weight (percent units)


def team_key(team) -> str:
    """The comparable identity of a pool entry: pool entries are repo-relative paths
    (``teams/Champions/M-B/The_Big_6_v2``) while CLI flags carry bare names - compare by file name."""
    return Path(str(team)).name.lower()


def canonical_own_team(own_team, team_pool):
    """The pool's own entry for ``own_team`` when it is in the pool (so chunks / replay routing see
    one consistent string), else ``own_team`` verbatim (``resolve_team_path`` finds it by name)."""
    k = team_key(own_team)
    for t in team_pool or []:
        if team_key(t) == k:
            return t
    return own_team


def own_team_matchups(own_team, team_pool: Sequence[str], n_battles: int, *,
                      mirror_frac: float = 0.2, weights: Optional[Dict[str, float]] = None,
                      seed: int = 0) -> List[Tuple[str, str, int]]:
    """Split ``n_battles`` with the OWN team ALWAYS on the model seat (``team_a``).

    ``round(mirror_frac * n)`` games are the mirror ``(own, own, k)`` (where Trick Room / Helping
    Hand self-coordination is learned); the rest go to the pool's OTHER teams (own excluded) in
    proportion to ``weights`` (``{pool_entry: weight}``; None / all-zero = uniform) - floor first,
    then the remainder by largest fractional part, ties broken by a ``seed``-shuffled order, so
    the split is REPRODUCIBLE per seed (the same guarantee ``team_matchups`` gives the A/B) and
    sums to exactly ``n_battles``. Opponents that round to 0 games are omitted."""
    n = max(0, int(n_battles))
    if n == 0:
        return []
    own = canonical_own_team(own_team, team_pool)
    ok = team_key(own)
    opps, seen = [], set()
    for t in team_pool or []:
        if team_key(t) != ok and team_key(t) not in seen:
            seen.add(team_key(t))
            opps.append(t)
    if not opps:
        return [(own, own, n)]
    frac = max(0.0, min(1.0, float(mirror_frac)))
    k = int(round(frac * n))
    rest = n - k
    w = [max(0.0, float((weights or {}).get(t, 0.0))) for t in opps] if weights else [1.0] * len(opps)
    if sum(w) <= 0.0:
        w = [1.0] * len(opps)
    tot = sum(w)
    exact = [rest * wi / tot for wi in w]
    counts = [int(math.floor(e)) for e in exact]
    rem = rest - sum(counts)
    order = list(range(len(opps)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: -(exact[i] - counts[i]))       # stable: ties keep the seeded order
    for i in order[:rem]:
        counts[i] += 1
    out: List[Tuple[str, str, int]] = [(own, own, k)] if k > 0 else []
    out.extend((own, t, c) for t, c in zip(opps, counts) if c > 0)
    return out


def _species_key(name) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def _team_file(team) -> Optional[Path]:
    """A light resolver (no poke-env import): a path, a repo-relative path, or a bare name found
    anywhere under teams/Champions/. None when nothing matches (the caller treats it as 0 usage)."""
    p = Path(str(team))
    if p.is_file():
        return p
    rp = _REPO_ROOT / str(team)
    if rp.is_file():
        return rp
    base = _REPO_ROOT / "teams" / "Champions"
    if base.exists():
        hits = sorted(q for q in base.rglob(p.name) if q.is_file())
        if hits:
            return hits[0]
    return None


def default_observed_meta_path() -> Path:
    """``data/observed_meta_<reg>.json`` for the active format (falls back to the M-B file)."""
    try:
        from v_dance.formats import reg_token, default_format
        tok = reg_token(default_format()) or "regmb"
    except Exception:
        tok = "regmb"
    p = _REPO_ROOT / "data" / f"observed_meta_{tok}.json"
    return p if p.exists() else _REPO_ROOT / "data" / "observed_meta_regmb.json"


def observed_team_weights(team_pool: Sequence[str], meta_path=None, *,
                          floor_pct: float = OBSERVED_FLOOR_PCT) -> Dict[str, float]:
    """``{pool_entry: weight}`` = the MEAN observed ``usage_pct`` of the team's species (the
    ladder dossiers' ``data/observed_meta_<reg>.json``), floored at ``floor_pct`` so a rare
    archetype keeps a minimum share of the opponent seat. Mega formes count under their base
    species (the meta file keys base names). A team whose paste cannot be read weighs the floor."""
    from v_dance.parser.vod_parser.team_sheet import parse_showdown_team, base_species
    path = Path(meta_path) if meta_path else default_observed_meta_path()
    try:
        meta = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    usage = {_species_key(k): float((v or {}).get("usage_pct") or 0.0)
             for k, v in (meta.get("pokemon") or {}).items()}
    out: Dict[str, float] = {}
    for t in team_pool or []:
        f = _team_file(t)
        vals: List[float] = []
        if f is not None:
            try:
                mons = parse_showdown_team(f.read_text(encoding="utf-8"))
            except Exception:
                mons = []
            vals = [usage.get(_species_key(base_species(m["species"])), 0.0)
                    for m in mons if m.get("species")]
        mean = sum(vals) / len(vals) if vals else 0.0
        out[t] = max(float(floor_pct), mean)
    return out




def subdivide_pairings(matchups, target_chunks):
    """Split ``(team_a, team_b, n)`` chunks into ~``target_chunks`` total so a SMALL pairing set can
    still saturate the worker pool (task #13; shared with the eval planners for W2, 2026-09-03).

    Each chunk may become several sub-chunks (``n_sub >= 1``, summing to ``n``); the extra splits go
    to the chunks with the most games-per-sub-chunk (largest-load-first), so the sub-chunks come out
    as even as possible. Motivating cases: a single-team MIRROR A/B, and the W2 own-vs-own champion
    mirror / HoF suspect - each ONE ``(own, own, N)`` chunk that would otherwise pin a 360-game eval
    to one sequential pairing. Each sub-chunk is later given a DISTINCT uid (distinct accounts +
    seeds), so they are INDEPENDENT seed-streams that aggregate to the identical verdict.

    NO-OP when the pool already yields ``>= target_chunks`` pairings or ``target_chunks <= 1``.
    Total games are preserved exactly and every sub-chunk keeps its parent's (team_a, team_b)."""
    chunks = [(a, b, n) for (a, b, n) in matchups if n > 0]
    total = sum(n for _, _, n in chunks)
    target = min(int(target_chunks), total)              # never more chunks than games
    if target <= len(chunks):
        return chunks
    ks = [1] * len(chunks)                               # sub-chunk count per original chunk (>= 1)
    ns = [n for _, _, n in chunks]
    for _ in range(target - len(chunks)):                # hand out the extra splits one at a time
        best, best_load = -1, -1.0
        for i in range(len(chunks)):
            if ks[i] >= ns[i]:                           # already 1 game / sub-chunk
                continue
            load = ns[i] / ks[i]                         # games per current sub-chunk
            if load > best_load:
                best, best_load = i, load
        if best < 0:
            break                                        # nothing left that can be split
        ks[best] += 1
    out = []
    for (a, b, n), k in zip(chunks, ks):
        base, rem = divmod(n, k)                         # k near-equal parts summing to n
        for j in range(k):
            out.append((a, b, base + (1 if j < rem else 0)))
    return out


def collection_pairings(team_pool: Sequence[str], n_games: int, *, seed: int = 0, own_team=None,
                        own_mirror_frac: float = 0.2, opp_weights=None) -> List[Tuple[str, str, int]]:
    """The collection batch's ``(team_a, team_b, n)`` plan: symmetric ``team_matchups`` by default;
    with ``own_team`` the W2 own-seat split (``own_team_matchups``)."""
    if own_team:
        return own_team_matchups(own_team, team_pool, n_games, mirror_frac=own_mirror_frac,
                                 weights=opp_weights, seed=seed)
    return team_matchups(team_pool, n_games, seed=seed)


def eval_pairings(kind: str, team_pool: Sequence[str], n_battles: int, *, seed: int = 0,
                  own_team=None) -> List[Tuple[str, str, int]]:
    """The gauntlet's pairing for one opponent ``kind``: symmetric ``team_matchups`` by default.
    With ``own_team`` (W2): the scripted anchors play OWN vs each eval team (uniform, no mirror -
    'judge what will be served'); the model-vs-model kinds (``prev_best`` = the champion mirror,
    and the HoF suspects that reuse it) play OWN vs OWN - both specialists on the same team."""
    if not own_team:
        return team_matchups(team_pool, n_battles, seed=seed)
    n = max(0, int(n_battles))
    if n == 0:
        return []
    own = canonical_own_team(own_team, team_pool)
    if kind == "prev_best":
        return [(own, own, n)]
    return own_team_matchups(own, team_pool, n, mirror_frac=0.0, weights=None, seed=seed)


# ══════════════════════════════════════════════════════════════════════════════
# Async orchestrator (reuses run_local_battle) — exercised by the live smoke
# ══════════════════════════════════════════════════════════════════════════════
# ── saved-replay routing (task E): organise the eval HTML replays by opponent ──
# Scripted opponents -> eval/<kind>/ ; the gen-vs-gen battles (the prev_best promotion mirror AND
# the HoF past-champion battles, both kind='prev_best') -> eval/<LEAGUE_SUBDIR>/ . Change this one
# constant to rename the gen-vs-gen folder (e.g. "championship").
LEAGUE_SUBDIR = "league"
_CKPT_GEN_RE = re.compile(r"gen(\d+)\.pt$")


def _ckpt_gen(path) -> Optional[int]:
    """Generation number parsed from a ``…/gen<N>.pt`` checkpoint path, or None."""
    if not path:
        return None
    m = _CKPT_GEN_RE.search(str(path))
    return int(m.group(1)) if m else None


def eval_replay_routing(kind: str, candidate_gen, *, opp_ref=None):
    """Where a saved eval replay goes + how it's named (task E). Returns ``(subdir, label)``:
      * scripted opponent  -> ``(<kind>, "gen<N>_vs_<kind>")``  e.g. ``("heuristic", "gen3_vs_heuristic")``
      * prev_best / HoF    -> ``(LEAGUE_SUBDIR, "gen<N>_vs_gen<M>")`` where M is the opponent
        checkpoint's generation (``opp_ref``), or ``"champion"`` when it isn't a ``gen<M>.pt``.
    ``candidate_gen`` is the candidate's generation N (``"?"`` if unknown). Pure — offline-tested."""
    n = "?" if candidate_gen is None else candidate_gen
    if kind == "prev_best":
        m = _ckpt_gen(opp_ref)
        opp = f"gen{m}" if m is not None else "champion"
        return LEAGUE_SUBDIR, f"gen{n}_vs_{opp}"
    return kind, f"gen{n}_vs_{kind}"


def _make_opponent(kind: str, username: str, team: str, model_path=None,
                   team_chooser_path=None, max_concurrent_battles: int = 1, port=None):
    """Construct one opponent player of the given ``kind``. ``max_concurrent_battles`` > 1
    enables parallel battles vs this opponent (3c.8c). ``port`` (22f) binds it to the assigned
    pool server (``None`` = poke-env's localhost:8000 default)."""
    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    if kind == "random":
        return R.make_player(username, team, model_path=None,
                             max_concurrent_battles=max_concurrent_battles, port=port)
    if kind in ("max_damage", "heuristic"):
        from v_dance.eval.eval_opponents import MaxDamageVGCPlayer, HeuristicVGCPlayer
        cls = MaxDamageVGCPlayer if kind == "max_damage" else HeuristicVGCPlayer
        _server = {"server_configuration": R.localhost_server_config(port)} if port is not None else {}
        return cls(
            replay_path=_REPO_ROOT / "artifacts" / "replay_buffer" / f"{username}.jsonl",
            account_configuration=AccountConfiguration(username, None),
            battle_format=R.BATTLE_FORMAT, team=team,
            max_concurrent_battles=max_concurrent_battles,
            log_level=logging.WARNING,
            **_server,
        )
    if kind == "prev_best":
        return R.make_player(username, team, model_path=model_path,
                             team_chooser_path=team_chooser_path,
                             max_concurrent_battles=max_concurrent_battles, port=port)
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
    n_workers: int = 1,
    mirror_battles: Optional[int] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    live_dir=None,
    save_replays: bool = False,
    name_salt: str = "",
    own_team=None,
) -> Dict[str, Tuple[int, int]]:
    """Play the model vs each opponent over the rotating team pool and return
    ``{opponent_name: (model_wins, n_finished)}``.

    ``n_workers`` (3c.8c) runs up to that many (opponent, team-pairing) CHUNKS concurrently
    via the shared bounded runner (``play.parallel_battles.run_jobs``, task #13), so the eval
    gate isn't the throughput bottleneck once collection is parallel. Each chunk keeps the
    player at max_concurrent_battles=1, so the total in-flight battle count == ``n_workers``
    (respects the CPU cap). ``stop_check`` (a sync predicate) drains the QUEUED chunks on a soft
    stop, so a mid-eval Ctrl-C during the overnight run doesn't launch battles at a torn-down
    server. The per-kind (wins, finished) accumulation + the source tally run synchronously in
    ``finally`` (no await between the reads/writes), so they're atomic under asyncio interleaving.

    ``name_salt`` (22d) folds the generation (and, for the HoF, the suspect) into the ``BC…`` /
    ``OP…`` account names so they don't repeat across gens — a stale server-side challenge from one
    gen can't collide with the next gen's reuse (``already a challenge between you and OPprev…``).
    ``""`` (the standalone-CLI default) keeps the legacy ``BC{uid}`` / ``OP{kind4}{uid}`` names.

    ``own_team`` (W2, 2026-09-03): the model seat plays THIS team every battle - scripted kinds vs
    each pool team, the champion mirror own-vs-own (``eval_pairings``)."""
    import v_dance.play.run_local_battle as R
    from v_dance.play import parallel_battles as PB
    server = R.start_showdown() if manage_server else None
    acc: Dict[str, list] = {kind: [0, 0] for kind in opponents}   # kind -> [wins, finished]
    source_totals: Counter = Counter()      # model vs retry/default/forfeit fallbacks
    spect = {"open": bool(spectate)}        # open the spectator on the FIRST chunk only

    descriptors = []
    uid = 0
    for kind in opponents:
        # The prev_best/champion MIRROR can run more battles than the scripted anchors
        # (the v2 gate's 70% bar needs >=200 games to be reliable; gate_sim), while the cheap
        # scripted ladder stays small — mirror_battles overrides just that opponent's count.
        nb = mirror_battles if (kind == "prev_best" and mirror_battles) else battles_per_opponent
        _pairs = eval_pairings(kind, team_pool, nb, seed=matchup_seed, own_team=own_team)
        if own_team:                    # W2: own-vs-own is ONE pairing - fill the worker slots
            _pairs = subdivide_pairings(_pairs, max(1, int(n_workers)))
        for model_team_name, opp_team_name, n in _pairs:
            uid += 1
            descriptors.append({"kind": kind, "mt": model_team_name,
                                "ot": opp_team_name, "n": n, "uid": uid})

    cand_gen = _ckpt_gen(ckpt)                         # task E: candidate gen N for replay names

    async def _run(d):
        kind, n, uid = d["kind"], d["n"], d["uid"]
        model_team = R.load_team(R.resolve_team_path(d["mt"]))
        opp_team = R.load_team(R.resolve_team_path(d["ot"]))
        # task E: save this match's HTML under eval/<kind>/ (scripted) or eval/league/ (gen-vs-gen),
        # named gen<N>_vs_<kind|genM>. The live spectate JSON stays flat in live_dir (dashboard).
        _subdir, _label = eval_replay_routing(
            kind, cand_gen, opp_ref=(prev_best_ckpt if kind == "prev_best" else None))
        _rdir = str(Path(live_dir) / _subdir) if (live_dir and save_replays) else None
        model_name, opp_name = PB.eval_account_names(kind, uid, salt=name_salt)  # 22d
        # #audit 2026-06-30: create players INSIDE the try so a raise from _make_opponent (unknown kind /
        # bad prev_best checkpoint) after make_player() succeeds still tears down model_player's socket.
        model_player = opp = None
        try:
            model_player = R.make_player(
                model_name, model_team, model_path=ckpt, team_chooser_path=team_chooser,
                live_dir=live_dir, save_replays=save_replays,   # #18b: eval match spectate
                replay_dir=_rdir, replay_label=_label)
            opp = _make_opponent(
                kind, opp_name, opp_team,
                model_path=prev_best_ckpt, team_chooser_path=team_chooser)
            if spect["open"]:                 # atomic check+clear (no await between)
                spect["open"] = False
                asyncio.ensure_future(_open_spectator(model_player))
            w, f = await PB.play_pairing(model_player, opp, n,
                                         battle_timeout=battle_timeout, label=f"vs {kind}")
            # #01: discount backstop-forfeits exactly like mp_eval (shared helper) — the gauntlet had
            # NO discount, so an opp (champion) loop-guard/abandon forfeit counted as a spurious
            # candidate WIN -> a one-directional PROMOTE bias in the default (collect_procs<=1) gate.
            w, f = PB.discount_forfeits(w, f, model_player, opp)
            acc[kind][0] += w
            acc[kind][1] += f
        finally:
            source_totals.update(getattr(model_player, "_source_counts", {}) or {})
            for k, v in (getattr(model_player, "_tp_source", {}) or {}).items():
                source_totals[f"tp_{k}"] += v          # team-preview tally (#4)
            await PB.close_players(model_player, opp)

    # task #13: run up to n_workers (opponent, team-pairing) chunks at once via the shared
    # bounded runner; stop_check drains the queued backlog on a soft stop (mid-eval Ctrl-C).
    try:
        await PB.run_jobs([lambda d=d: _run(d) for d in descriptors],
                          workers=n_workers, stop_check=stop_check)
    finally:
        if server is not None:
            R.stop_showdown(server)
    results = {kind: (acc[kind][0], acc[kind][1]) for kind in opponents}
    for kind in opponents:
        log.info("vs %s: %d/%d", kind, *results[kind])
    return results, source_totals


# ══════════════════════════════════════════════════════════════════════════════
# Team-pool selection (default = the whole reg pool; pick specific teams / a folder /
# a count via the CLI or a native file-explorer dialog).
# ══════════════════════════════════════════════════════════════════════════════
def _pick_team_files(initialdir: Path) -> List[str]:
    """Native OS multi-select file picker for team pastes — thin alias over the shared
    run_local_battle.pick_team_files (also used by the self-play eval --pick-eval-teams)."""
    import v_dance.play.run_local_battle as R
    return R.pick_team_files(initialdir)


def resolve_team_pool(args) -> List[str]:
    """Decide the gauntlet's rotating team pool from the CLI, in priority order:
      --pick-teams  : native file-explorer multi-select (Cancel -> the discovered pool)
      --teams-dir D : every team paste under D
      --teams a b c : explicit names/paths (the old behaviour)
      (default)     : EVERY team under teams/Champions/<reg> — auto-grows as you add files
    then capped/sampled to --n-teams (deterministic by --matchup-seed) when the pool is bigger.
    """
    import v_dance.play.run_local_battle as R
    from v_dance import formats as _formats
    reg = args.battle_format or _formats.DEFAULT_FORMAT
    if args.pick_teams:
        pool = _pick_team_files(R.CHAMPIONS_DIR)
        if not pool:
            print("[gauntlet] no teams picked -> using the discovered reg pool.")
            pool = R.discover_teams(reg=reg)
    elif args.teams_dir:
        pool = R.discover_teams(root=Path(args.teams_dir))
        if not pool:
            raise SystemExit(f"[gauntlet] no team files found under --teams-dir {args.teams_dir}")
    elif args.teams:
        pool = list(args.teams)
    else:
        pool = R.discover_teams(reg=reg)             # DEFAULT: the whole reg pool
    if not pool:
        raise SystemExit("[gauntlet] no teams found; pass --teams / --teams-dir / --pick-teams, "
                         "or add team files under teams/Champions/.")
    if args.n_teams and len(pool) > args.n_teams:    # cap/sample to N (seeded -> reproducible)
        rng = random.Random(args.matchup_seed)
        pool = sorted(rng.sample(pool, args.n_teams))
    if len(pool) < 2:
        print(f"[gauntlet] WARNING: only {len(pool)} team(s); the side-balanced rotation wants "
              ">=2 (>=4 ideal, else deterministic opponents replay the same few matchups).")
    shown = ", ".join(Path(t).name for t in pool[:12])
    print(f"[gauntlet] team pool: {len(pool)} teams -> {shown}{' ...' if len(pool) > 12 else ''}")
    return pool


# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT, DEFAULT_TP_CHECKPOINT  # shared prod paths
    ap = argparse.ArgumentParser(description="BC win-rate eval gauntlet (#3)")
    ap.add_argument("--battles", "-n", type=int, default=20,
                    help="battles per opponent (split across the team pool)")
    ap.add_argument("--opponents", nargs="+",
                    default=["random", "max_damage", "heuristic"],
                    help="opponent ladder (subset of random/max_damage/heuristic/prev_best)")
    ap.add_argument("--teams", nargs="+", default=None,
                    help="explicit rotating team pool (names under teams/Champions/ or paths). "
                         "DEFAULT (omit this) = EVERY team under teams/Champions/<reg> — the pool "
                         "auto-grows as you add team files. Use >=4 teams: with DETERMINISTIC "
                         "opponents a tiny pool replays the same matchups.")
    ap.add_argument("--teams-dir", default=None,
                    help="use every team paste under this folder as the pool (e.g. a custom set).")
    ap.add_argument("--pick-teams", action="store_true",
                    help="pick the team files via the native OS file explorer "
                         "(multi-select; Cancel falls back to the discovered pool).")
    ap.add_argument("--n-teams", type=int, default=None,
                    help="cap the pool to this many teams (randomly sampled, seeded by "
                         "--matchup-seed so it's reproducible). Default: use the whole pool.")
    ap.add_argument("--ckpt", default=str(DEFAULT_BC_CHECKPOINT))
    ap.add_argument("--team-chooser", default=str(DEFAULT_TP_CHECKPOINT))
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
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many (opponent, team-pairing) chunks concurrently "
                         "(3c.8c); total in-flight battles == workers. Default 1 (sequential)")
    ap.add_argument("--spectate", action="store_true",
                    help="open a browser tab spectating the first battle (live view)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="DEBUG logging (incl. poke-env) for diagnosing stalls")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--timestamp", default=None)
    from v_dance import formats as _formats
    ap.add_argument("--format", default=None, dest="battle_format",
                    help="Champions-doubles format id to eval in (default: active = "
                         f"{_formats.DEFAULT_FORMAT}). SPAWN-SAFE: env-propagated to workers.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from datetime import datetime
    args = parse_args(argv)
    if args.battle_format:                       # select format for this proc + mp workers (env)
        from v_dance import formats as _formats
        if not _formats.is_champions_doubles(args.battle_format):
            raise SystemExit(f"--format {args.battle_format!r} is not a Champions-doubles id "
                             f"(known: {_formats.known_formats()})")
        _formats.set_active_format(args.battle_format)
        import v_dance.play.run_local_battle as _R
        _R.BATTLE_FORMAT = _formats.DEFAULT_FORMAT
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

    team_pool = resolve_team_pool(args)          # default = whole reg pool; --pick-teams/-dir/-n-teams

    results, sources = asyncio.run(run_gauntlet(
        opponents=args.opponents,
        team_pool=team_pool,
        battles_per_opponent=args.battles,
        ckpt=ckpt,
        team_chooser=Path(args.team_chooser),
        prev_best_ckpt=Path(args.prev_best) if args.prev_best else None,
        manage_server=not args.no_server,
        matchup_seed=args.matchup_seed,
        battle_timeout=args.battle_timeout,
        spectate=args.spectate,
        n_workers=args.workers,
    ))

    row = build_run_row(results, ckpt=str(ckpt), run_id=run_id, timestamp=timestamp,
                        extra={"sources": dict(sources)})
    history = load_history(args.history)            # BEFORE appending current
    gate = regression_gate(history, row)
    # audit: a NO-RESULT eval (model_elo None = every scripted anchor had 0 finished battles — total server
    # stall / mask flood / all chunks abandoned) must NOT exit 0. regression_gate folds None into
    # regressed=False, so without this an automated promotion/CI harness reads a catastrophically-broken eval
    # as a clean non-regressing pass. Distinct non-zero code (3) so it's not confused with a real regression (1).
    # audit: also do NOT persist the None-elo row — a poison baseline would make the NEXT good run's
    # regression_gate see history[-1].model_elo==None and skip its comparison ("no baseline").
    if row.get("model_elo") is None:
        print_report(row, gate)
        print_sources(sources)
        if any(o in ANCHORS for o in (args.opponents or [])):   # anchors WERE requested but all stalled
            print("[gauntlet] FATAL: no measurable scripted result (model_elo is None — every requested "
                  "anchor had 0 finished battles). Not a clean pass; NOT recorded to history.", file=sys.stderr)
            return 3
        # no scripted anchors requested (e.g. --opponents prev_best): a VALID anchor-free run. model_elo
        # is undefined by design — show the mirror results but do NOT append a None-elo poison baseline.
        print("[gauntlet] anchor-free run (no scripted anchors requested): model_elo not measured; "
              "mirror results above, history NOT updated.", file=sys.stderr)
        return 0
    append_run(args.history, row)
    print_report(row, gate)
    print_sources(sources)
    return 1 if gate["regressed"] else 0


if __name__ == "__main__":
    sys.exit(main())
