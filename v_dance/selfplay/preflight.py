"""Self-play PREFLIGHT (2026-09-30) — catch, in minutes, the errors that used to surface hours into a run.

The old smoke (``--generations 1``) never reached the code that only runs from generation 1 on: the
past-snapshot opponents, the champion mirror + promote, the HoF, server recycling, the per-gen save and
the RESUME. The preflight runs the REAL launch path with the operator's own flags, shrunk:

  * 3 generations, then a RESUMED 4th (``--resume-gen latest``) from the snapshot the first part wrote;
  * every opponent in ROTATION — latest, a past snapshot, EACH clone checkpoint, each scripted anchor;
  * the gate loosened so later generations promote (champion mirror, HoF, arm registration into a
    SANDBOX copy of the bandit config — never the live one), server recycling every 2 gens;
  * a throwaway archive under ``artifacts/preflight/<stamp>/`` (removed on PASS, kept on FAIL).

Then it CHECKS what actually happened. Collection swallows per-chunk errors (a warning, the run goes
on), so a broken clone checkpoint would otherwise just play zero games all night — the preflight
counts finished games per opponent (``OpponentLeague.played``) and fails on any zero."""
from __future__ import annotations

import copy
import shutil
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_DIR = _REPO_ROOT / "artifacts" / "preflight"
PREFLIGHT_GENS = 3
# Promote whenever the candidate wins ANY mirror game; never revert (a 4-game sample is noise).
_GATE_LOOSE = {"promote_threshold": 0.0, "promote_z": -1e6, "min_h2h_games": 1,
               "mirror_collapse_min_games": 10 ** 9, "floor_margin": 1.0}


def wanted(args) -> bool:
    """``--preflight on|off|auto`` (auto = on for runs longer than the preflight itself)."""
    if getattr(args, "preflight_only", False):
        return True
    mode = getattr(args, "preflight", "auto") or "auto"
    if mode in ("on", "off"):
        return mode == "on"
    g = int(getattr(args, "generations", 0) or 0)
    return g <= 0 or g > PREFLIGHT_GENS


def _n_clones(args) -> int:
    return len(getattr(args, "league_clones", None) or []) if (getattr(args, "clone_frac", 0) or 0) > 0 else 0


def preflight_args(args, root: Path):
    """The operator's namespace, shrunk to the preflight run (pure: no IO)."""
    a = copy.copy(args)
    a.generations = PREFLIGHT_GENS
    a.games = _n_clones(args) + 8          # >= one chunk per opponent per gen, with slack for grouped mirrors
    a.eval_battles = 2
    a.mirror_battles = 4
    a.allow_undersized_mirror = True
    a.hof_games = 2
    a.warmup = min(int(getattr(args, "warmup", 1) or 1), 1)
    a.archive = str(root)
    a.resume = None
    a.resume_gen = None
    a.hours = None
    a.stop_after_stale = 0
    a.restart_server_every = 2
    a.save_replays = False
    a.keep_snapshots = 0
    a.cover_kinds = True
    a.panel_battles = 4                    # one 4-game pairing vs each panel checkpoint per gen
    a.panel_patience = 0
    a.preflight = "off"
    a.preflight_only = False
    a.run_cfg_gate = {**(getattr(args, "run_cfg_gate", None) or {}), **_GATE_LOOSE}
    if getattr(args, "register_arms", False):
        a.bandit_config = str(root / "serve_bandit_preflight.json")
        a.register_prefix = "preflight_g"      # a copied live config may already hold era5b_g1 etc.
    return a


def resume_args(a):
    b = copy.copy(a)
    b.generations = 1
    b.resume_gen = "latest"
    return b


def _short(key: str) -> str:
    if key.startswith("clone:"):
        p = Path(key[len("clone:"):])
        return f"clone:{p.parent.name}/{p.name}" if p.parent.name else f"clone:{p.name}"
    return key


def check(fresh: Optional[dict], resumed: Optional[dict], *, hof_on: bool = False,
          arm_names: Optional[List[str]] = None, register_on: bool = False,
          panel_names: Optional[List[str]] = None) -> Tuple[List[str], List[str], dict]:
    """``(problems, warnings, played)`` from the two run results. Any problem = FAIL."""
    problems: List[str] = []
    warns: List[str] = []
    if not fresh or not fresh.get("history"):
        return ["the preflight run returned nothing"], warns, {}
    recs = fresh["history"].records
    if len(recs) < PREFLIGHT_GENS:
        problems.append(f"only {len(recs)} of {PREFLIGHT_GENS} generations ran")
    for r in recs:
        if r.n_trajectories <= 0:
            problems.append(f"gen {r.generation}: collected no trajectories")
        if "loss" not in (r.update_stats or {}):
            problems.append(f"gen {r.generation}: the PPO update did not run")
    if not any(r.promoted for r in recs[1:]):
        warns.append("no promotion after gen 0 — the champion-mirror promote path was not exercised")

    rrecs = resumed["history"].records if resumed and resumed.get("history") else []
    if len(rrecs) != len(recs) + 1 or (rrecs and rrecs[-1].generation != len(recs)):
        problems.append(f"the resume did not continue the run (history {len(recs)} -> {len(rrecs)} records)")
    elif "loss" not in (rrecs[-1].update_stats or {}):
        problems.append("resumed gen: the PPO update did not run")

    played: dict = {}
    expected: List[str] = []
    for res in (fresh, resumed):
        lg = (res or {}).get("league")
        if lg is None:
            continue
        for k, v in getattr(lg, "played", {}).items():
            played[k] = played.get(k, 0) + int(v)
        expected += [k for k in lg.opponent_keys() if k not in expected]
    for k in expected:
        if played.get(k, 0) <= 0:
            problems.append(f"opponent {_short(k)} finished 0 games (its chunks failed — see the warnings above)")

    for r in recs + rrecs[len(recs):]:
        for name in panel_names or ():
            if int(((r.panel or {}).get(name) or (0, 0))[1]) <= 0:
                problems.append(f"gen {r.generation}: panel opponent {name} finished 0 games")
    if hof_on and not any((r.hof or {}).get("reason") not in (None, "thin_pool_skip") for r in recs + rrecs):
        warns.append("the Hall-of-Fame check never ran (needs 2 past champions) — not exercised")
    if register_on and not arm_names:
        warns.append("--register-arms: no arm reached the sandbox bandit config")
    return problems, warns, played


def _registered_arms(path: Path) -> List[str]:
    try:
        import json
        d = json.loads(path.read_text(encoding="utf-8"))
        arms = d.get("arms", d) if isinstance(d, dict) else d
        return [a.get("name") if isinstance(a, dict) else str(a) for a in arms] if isinstance(arms, list) \
            else list(arms.keys()) if isinstance(arms, dict) else []
    except Exception:
        return []


def run_preflight(args, launch_fn) -> bool:
    """Run the shrunk launch twice (fresh 3 gens + a resumed 4th) and check it. True = PASS."""
    root = PREFLIGHT_DIR / time.strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    a = preflight_args(args, root)
    arm_file = Path(a.bandit_config) if getattr(args, "register_arms", False) else None
    arms_before: set = set()
    if arm_file is not None:                           # sandbox copy of the live config — never the real one
        try:
            from v_dance.ladder.update import DEFAULT_BANDIT_CONFIG
            src = Path(getattr(args, "bandit_config", None) or DEFAULT_BANDIT_CONFIG)
            if src.exists():
                shutil.copy2(src, arm_file)
            arms_before = set(_registered_arms(arm_file))
        except Exception:
            traceback.print_exc()
    print("\n" + "=" * 78)
    print(f"PREFLIGHT: {PREFLIGHT_GENS} gens x {a.games} games + 1 resumed gen, every opponent in rotation, "
          f"gate loosened ON PURPOSE (ignore its 'undersized mirror' warning)\n  archive: {root}")
    print("=" * 78)
    t0 = time.perf_counter()
    fresh = resumed = None
    try:
        fresh = launch_fn(a)
        print("\n[preflight] --- resuming from the snapshot just written ---")
        resumed = launch_fn(resume_args(a))
    except (Exception, SystemExit):                    # SystemExit: a launch-time sys.exit(2) is a FAIL too
        traceback.print_exc()
        print(f"\n[preflight] FAIL — the run CRASHED (traceback above). Nothing long was started. "
              f"Artifacts kept: {root}")
        return False
    arms = _registered_arms(arm_file) if arm_file is not None else []
    problems, warns, played = check(fresh, resumed, hof_on=bool(getattr(args, "hof", False)),
                                    arm_names=[n for n in arms if n not in arms_before],
                                    register_on=arm_file is not None,
                                    panel_names=[str(p).partition("=")[0] for p in (getattr(args, "panel", None) or [])])
    mins = (time.perf_counter() - t0) / 60.0
    print("\n" + "=" * 78)
    print(f"PREFLIGHT {'PASS' if not problems else 'FAIL'} in {mins:.1f} min")
    print("  games finished per opponent: " + ", ".join(f"{_short(k)}={v}" for k, v in sorted(played.items())))
    for p in problems:
        print(f"  ✗ {p}")
    for w in warns:
        print(f"  ! {w}")
    if not problems:
        if root.resolve().parent == PREFLIGHT_DIR.resolve():   # only ever its own throwaway folder
            shutil.rmtree(root, ignore_errors=True)
        print("  -> preflight only: done." if getattr(args, "preflight_only", False)
              else "  -> starting the real run.")
    else:
        print(f"  -> the real run was NOT started. Artifacts kept: {root}")
    print("=" * 78 + "\n")
    return not problems
