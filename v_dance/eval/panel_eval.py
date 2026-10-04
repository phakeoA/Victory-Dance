"""Fresh, PARALLEL head-to-head re-test of candidate checkpoints vs one opponent (2026-10-01).

Picking the best generation by its in-run panel score carries selection luck (the winner's curse); this plays each
candidate a FRESH sample (a new matchup seed; the SAME pairings for every candidate, so they compare paired) against
the opponent, fanned across a worker-process pool + a Showdown server pool — the self-play eval machinery
(``mp_eval.eval_with_pool``'s panel path), so 400 games take minutes, not the single-process gauntlet's ~20 min.

Two team shapes: ``random`` (both sides random pool teams = general skill) and ``own`` (the candidate always plays
``--own-team`` vs random opponent teams = the LADDER shape). Put the opponent itself in ``--candidates`` as the
control: in ``random`` it must read ~50 % (a tool-bias check), in ``own`` it is the baseline the others must beat.

    .venv/Scripts/python.exe -X utf8 -m v_dance.eval.panel_eval --opponent era2=<ckpt> \\
        --candidates era2=<ckpt> g42=<ckpt> ... --battles 400 --modes random own --own-team Baltimore_Sand_Psy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def wilson(w: int, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


def resolve_candidate_tp(cands: dict, candidate_tp, paired=None):
    """2026-10-03 USER RULE — the PAIR ("treat them like 1 neural network"): ``(picker for the candidate seat, note)``.
    An explicit ``--candidate-tp`` wins ('default' = the eval default picker, explicitly → None). Unset: a candidate
    with a co-trained picker (its verified pairing sidecar) plays WITH it; candidates without one keep the eval
    default. The candidate picker is ONE setting for the whole eval, so candidates that need DIFFERENT pickers raise
    ValueError (evaluate them one per call). ``paired`` is injectable for tests."""
    if candidate_tp is not None:
        return (None, "candidate team picker: the eval default (explicit)") \
            if str(candidate_tp).strip().lower() == "default" else (candidate_tp, "")
    if paired is None:
        from v_dance.selfplay.tp_learning import paired_tp_for as paired
    pairs = {n: paired(p) for n, p in cands.items()}
    got = {v for v in pairs.values() if v}
    if not got:
        return None, ""
    if len(got) > 1 or any(v is None for v in pairs.values()):
        raise ValueError("the PAIR rule: these candidates play with DIFFERENT pickers "
                         f"({ {n: (Path(v).name if v else 'eval default') for n, v in pairs.items()} }) — evaluate one "
                         "candidate per call, or pass --candidate-tp")
    tp = got.pop()
    return tp, f"the PAIR rule: the candidate plays with its co-trained picker {tp}"


def filter_pool(teams, own_team, has=None, lacks=None, species_of=None):
    """2026-10-03: keep the OPPONENT pool teams whose sheet shows any of ``has`` (all when empty) and none of
    ``lacks``; the own team always stays (canonical_own_team needs it in the pool). ``species_of(team) ->
    [species]`` is injectable for tests; default = the team paste parsed with the live encoder's parser."""
    from v_dance.dex.pokedex import norm_species
    if not has and not lacks:
        return list(teams)
    if species_of is None:
        import v_dance.play.run_local_battle as R
        from v_dance.encoders.live_state_encoder import team_species_from_paste

        def species_of(t):
            return team_species_from_paste(Path(R.resolve_team_path(t)).read_text(encoding="utf-8"))
    want = {norm_species(s) for s in (has or ())}
    ban = {norm_species(s) for s in (lacks or ())}
    own = Path(str(own_team)).name.lower() if own_team else None
    out = []
    for t in teams:
        if own and Path(str(t)).name.lower() == own:
            out.append(t)
            continue
        sp = {norm_species(s) for s in species_of(t)}
        if (not want or sp & want) and not (sp & ban):
            out.append(t)
    return out


def main(argv=None) -> int:
    from v_dance.selfplay.generation import parse_panel
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", required=True, metavar="NAME=CKPT")
    ap.add_argument("--opponent", required=True, metavar="NAME=CKPT")
    ap.add_argument("--battles", type=int, default=400, help="games per candidate per mode")
    ap.add_argument("--modes", nargs="+", default=["random"], choices=["random", "own"])
    ap.add_argument("--own-team", default=None, help="the candidate's team in mode 'own' (the ladder team)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--async-per-proc", type=int, default=3)
    ap.add_argument("--servers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1000, help="matchup seed (self-play runs use 0 + generation)")
    ap.add_argument("--out", default=str(_REPO / "scratch" / "panel_eval.json"))
    # 2026-10-03 diagnostics (USER: teach the Rillaboom lesson to both nets)
    ap.add_argument("--opp-has", nargs="+", default=None, metavar="SPECIES",
                    help="keep only opponent pool teams showing ANY of these species (own mode)")
    ap.add_argument("--opp-lacks", nargs="+", default=None, metavar="SPECIES",
                    help="drop opponent pool teams showing ANY of these species")
    ap.add_argument("--candidate-tp", default=None,
                    help="the CANDIDATE's team picker: 'none' = the first-4 roster heuristic (the self-play "
                         "learner's team preview), a path = that picker, 'default' = the eval default picker; "
                         "unset = the PAIR rule (2026-10-03): the candidate's co-trained picker when it has one, "
                         "else the eval default")
    ap.add_argument("--candidate-rules", default=None,
                    help="comma-separated matchup rules for the CANDIDATE (e.g. rillaboom_salamence; needs a picker)")
    a = ap.parse_args(argv)
    if "own" in a.modes and not a.own_team:
        ap.error("--modes own needs --own-team")
    cands, opp = parse_panel(a.candidates), parse_panel([a.opponent])
    import os
    if a.candidate_rules:
        from v_dance.play.matchup_rules import RULES
        bad = [r for r in a.candidate_rules.split(",") if r.strip() and r.strip() not in RULES]
        if bad:
            ap.error(f"unknown matchup rule(s) {bad}; known: {sorted(RULES)}")
        if (a.candidate_tp or "").strip().lower() == "none":
            ap.error("--candidate-rules act through the team picker — drop --candidate-tp none")
    try:                                               # 2026-10-03 USER RULE: a pair plays as a pair
        a.candidate_tp, _note = resolve_candidate_tp(cands, a.candidate_tp)
    except ValueError as exc:
        ap.error(str(exc))
    if _note:
        print(f"[panel_eval] {_note}")
    # set BEFORE the worker pool exists: spawned workers inherit the environment (mp_eval.candidate_overrides)
    if a.candidate_tp:
        os.environ["VD_EVAL_CANDIDATE_TP"] = a.candidate_tp
    if a.candidate_rules:
        os.environ["VD_EVAL_CANDIDATE_RULES"] = a.candidate_rules

    import v_dance.play.run_local_battle as R
    from v_dance.formats import default_format
    from v_dance.play.model_io import DEFAULT_TP_CHECKPOINT
    from v_dance.selfplay import mp_collect as MP
    from v_dance.selfplay import mp_eval as ME
    teams = sorted(R.discover_teams(reg=default_format()))
    if a.opp_has or a.opp_lacks:
        _n0 = len(teams)
        teams = filter_pool(teams, a.own_team, a.opp_has, a.opp_lacks)
        print(f"[panel_eval] opponent pool filter has={a.opp_has} lacks={a.opp_lacks}: {_n0} -> {len(teams)} teams "
              f"(own team kept)")
        if len(teams) < 2:
            ap.error("the opponent filter left no opponent team")
    (oname, _), = opp.items()
    print(f"[panel_eval] {len(cands)} candidate(s) x {a.modes} x {a.battles} games vs {oname}; {len(teams)} "
          f"{default_format()} teams; {a.procs} procs x {a.async_per_proc} async, {a.servers} server(s); seed {a.seed}")
    from v_dance.encoders.battle_mechanics import terrain_values_banner
    print("[panel_eval] " + terrain_values_banner())          # 2026-10-02 (VD_TERRAIN_V19D)
    print(f"[panel_eval] candidate team picker: {a.candidate_tp or 'default (' + str(DEFAULT_TP_CHECKPOINT.name) + ')'}"
          f" · candidate matchup rules: {a.candidate_rules or 'none'}")
    pool = MP.CollectionPool(a.procs)
    servers = R.ServerPool(max(1, a.servers), manage=True).start_all()
    ports = servers.ports if a.servers > 1 else None
    rows = []
    try:
        for mode in a.modes:
            for i, (name, ck) in enumerate(cands.items()):
                t0 = time.perf_counter()
                res, _src = ME.eval_with_pool(
                    ck, opponents=[], team_pool=teams, battles_per_opponent=0, team_chooser=str(DEFAULT_TP_CHECKPOINT),
                    submit_fn=pool.submit, n_procs=a.procs, async_per_proc=a.async_per_proc,
                    generation=900 + 10 * a.modes.index(mode) + i, ports=ports, matchup_seed=a.seed,
                    panel=opp, panel_battles=a.battles, panel_team_pool=teams,
                    panel_own_team=(a.own_team if mode == "own" else None))
                w, f = res.get(ME.PANEL_PREFIX + oname, (0, 0))
                lo, hi = wilson(w, f)
                dt = time.perf_counter() - t0
                rows.append({"mode": mode, "candidate": name, "ckpt": ck, "wins": w, "games": f,
                             "rate": (w / f) if f else None, "ci95": [lo, hi], "secs": round(dt, 1)})
                print(f"  [{mode:6s}] {name:10s} vs {oname}: {w:4d}/{f:<4d} = {w / max(1, f) * 100:5.1f}%  "
                      f"95% CI [{lo * 100:4.1f}, {hi * 100:4.1f}]  ({f / dt * 60 if dt else 0:.0f} games/min)", flush=True)
    finally:
        pool.close()
        servers.stop_all()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"opponent": opp, "seed": a.seed, "rows": rows}, indent=2), encoding="utf-8")
    print(f"[panel_eval] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
