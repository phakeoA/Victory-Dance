"""Nightly ladder PPO update — W3b-2 (2026-09-02). THE USER LAUNCHES THIS (box quiet).

    .venv/Scripts/python.exe -m v_dance.ladder.ppo_update --dry-run          # what would train, and why not
    .venv/Scripts/python.exe -m v_dance.ladder.ppo_update                    # train + in-process gates + save
    .venv/Scripts/python.exe -m v_dance.ladder.ppo_update --run-gates --register   # + ruler/type-eff, then the arm
    (moved from scratch/ladder_ppo_update.py in the 2026-09-10 refactor, Phase 1; the flags are unchanged)

Defaults: base = the bandit's incumbent checkpoint, arms = every τ > 0 arm that plays it, the last
1 day of ``artifacts/ladder_rl/<format>/``, τ = the one with the most turn steps, the design §5
recipe. Output ``ai_train_scripts/BC_model/checkpoints_attn_ladder_ppo_<YYYYMMDD>/battle_base.pt``
(+ ``ladder_ppo_meta.json``, ``gates/``). ``--register`` appends arm ``ppo_<YYYYMMDD>`` to
``config/serve_bandit.json`` ONLY when every gate that ran passed — then restart the bot (lanes).
A registered arm is ALSO deployed to ``.env VD_BATTLE_CKPT`` (2026-09-04, USER: the newest PPO model
is the .env default; the old value stays as a commented rollback line) — ``--no-deploy-env`` skips that.
Mission Control's Train tab carries this command as a card ("W3b chain update"); the bot must be DOWN.
``--twin`` (2026-09-04, design B3) also registers ``ppo_<stamp>_t0`` — the new head at τ 0 next to the incumbent,
the previous twin benched — because the learning arm's own record can only ever measure the τ-0.3 cost.

Exit codes: 0 = candidate saved (and registered if asked) · 2 = refused (no trainable data) ·
3 = gates failed (candidate saved, NOT registered).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[2]          # v_dance/ladder/ -> repo root
assert (_REPO / "pyproject.toml").is_file(), _REPO

from v_dance.ladder import update as LU   # noqa: E402


def _default_base(arms: dict):
    from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT
    from v_dance.play.serve_bandit import _resolve
    inc = next((a for a in arms.values() if a.incumbent), None)
    if inc is not None and not inc.uses_default("battle"):
        return _resolve(inc.battle_ckpt)
    return Path(DEFAULT_BC_CHECKPOINT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, help="checkpoint to warm-start from (default: the bandit incumbent); "
                    "'learning' = the LEARNING arm's checkpoint (L3 chain mode: train from last night's arm on its games, "
                    "register the result as the new learning arm, bench the old one)")
    ap.add_argument("--anchor", default=None, help="checkpoint for the ABSOLUTE ruler floor (default: the incumbent's "
                    "in chain mode, none otherwise)")
    ap.add_argument("--no-rotate", action="store_true", help="chain mode: keep the previous learning arm served (default: bench it)")
    ap.add_argument("--twin", action="store_true",
                    help="after a successful --register, ALSO register ppo_<stamp>_t0 = the same checkpoint at tau 0 "
                         "(argmax) next to the incumbent and bench the previous twin — the clean read of the chain's "
                         "WEIGHTS on the ladder (the learning arm's record carries the tau-0.3 sampling cost)")
    ap.add_argument("--arms", nargs="*", default=None, help="τ arms to learn from (default: every τ>0 arm on the base)")
    ap.add_argument("--config", default=str(LU.DEFAULT_BANDIT_CONFIG))
    ap.add_argument("--fmt", default=None, help="format folder under artifacts/ladder_rl (default: the served format)")
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--tau", default="auto", help="'auto' (most turn steps) or a number")
    ap.add_argument("--min-steps", type=int, default=200, help="refuse below this many turn steps")
    ap.add_argument("--out", default=None, help="candidate dir (default: checkpoints_attn_ladder_ppo_<date>)")
    ap.add_argument("--name", default=None, help="bandit arm name (default: ppo_<date>)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="select + report only; no training")
    ap.add_argument("--run-gates", action="store_true", help="also run the ruler + type-eff probe (minutes)")
    ap.add_argument("--register", action="store_true", help="append the arm to the bandit config when the gates pass")
    ap.add_argument("--force-register", action="store_true", help="register even when a gate failed (NOT recommended)")
    ap.add_argument("--regate", default=None, metavar="CANDIDATE_DIR",
                    help="re-run the gates on an EXISTING candidate dir (no selection, no training) and register it when "
                         "they pass — for a gate that CRASHED rather than failed (2026-09-05: the era2 anchor gate's "
                         "val-report died on a 199 MiB allocation with the box at its commit limit). Only the gates that "
                         "did not pass are re-run; the passed ones are kept from ladder_ppo_meta.json")
    ap.add_argument("--no-deploy-env", action="store_true",
                    help="after a successful --register, do NOT point .env VD_BATTLE_CKPT at the new checkpoint "
                         "(default: deploy it — the newest PPO model becomes the .env default battle net)")
    ap.add_argument("--env-path", default=str(_REPO / ".env"), help=argparse.SUPPRESS)
    for k, v in LU.RECIPE.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None, help=f"recipe override (default {v})")
    for k, v in LU.GATES.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):      # a cp1252 console must never crash the report
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    arms_cfg = LU.arm_table(args.config)
    regate_dir = Path(args.regate).resolve() if args.regate else None
    if regate_dir is not None:                    # 2026-09-05: gates + registration only, on an existing candidate
        tail = regate_dir.name.rsplit("ladder_ppo_", 1)[-1] if "ladder_ppo_" in regate_dir.name else ""
        if not (len(tail) in (8, 9) and tail[:8].isdigit()) or not (regate_dir / "battle_base.pt").is_file() \
                or not (regate_dir / "ladder_ppo_meta.json").is_file():
            print(f"[ladder-ppo] --regate: {regate_dir} is not a candidate dir (needs battle_base.pt + "
                  "ladder_ppo_meta.json under a checkpoints_attn_ladder_ppo_<stamp> folder)")
            return 2
        stamp = tail
        print(f"[ladder-ppo] REGATE {regate_dir.name}: no selection, no training — the gates that did not pass are "
              f"re-run, the passed ones kept; ppo_{stamp} is registered when every gate passes")
    else:
        stamp = LU.next_run_stamp(time.strftime("%Y%m%d"), arms_cfg.keys())   # a 2nd same-day run gets '…b'
        if not stamp.isdigit():
            print(f"[ladder-ppo] same-day run: arm ppo_{stamp} / candidate checkpoints_attn_ladder_ppo_{stamp} "
                  f"(today's earlier run keeps its name and folder)")
    chain = (args.base or "").strip().lower() == "learning"
    if chain:
        from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT
        try:
            base = LU.learning_base(arms_cfg, default_battle=Path(DEFAULT_BC_CHECKPOINT))
        except ValueError as exc:
            print(f"[ladder-ppo] chain mode REFUSED: {exc}")
            return 2
        print(f"[ladder-ppo] chain mode: base = the learning arm {LU.learning_arms(arms_cfg)[0]}'s checkpoint")
    else:
        base = Path(args.base) if args.base else _default_base(arms_cfg)
    if not base.is_file():
        print(f"[ladder-ppo] base checkpoint not found: {base}")
        return 2
    anchor = Path(args.anchor) if args.anchor else (_default_base(arms_cfg) if chain else None)
    if anchor is not None and (not anchor.is_file() or anchor.resolve() == base.resolve()):
        anchor = None                                 # nothing to anchor against (first chain step from the incumbent)
    fmt = args.fmt
    if fmt is None:
        from v_dance.formats import DEFAULT_FORMAT
        fmt = DEFAULT_FORMAT
    if regate_dir is not None:                    # 2026-09-05: pick the candidate up where the crashed gate left it
        report = json.loads((regate_dir / "ladder_ppo_meta.json").read_text(encoding="utf-8"))
        ss = report.get("selection") or {}
        sel = SimpleNamespace(n_games=int(ss.get("games") or report.get("n_games") or 0),
                              n_turn_steps=int(ss.get("turn_steps") or 0), per_arm=dict(ss.get("per_arm") or {}),
                              tau=float(ss.get("tau") if ss.get("tau") is not None else (report.get("tau") or 0.3)))
        g = report.get("gates") or {}
        ok, fails, warns = bool(g.get("in_process_ok")), list(g.get("failures") or []), list(g.get("warnings") or [])
        prev = dict(g.get("external") or {})
        ckpt = regate_dir / "battle_base.pt"
        print(f"[ladder-ppo] candidate {ckpt}  (in-process gates {'ok' if ok else 'FAILED'}; earlier external verdicts: "
              + (", ".join(f"{k[:-3]} {'ok' if v else 'FAIL'}" for k, v in prev.items() if k.endswith("_ok")) or "none")
              + ")")
        return _gates_and_register(args, base=base, anchor=anchor, chain=chain, arms_cfg=arms_cfg, stamp=stamp,
                                   out_dir=regate_dir, ckpt=ckpt, report=report, sel=sel, ok=ok, warns=warns,
                                   fails=fails, prev_ext=prev)

    arms = args.arms if args.arms else LU.exploration_arms(arms_cfg, base)
    from v_dance.encoders.state_encoder import get_state_dim
    files = LU.trajectory_files(LU.LADDER_RL_DIR, fmt, days=args.days)
    print(f"[ladder-ppo] base {base}\n[ladder-ppo] arms {arms}  window {args.days:g} d  format {fmt}")
    sel = LU.select_trajectories(files, base=base, arms=arms, arm_table=arms_cfg, days=args.days,
                                 tau=args.tau, expected_state_dim=get_state_dim())
    print("[ladder-ppo] selection\n" + LU.format_selection(sel))
    if sel.n_games == 0:
        print("[ladder-ppo] REFUSED — no trainable games (see the skipped reasons above)")
        return 2
    if sel.n_turn_steps < args.min_steps:
        print(f"[ladder-ppo] REFUSED — {sel.n_turn_steps} turn steps < --min-steps {args.min_steps}")
        return 2
    over = {k: getattr(args, k) for k in LU.RECIPE}
    ppo_cfg, train_cfg = LU.build_configs(sel, **over)
    print("[ladder-ppo] recipe  " + json.dumps({**{k: v for k, v in vars(ppo_cfg).items()
                                                    if k in ("tau", "clip_eps", "entropy_coef", "kl_coef",
                                                             "pair_decode", "gimmick_terms", "replacement_policy")},
                                                 **{k: v for k, v in vars(train_cfg).items()
                                                    if k in ("actor_lr", "critic_lr", "ppo_epochs", "minibatch_size",
                                                             "target_kl_from_bc", "approx_kl_stop",
                                                             "actor_weight_decay", "backbone_lr_scale")}}))
    # 2026-09-04 B2: games against stronger opponents weigh more (centred on the batch mean; 0 = off)
    scale = over.get("opp_weight_scale")
    tw, tw_info = LU.opp_rating_weights(sel, scale=(LU.RECIPE["opp_weight_scale"] if scale is None else float(scale)))
    print(f"[ladder-ppo] opp-rating weights  scale {tw_info['scale']:g}  known {tw_info['n_known']}/{sel.n_games}"
          f"  mean opp {tw_info.get('mean_opp_rating')}  w mean/min/max {tw_info.get('w_mean')}/{tw_info.get('w_min')}"
          f"/{tw_info.get('w_max')}")
    if args.dry_run:
        print("[ladder-ppo] dry run — nothing trained")
        return 0

    t0 = time.time()
    ac, report = LU.run_update(base, sel, ppo_cfg, train_cfg, device=args.device, seed=args.seed,
                               warmup_updates=int(over.get("warmup_updates") or LU.RECIPE["warmup_updates"]),
                               traj_weights=tw)
    report["selection"] = sel.summary()
    report["opp_weights"] = tw_info
    report["elapsed_s"] = round(time.time() - t0, 1)
    u = report["update"]
    print(f"[ladder-ppo] update  steps {report['n_steps']}  epochs {u.get('epochs_run')}  "
          f"loss {u.get('loss', float('nan')):.4f}  ratio {u.get('ratio_mean', float('nan')):.3f}  "
          f"approx_kl {u.get('approx_kl_old_new', float('nan')):.4f}  KL-to-base {report['kl_to_base_after']:.4f}  "
          f"EV {report['explained_variance']:.3f}  pair_flips {report.get('pair_flips')}  "
          f"halted {report['halted']} ({report['halt_reason']})  {report['elapsed_s']} s")
    ok, fails, warns = LU.gate(report, max_kl=args.max_kl, min_ev=args.min_ev, max_pair_flips=args.max_pair_flips)
    report["gates"] = {"in_process_ok": ok, "failures": fails, "warnings": warns}
    out_dir = Path(args.out) if args.out else (LU.CKPT_ROOT / f"checkpoints_attn_ladder_ppo_{stamp}")
    ckpt = LU.save_candidate(ac, out_dir, report)
    print(f"[ladder-ppo] candidate -> {ckpt}")
    for w in warns:
        print(f"[ladder-ppo] warning: {w}")
    for f in fails:
        print(f"[ladder-ppo] GATE FAILED: {f}")
    del ac                                         # 2026-09-05: the trained nets are on disk — free them before the gate
    gc.collect()                                   # subprocesses run (they need the commit charge the trainer held)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return _gates_and_register(args, base=base, anchor=anchor, chain=chain, arms_cfg=arms_cfg, stamp=stamp,
                               out_dir=out_dir, ckpt=ckpt, report=report, sel=sel, ok=ok, warns=warns, fails=fails)


_EXT_KEY_SUFFIXES = ("_delta_pp", "_ok", "_note", "_returncode", "_verdict", "_retries")


def _gates_and_register(args, *, base, anchor, chain, arms_cfg, stamp, out_dir, ckpt, report, sel, ok, warns, fails,
                        prev_ext=None) -> int:
    """The tail shared by a fresh run and ``--regate``: the external gates (with ``prev_ext`` — the earlier verdicts
    from ladder_ppo_meta.json — only the gates that did not pass are re-run), the meta file, then registration /
    rotation / twin / .env deploy exactly as before."""
    out_dir = Path(out_dir)
    cmds = LU.external_gate_commands(base, ckpt, out_dir)
    if args.run_gates:
        ext = dict(prev_ext or {})
        which = None
        if prev_ext:                                    # 2026-09-05 --regate: keep what passed, re-run the rest
            all_gates = ("ruler", "type_eff") + (("ruler_anchor",) if anchor is not None else ())
            which = tuple(n for n in all_gates if not prev_ext.get(f"{n}_ok"))
            print(f"[ladder-ppo] gates to re-run: {', '.join(which) or 'none (every gate passed before)'}")
            for n in which:
                for suf in _EXT_KEY_SUFFIXES:
                    ext.pop(f"{n}{suf}", None)
        if which is None or which:
            fresh = LU.run_external_gates(base, ckpt, out_dir, ruler_floor_pp=args.ruler_floor_pp,
                                          anchor=anchor, ruler_abs_floor_pp=args.ruler_abs_floor_pp, which=which)
            ext.update(fresh)
        report.setdefault("gates", {})["external"] = ext
        print(f"[ladder-ppo] ruler delta {ext.get('ruler_delta_pp')} pp (floor {args.ruler_floor_pp}) "
              f"-> {'ok' if ext.get('ruler_ok') else 'FAIL'};  type-eff {ext.get('type_eff_verdict')} "
              f"-> {'ok' if ext.get('type_eff_ok') else 'FAIL'}")
        if anchor is not None:                          # L3: the absolute floor vs the incumbent
            print(f"[ladder-ppo] ruler vs anchor {LU.repo_relative(anchor)}: {ext.get('ruler_anchor_delta_pp')} pp "
                  f"(floor {args.ruler_abs_floor_pp}) -> {'ok' if ext.get('ruler_anchor_ok') else 'FAIL'}")
        for k in ("ruler_note", "type_eff_note", "ruler_anchor_note"):   # a gate that did not RUN says so
            if ext.get(k):
                print(f"[ladder-ppo] {k.split('_note')[0]}: {ext[k]}  (full output: {out_dir / 'gates'})")
        ok = ok and LU.external_ok(ext)
        (out_dir / "ladder_ppo_meta.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    else:
        print("[ladder-ppo] external gates NOT run (add --run-gates) — the commands:\n  " +
              "\n  ".join(cmds.values()))
    print(f"[ladder-ppo] suite: {cmds['suite']}")
    if args.register or args.force_register:
        if ok or args.force_register:
            name = args.name or f"ppo_{stamp}"
            note = (f"W3b-2 nightly ladder PPO {stamp}: base {LU.repo_relative(base)}, {sel.n_games} games / "
                    f"{sel.n_turn_steps} turn steps from {sorted(sel.per_arm)} at tau {sel.tau:g}; KL-to-base "
                    f"{report['kl_to_base_after']:.3f}, EV {report['explained_variance']:.2f}"
                    + ("" if ok else " — FORCE-REGISTERED with failed gates"))
            if chain:
                note += " — LEARNING arm (W3b chain head)"
            # 2026-09-04: in chain mode the new head inherits its predecessor's ladder record as a Thompson prior
            # (serve_bandit warm-start) — the arm it was trained from is the previous learning arm.
            entry = LU.register_arm(args.config, name=name, battle_ckpt=ckpt, tau=sel.tau, note=note, learning=chain,
                                    prior_from=(LU.learning_arms(arms_cfg)[0] if chain else None))
            print(f"[ladder-ppo] arm registered: {json.dumps(entry)}")
            if chain and not args.no_rotate:
                benched = LU.rotate_learning_arm(args.config, keep=name, stamp=stamp)
                print(f"[ladder-ppo] chain: previous learning arm(s) benched: {benched or 'none'}")
            if args.twin:                                   # 2026-09-04 B3: the argmax twin of the new head
                tw = LU.register_twin_arm(args.config, head=name, battle_ckpt=ckpt, stamp=stamp)
                print(f"[ladder-ppo] twin registered: {json.dumps(tw)}")
                tb = LU.rotate_twin_arm(args.config, keep=tw["name"], stamp=stamp)
                print(f"[ladder-ppo] twin: previous twin(s) benched: {tb or 'none'}")
            if args.no_deploy_env:
                print(f"[ladder-ppo] .env {LU.ENV_BATTLE_KEY} left alone (--no-deploy-env)")
            else:
                dep = LU.deploy_env_battle_ckpt(args.env_path, ckpt, stamp=stamp, arm=name)
                print(f"[ladder-ppo] .env {dep['key']} -> {dep['new']}"
                      + (f" (was {dep['old']})" if dep["changed"] else " (already current)"))
            print(f"[ladder-ppo] restart the bot (lanes) — the ladder decides (retire at ~40 g, promote at >= 200 g)")
        else:
            print("[ladder-ppo] NOT registered — a gate failed (use --force-register to override)")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
