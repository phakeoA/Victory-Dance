"""Ladder-wins BC fine-tune — W3b B4 (2026-09-04, USER: "learning is the priority"). THE USER LAUNCHES THIS (GPU, ~1.5 h).

    .venv/Scripts/python.exe -m v_dance.ladder.bc_finetune --dry-run                 # counts + the exact commands
    .venv/Scripts/python.exe -m v_dance.ladder.bc_finetune --run-gates --register    # export -> train -> gates -> arm
    .venv/Scripts/python.exe -m v_dance.ladder.bc_finetune --smoke                   # CPU plumbing check (~2 min)
    (moved from scratch/ladder_bc_finetune.py in the 2026-09-10 refactor, Phase 1; the flags are unchanged)

The VOLUME complement to the nightly ladder PPO (docs/w3b_ladder_ppo_design.md section 12, B4). Every rated ladder game
the bot plays lands as a replay in data/vods/Type_C; the Type-C export keeps each game's WINNING perspective (the bot's
wins teach its own play, its losses teach the human who beat it) into Prepared_training_data/<Regulation>/Jsonl_TypeC
(two passes: the rated main pass + the hand-approved private games); then train_bc fine-tunes the chain head on the
era-2 lineage recipe (docs/era3_kickoff_design.md Arm A: the HF corpus + Type-C, exp advantage weighting off the base's
value head, demonstrator-rating weighting, the aux opponent head, move-order augmentation). Gates = the same ruler /
type-eff pair the PPO chain uses (vs the base AND vs the incumbent). --register adds arm ``bcft_<stamp>`` at tau 0
with the launch-default adapt-rules (like era2) and a FIXED share (the fixed-share allocation gives it games; shares
above 1 are scaled proportionally by the bandit). Promotion stays human.

Exit codes: 0 ok · 2 refused (no base / export failed / nothing to train) · 3 gates failed (candidate saved, NOT
registered) · 4 training failed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]          # v_dance/ladder/ -> repo root
assert (_REPO / "pyproject.toml").is_file(), _REPO

from v_dance.ladder import update as LU   # noqa: E402

PY = sys.executable
PREP = _REPO / "data" / "vods" / "Prepared_training_data"
TYPE_C_DIR = _REPO / "data" / "vods" / "Type_C"
APPROVED_DIR = _REPO / "data" / "vods" / "Type_C_private" / "approved"
CKPT_ROOT = LU.CKPT_ROOT

# The era-2 lineage recipe (docs/era3_kickoff_design.md, Arm A): the four HF open-sheet corpora + Type-C for training,
# the four pinned human val folders. The M-B Type-C folder is derived from the served format (see typec_folder).
TRAIN_FOLDERS = ["Regulation_MA/Jsonl_HF_OTS", "Regulation_MB/Jsonl_HF_OTS",
                 "Regulation_MB_Bo3/Jsonl_HF_OTS", "Regulation_MA_Bo3/Jsonl_HF_OTS"]
VAL_FOLDERS = ["Regulation_MA/Jsonl_TypeA", "Regulation_MA/Jsonl_TypeB",
               "Regulation_MB/Jsonl_TypeB", "Regulation_MB_Bo3/Jsonl_TypeB"]
RECIPE_FLAGS = ["--adv-weight", "exp", "--adv-beta", "1.6", "--rating-weight", "--aux-opp-head",
                "--d-model", "256", "--n-heads", "8", "--n-layers", "4", "--augment-move-order"]


def reg_folder(fmt: str) -> str:
    """``gen9championsvgc2026regmb`` -> ``Regulation_MB`` (``regmc`` -> ``Regulation_MC`` when it lands)."""
    m = re.search(r"reg([a-z]+)$", (fmt or "").lower())
    return f"Regulation_{m.group(1).upper()}" if m else "Regulation_MB"


def typec_folder(fmt: str) -> Path:
    return PREP / reg_folder(fmt) / "Jsonl_TypeC"


def export_commands(out_dir, *, fmt=None, limit=None, py: str = PY) -> dict:
    """The two Type-C passes (STANDING pipeline 2026-07-19): rated games from data/vods/Type_C with the
    ``--rated-only`` purge gate; hand-approved private games from Type_C_private/approved without it (moving the
    file IS the approval), ``--overwrite`` so a re-approval refreshes. Both winner-only, same output folder;
    already-exported replays are skipped by the main pass (incremental)."""
    common = [py, "-X", "utf8", "-u", "-m", "v_dance.datatools.bulk_parse_replays",
              "--output", str(out_dir), "--type", "C", "--winner-only"]
    main = common[:8] + ["--input", str(TYPE_C_DIR)] + common[8:] + ["--rated-only"]
    approved = common[:8] + ["--input", str(APPROVED_DIR)] + common[8:] + ["--overwrite"]
    if fmt:                                   # Type_C mixes regulations (B5 2026-09-29): export only this one
        main += ["--format", fmt]
        approved += ["--format", fmt]
    if limit:
        main += ["--limit", str(int(limit))]
        approved += ["--limit", str(int(limit))]
    return {"main": main, "approved": approved}


def train_command(base, out_dir, *, typec, epochs: int, lr: float, patience: int, device: str,
                  loader_workers: int = 4, limit_files=None, py: str = PY) -> list:
    """The train_bc argv: warm-start = the base (its value head scores the advantages too). A smoke run
    (``limit_files``) bypasses the encoded cache and the worker loader — both need cache-backed data."""
    argv = [py, "-X", "utf8", "-u", "-m", "v_dance.training.train_bc",
            "--data", *[str(PREP / f) for f in TRAIN_FOLDERS], str(typec),
            "--val-data", *[str(PREP / f) for f in VAL_FOLDERS],
            "--warm-start", str(base), "--adv-value-ckpt", str(base), *RECIPE_FLAGS,
            "--epochs", str(int(epochs)), "--lr", str(float(lr)), "--patience", str(int(patience)),
            "--seed", "0", "--device", device, "--progress-secs", "300", "--out", str(out_dir)]
    if limit_files:
        argv += ["--limit-files", str(int(limit_files))]
    else:
        argv += ["--mmap-cache", "--loader-workers", str(int(loader_workers))]
    return argv


def free_stamp(stamp: str, arm_names, ckpt_root=CKPT_ROOT) -> str:
    """``bcft_<stamp>`` / ``checkpoints_attn_bcft_<stamp>`` must both be free (a same-day second run gets ``b``…)."""
    names = set(arm_names or ())
    root = Path(ckpt_root)
    for suffix in [""] + [chr(c) for c in range(ord("b"), ord("z") + 1)]:
        cand = f"{stamp}{suffix}"
        if f"bcft_{cand}" not in names and not (root / f"checkpoints_attn_bcft_{cand}").exists():
            return cand
    raise ValueError(f"no free stamp for {stamp}")


def count_replays(typec) -> dict:
    return {"type_c_replays": len(list(TYPE_C_DIR.glob("*.html"))) if TYPE_C_DIR.is_dir() else 0,
            "approved_replays": len(list(APPROVED_DIR.glob("*.html"))) if APPROVED_DIR.is_dir() else 0,
            "jsonl_exported": len(list(Path(typec).glob("*.jsonl"))) if Path(typec).is_dir() else 0}


def resolve_base(arg: str, arms_cfg: dict):
    """'learning' = the chain head's checkpoint, 'incumbent' = the bandit incumbent's, else a path."""
    from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT
    from v_dance.play.serve_bandit import _resolve
    key = (arg or "learning").strip().lower()
    if key == "learning":
        return LU.learning_base(arms_cfg, default_battle=Path(DEFAULT_BC_CHECKPOINT))
    if key == "incumbent":
        inc = next((a for a in arms_cfg.values() if a.incumbent), None)
        if inc is not None and not inc.uses_default("battle"):
            return _resolve(inc.battle_ckpt)
        return Path(DEFAULT_BC_CHECKPOINT)
    return Path(arg)


def incumbent_ckpt(arms_cfg: dict):
    from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT
    from v_dance.play.serve_bandit import _resolve
    inc = next((a for a in arms_cfg.values() if a.incumbent), None)
    if inc is not None and not inc.uses_default("battle"):
        return _resolve(inc.battle_ckpt)
    return Path(DEFAULT_BC_CHECKPOINT)


def _run(argv, label: str) -> int:
    print(f"[bcft] {label}: " + " ".join(Path(argv[0]).name if i == 0 else a for i, a in enumerate(argv)), flush=True)
    t0 = time.time()
    rc = subprocess.run(argv, cwd=str(_REPO)).returncode
    print(f"[bcft] {label} exit {rc} in {time.time() - t0:.0f} s", flush=True)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="learning", help="'learning' (the chain head, default), 'incumbent', or a checkpoint path")
    ap.add_argument("--config", default=str(LU.DEFAULT_BANDIT_CONFIG))
    ap.add_argument("--fmt", default=None, help="served format (default: v_dance.formats.DEFAULT_FORMAT)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loader-workers", type=int, default=4)
    ap.add_argument("--share", type=float, default=0.15, help="fixed bandit share for the registered arm (0 = none)")
    ap.add_argument("--name", default=None, help="arm name (default bcft_<stamp>)")
    ap.add_argument("--out", default=None, help="candidate dir (default checkpoints_attn_bcft_<stamp>)")
    ap.add_argument("--no-export", action="store_true", help="skip the Type-C export passes (reuse the folder as is)")
    ap.add_argument("--export-only", action="store_true", help="run the export passes and stop")
    ap.add_argument("--dry-run", action="store_true", help="print the counts + commands; nothing runs")
    ap.add_argument("--run-gates", action="store_true", help="ruler vs base + vs incumbent, type-eff probe (minutes)")
    ap.add_argument("--register", action="store_true", help="register bcft_<stamp> when every gate that ran passed")
    ap.add_argument("--force-register", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing check: no export, --limit-files 40, 1 epoch, CPU, temp out dir, no gates/register")
    for k, v in (("ruler_floor_pp", LU.GATES["ruler_floor_pp"]), ("ruler_abs_floor_pp", LU.GATES["ruler_abs_floor_pp"])):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=v)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    fmt = args.fmt
    if fmt is None:
        from v_dance.formats import DEFAULT_FORMAT
        fmt = DEFAULT_FORMAT
    typec = typec_folder(fmt)
    arms_cfg = LU.arm_table(args.config)
    try:
        base = resolve_base(args.base, arms_cfg)
    except ValueError as exc:
        print(f"[bcft] REFUSED: {exc}")
        return 2
    if not base.is_file():
        print(f"[bcft] REFUSED: base checkpoint not found: {base}")
        return 2
    anchor = incumbent_ckpt(arms_cfg)
    if anchor.resolve() == base.resolve():
        anchor = None
    stamp = free_stamp(time.strftime("%Y%m%d"), arms_cfg.keys())
    if args.smoke:
        out_dir = _REPO / "artifacts" / "smoke" / f"bcft_smoke_{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        out_dir = Path(args.out) if args.out else (CKPT_ROOT / f"checkpoints_attn_bcft_{stamp}")
    counts = count_replays(typec)
    print(f"[bcft] base {LU.repo_relative(base)}   anchor {LU.repo_relative(anchor) if anchor else '-'}   format {fmt}")
    print(f"[bcft] replays: Type_C {counts['type_c_replays']}  approved {counts['approved_replays']}  "
          f"exported so far {counts['jsonl_exported']} -> {LU.repo_relative(typec)}")
    exp = export_commands(typec, fmt=fmt)
    train_argv = train_command(base, out_dir, typec=typec, epochs=(1 if args.smoke else args.epochs), lr=args.lr,
                               patience=args.patience, device=("cpu" if args.smoke else args.device),
                               loader_workers=args.loader_workers, limit_files=(40 if args.smoke else None))
    if args.dry_run:
        print("[bcft] plan (dry run, nothing runs):")
        for k, v in exp.items():
            print(f"  export {k}: " + " ".join(v[1:]))
        print("  train: " + " ".join(train_argv[1:]))
        print(f"  then: gates (ruler vs base, vs {LU.repo_relative(anchor) if anchor else '-'}, type-eff) -> arm bcft_{stamp} "
              f"(tau 0, share {args.share:g}) -> restart ONE bot")
        return 0

    if not args.no_export and not args.smoke:
        for k in ("main", "approved"):
            if k == "approved" and not APPROVED_DIR.is_dir():
                print("[bcft] export approved: folder absent, skipped")
                continue
            rc = _run(exp[k], f"export {k}")
            if rc != 0:
                print(f"[bcft] REFUSED: export {k} failed (exit {rc})")
                return 2
        after = count_replays(typec)
        print(f"[bcft] export done: {after['jsonl_exported']} jsonl in {LU.repo_relative(typec)} "
              f"(+{after['jsonl_exported'] - counts['jsonl_exported']})")
        if after["jsonl_exported"] == 0:
            print("[bcft] REFUSED: nothing to train on")
            return 2
        if args.export_only:
            return 0

    t0 = time.time()
    rc = _run(train_argv, "train_bc")
    ckpt = out_dir / "battle_base.pt"
    if rc != 0 or not ckpt.is_file():
        print(f"[bcft] TRAINING FAILED (exit {rc}, checkpoint {'present' if ckpt.is_file() else 'missing'}) -> {out_dir}")
        return 4
    from v_dance.play.model_io import load_bc_policy
    pol, heads = load_bc_policy(ckpt, device="cpu")           # what the bandit will load
    print(f"[bcft] candidate -> {ckpt}  (loads: heads {tuple(heads)}, pair_cond {bool(getattr(pol, 'pair_cond', False))}, "
          f"{time.time() - t0:.0f} s)")
    meta = {"base": str(base), "anchor": (str(anchor) if anchor else None), "format": fmt, "stamp": stamp,
            "counts_before": counts, "train_argv": train_argv, "elapsed_s": round(time.time() - t0, 1), "smoke": args.smoke}
    ok = True
    if args.run_gates and not args.smoke:
        ext = LU.run_external_gates(base, ckpt, out_dir, ruler_floor_pp=args.ruler_floor_pp, anchor=anchor,
                                    ruler_abs_floor_pp=args.ruler_abs_floor_pp)
        meta["gates"] = ext
        print(f"[bcft] ruler delta {ext.get('ruler_delta_pp')} pp (floor {args.ruler_floor_pp}) "
              f"-> {'ok' if ext.get('ruler_ok') else 'FAIL'};  type-eff {ext.get('type_eff_verdict')} "
              f"-> {'ok' if ext.get('type_eff_ok') else 'FAIL'}")
        if anchor is not None:
            print(f"[bcft] ruler vs anchor {LU.repo_relative(anchor)}: {ext.get('ruler_anchor_delta_pp')} pp "
                  f"(floor {args.ruler_abs_floor_pp}) -> {'ok' if ext.get('ruler_anchor_ok') else 'FAIL'}")
        for k in ("ruler_note", "type_eff_note", "ruler_anchor_note"):
            if ext.get(k):
                print(f"[bcft] {k.split('_note')[0]}: {ext[k]}  (full output: {out_dir / 'gates'})")
        ok = LU.external_ok(ext)
    elif not args.smoke:
        print("[bcft] external gates NOT run (add --run-gates)")
    (out_dir / "bcft_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if args.smoke:
        print(f"[bcft] SMOKE OK -> {out_dir} (delete it at will)")
        return 0
    if args.register or args.force_register:
        if ok or args.force_register:
            name = args.name or f"bcft_{stamp}"
            note = (f"B4 ladder-wins BC fine-tune {stamp}: base {LU.repo_relative(base)}, era-2 lineage recipe (HF corpus + "
                    f"Type-C {counts['type_c_replays']} replays, exp adv + rating weight), {args.epochs} ep lr {args.lr:g}"
                    + ("" if ok else " — FORCE-REGISTERED with failed gates") + ". Argmax candidate next to the incumbent; "
                    "promotion = human (swap 'incumbent').")
            entry = LU.register_arm(args.config, name=name, battle_ckpt=ckpt, tau=0.0, note=note, adapt_rules=None,
                                    share=(args.share if args.share > 0 else None))
            print(f"[bcft] arm registered: {json.dumps(entry)}")
            print("[bcft] restart ONE bot — the fixed shares now include the candidate (scaled if they exceed 1)")
        else:
            print("[bcft] NOT registered — a gate failed (use --force-register to override)")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
