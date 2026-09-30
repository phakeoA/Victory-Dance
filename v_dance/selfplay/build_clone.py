"""Build league CLONE opponents by behaviour cloning (docs/league_design_2026-09-29.md, P1). THE USER LAUNCHES THE TRAINING.

``nemesis``: the humans who BEAT the bot. The Type_C export is --winner-only, so every bot LOSS is already on disk as the
winning human's perspective; this selects those files (row field ``winner`` != the bot account), hard-links them into
``Prepared_training_data/Nemesis_<regs>/`` (no extra disk; several regs pool) and fine-tunes the era2 BC anchor on them
with the era-2 architecture. Pure imitation: no advantage weighting (the goal is to copy those players, not to beat
them). The checkpoint is an OPPONENT for the league — it is never registered as a ladder bandit arm.

    python -m v_dance.selfplay.build_clone --kind nemesis --reg regmb regmc --dry-run   # counts + the exact train command
    python -m v_dance.selfplay.build_clone --kind nemesis --reg regmb regmc             # link + train (GPU, minutes)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PTD = _REPO / "data" / "vods" / "Prepared_training_data"
_BC = _REPO / "ai_train_scripts" / "BC_model"
ANCHOR = _BC / "checkpoints_attn_era2" / "battle_base.pt"
_REG_DIR = {"regma": "Regulation_MA", "regmb": "Regulation_MB", "regmc": "Regulation_MC"}
# era-2 lineage architecture + recipe (as bc_finetune's train command, minus advantage weighting)
_ARCH = ["--d-model", "256", "--n-heads", "8", "--n-layers", "4", "--aux-opp-head", "--augment-move-order",
         "--rating-weight"]


def bot_accounts() -> set[str]:
    """Lower-cased account names the bot has played as (.env PS_USERNAME + the historical one)."""
    names = {"victoriousdancing"}
    env = _REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("PS_USERNAME="):
                names.add(line.split("=", 1)[1].strip().lower())
    return names


def loss_files(typec_dir: Path, bots: set[str]) -> list[Path]:
    """Files whose first row's ``winner`` is NOT the bot = the human side of a bot loss."""
    out = []
    for f in sorted(typec_dir.glob("*.jsonl")):
        try:
            with open(f, encoding="utf-8") as fh:
                winner = (json.loads(fh.readline()).get("winner") or "").lower()
        except (OSError, ValueError):
            continue
        if winner and winner not in bots:
            out.append(f)
    return out


def link_into(files: list[Path], dest: Path) -> int:
    """Hard-link ``files`` into ``dest`` (same volume; copy as a fallback). Idempotent; returns new links made."""
    dest.mkdir(parents=True, exist_ok=True)
    made = 0
    for f in files:
        target = dest / f.name
        if target.exists():
            continue
        try:
            os.link(f, target)
        except OSError:
            target.write_bytes(f.read_bytes())
        made += 1
    return made


def train_cmd(data_dirs: list[Path], val_dirs: list[Path], out: Path, epochs: int, device: str,
              batch_size: int | None = None) -> list[str]:
    extra = ["--batch-size", str(batch_size)] if batch_size else []
    return [sys.executable, "-X", "utf8", "-u", "-m", "v_dance.training.train_bc",
            "--data", *map(str, data_dirs), "--val-data", *map(str, val_dirs),
            "--warm-start", str(ANCHOR), *_ARCH, "--epochs", str(epochs), "--lr", "5e-4", "--patience", "3",
            "--seed", "0", "--device", device, "--progress-secs", "300", "--out", str(out),
            "--mmap-cache", "--loader-workers", "4", *extra]


# ── archetype clones (league P1, 2026-09-30) ──────────────────────────────────────────────────────────────────────
# One clone per team ARCHETYPE (team_archetypes k10 centroids), each imitating only the SIDES whose team belongs to it,
# so the league meets distinct game plans (rain, Trick Room, sand, goodstuffs …). Sources = the M-B + M-C corpora only
# (M-A is two regs old; its 75 GB Bo3 set is too heavy to duplicate). A replay file holds BOTH perspectives, so the
# split writes one file per SIDE under Archetypes_k10/aNN/{train,val} (~10 GB for these sources; the folders can go to
# the Recycle Bin once the clones are trained). 10 % of sides (hash of replay+side) are held out for validation.
ARCH_SOURCES = ["Regulation_MB/Jsonl_HF_OTS", "Regulation_MB_Bo3/Jsonl_HF_OTS", "Regulation_MB/Jsonl_TypeB",
                "Regulation_MB_Bo3/Jsonl_TypeB", "Regulation_MC/Jsonl_TypeB"]
ARCH_ARTIFACT = _BC / "team_archetypes_k10_full.json"
ARCH_ROOT = _PTD / "Archetypes_k10"
ARCH_MIN_TRAIN = 150                                   # sides; a smaller archetype is too thin to clone


def _is_val(replay_id: str, perspective: str) -> bool:
    import hashlib
    return int(hashlib.md5(f"{replay_id}|{perspective}".encode()).hexdigest(), 16) % 10 == 0


def side_archetypes(sources: list[Path], artifact_path: Path, limit_files: int | None = None) -> dict:
    """{(replay_id, perspective): archetype} — every team-side assigned to its nearest k10 centroid."""
    from v_dance.datatools.team_archetypes import assign, collect_team_records, load_artifact
    art = load_artifact(str(artifact_path))
    side = {}
    for rec in collect_team_records([str(s) for s in sources], limit_files=limit_files).values():
        a, _ = assign(rec["feats"], art)
        for rid, persp in rec["sightings"]:
            side[(str(rid), str(persp))] = int(a)
    return side


def split_by_archetype(sources: list[Path], side: dict, out_root: Path) -> dict:
    """Write each assigned SIDE's rows to out_root/aNN/{train,val}/<replay>_<side>.jsonl; returns {(a, split): n}."""
    from collections import Counter, defaultdict

    from v_dance.training.bc_dataset import iter_jsonl_files
    counts = Counter()
    for folder in sources:
        for f in iter_jsonl_files(str(folder)):
            groups = defaultdict(list)
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    groups[(str(row.get("replay_id")), str(row.get("perspective")))].append(line)
            for (rid, persp), lines in groups.items():
                a = side.get((rid, persp))
                if a is None:
                    continue
                split = "val" if _is_val(rid, persp) else "train"
                dest = out_root / f"a{a:02d}" / split / f"{Path(f).stem}_{persp}.jsonl"
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(".tmp")
                    tmp.write_text("".join(lines), encoding="utf-8")
                    os.replace(tmp, dest)
                counts[(a, split)] += 1
    return dict(counts)


def _run_archetype(a) -> None:
    if a.split:
        sources = [_PTD / s for s in ARCH_SOURCES if (_PTD / s).is_dir()]
        print(f"[clone] archetype split: {len(sources)} source folders, artifact {ARCH_ARTIFACT.name} -> "
              f"{ARCH_ROOT.relative_to(_REPO)}")
        side = side_archetypes(sources, ARCH_ARTIFACT, a.limit_files)
        print(f"[clone] {len(side)} team-sides assigned")
        counts = split_by_archetype(sources, side, ARCH_ROOT)
        for k in sorted({x for x, _ in counts}):
            print(f"  a{k:02d}: train {counts.get((k, 'train'), 0):5d}  val {counts.get((k, 'val'), 0):4d}")
        return
    todo = []
    for d in sorted(ARCH_ROOT.glob("a[0-9][0-9]")):
        k = int(d.name[1:])
        if a.only and k not in a.only:
            continue
        n = len(list((d / "train").glob("*.jsonl")))
        if n < ARCH_MIN_TRAIN:
            print(f"[clone] a{k:02d}: {n} train sides < {ARCH_MIN_TRAIN} — skipped (too thin)")
            continue
        out = _BC / f"checkpoints_attn_clone_arch{k:02d}_k10_{datetime.now():%Y%m%d}"
        todo.append((k, n, train_cmd([d / "train"], [d / "val"], out, a.epochs, a.device, a.batch_size), out))
    if not todo:
        raise SystemExit("[clone] nothing to train — run --kind archetype --split first")
    for k, n, cmd, out in todo:
        print(f"[clone] a{k:02d}: {n} train sides -> {out.relative_to(_REPO)}")
    if a.dry_run:
        print("[clone] train (first):", " ".join(todo[0][2][1:]))
        return
    for k, n, cmd, out in todo:
        rc = subprocess.call(cmd, cwd=_REPO)
        print(f"[clone] a{k:02d} exit {rc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["nemesis", "archetype"], required=True)
    ap.add_argument("--split", action="store_true", help="archetype: build the per-archetype data folders (CPU)")
    ap.add_argument("--only", type=int, nargs="+", default=None, help="archetype: train only these ids")
    ap.add_argument("--limit-files", type=int, default=None, help="archetype split: cap source files (smoke)")
    ap.add_argument("--reg", nargs="+", default=["regmb"], choices=sorted(_REG_DIR),
                    help="one or more regs whose Type_C losses are pooled (e.g. regmb regmc)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="train_bc batch size (default: train_bc's); lower it on a CUDA out-of-memory")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.kind == "archetype":
        if not a.split and not ANCHOR.exists():
            raise SystemExit(f"[clone] anchor missing: {ANCHOR}")
        return _run_archetype(a)

    bots, losses, val = bot_accounts(), [], []
    for reg in a.reg:
        reg_dir = _PTD / _REG_DIR[reg]
        typec = reg_dir / "Jsonl_TypeC"
        if not typec.is_dir():
            raise SystemExit(f"[clone] no Type_C export at {typec}")
        found = loss_files(typec, bots)
        print(f"[clone] {reg}: {len(found)} bot-loss files in {typec.relative_to(_PTD)}")
        losses += found
        val += [d for d in (reg_dir / "Jsonl_TypeB",) if d.is_dir()]
    tag = "_".join(a.reg)
    dest = _PTD / f"Nemesis_{tag}"
    out = _BC / f"checkpoints_attn_clone_nemesis_{tag}_{datetime.now():%Y%m%d}"
    cmd = train_cmd([dest], val, out, a.epochs, a.device, a.batch_size)
    print(f"[clone] nemesis: {len(losses)} files -> {dest.relative_to(_REPO)}")
    print(f"[clone] warm-start {ANCHOR.relative_to(_REPO)} -> {out.relative_to(_REPO)}")
    print("[clone] train:", " ".join(cmd[1:]))
    if a.dry_run:
        return
    if not ANCHOR.exists():
        raise SystemExit(f"[clone] anchor missing: {ANCHOR}")
    print(f"[clone] linked {link_into(losses, dest)} new files")
    raise SystemExit(subprocess.call(cmd, cwd=_REPO))


if __name__ == "__main__":
    main()
