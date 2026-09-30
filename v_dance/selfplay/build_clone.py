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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["nemesis"], required=True)
    ap.add_argument("--reg", nargs="+", default=["regmb"], choices=sorted(_REG_DIR),
                    help="one or more regs whose Type_C losses are pooled (e.g. regmb regmc)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="train_bc batch size (default: train_bc's); lower it on a CUDA out-of-memory")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

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
