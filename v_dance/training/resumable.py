"""Shared epoch-granular resume machinery (2026-07-19, USER: EVERY long training must be
pausable at the USER's convenience — overnight runs in chunks, no ceremony).

Contract (used by train_bc + train_teampreview; the exploiter has its own iteration-style
resume in v_dance/selfplay/exploiter.py):
  * ``save_resume_state`` is called at EVERY epoch boundary and writes ATOMICALLY
    (tmp + os.replace) — a kill / Ctrl-C / power cut at ANY moment costs at most the
    in-progress epoch and can never corrupt the state file.
  * ``--resume`` restores model + optimizer + counters + history + all four RNG streams
    and is guarded by ``config_fingerprint`` — resuming into a DIFFERENT recipe fails
    loud instead of silently training a chimera.
  * Re-running a FINISHED run with ``--resume`` and a higher ``--epochs`` extends it.
  * A resumed run is scientifically equivalent but not bit-identical to an uninterrupted
    one (DataLoader worker streams re-fork at the seam).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

RESUME_NAME = "resume_state.pt"


def config_fingerprint(config: dict) -> str:
    """Deterministic hash of the run's stamped config — the resume-compatibility key."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def rng_states() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(st: dict) -> None:
    # torch's RNG setters require CPU ByteTensors; a state loaded with
    # map_location='cuda' hands back CUDA tensors — coerce back.
    torch.set_rng_state(st["torch"].cpu())
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])


def save_resume_state(path: Path, *, epoch: int, model, optimizer,
                      best: float, epochs_no_improve: int,
                      history: list, config_fp: str,
                      extra: Optional[dict] = None) -> None:
    """Atomic write (tmp + os.replace) so a mid-write kill never corrupts the state.
    ``extra`` carries trainer-specific integrity guards (e.g. the TP vocab)."""
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best": best,
        "epochs_no_improve": epochs_no_improve,
        "history": history,
        "config_fp": config_fp,
        "rng": rng_states(),
        "extra": extra or {},
    }, tmp)
    os.replace(tmp, path)


def load_resume_state(path: Path, config_fp: str, device: str = "cpu",
                      label: str = "train") -> dict:
    """Load + fingerprint-guard a resume state; SystemExit with a clear message on any
    mismatch (the caller applies model/optimizer/counters/RNG itself)."""
    import sys
    if not Path(path).is_file():
        sys.exit(f"[{label}] --resume: no {Path(path).name} in {Path(path).parent} — "
                 f"nothing to resume (state is written after every completed epoch).")
    rs = torch.load(path, map_location=device, weights_only=False)
    if rs.get("config_fp") != config_fp:
        sys.exit(f"[{label}] --resume REFUSED: the saved state was produced by a DIFFERENT "
                 f"recipe (config fingerprint mismatch). Re-run with the exact same "
                 f"flags/data as the interrupted run, or start fresh without --resume "
                 f"(state: {path}).")
    return rs
