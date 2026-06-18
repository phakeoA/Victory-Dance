"""Resumable multi-session training — task 3c.4 (the "can't run 24/7" essential).

Training runs on a personal PC that can't stay on continuously, so the design constraint
is FULL RESUMABILITY: stop any time, lose almost nothing, and a chunked run is
mathematically identical to a continuous one (docs/ppo_reward_design.md sec 17).

PPO is ON-POLICY, so there is NO large replay buffer to persist — an interrupted
generation's collection is simply dropped and re-collected. The resume snapshot is just
the TRAINING STATE: actor+critic weights, BOTH optimiser states, the trainer RNG, the
update/warmup counters, the league pool, the generation history, and the seed. The
frozen-BC reference (for KL-to-BC) and the architecture are NOT persisted — they are
re-derived from the SAME base checkpoint passed on resume (so resume with the same
``--ckpt``).

Snapshots are small + written ATOMICALLY (tmp + os.replace) after every generation, so a
power blip costs at most the in-flight generation. ``StopController`` adds graceful Ctrl-C
and an optional wall-clock budget; the loop checks ``should_stop()`` between generations.
"""
from __future__ import annotations

import os
import re
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from v_dance.selfplay.generation import GenerationHistory
from v_dance.selfplay.league import OpponentLeague

SNAPSHOT_FORMAT = 1


def _gen_cfg_obj(gc) -> dict:
    return {"n_games": gc.n_games, "warmup_updates": gc.warmup_updates,
            "gate": asdict(gc.gate)}


def save_snapshot(path, *, actor_critic, trainer, league, history,
                  ppo_cfg=None, train_cfg=None, gen_cfg=None, seed: int = 0) -> Path:
    """Write the resume snapshot ATOMICALLY (tmp file + os.replace) so a crash mid-write
    can never corrupt the existing snapshot. Returns the path."""
    snap = {
        "format": SNAPSHOT_FORMAT,
        "generation": history.generation,
        "ac_state": actor_critic.state_dict(),
        "actor_opt": trainer.actor_opt.state_dict(),
        "critic_opt": trainer.critic_opt.state_dict(),
        "rng_state": trainer.rng.bit_generator.state,
        "updates": trainer.updates, "warmups": trainer.warmups,
        "league": league.to_obj(), "history": history.to_obj(), "seed": seed,
        # configs are recorded for provenance; the run uses the live (CLI) configs.
        "ppo_cfg": asdict(ppo_cfg) if ppo_cfg is not None else None,
        "train_cfg": asdict(train_cfg) if train_cfg is not None else None,
        "gen_cfg": _gen_cfg_obj(gen_cfg) if gen_cfg is not None else None,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    torch.save(snap, tmp)
    os.replace(tmp, p)            # atomic on the same filesystem
    return p


def load_into(path, *, actor_critic, trainer, device: str = "cpu"):
    """Restore the training state into a FRESHLY-BUILT actor_critic + trainer (both
    constructed from the SAME base checkpoint as the original run, so the architecture
    and the frozen-BC reference already match). Returns ``(league, history, snap)`` rebuilt
    from the file."""
    snap = torch.load(path, map_location=device, weights_only=False)
    actor_critic.load_state_dict(snap["ac_state"])
    trainer.actor_opt.load_state_dict(snap["actor_opt"])
    trainer.critic_opt.load_state_dict(snap["critic_opt"])
    trainer.rng.bit_generator.state = snap["rng_state"]
    trainer.updates = int(snap.get("updates", 0))
    trainer.warmups = int(snap.get("warmups", 0))
    league = OpponentLeague.from_obj(snap["league"])
    history = GenerationHistory.from_obj(snap["history"])
    return league, history, snap


# ── per-generation snapshots (task #20: resume from any generation) ────────────
_SNAP_RE = re.compile(r"^snap_gen(\d+)\.pt$")


def snapshot_path_for(archive, generation: int) -> Path:
    """Path of the full-state snapshot taken AFTER ``generation`` completed
    (``<archive>/snap_gen{N}.pt``). Resuming it continues at gen N+1."""
    return Path(archive) / f"snap_gen{int(generation)}.pt"


def list_snapshots(archive) -> List[Tuple[int, Path]]:
    """All per-gen snapshots in ``archive`` as ``(generation, path)``, sorted by generation."""
    out: List[Tuple[int, Path]] = []
    d = Path(archive)
    if not d.is_dir():
        return out
    for p in d.glob("snap_gen*.pt"):
        m = _SNAP_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda t: t[0])


def latest_snapshot(archive) -> Optional[Path]:
    """The highest-generation snapshot in ``archive``, or ``None`` if there are none."""
    snaps = list_snapshots(archive)
    return snaps[-1][1] if snaps else None


def resolve_resume(archive, resume_gen=None, explicit_path=None) -> Optional[Path]:
    """Resolve a resume request to a snapshot path (task #20):
      * ``explicit_path`` (the legacy ``--resume <file>``) wins if given;
      * ``resume_gen`` ``None`` -> ``None`` (FRESH start from the base ``--ckpt``);
      * ``resume_gen`` ``"latest"``/``"last"``/``""`` -> the highest-gen snapshot;
      * ``resume_gen`` an int / numeric string ``N`` -> ``snap_gen{N}.pt`` (whether or not it
        exists — the caller validates + errors loudly so a typo never silently starts fresh)."""
    if explicit_path:
        return Path(explicit_path)
    if resume_gen is None:
        return None
    s = str(resume_gen).strip().lower()
    if s in ("", "latest", "last"):
        return latest_snapshot(archive)
    return snapshot_path_for(archive, int(s))


def prune_snapshots(archive, keep: int) -> List[Path]:
    """Keep only the last ``keep`` per-gen snapshots (the per-gen snapshots are ~21 MB each, so an
    unbounded run would accumulate them). ``keep`` <= 0 keeps ALL. Returns the deleted paths."""
    if not keep or keep <= 0:
        return []
    snaps = list_snapshots(archive)
    deleted: List[Path] = []
    for _gen, p in snaps[:-int(keep)]:
        try:
            p.unlink()
            deleted.append(p)
        except Exception:
            pass
    return deleted


class StopController:
    """Graceful stop for chunked training. Ctrl-C (SIGINT) requests a CLEAN stop (the loop
    finishes the current step, checkpoints, and exits); an optional ``max_hours`` wall-clock
    budget stops between generations. The in-flight (on-policy) collection is cheap to drop,
    so a stop costs at most one generation. ``clock`` is injectable for testing."""

    def __init__(self, max_hours=None, install_signal: bool = True, clock=time.monotonic):
        self._stop = False
        self._clock = clock
        self._deadline = (clock() + float(max_hours) * 3600.0) if max_hours else None
        if install_signal:
            self._install()

    def _install(self) -> None:
        def handler(_sig, _frame):
            self._stop = True
            print("\n[3c.4] stop requested (Ctrl-C) — checkpoint + exit after the current step.")
        try:
            signal.signal(signal.SIGINT, handler)
        except (ValueError, OSError):      # not the main thread / unsupported
            pass

    def request_stop(self) -> None:
        self._stop = True

    def time_left(self):
        return None if self._deadline is None else max(0.0, self._deadline - self._clock())

    def should_stop(self) -> bool:
        if self._stop:
            return True
        return self._deadline is not None and self._clock() >= self._deadline
