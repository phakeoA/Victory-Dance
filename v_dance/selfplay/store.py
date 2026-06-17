"""Type-C self-play replay store: Trajectory <-> jsonl (task 3a.4).

The on-disk buffer the PPO trainer reads, and the converter between the in-memory
``Trajectory`` (produced by the collector, task 3a.2) and disk. One ``Trajectory`` per
jsonl line (each game appends TWO lines — own + opponent perspective, sec 6).

DESIGN: this is a SEPARATE store from the BC-era ``vgc_base.ReplayBuffer`` (which
writes per-TURN lines with state/action/source/outcome and NO logprob/value). The BC
recorder is left untouched for the BC path; the RL path gets this clean store so PPO's
behaviour-logprob + collection-time value are persisted natively (they cannot be
recomputed later). Torch-free.

The sec 13 env-build asserts (no VecNormalize/reward-clip wrapper; live action-mask
keeps edge-case moves legal) are RUNTIME checks needing the env/player — they live in
the 3c self-play loop, not in this data converter. The data-level reward-integrity
check below (``assert_terminal_rewards_clean``) belongs here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from v_dance.selfplay.schema import Trajectory


class TrajectoryStore:
    """Append-only jsonl sink for self-play Trajectories (one per line)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self.n_written = 0

    def append(self, traj: Trajectory) -> None:
        self._handle.write(json.dumps(traj.to_obj()) + "\n")
        self._handle.flush()
        self.n_written += 1

    def append_all(self, trajs: Iterable[Trajectory]) -> None:
        for t in trajs:
            self.append(t)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TrajectoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_trajectories(path, trajectories: Iterable[Trajectory]) -> int:
    """Write trajectories to ``path`` (append). Returns the number written."""
    store = TrajectoryStore(path)
    try:
        store.append_all(trajectories)
        return store.n_written
    finally:
        store.close()


def iter_trajectories(path, *, expected_state_dim: Optional[int] = None) -> Iterator[Trajectory]:
    """Stream trajectories from a jsonl buffer (memory-friendly for large buffers).
    Validates each transition's state dim when ``expected_state_dim`` is given."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            traj = Trajectory.from_obj(json.loads(line))
            if expected_state_dim is not None:
                for t in traj.transitions:
                    assert t.state.shape == (expected_state_dim,), (
                        f"{p}:{ln} state dim {t.state.shape} != ({expected_state_dim},) "
                        f"— stale-layout buffer?")
            yield traj


def read_trajectories(path, *, expected_state_dim: Optional[int] = None) -> List[Trajectory]:
    """Read all trajectories from a jsonl buffer into a list (missing file -> [])."""
    return list(iter_trajectories(path, expected_state_dim=expected_state_dim))


def assert_terminal_rewards_clean(trajectories: Iterable[Trajectory], tol: float = 1e-6) -> None:
    """Data-integrity guard (sec 3/sec 13) for a SPARSE-phase buffer (PBRS off): every
    per-step reward is 0 except the LAST, and the last is the clean terminal value in
    [-1, 1] (+-1 win/loss/adjudicated, 0 draw/horizon). Catches a reward-normalization
    wrapper or a misplaced reward. NOTE: do NOT run this once PBRS shaping is enabled
    (intermediate F_t rewards are then nonzero by design)."""
    for traj in trajectories:
        ts = traj.transitions
        if not ts:
            continue
        for t in ts[:-1]:
            assert abs(t.reward) <= tol, (
                f"{traj.meta.battle_id}: non-terminal reward {t.reward} != 0 "
                f"(reward wrapper or misplaced reward?)")
        last = ts[-1].reward
        assert -1.0 - tol <= last <= 1.0 + tol, (
            f"{traj.meta.battle_id}: terminal reward {last} outside [-1, 1] "
            f"(reward normalization applied?)")
