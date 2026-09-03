"""PPO self-play two-perspective collector + zero-sum symmetry guard (task 3a.2).

Each self-play GAME yields TWO trajectories — one per perspective (own side via
``_request_own_state``, opponent side via the gap-#6 ``opp_snapshot``; see
docs/ppo_reward_design.md sec 6). A ``TrajectoryCollector`` accumulates one
perspective's decision steps during play (the player calls ``add_step`` each turn,
recording the BEHAVIOUR-policy logprob + collection-time value), and ``finish``
finalises it into a ``Trajectory``.

``assert_zero_sum`` is the sec 6 invariant: the two perspectives must be the same game,
share the turn clock (so gamma^T cancels), and carry NEGATED outcome rewards (r_us ==
-r_opp; both 0 on a draw). It is THE guard against a discount-asymmetry / perspective-flip
self-collusion bug. NOTE (audit): it is enforced by the **Phase-0 plumbing smoke**
(``game_runner.phase0_report``, run before a self-play launch), NOT per-pair inside the
live ``PPOTrainer`` — the live trainer flattens both perspectives and trains on the pool
without re-asserting zero-sum each update. So run the Phase-0 smoke after any change to the
recording/finalisation path to re-validate this invariant.

This module places NO per-step reward (that is task 3a.3); ``finish`` only marks the
last step ``done`` and attaches the episode metadata.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from v_dance.selfplay.schema import (
    PASS_ACTION, Transition, EpisodeMeta, Trajectory, TerminalType,
)


class TrajectoryCollector:
    """Accumulates ONE perspective's decision steps for ONE self-play game."""

    def __init__(self, battle_id: str, own_role: str):
        self.battle_id = battle_id
        self.own_role = own_role
        self._steps: List[Transition] = []

    def add_step(
        self, *, state, action_s0: int, action_s1: int,
        gimmick_s0: int = 0, gimmick_s1: int = 0,
        logprob: float = 0.0, value: float = 0.0,
        decision_type: str = "turn", turn: int = 0,
        mask_s0=None, mask_s1=None, gmask_s0=None, gmask_s1=None,
        pair_first: Optional[int] = None,
    ) -> None:
        """Append one decision step. ``logprob`` is the joint log-prob of the chosen
        (a0,g0,a1,g1) under the BEHAVIOUR policy; ``value`` is the critic V(s) at
        collection time (value_pm space) — both captured now because PPO can't
        recompute them later. ``mask_s*`` / ``gmask_s*`` are the decision-time LEGAL
        masks (build_action_mask / build_gimmick_mask rows for each slot) so PPO can
        recompute the masked log-prob with serve-identical legality (3b.5)."""
        self._steps.append(Transition(
            state=np.asarray(state, dtype=np.float32),
            action_s0=int(action_s0), action_s1=int(action_s1),
            gimmick_s0=int(gimmick_s0), gimmick_s1=int(gimmick_s1),
            logprob=float(logprob), value=float(value),
            reward=0.0, done=False, decision_type=decision_type, turn=int(turn),
            mask_s0=mask_s0, mask_s1=mask_s1, gmask_s0=gmask_s0, gmask_s1=gmask_s1,
            pair_first=(None if pair_first is None else int(pair_first)),   # W3b-1b
        ))

    def __len__(self) -> int:
        return len(self._steps)

    def last_step(self) -> Optional[Transition]:
        """The most recently added step, or None. Lets the collection player detect a
        rejection re-call (poke-env re-invokes for the SAME turn) and replace / discard
        the non-executed pick (3c.1c)."""
        return self._steps[-1] if self._steps else None

    def pop_step(self) -> Optional[Transition]:
        """Remove + return the last step (a rejected order that did NOT execute)."""
        return self._steps.pop() if self._steps else None

    def finish(
        self, *, own_team: Sequence[str], opp_team: Sequence[str],
        tp_bring: Sequence[int], tp_leads: Sequence[int],
        won: Optional[bool], terminal_type, n_turns: Optional[int] = None,
        sampling: Optional[dict] = None,
    ) -> Trajectory:
        """Finalise: mark the last step ``done`` and attach ``EpisodeMeta``. Does NOT
        place the terminal reward (task 3a.3). ``n_turns`` defaults to the last step's
        turn (the shared clock used by the symmetry / discount checks). ``sampling``
        labels the behaviour-policy params (tau/top_p) the recorded logprobs assume."""
        if self._steps:
            self._steps[-1].done = True
        if n_turns is None:
            n_turns = self._steps[-1].turn if self._steps else 0
        meta = EpisodeMeta(
            battle_id=self.battle_id, own_role=self.own_role,
            own_team=list(own_team), opp_team=list(opp_team),
            tp_bring=[int(i) for i in tp_bring], tp_leads=[int(i) for i in tp_leads],
            won=won, terminal_type=str(TerminalType(terminal_type).value),
            n_turns=int(n_turns), sampling=sampling,
        )
        return Trajectory(meta=meta, transitions=list(self._steps))


def assert_zero_sum(us: Trajectory, opp: Trajectory, tol: float = 1e-6) -> None:
    """sec 6 zero-sum symmetry guard for the two perspectives of one game. Raises
    AssertionError on violation. (WIN/LOSS are MIRROR terminal types across the two
    sides, so we check the outcome-reward NEGATION, not type-equality.)"""
    assert us.meta.battle_id == opp.meta.battle_id, \
        f"battle_id mismatch: {us.meta.battle_id!r} vs {opp.meta.battle_id!r}"
    assert us.meta.own_role != opp.meta.own_role, \
        f"both perspectives share role {us.meta.own_role!r} (perspective-flip bug)"
    assert us.meta.n_turns == opp.meta.n_turns, \
        f"turn-clock mismatch {us.meta.n_turns} vs {opp.meta.n_turns} (gamma^T won't cancel)"
    ru, ro = us.meta.outcome_reward(), opp.meta.outcome_reward()
    if ru is None or ro is None:
        # outcome-less terminals (HORIZON_CUT bootstrap / FALLBACK discard) — both
        # sides must be the SAME outcome-less kind (it's one game).
        assert ru is None and ro is None, \
            f"outcome-reward presence mismatch: r_us={ru} r_opp={ro}"
        assert us.meta.terminal_type == opp.meta.terminal_type, \
            f"outcome-less terminal mismatch: {us.meta.terminal_type} vs {opp.meta.terminal_type}"
        return
    assert abs(ru + ro) <= tol, \
        f"not zero-sum: r_us={ru} r_opp={ro} (sum {ru + ro}, tol {tol})"


# ── Level-A opponent-action alignment (Piece 3) ───────────────────────────────
def _steps_by_key(traj: Trajectory) -> dict:
    """Index a trajectory's steps by ``(turn, decision_type)``. After the 3c.1c de-dup
    there is exactly one step per key; if a stray duplicate survives, the LAST (the
    executed pick) wins — matching the de-dup's keep-the-executed semantics."""
    idx: dict = {}
    for t in traj.transitions:
        idx[(t.turn, t.decision_type)] = t
    return idx


def align_opponent_actions(traj_a: Trajectory, traj_b: Trajectory) -> int:
    """Fill each perspective's ``opp_a_action`` / ``opp_b_action`` from the OTHER
    perspective's same-``(turn, decision_type)`` ``action_s0`` / ``action_s1`` — the
    Level-A opponent-prediction target (Piece 3).

    The action codec is SEAT-SYMMETRIC (each player encodes from its own seat, and the
    BC aux target ``opp_action_to_index`` just re-encodes the foe from that same seat),
    so the opponent player's OWN ``action_s0`` / ``action_s1`` ARE the ``opp_a`` / ``opp_b``
    target space directly — a copy, no re-flip. A slot the opponent PASSED
    (``PASS_ACTION``) or a ``(turn, decision_type)`` with no matching opposite-perspective
    step leaves the ``PASS_ACTION`` sentinel, so the Piece-2 aux loss masks it
    (``opp_valid = 0``).

    Mutates both trajectories in place. Returns the number of slot-targets filled."""
    a_idx = _steps_by_key(traj_a)
    b_idx = _steps_by_key(traj_b)
    filled = 0
    for src, dst_idx in ((traj_a, b_idx), (traj_b, a_idx)):
        for t in src.transitions:
            opp = dst_idx.get((t.turn, t.decision_type))
            if opp is None:
                continue
            if opp.action_s0 != PASS_ACTION:
                t.opp_a_action = int(opp.action_s0)
                filled += 1
            if opp.action_s1 != PASS_ACTION:
                t.opp_b_action = int(opp.action_s1)
                filled += 1
    return filled


def align_paired_trajectories(our_trajs: dict, opp_trajs: dict) -> int:
    """Align opp-action targets for every ``battle_tag`` present in BOTH perspective
    dicts. Only ``latest`` mirror chunks carry both perspectives — ``snapshot`` /
    ``scripted`` opponents are NON-recording, so those transitions keep the sentinel and
    Piece 2 masks them. Mutates in place; returns total slot-targets filled."""
    total = 0
    for tag, a in our_trajs.items():
        b = opp_trajs.get(tag)
        if b is not None:
            total += align_opponent_actions(a, b)
    return total
