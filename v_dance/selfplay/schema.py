"""PPO self-play trajectory schema (task 3a.1).

The in-memory + on-disk structures the self-play collector produces and the PPO
trainer consumes. See docs/ppo_reward_design.md sec 1 (terminal reward + terminal
types), sec 6 (two-perspective + TP decision recorded), sec 19 (3a breakdown).

Deliberately torch-free (numpy + stdlib only) so it can be built and tested without
the model. Per-step ``value`` and ``logprob`` are recorded AT COLLECTION TIME: PPO's
importance ratio needs the BEHAVIOUR-policy log-prob, and the GAE baseline needs the
collection-time critic value — neither can be faithfully recomputed later. The legacy
``vgc_base.ReplayBuffer`` (BC-style: state/action/source/outcome) does NOT capture
these, so this schema supersedes it for the RL path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

PASS_ACTION = -1  # action_s0/s1 sentinel: this slot passed / had no decodable action


def _mask_to_list(m) -> Optional[List[int]]:
    """Normalise a legal-action mask (sequence of 0/1, an np.ndarray, or None) to a
    plain ``list[int]`` for JSON, or None when absent."""
    if m is None:
        return None
    return [int(x) for x in np.asarray(m).ravel().tolist()]


class TerminalType(str, Enum):
    """How an episode ended — drives the terminal reward (sec 1)."""
    WIN = "win"                  # won outright (all opp brought fainted / opp forfeit) -> +1
    LOSS = "loss"                # lost outright -> -1
    DRAW = "draw"                # server declares NO winner -> reward 0, NO bootstrap
    ADJUDICATED = "adjudicated"  # Showdown timer winner by mon-count/HP -> +-1 via `won`
    HORIZON_CUT = "horizon_cut"  # OUR training step-cap (not a real end) -> bootstrap gamma*V, NOT +-1
    FALLBACK = "fallback"        # engineering failure / fallback-forfeit -> DISCARD the trajectory


@dataclass
class Transition:
    """One decision step (one turn OR one forced replacement) from ONE perspective."""
    state: np.ndarray                # encoded board at decision time, shape (STATE_DIM,)
    action_s0: int                   # slot-0 action 0..ACTION_DIM-1; PASS_ACTION(-1) = pass/no-decode
    action_s1: int                   # slot-1 action; PASS_ACTION(-1) = pass/no-decode
    gimmick_s0: int = 0              # GIMMICK_NONE(0) / GIMMICK_MEGA(1)
    gimmick_s1: int = 0
    # Level-A opp-prediction targets (Piece 3): the OPPONENT's same-turn action per slot,
    # in the opponent's own-perspective 0..ACTION_DIM-1 codec. The action codec is
    # seat-symmetric, so the opponent player's OWN action_s0/action_s1 ARE the opp_a/opp_b
    # aux-head target space directly (no re-flip). PASS_ACTION(-1) = unknown (opp passed, or
    # no matching opposite-perspective step) -> the Piece-2 aux loss masks it (opp_valid=0).
    opp_a_action: int = PASS_ACTION
    opp_b_action: int = PASS_ACTION
    logprob: float = 0.0             # joint log-prob of (a0,g0,a1,g1) under the BEHAVIOUR policy
    value: float = 0.0               # critic V(s) at collection time (value_pm space, [-1,1])
    reward: float = 0.0              # per-step reward (0 except terminal; + PBRS F_t if enabled)
    done: bool = False               # True only on the last transition
    decision_type: str = "turn"      # "turn" | "replacement"
    turn: int = 0                    # Showdown turn (shared clock across both perspectives)
    # Decision-time LEGAL masks (serve-parity), recorded so PPO can recompute the
    # masked log-prob with the EXACT same legality (3b.5). Per slot: action mask
    # over ACTION_DIM, gimmick mask over GIMMICK_DIM (build_action_mask /
    # build_gimmick_mask rows). None = unknown -> treated as all-legal / no-gimmick.
    mask_s0: Optional[List[int]] = None
    mask_s1: Optional[List[int]] = None
    gmask_s0: Optional[List[int]] = None
    gmask_s1: Optional[List[int]] = None

    def to_obj(self) -> dict:
        return {
            "state": np.asarray(self.state, dtype=np.float32).tolist(),
            "action_s0": int(self.action_s0), "action_s1": int(self.action_s1),
            "gimmick_s0": int(self.gimmick_s0), "gimmick_s1": int(self.gimmick_s1),
            "opp_a_action": int(self.opp_a_action), "opp_b_action": int(self.opp_b_action),
            "logprob": float(self.logprob), "value": float(self.value),
            "reward": float(self.reward), "done": bool(self.done),
            "decision_type": self.decision_type, "turn": int(self.turn),
            "mask_s0": _mask_to_list(self.mask_s0), "mask_s1": _mask_to_list(self.mask_s1),
            "gmask_s0": _mask_to_list(self.gmask_s0), "gmask_s1": _mask_to_list(self.gmask_s1),
        }

    @classmethod
    def from_obj(cls, d: dict) -> "Transition":
        return cls(
            state=np.asarray(d["state"], dtype=np.float32),
            action_s0=int(d["action_s0"]), action_s1=int(d["action_s1"]),
            gimmick_s0=int(d.get("gimmick_s0", 0)), gimmick_s1=int(d.get("gimmick_s1", 0)),
            opp_a_action=int(d.get("opp_a_action", PASS_ACTION)),
            opp_b_action=int(d.get("opp_b_action", PASS_ACTION)),
            logprob=float(d.get("logprob", 0.0)), value=float(d.get("value", 0.0)),
            reward=float(d.get("reward", 0.0)), done=bool(d.get("done", False)),
            decision_type=d.get("decision_type", "turn"), turn=int(d.get("turn", 0)),
            mask_s0=_mask_to_list(d.get("mask_s0")), mask_s1=_mask_to_list(d.get("mask_s1")),
            gmask_s0=_mask_to_list(d.get("gmask_s0")), gmask_s1=_mask_to_list(d.get("gmask_s1")),
        )


@dataclass
class EpisodeMeta:
    """One self-play episode from ONE perspective (each game yields TWO — sec 6)."""
    battle_id: str
    own_role: str                    # "p1" | "p2"
    own_team: List[str]              # the 6 own roster species
    opp_team: List[str]              # opp teampreview roster (seen at preview)
    tp_bring: List[int]              # indices into own_team brought (<=4) — recorded even w/ frozen TP (sec 14)
    tp_leads: List[int]              # indices into own_team led
    won: Optional[bool]              # True win / False loss / None draw-or-undecided
    terminal_type: str               # a TerminalType value
    n_turns: int                     # shared turn clock at end (for the symmetry/discount checks)
    sampling: Optional[dict] = None  # behaviour-policy params the recorded logprobs assume
                                     # ({"tau": float, "top_p": float}); τ<=0 = deterministic argmax
                                     # seat (logprob stamped 0.0); top_p<1 = the selection was
                                     # nucleus-truncated but the stored logprob is the UNtruncated
                                     # softmax — offline consumers must recompute if they need exact.

    def to_obj(self) -> dict:
        return {
            "battle_id": self.battle_id, "own_role": self.own_role,
            "own_team": list(self.own_team), "opp_team": list(self.opp_team),
            "tp_bring": [int(i) for i in self.tp_bring],
            "tp_leads": [int(i) for i in self.tp_leads],
            "won": self.won, "terminal_type": self.terminal_type,
            "n_turns": int(self.n_turns),
            "sampling": self.sampling,
        }

    @classmethod
    def from_obj(cls, d: dict) -> "EpisodeMeta":
        return cls(
            battle_id=d["battle_id"], own_role=d["own_role"],
            own_team=list(d.get("own_team", [])), opp_team=list(d.get("opp_team", [])),
            tp_bring=[int(i) for i in d.get("tp_bring", [])],
            tp_leads=[int(i) for i in d.get("tp_leads", [])],
            won=d.get("won"), terminal_type=d["terminal_type"],
            n_turns=int(d.get("n_turns", 0)),
            sampling=d.get("sampling"),
        )

    @property
    def is_trainable(self) -> bool:
        """FALLBACK trajectories are discarded from the PPO buffer (sec 1): an
        engineering desync must never become a learned target."""
        return TerminalType(self.terminal_type) is not TerminalType.FALLBACK

    @property
    def bootstraps(self) -> bool:
        """HORIZON_CUT is not a real game end -> bootstrap gamma*V(s_cut) instead of
        a +-1 terminal (NEVER hand +-1 at the artificial cut)."""
        return TerminalType(self.terminal_type) is TerminalType.HORIZON_CUT

    def outcome_reward(self) -> Optional[float]:
        """The terminal reward to place on the last step, or None when the episode
        does NOT carry an outcome reward (HORIZON_CUT bootstraps; FALLBACK is
        discarded). DRAW is a real terminal 0 (no bootstrap)."""
        tt = TerminalType(self.terminal_type)
        if tt is TerminalType.DRAW:
            return 0.0
        if tt in (TerminalType.WIN, TerminalType.LOSS, TerminalType.ADJUDICATED):
            return 1.0 if self.won else -1.0
        return None  # HORIZON_CUT -> bootstrap, FALLBACK -> discard


@dataclass
class Trajectory:
    """One perspective's full episode: metadata + ordered decision steps."""
    meta: EpisodeMeta
    transitions: List[Transition] = field(default_factory=list)

    def to_obj(self) -> dict:
        return {"meta": self.meta.to_obj(),
                "transitions": [t.to_obj() for t in self.transitions]}

    @classmethod
    def from_obj(cls, d: dict) -> "Trajectory":
        return cls(meta=EpisodeMeta.from_obj(d["meta"]),
                   transitions=[Transition.from_obj(t) for t in d.get("transitions", [])])
