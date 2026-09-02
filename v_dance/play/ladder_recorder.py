"""Ladder trajectory recorder — W3b-0 (2026-09-02, docs/w3b_ladder_ppo_design.md §4).

The learning half of the USER's live-training idea needs the ladder's games as RL trajectories, not
only as winner-filtered replays. This records every MODEL decision the online bot makes into the
self-play ``Trajectory`` schema (state, actions, gimmicks, the EFFECTIVE legal masks the sampler
used, decision type, turn, the served model's value-head estimate) and seals each game with the
±1 terminal reward (the reward bible: terminal only, no shaping) plus per-game metadata — which
bandit ARM played it, pinned or not, τ / top-p, pair decode, adapt-rules, opponent + ratings.

⚠ ``logprob`` is a PLACEHOLDER (0.0) and every game is stamped ``sampling.logprob_valid = false``
until W3b-1a records the behaviour log-prob at the sampling site (model_io). Trajectories so stamped
must never be trained on — the update script refuses them. Everything else needed for an exact
recompute later is here.

Zero-impact when disabled (``VD_LADDER_RECORD=0``): the hooks stay the base no-ops. Every hook is
guarded — recording must never cost a turn.
"""
from __future__ import annotations

import logging
import time
import types
from pathlib import Path
from typing import Callable, Dict, Optional

from v_dance.selfplay.collector import TrajectoryCollector
from v_dance.selfplay.reward import _MODEL_DRIVEN_SOURCES, place_terminal_reward
from v_dance.selfplay.schema import PASS_ACTION
from v_dance.selfplay.store import TrajectoryStore

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
LADDER_RL_DIR = _REPO / "artifacts" / "ladder_rl"


def terminal_type_for(won: Optional[bool], lost: Optional[bool]) -> str:
    """Mirror of ``selfplay.game_runner.terminal_type_for`` (kept torch-free here): a real win / loss,
    else a true draw (reward 0, no bootstrap)."""
    if won:
        return "win"
    if lost:
        return "loss"
    return "draw"


class LadderRecorder:
    """Installs the ``_record_rl_decision`` / ``_discard_rl_decision`` hooks on the served player and
    seals one ``Trajectory`` per finished game into ``artifacts/ladder_rl/<fmt>/<session>.jsonl``."""

    def __init__(self, player, store_path, *, session_id: str, fmt: str,
                 arm_info: Optional[Callable[[str], Optional[dict]]] = None,
                 adapt_rules: bool = False, mask_builders=None, now=None) -> None:
        self.player = player
        self.path = Path(store_path)
        self.store = TrajectoryStore(self.path)
        self.session_id, self.fmt = str(session_id), str(fmt)
        self.arm_info = arm_info                    # tag -> {arm, pinned, tau, top_p, pair_decode}
        self.adapt_rules = bool(adapt_rules)
        self._mask_builders = mask_builders         # (legal, replacement, gimmick) — injectable
        self._now = now or time.time
        self._collectors: Dict[str, TrajectoryCollector] = {}
        self.games = 0
        self.steps = 0
        self.rejected = 0                           # rejection re-calls replaced (same turn + type)
        self.failed = 0                             # guarded hook failures (never cost a turn)
        self.skipped_empty = 0                      # games that ended with no model decision recorded
        self.install()

    # ── hooks ────────────────────────────────────────────────────────────────
    def install(self) -> None:
        p = self.player
        p._record_masks = True                      # stash the EFFECTIVE per-slot masks (#10/#11)
        if not isinstance(getattr(p, "_sampling_masks", None), dict):
            p._sampling_masks = {}
        rec = self
        p._record_rl_decision = types.MethodType(lambda _p, *a, **k: rec.record(*a, **k), p)
        p._discard_rl_decision = types.MethodType(lambda _p, *a, **k: rec.discard(*a, **k), p)

    def _builders(self):
        if self._mask_builders is None:
            from v_dance.play.vgc_base import (build_gimmick_legal_mask, build_legal_action_mask,
                                               build_replacement_mask)
            self._mask_builders = (build_legal_action_mask, build_replacement_mask,
                                   build_gimmick_legal_mask)
        return self._mask_builders

    @staticmethod
    def base_tag(tag: str) -> str:
        """``battle-<fmt>-<id>`` without a private-room password suffix — the id the bench rows and
        the bandit use, so trajectories join on the same key (live check 09-02: one private game
        sealed under its 33-char suffixed id)."""
        parts = (tag or "").lstrip(">").split("-")
        return "-".join(parts[:3]) if len(parts) >= 3 else (tag or "")

    def _collector_for(self, battle) -> TrajectoryCollector:
        tag = battle.battle_tag                     # keyed by the tag the hooks see (suffix included)
        c = self._collectors.get(tag)
        if c is None:
            c = TrajectoryCollector(self.base_tag(tag), getattr(battle, "player_role", None) or "p?")
            self._collectors[tag] = c
        return c

    def record(self, battle, state_vec, a0, a1, g0, g1, source, decision_type) -> None:
        """One FINAL model decision (turn or model-driven forced replacement)."""
        if source not in _MODEL_DRIVEN_SOURCES:
            return                                  # retry / default / escape are not the model's pick
        try:
            tag = battle.battle_tag
            stash = None
            sm = getattr(self.player, "_sampling_masks", None)
            if isinstance(sm, dict):
                stash = sm.pop((tag, decision_type), None)
            legal, repl, gim = self._builders()
            if decision_type == "replacement":
                m0 = stash[0] if (stash and stash[0] is not None) else repl(battle, 0)
                m1 = stash[1] if (stash and stash[1] is not None) else repl(battle, 1)
                gm0 = gm1 = None
            else:
                m0 = stash[0] if (stash and stash[0] is not None) else legal(battle, 0)
                m1 = stash[1] if (stash and stash[1] is not None) else legal(battle, 1)
                gm0, gm1 = gim(battle, 0), gim(battle, 1)
            c = self._collector_for(battle)
            turn = int(getattr(battle, "turn", 0) or 0)
            last = c.last_step()
            if last is not None and last.turn == turn and last.decision_type == decision_type:
                c.pop_step()                        # Showdown rejected the prior order → keep the executed one
                self.rejected += 1
            wp = getattr(self.player, "_last_value", None)
            value = (2.0 * float(wp) - 1.0) if isinstance(wp, (int, float)) else 0.0
            c.add_step(state=state_vec,
                       action_s0=(PASS_ACTION if a0 is None else int(a0)),
                       action_s1=(PASS_ACTION if a1 is None else int(a1)),
                       gimmick_s0=int(g0 or 0), gimmick_s1=int(g1 or 0),
                       logprob=0.0,                 # placeholder until W3b-1a (logprob_valid=false)
                       value=value, decision_type=decision_type, turn=turn,
                       mask_s0=m0, mask_s1=m1, gmask_s0=gm0, gmask_s1=gm1)
            self.steps += 1
        except Exception:
            self.failed += 1
            log.debug("ladder record failed (non-fatal)", exc_info=True)

    def discard(self, battle, decision_type) -> None:
        """The model's pick for this (turn, type) did NOT execute — drop it."""
        try:
            c = self._collectors.get(battle.battle_tag)
            if c is None:
                return
            last = c.last_step()
            turn = int(getattr(battle, "turn", 0) or 0)
            if last is not None and last.turn == turn and last.decision_type == decision_type:
                c.pop_step()
        except Exception:
            self.failed += 1
            log.debug("ladder discard failed (non-fatal)", exc_info=True)

    # ── sealing a game ───────────────────────────────────────────────────────
    def finish(self, tag: str, battle=None, *, won: Optional[bool] = None,
               lost: Optional[bool] = None, opponent: Optional[str] = None,
               rating_before=None, opp_rating_before=None, turn: Optional[int] = None,
               lane: Optional[int] = None):
        """Seal ``tag``'s trajectory: ±1 terminal reward, per-game metadata, append to the store.
        Returns the Trajectory, or None when nothing was recorded for the game."""
        c = self._collectors.pop(tag, None)
        if c is None or len(c) == 0:
            self.skipped_empty += 1
            return None
        tp = ((getattr(self.player, "_tp_decision", None) or {}).get(tag)) or {}
        own_team = list(tp.get("own_team") or self._roster(battle, own=True))
        opp_team = list(tp.get("opp_team") or self._roster(battle, own=False))
        bring = list(tp.get("bring") or range(min(4, len(own_team))))
        leads = list(tp.get("leads") or bring[:2])
        info: dict = {}
        if self.arm_info is not None:
            try:
                info = dict(self.arm_info(tag) or {})
            except Exception:
                info = {}
        model = getattr(self.player, "_model", None)
        sampling = {
            "source": "ladder", "session": self.session_id, "fmt": self.fmt,
            "arm": info.get("arm"), "pinned": bool(info.get("pinned", False)),
            "tau": float(info.get("tau", getattr(self.player, "_temperature", 0.0) or 0.0)),
            "top_p": float(info.get("top_p", getattr(self.player, "_top_p", 1.0) or 1.0)),
            "pair_decode": bool(info.get("pair_decode", getattr(model, "_pair_decode", False))),
            "adapt_rules": bool(info.get("adapt_rules", self.adapt_rules)),
            "logprob_valid": False,                 # W3b-1a flips this when the sampler records it
            "value_source": "serve_value_head",
            "opponent": opponent, "rating_before": rating_before,
            "opp_rating_before": opp_rating_before, "lane": lane,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._now())),
        }
        n_turns = int(turn if turn is not None else (getattr(battle, "turn", 0) or 0))
        traj = c.finish(own_team=own_team, opp_team=opp_team, tp_bring=bring, tp_leads=leads,
                        won=(None if won is None else bool(won)),
                        terminal_type=terminal_type_for(won, lost), n_turns=n_turns,
                        sampling=sampling)
        if traj.meta.is_trainable:
            place_terminal_reward(traj)
        self.store.append(traj)
        self.games += 1
        return traj

    @staticmethod
    def _roster(battle, *, own: bool) -> list:
        if battle is None:
            return []
        src = (getattr(battle, "teampreview_team", None) if own
               else getattr(battle, "teampreview_opponent_team", None))
        if not src:
            src = getattr(battle, "team", None) if own else getattr(battle, "opponent_team", None)
        mons = list(src.values()) if isinstance(src, dict) else list(src or [])
        return [sp for sp in (getattr(m, "species", None) for m in mons) if sp]

    # ── reporting ────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        return {"games": self.games, "steps": self.steps, "rejected": self.rejected,
                "failed": self.failed, "skipped_empty": self.skipped_empty,
                "open": len(self._collectors), "path": str(self.path)}

    def banner(self) -> str:
        try:
            rel = self.path.relative_to(_REPO)
        except ValueError:
            rel = self.path
        return (f"[online] ladder RECORDER ACTIVE — trajectories → {rel} (states / actions / masks / "
                f"value / arm per game; logprob placeholder until W3b-1a); VD_LADDER_RECORD=0 disables")

    def close(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass
