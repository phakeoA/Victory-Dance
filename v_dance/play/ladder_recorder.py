"""Ladder trajectory recorder — W3b-0 (2026-09-02, docs/w3b_ladder_ppo_design.md §4).

The learning half of the USER's live-training idea needs the ladder's games as RL trajectories, not
only as winner-filtered replays. This records every MODEL decision the online bot makes into the
self-play ``Trajectory`` schema (state, actions, gimmicks, the EFFECTIVE legal masks the sampler
used, decision type, turn, the served model's value-head estimate) and seals each game with the
±1 terminal reward (the reward bible: terminal only, no shaping) plus per-game metadata — which
bandit ARM played it, pinned or not, τ / top-p, pair decode, adapt-rules, opponent + ratings.

W3b-1a (2026-09-02): ``logprob`` is the step's joint BEHAVIOUR log-prob, summed from the per-slot
log-probs the sampler recorded at the sampling site (``model_io.masked_sample_logp`` → the decode
record → ``player._sampling_logp``): under the 2b decode that is log p(a_first) + log p(a_second |
a_first) over the EFFECTIVE masks; the gimmick and forced-replacement heads are argmax in serve
(log-prob 0 by the τ→0 convention — ``sampling.gimmick_sampled`` / ``replacement_sampled`` say so,
the evaluator must skip those terms). A game seals ``sampling.logprob_valid = true`` ONLY when every
turn step carried a sampler log-prob AND the arm is clean (τ > 0, top-p 1.0, adapt-rules off, one τ
for the whole game); otherwise ``logprob_reason`` says why and the update script refuses it.
``logprob_inexact_steps`` counts cross-slot switch-dedup re-decodes (slot 1's term is the re-decode's
own distribution — a close approximation, model_io.merge_dedup_records).

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
        # W3b-1a: behaviour log-prob accounting — per open game and session totals
        self._lp: Dict[str, dict] = {}              # tag -> {turn, sampler, missing, inexact, repl, taus, top_ps}
        self.lp_steps = 0                           # turn steps whose log-prob came from the sampler
        self.lp_inexact = 0                         # ... of which cross-slot dedup re-decodes
        self.valid_games = 0                        # games sealed logprob_valid=true
        self.install()

    # ── hooks ────────────────────────────────────────────────────────────────
    def install(self) -> None:
        p = self.player
        p._record_masks = True                      # stash the EFFECTIVE per-slot masks (#10/#11)
        if not isinstance(getattr(p, "_sampling_masks", None), dict):
            p._sampling_masks = {}
        if not isinstance(getattr(p, "_sampling_logp", None), dict):   # W3b-1a behaviour log-probs
            p._sampling_logp = {}
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
            # W3b-1a: the behaviour log-prob the sampler recorded for THIS decision (None = none)
            lp = None
            slp = getattr(self.player, "_sampling_logp", None)
            if isinstance(slp, dict):
                lp = slp.pop((tag, decision_type), None)
            c = self._collector_for(battle)
            turn = int(getattr(battle, "turn", 0) or 0)
            last = c.last_step()
            if last is not None and last.turn == turn and last.decision_type == decision_type:
                c.pop_step()                        # Showdown rejected the prior order → keep the executed one
                self.rejected += 1
                self._uncount(tag, last.decision_type)
            wp = getattr(self.player, "_last_value", None)
            value = (2.0 * float(wp) - 1.0) if isinstance(wp, (int, float)) else 0.0
            logprob = self._joint_logprob(tag, decision_type, lp)
            c.add_step(state=state_vec,
                       action_s0=(PASS_ACTION if a0 is None else int(a0)),
                       action_s1=(PASS_ACTION if a1 is None else int(a1)),
                       gimmick_s0=int(g0 or 0), gimmick_s1=int(g1 or 0),
                       logprob=logprob, value=value, decision_type=decision_type, turn=turn,
                       mask_s0=m0, mask_s1=m1, gmask_s0=gm0, gmask_s1=gm1,
                       # W3b-1b: the behaviour decode's first slot (pair decode only)
                       pair_first=(lp.get("first") if (isinstance(lp, dict) and lp.get("pair")
                                                       and decision_type == "turn") else None))
            self.steps += 1
        except Exception:
            self.failed += 1
            log.debug("ladder record failed (non-fatal)", exc_info=True)

    # ── W3b-1a: behaviour log-prob accounting ────────────────────────────────
    def _lp_for(self, tag: str) -> dict:
        st = self._lp.get(tag)
        if st is None:
            st = self._lp[tag] = {"turn": 0, "sampler": 0, "missing": 0, "inexact": 0, "repl": 0,
                                  "taus": set(), "top_ps": set(), "last": []}
        return st

    def _joint_logprob(self, tag: str, decision_type: str, lp: Optional[dict]) -> float:
        """Sum the per-slot sampler log-probs into the step's joint behaviour log-prob (a slot with
        no pick contributes 0, mirroring the evaluator's PASS handling); the gimmick term is 0
        (argmax in serve). Books the step so ``finish`` can judge ``logprob_valid``."""
        st = self._lp_for(tag)
        terms = list((lp or {}).get("logp") or ()) if isinstance(lp, dict) else []
        have = bool(lp) and any(t is not None for t in terms)
        total = float(sum(float(t) for t in terms if t is not None)) if have else 0.0
        if decision_type == "replacement":
            st["repl"] += 1
            st["last"].append(("replacement", False, False))
            return total                            # argmax in serve → 0.0 by convention
        st["turn"] += 1
        if have:
            st["sampler"] += 1
            self.lp_steps += 1
            st["taus"].add(round(float(lp.get("tau", 0.0) or 0.0), 6))
            st["top_ps"].add(round(float(lp.get("top_p", 1.0) or 1.0), 6))
            inexact = not bool(lp.get("exact", True))
            if inexact:
                st["inexact"] += 1
                self.lp_inexact += 1
            st["last"].append(("turn", True, inexact))
        else:
            st["missing"] += 1
            st["last"].append(("turn", False, False))
        return total

    def _uncount(self, tag: str, decision_type: str) -> None:
        """A step was popped (rejection re-call / discard) — undo its booking."""
        st = self._lp.get(tag)
        if not st or not st["last"]:
            return
        kind, sampler, inexact = st["last"].pop()
        if kind == "replacement":
            st["repl"] = max(0, st["repl"] - 1)
            return
        st["turn"] = max(0, st["turn"] - 1)
        if sampler:
            st["sampler"] = max(0, st["sampler"] - 1)
            self.lp_steps = max(0, self.lp_steps - 1)
            if inexact:
                st["inexact"] = max(0, st["inexact"] - 1)
                self.lp_inexact = max(0, self.lp_inexact - 1)
        else:
            st["missing"] = max(0, st["missing"] - 1)

    @staticmethod
    def logprob_verdict(st: Optional[dict], *, tau: float, top_p: float, adapt_rules: bool) -> dict:
        """The seal's log-prob fields for one game (pure — testable): ``logprob_valid`` is true only
        when every turn step carried a sampler log-prob AND the arm is clean (τ > 0, top-p 1.0,
        adapt-rules off, one τ for the whole game that matches the arm's). Otherwise
        ``logprob_reason`` lists why. ``logprob_source`` = sampler / mixed / placeholder."""
        st = st or {}
        turn, sampler = int(st.get("turn", 0)), int(st.get("sampler", 0))
        missing, inexact = int(st.get("missing", 0)), int(st.get("inexact", 0))
        taus, top_ps = set(st.get("taus") or ()), set(st.get("top_ps") or ())
        reasons = []
        if tau <= 0.0:
            reasons.append("argmax arm (tau 0): behaviour log-prob is 0 by convention")
        if top_p < 1.0:
            reasons.append(f"top-p {top_p:g} truncation (PPO needs top-p 1.0)")
        if adapt_rules:
            reasons.append("adapt-rules logit bias not reproducible")
        if turn == 0:
            reasons.append("no turn decisions recorded")
        if missing:
            reasons.append(f"{missing} turn step(s) without a sampler log-prob")
        if len(taus) > 1:
            reasons.append(f"tau changed mid-game ({sorted(taus)})")
        elif taus and abs(next(iter(taus)) - round(float(tau), 6)) > 1e-6:
            reasons.append(f"sampler tau {next(iter(taus))} differs from the arm's {tau:g}")
        if any(tp < 1.0 for tp in top_ps) and top_p >= 1.0:
            reasons.append("sampler used top-p < 1.0")
        source = "sampler" if (turn and not missing) else ("mixed" if sampler else "placeholder")
        return {"logprob_valid": not reasons, "logprob_source": source,
                "logprob_reason": ("; ".join(reasons) if reasons else None),
                "logprob_inexact_steps": inexact, "turn_steps": turn,
                "replacement_steps": int(st.get("repl", 0))}

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
                self._uncount(battle.battle_tag, decision_type)
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
            self._lp.pop(tag, None)                 # W3b-1a: nothing to seal → drop its booking
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
        tau = float(info.get("tau", getattr(self.player, "_temperature", 0.0) or 0.0))
        top_p = float(info.get("top_p", getattr(self.player, "_top_p", 1.0) or 1.0))
        adapt = bool(info.get("adapt_rules", self.adapt_rules))
        verdict = self.logprob_verdict(self._lp.pop(tag, None), tau=tau, top_p=top_p, adapt_rules=adapt)
        if verdict["logprob_valid"]:
            self.valid_games += 1
        sampled_heads = bool(getattr(self.player, "_collect_sample", False))
        sampling = {
            "source": "ladder", "session": self.session_id, "fmt": self.fmt,
            "arm": info.get("arm"), "pinned": bool(info.get("pinned", False)),
            "tau": tau, "top_p": top_p,
            "pair_decode": bool(info.get("pair_decode", getattr(model, "_pair_decode", False))),
            "adapt_rules": adapt,
            # W3b-1a: the behaviour log-prob's provenance + the evaluator's parity contract
            **verdict,
            "logprob_site": "model_io.masked_sample_logp",
            "gimmick_sampled": sampled_heads,       # argmax in serve → its term is 0 (skip it)
            "replacement_sampled": sampled_heads,   # argmax in serve → replacement steps carry 0
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
                "open": len(self._collectors), "path": str(self.path),
                # W3b-1a: how much of the session is PPO-trainable
                "lp_steps": self.lp_steps, "lp_inexact": self.lp_inexact,
                "valid_games": self.valid_games}

    def banner(self) -> str:
        try:
            rel = self.path.relative_to(_REPO)
        except ValueError:
            rel = self.path
        return (f"[online] ladder RECORDER ACTIVE — trajectories → {rel} (states / actions / masks / "
                f"value / arm per game; behaviour log-prob from the sampler [W3b-1a] — clean τ arms seal "
                f"logprob_valid=true); VD_LADDER_RECORD=0 disables")

    def close(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass
