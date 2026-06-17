"""
local_battle/live_vgc_base.py  —  gap-#6 opponent-splice player base
====================================================================
A drop-in replacement for the repo-root ``vgc_base.VGCPlayerBase`` that wires
the gap-#6 opponent fix into LIVE play.

WHY
---
The root base calls ``LiveStateEncoder.encode(battle)``, which builds the
opponent side from poke-env's view.  poke-env keys opponent mons BY SPECIES, so
during a duplicate-species Zoroark illusion (a Zoroark disguised as a brought
teammate, e.g. two "Charizard" on the field) it MERGES the disguise with the
real same-species mon and loses one off the field — the bot then sees the wrong
opponent composition (Gap #6).

This base instead reconstructs the OPPONENT side from the public protocol log
with the SAME ``vod_parser`` the TRAINING data was built with, in REAL-TIME
(prefix-parse up to the current turn — no future ``|replace|`` can leak back),
and feeds it to the encoder via ``encode(battle, opp_snapshot=...)``.  The own
side + globals stay poke-env-derived (the own side carries private |request|
data).  Result: the live opponent view matches training, immune to the merge.

HOW
---
* ``_handle_battle_message`` is overridden to ACCUMULATE the raw protocol lines
  per battle (poke-env hands us the already-split protocol messages).
* ``choose_move`` builds an opponent snapshot from that log (synthesising the
  teampreview ``|poke|`` roster from poke-env, because this format ships open
  team sheets via ``|showteam|`` rather than ``|poke|``) and passes it to the
  encoder.  If anything fails it degrades silently to poke-env's view.

Everything else (teampreview, forceSwitch, replay recording, order building,
the action helpers) is inherited UNCHANGED from the root base, so this is a
narrow, low-risk addition.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Ensure local_battle/ is FIRST (so `import player`/`random_player` resolve to
# the spliced local versions, not the repo-root ones), then repo root (for
# vgc_base) + data/scripts (encoders).  remove-then-prepend so the order holds
# no matter which local module bootstraps first or how the script is launched.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
from poke_env.battle import DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import (
    DefaultBattleOrder, DoubleBattleOrder, PassBattleOrder,
)

# After this many perturbations on one turn (all rejected by Showdown), our legal
# mask is out of sync with the real board (an illusion/transform desync) — stop the
# 'retry storm' and let Showdown resolve the turn legally via /choose default.
_MAX_TURN_RETRIES = 10

# Reuse the tested root base + its pure helpers unchanged.
from v_dance.play.vgc_base import (  # noqa: E402
    VGCPlayerBase as _RootVGCPlayerBase,
    build_legal_action_mask,
    build_replacement_mask,
    random_legal_action,
    _switch_order_target,
)
import random as _random  # noqa: E402  (retry-exploration on rejected choices)

from v_dance.encoders.live_state_encoder import (  # noqa: E402
    opp_snapshot_from_log_prefix, opp_snapshot_current, own_bench_mons,
)
from v_dance.encoders.state_encoder import SWITCH_OFFSET, GIMMICK_NONE  # noqa: E402

log = logging.getLogger(__name__)

# poke-env id → Showdown display name (the vod_parser roster + seen-mon keys are
# in display space, e.g. "Charizard-Mega-Y"/"Arcanine-Hisui"; poke-env species
# ids are lowercase, e.g. "charizardmegay").  Used to synthesise |poke| lines.
try:
    from poke_env.data import GenData
    _DEX = GenData.from_gen(9).pokedex
except Exception:  # pragma: no cover - degrade if poke-env data unavailable
    _DEX = {}

# Client-only / private protocol messages that never appear in a replay log and
# would only confuse the vod_parser.  We skip them when capturing so the
# reconstructed log matches the request-less replay logs the parser was built on.
_CAPTURE_SKIP = {
    "", "request", "inactive", "inactiveoff", "t:", "j", "J", "l", "L", "n", "N",
    "c", "c:", "chat", "raw", "html", "uhtml", "uhtmlchange", "init", "title",
    "join", "leave", "popup", "updatesearch", "updateuser", "queryresponse",
    # open team sheets — the vod_parser can't read packed teams; we synthesise
    # |poke| roster lines from poke-env's resolved teampreview instead.
    "showteam",
}


def _norm_tag(s: Optional[str]) -> str:
    """Battle-room id with the leading '>' (and whitespace) stripped."""
    return (s or "").strip().lstrip(">").strip()


def _display_species(mon) -> Optional[str]:
    """Showdown display name for a poke-env mon's CURRENT species id."""
    sid = getattr(mon, "species", None)
    if not sid:
        return None
    entry = _DEX.get(sid)
    if entry and entry.get("name"):
        return entry["name"]
    return sid  # last-resort: the id itself (matches if no forme suffix)


class SplicingVGCPlayerBase(_RootVGCPlayerBase):
    """Root VGCPlayerBase + gap-#6 opponent splice (see module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # raw protocol lines accumulated per battle tag (real-time)
        self._proto_log: Dict[str, List[str]] = {}
        # per-(battle,turn) actions already tried, so a DETERMINISTIC policy
        # doesn't loop forever when Showdown reports a choice "[Unavailable]"
        # (e.g. a trapped switch / disabled move our approximate mask allows).
        self._tried_actions: dict = {}
        # per-source decision tally (model / retry / forced_switch / model_error …)
        # so a run can show, at a glance, that the AI — not randomness — is driving.
        self._source_counts: "Counter[str]" = Counter()
        # latest Showdown |error| text per battle tag — surfaced in the rejection
        # warnings so a mask desync shows the REASON ("Can't move: X is disabled",
        # "trapped", "Invalid choice", …) instead of an opaque "REJECTED".
        self._last_error: Dict[str, str] = {}

    # ── Protocol capture ───────────────────────────────────────────────────────
    async def _handle_battle_message(self, split_messages):  # type: ignore[override]
        """Accumulate the public protocol per battle, then defer to poke-env.

        Wrapped so a capture hiccup can never break message handling — the splice
        just degrades to poke-env's opponent view for that turn.
        """
        try:
            if split_messages and split_messages[0]:
                tag = _norm_tag(split_messages[0][0])
                if tag:
                    buf = self._proto_log.setdefault(tag, [])
                    for sm in split_messages[1:]:
                        if not sm or len(sm) < 2:
                            continue
                        if sm[1] == "error":     # Showdown's rejection reason (for diagnosis)
                            self._last_error[tag] = "|".join(sm[2:])[:200]
                        if sm[1] in _CAPTURE_SKIP:
                            continue
                        buf.append("|".join(sm))
        except Exception:
            log.debug("proto capture failed (non-fatal)", exc_info=True)
        return await super()._handle_battle_message(split_messages)

    # ── Decision: encode with the spliced opponent snapshot ─────────────────────
    def choose_move(self, battle: DoubleBattle):
        """Same control flow as the root base, but the normal-turn state vector
        is built with the gap-#6 opponent splice."""
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        if any(battle.force_switch):
            return self._handle_force_switch(battle)

        opp_snapshot = self._build_opp_snapshot(battle)
        state_vec = self._encoder.encode(battle, opp_snapshot=opp_snapshot)
        # gap-#6 reconstructed opponent slot occupancy → lets the codec target
        # opp_a vs opp_b DELIBERATELY when a same-species illusion makes poke-env
        # lose a foe slot (#15).  None when no reconstruction (legacy targeting).
        _oa = (opp_snapshot or {}).get("opp_active") or {}
        opp_present_recon = ({0: bool(_oa.get("opp_a")), 1: bool(_oa.get("opp_b"))}
                             if opp_snapshot else None)
        action_s0, action_s1, source = self._select_actions(battle, state_vec)

        # ── Forced-move escape (no representable legal action) ──────────────
        # If an ACTIVE, non-fainted slot has an EMPTY legal mask, its only usable order is one
        # the 16-action codec can't express — Struggle (all real moves unusable), a recharge /
        # 2-turn move's forced continuation, or a move whose live id != mon.moves. Passing such
        # a slot is ILLEGAL ("must make a move/switch") and the model has NO real choice, so
        # resolve the whole turn with /choose default (Showdown plays the forced move) instead
        # of the retry-storm + rejected Pass. Normal turns (non-empty mask) are unaffected.
        if self._active_empty_mask(battle):
            self._source_counts["forced_default"] += 1
            self._discard_rl_decision(battle, "turn")
            return DefaultBattleOrder()

        # ── Retry exploration ───────────────────────────────────────────────
        # poke-env re-calls choose_move (without the battle advancing) when
        # Showdown rejects the order as "[Unavailable choice]".  A deterministic
        # policy would return the SAME order forever, so once we've already tried
        # this exact (a0,a1) for this (battle,turn) we perturb each slot to a
        # fresh legal action — exactly how the random player escapes the loop.
        key = (battle.battle_tag, battle.turn)
        if key not in self._tried_actions:           # new turn → forget old keys
            self._tried_actions = {key: {0: set(), 1: set(), "n": 0}}
        tried = self._tried_actions[key]
        if action_s0 in tried[0] and action_s1 in tried[1] and (tried[0] or tried[1]):
            tried["n"] += 1
            # If every legal action for BOTH slots has already been tried + rejected,
            # or we've perturbed too many times, our mask is out of sync with the real
            # board (an illusion/transform desync) — STOP thrashing (the 'retry storm'
            # that floods the log + burns the watchdog) and let Showdown resolve the
            # turn legally via /choose default.  The model did NOT drive this turn.
            exhausted = (not self._has_fresh_legal(battle, 0, tried[0])
                         and not self._has_fresh_legal(battle, 1, tried[1]))
            if tried["n"] > _MAX_TURN_RETRIES or exhausted:
                log.warning("Turn %d [%s] order rejected %d× / legal actions exhausted "
                            "— '/choose default' (mask desync; model not driving). reason=%r",
                            battle.turn, battle.battle_tag, tried["n"],
                            self._last_error.get(battle.battle_tag, "?"))
                self._source_counts["retry_default"] += 1
                self._discard_rl_decision(battle, "turn")   # default executes, not the model
                return DefaultBattleOrder()
            action_s0 = self._fresh_legal(battle, 0, tried[0], action_s0)
            action_s1 = self._fresh_legal(battle, 1, tried[1], action_s1)
            source = "retry"
            # A retry means Showdown REJECTED the model's order — surface it loudly
            # (the bench/disabled-move mask fixes should make this ~never fire; if
            # it does it's an unmodelled lock to investigate, NOT normal play).
            log.warning(
                "Turn %d [%s] order REJECTED by Showdown (reason=%r) — perturbing to a "
                "fresh legal action (src=retry). The model did NOT drive this slot.",
                battle.turn, battle.battle_tag,
                self._last_error.get(battle.battle_tag, "?"),
            )
            # The model's pick was rejected; a random perturbation will execute instead
            # — drop the rejected step so the trajectory holds only executed decisions.
            self._discard_rl_decision(battle, "turn")
        tried[0].add(action_s0)
        tried[1].add(action_s1)

        # Gimmick (mega) decision for the FINAL actions.  A retry order is an
        # emergency perturbation, not the model's pick, so it never gimmicks.
        if source == "retry":
            g0 = g1 = GIMMICK_NONE
        else:
            g0, g1 = self._select_gimmicks(battle, state_vec, action_s0, action_s1)

        self._source_counts[source] += 1

        log.debug(
            "Turn %d [%s] a0=%d a1=%d src=%s opp_splice=%s",
            battle.turn, battle.battle_tag, action_s0, action_s1, source,
            "on" if opp_snapshot is not None else "off",
        )

        self._replay.record(
            battle_id=battle.battle_tag,
            turn=battle.turn,
            state=state_vec,
            action_s0=action_s0,
            action_s1=action_s1,
            source=source,
        )

        # Self-play collection hook (3c.1): records this decision into an RL
        # trajectory. A NO-OP in the base — zero behaviour change for live/gauntlet
        # play; only SelfPlayVGCPlayer overrides it.
        self._record_rl_decision(battle, state_vec, action_s0, action_s1,
                                 g0, g1, source, "turn")

        order_s0 = self._safe_order(action_s0, battle, slot=0, gimmick=g0,
                                    opp_present_recon=opp_present_recon)
        # Slot-1 must not switch in the same mon slot-0 is switching in ("slot N can
        # only switch in once").  Thread slot-0's resolved switch command so slot-1's
        # decode AND its fallback scan both avoid it — closes BOTH the retry-
        # perturbation collision and the under-illusion two-moves-fall-back-to-the-
        # same-switch collision the action-level dedup in _select_actions can't see.
        taken = _switch_order_target(order_s0)
        order_s1 = self._safe_order(action_s1, battle, slot=1, gimmick=g1,
                                    opp_present_recon=opp_present_recon,
                                    taken_switch_targets={taken} if taken else None)
        # _safe_order returns None when an ACTIVE, non-fainted slot has no representable
        # legal order (its only codec action — a switch — collides with the ally's, so
        # only a forced move the 16-action codec can't express remains, e.g. Struggle).
        # Passing it is illegal ("must make a move/switch") and would trigger the
        # retry-storm + a rejected order (the Sneasler desync the user's live run hit);
        # resolve the whole turn via /choose default instead — the cross-slot variant of
        # the empty-mask escape above.  The model did NOT drive this turn → drop the step.
        if order_s0 is None or order_s1 is None:
            self._source_counts["forced_default"] += 1
            self._discard_rl_decision(battle, "turn")
            return DefaultBattleOrder()
        return DoubleBattleOrder(order_s0, order_s1)

    @staticmethod
    def _fresh_legal(battle: DoubleBattle, slot: int, tried: set, fallback: int) -> int:
        """A random legal action for ``slot`` not already tried this turn; the
        fallback (the model's pick) if every legal action has been exhausted."""
        mask = build_legal_action_mask(battle, slot)
        legal = [i for i, ok in enumerate(mask) if ok and i not in tried]
        if legal:
            return _random.choice(legal)
        # all legal actions tried already → let poke-env's default break the tie
        return random_legal_action(battle, slot)

    @staticmethod
    def _has_fresh_legal(battle: DoubleBattle, slot: int, tried: set) -> bool:
        """Whether ``slot`` has any legal action NOT yet tried this turn.  When both
        slots have none, every legal action has been rejected → escape to default
        instead of re-trying rejected actions forever (the retry storm)."""
        mask = build_legal_action_mask(battle, slot)
        return any(ok and i not in tried for i, ok in enumerate(mask))

    # ── Opponent reconstruction (gap #6) ────────────────────────────────────────
    def _assemble_log(self, battle: DoubleBattle):
        """Return ``(log_str, own_role)`` for the captured protocol of this
        battle (with synthesised |poke| roster lines if the format shipped open
        team sheets instead), or ``(None, None)`` if prerequisites are missing."""
        own_role = getattr(battle, "player_role", None)
        if not own_role:
            return None, None
        lines = self._proto_log.get(_norm_tag(battle.battle_tag))
        if not lines:
            return None, None
        # This format ships open team sheets (|showteam|), not |poke|, so
        # synthesise the teampreview rosters poke-env resolved for us — but ONLY
        # if the captured log has no |poke| of its own (duplicating the roster
        # would double-list bench mons).
        has_poke = any(ln.startswith("|poke|") for ln in lines)
        poke_lines = [] if has_poke else self._synth_poke_lines(battle)
        return "\n".join(poke_lines + lines), own_role

    def _build_opp_snapshot(self, battle: DoubleBattle) -> Optional[dict]:
        """Opponent side as of the START of the current turn (normal-turn
        decisions).  Returns None (→ poke-env view) on any failure."""
        try:
            log_str, own_role = self._assemble_log(battle)
            if not log_str:
                return None
            return opp_snapshot_from_log_prefix(log_str, own_role, battle.turn)
        except Exception:
            log.debug("opp_snapshot build failed (degrading to poke-env)", exc_info=True)
            return None

    def _build_opp_snapshot_current(self, battle: DoubleBattle) -> Optional[dict]:
        """Opponent side as of the CURRENT (mid-turn / post-faint) board, for a
        forced-replacement decision — the start-of-turn snapshot is stale once a
        faint has happened this turn.  Returns None (→ poke-env view) on failure."""
        try:
            log_str, own_role = self._assemble_log(battle)
            if not log_str:
                return None
            return opp_snapshot_current(log_str, own_role)
        except Exception:
            log.debug("current opp_snapshot build failed (degrading to poke-env)", exc_info=True)
            return None

    # ── Forced replacement (post-faint) — encode + let the POLICY choose ────────
    def _handle_force_switch(self, battle: DoubleBattle):
        """Encode the post-faint board with the gap-#6 opponent splice (so the
        replacement decision sees the SAME opponent composition training's
        ``decision_type='replacement'`` transitions did), then let the policy pick
        the replacement via the ``_select_replacement_actions`` hook.  The base hook
        returns None → the inherited RANDOM picker; ``VGCPlayer`` overrides it with
        the model (masked-argmax of the fainted slot's head over a switch-only
        replacement mask).  Degrades to random on any failure."""
        # Loop guard FIRST: if Showdown rejected our last forced-switch order it
        # re-sends the identical request; the model would re-pick the same rejected
        # switch forever.  Shared escape (root): /choose default, then FORFEIT after
        # too many handlings — guarantees the battle can't hang.
        escape = self._force_switch_escape(battle)
        if escape is not None:
            self._source_counts["forced_switch_escape"] += 1
            self._discard_rl_decision(battle, "replacement")   # escape executes, not the model
            return escape

        state_vec = None
        opp_snapshot = None
        try:
            opp_snapshot = self._build_opp_snapshot_current(battle)
            state_vec = self._encoder.encode(battle, opp_snapshot=opp_snapshot)
        except Exception:
            log.debug("forceSwitch encode failed (non-fatal)", exc_info=True)

        repl = None
        if state_vec is not None:
            try:
                repl = self._select_replacement_actions(battle, state_vec)
            except Exception:
                log.debug("replacement selection failed (→ random)", exc_info=True)
                repl = None

        if repl is not None:
            a0, a1, source = repl
            order_s0 = self._replacement_order(battle, 0, a0)
            order_s1 = self._replacement_order(battle, 1, a1)
            self._source_counts[source] += 1
            # Self-play collection hook (3c.1) — replacement decision. No-op in base.
            self._record_rl_decision(battle, state_vec, a0, a1,
                                     GIMMICK_NONE, GIMMICK_NONE, source, "replacement")
            try:
                self._replay.record(
                    battle_id=battle.battle_tag, turn=battle.turn, state=state_vec,
                    action_s0=a0 if a0 is not None else -1,
                    action_s1=a1 if a1 is not None else -1, source=source,
                )
            except Exception:
                log.debug("forceSwitch record failed (non-fatal)", exc_info=True)
            log.debug(
                "forceSwitch %s [%s] turn %d a0=%s a1=%s opp_splice=%s",
                source, battle.battle_tag, battle.turn, a0, a1,
                "on" if opp_snapshot is not None else "off",
            )
            return DoubleBattleOrder(order_s0, order_s1)

        # Fallback: record the state (if any) then the inherited RANDOM pick.
        self._source_counts["forced_switch"] += 1
        if state_vec is not None:
            try:
                self._replay.record(
                    battle_id=battle.battle_tag, turn=battle.turn, state=state_vec,
                    action_s0=-1, action_s1=-1, source="forced_switch",
                )
            except Exception:
                log.debug("forceSwitch record failed (non-fatal)", exc_info=True)
        # Use the BUILDER (not _handle_force_switch) so the loop guard isn't
        # double-counted — we already called it at the top of this method.
        return super()._build_force_switch_order(battle)

    def _record_rl_decision(self, battle, state_vec, action_s0, action_s1,
                            gimmick_s0, gimmick_s1, source, decision_type):
        """Self-play collection hook (3c.1). Called with the FINAL decision each
        normal turn ("turn") and each model-driven forced replacement ("replacement").
        NO-OP in the base so live / gauntlet play is byte-identical; SelfPlayVGCPlayer
        overrides it to record the step (state, actions, gimmicks, masks, log-prob,
        value) into a TrajectoryCollector."""
        return None

    def _discard_rl_decision(self, battle, decision_type):
        """Self-play collection hook (3c.1c): the model's pick for THIS (turn,
        decision_type) did NOT execute — Showdown rejected it and a random perturbation
        / ``/choose default`` / forced-switch escape took over. SelfPlayVGCPlayer drops
        the non-executed step so the trajectory holds only EXECUTED model decisions.
        NO-OP in the base."""
        return None

    def _select_replacement_actions(self, battle: DoubleBattle, state_vec):
        """Hook: return ``(a0, a1, source)`` — switch action indices (12..15) for the
        fainted slot(s), ``None`` for a slot that is not switching — or ``None`` to
        defer to the inherited RANDOM replacement.  Base = random (None); the
        model-driven override lives in ``VGCPlayer``."""
        return None

    @staticmethod
    def _replacement_order(battle: DoubleBattle, slot: int, action):
        """A SingleBattleOrder for a replacement switch action (or Pass).  ``action``
        is a switch index (12..15) → ``own_bench_mons(battle)[action-12]``; None → a
        non-switching slot → Pass."""
        if action is None:
            return PassBattleOrder()
        bench = own_bench_mons(battle)
        i = action - SWITCH_OFFSET
        if 0 <= i < len(bench):
            from v_dance.play.vgc_base import _log_switch_choice, build_switch_order
            _log_switch_choice(battle, slot, bench[i], "replacement")
            return build_switch_order(battle, bench[i])
        return PassBattleOrder()

    @staticmethod
    def _synth_poke_lines(battle: DoubleBattle) -> List[str]:
        """Build ``|poke|<role>|<DisplaySpecies>, L50|`` lines for both sides
        from poke-env's teampreview rosters, so the parser can fill unseen
        opponent-bench stubs exactly as it did for the training replays."""
        out: List[str] = []

        def roster(tp, full) -> list:
            mons = list(tp or [])
            if not mons:
                mons = list((full or {}).values())
            return mons

        pr = getattr(battle, "player_role", None)
        opr = getattr(battle, "opponent_role", None)
        if pr:
            for m in roster(getattr(battle, "teampreview_team", None),
                            getattr(battle, "team", None)):
                name = _display_species(m)
                if name:
                    out.append(f"|poke|{pr}|{name}, L50|")
        if opr:
            for m in roster(getattr(battle, "teampreview_opponent_team", None),
                            getattr(battle, "opponent_team", None)):
                name = _display_species(m)
                if name:
                    out.append(f"|poke|{opr}|{name}, L50|")
        return out
