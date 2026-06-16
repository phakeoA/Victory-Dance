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
for _p in (str(_REPO_ROOT / "data" / "scripts"), str(_REPO_ROOT), str(_HERE)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from poke_env.battle import DoubleBattle
from poke_env.player.battle_order import DoubleBattleOrder

# Reuse the tested root base + its pure helpers unchanged.
from vgc_base import (  # noqa: E402
    VGCPlayerBase as _RootVGCPlayerBase,
    build_legal_action_mask,
    random_legal_action,
)
import random as _random  # noqa: E402  (retry-exploration on rejected choices)

from live_state_encoder import opp_snapshot_from_log_prefix, opp_snapshot_current  # noqa: E402

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
        action_s0, action_s1, source = self._select_actions(battle, state_vec)

        # ── Retry exploration ───────────────────────────────────────────────
        # poke-env re-calls choose_move (without the battle advancing) when
        # Showdown rejects the order as "[Unavailable choice]".  A deterministic
        # policy would return the SAME order forever, so once we've already tried
        # this exact (a0,a1) for this (battle,turn) we perturb each slot to a
        # fresh legal action — exactly how the random player escapes the loop.
        key = (battle.battle_tag, battle.turn)
        if key not in self._tried_actions:           # new turn → forget old keys
            self._tried_actions = {key: {0: set(), 1: set()}}
        tried = self._tried_actions[key]
        if action_s0 in tried[0] and action_s1 in tried[1] and (tried[0] or tried[1]):
            action_s0 = self._fresh_legal(battle, 0, tried[0], action_s0)
            action_s1 = self._fresh_legal(battle, 1, tried[1], action_s1)
            source = "retry"
            # A retry means Showdown REJECTED the model's order — surface it loudly
            # (the bench/disabled-move mask fixes should make this ~never fire; if
            # it does it's an unmodelled lock to investigate, NOT normal play).
            log.warning(
                "Turn %d [%s] order REJECTED by Showdown — perturbing to a fresh "
                "legal action (src=retry). The model did NOT drive this slot.",
                battle.turn, battle.battle_tag,
            )
        tried[0].add(action_s0)
        tried[1].add(action_s1)

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

        order_s0 = self._safe_order(action_s0, battle, slot=0)
        order_s1 = self._safe_order(action_s1, battle, slot=1)
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

    # ── Forced replacement (post-faint) — encode the CURRENT opp board ──────────
    def _handle_force_switch(self, battle: DoubleBattle):
        """Encode the post-faint board with the gap-#6 opponent splice and record
        it (so replacement decisions carry the same opponent composition training's
        ``replacement`` transitions did), THEN defer the actual switch choice to
        the inherited handler."""
        try:
            self._source_counts["forced_switch"] += 1
            opp_snapshot = self._build_opp_snapshot_current(battle)
            state_vec = self._encoder.encode(battle, opp_snapshot=opp_snapshot)
            self._replay.record(
                battle_id=battle.battle_tag,
                turn=battle.turn,
                state=state_vec,
                action_s0=-1,            # replacement target chosen below; the
                action_s1=-1,            # forced-switch action codec is separate
                source="forced_switch",
            )
            log.debug(
                "forceSwitch encode [%s] turn %d opp_splice=%s",
                battle.battle_tag, battle.turn,
                "on" if opp_snapshot is not None else "off",
            )
        except Exception:
            log.debug("forceSwitch encode/record failed (non-fatal)", exc_info=True)
        return super()._handle_force_switch(battle)

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
