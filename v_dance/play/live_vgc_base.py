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
    DefaultBattleOrder, DoubleBattleOrder, ForfeitBattleOrder, PassBattleOrder,
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
    reconstruct_for_decision, reconstruct_full_for_decision,
)
from v_dance.encoders.state_encoder import SWITCH_OFFSET, GIMMICK_NONE, _WEATHER_SPEED_ABILITY  # noqa: E402
from v_dance.parser.vod_parser.pokedex import norm_species  # noqa: E402  (A3 belief-feed species keys)

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
        # #18 multi-battle spectate (popped BEFORE super — poke-env doesn't know these): when
        # live_dir is set, each battle writes its room + growing |-log to <live_dir>/<tag>.json,
        # and on finish the replay is SAVED (save_replays) or deleted. Everything is guarded on
        # self._live, so live/gauntlet play that doesn't set live_dir is byte-identical.
        live_dir = kwargs.pop("live_dir", None)
        save_replays = kwargs.pop("save_replays", False)
        # task E: where the SAVED html replay goes (else self._live.dir) + its filename prefix, so the
        # eval gauntlet can file replays into eval/<kind>/ and eval/league/ named gen<N>_vs_<...>.
        replay_dir = kwargs.pop("replay_dir", None)
        replay_label = kwargs.pop("replay_label", None)
        # 2026-07-10 (USER): SECOND copy of every saved replay into the training-corpus folder
        # (data/vods/Type_C) — session-prefixed names, see LiveBattles.save_html_replay.
        replay_copy_dir = kwargs.pop("replay_copy_dir", None)
        # Level C / A3 (live wiring): within-game opponent belief (MatchBelief) PRE-ENRICHMENT of the
        # gap-#6 opponent snapshot. OFF by default → prod serving stays byte-identical (the un-spliced
        # static-belief path); the A/B harness / overnight self-play flips it on. See _apply_match_belief.
        use_match_belief = kwargs.pop("use_match_belief", False)
        # Level C / B1 (search): belief-weighted expectimax in front of the served net. OFF by default →
        # prod serving stays byte-identical (the policy-argmax path). The A/B harness flips it on.
        use_search = kwargs.pop("use_search", False)
        # Phase-3 B-L1 (adapt-rules): serve-time pattern tilt (Wide-Guard streak → spread-move
        # logit bias). OFF by default → prod byte-identical. See v_dance/play/adapt_rules.py.
        adapt_rules = kwargs.pop("adapt_rules", False)
        # S1 L2b (dossier warm-start, 2026-07-12): fill unknown opp item/ability/moves from the
        # per-opponent dossier BEFORE belief enrichment. OFF by default → prod byte-identical.
        use_dossier = kwargs.pop("use_dossier", False)
        super().__init__(*args, **kwargs)
        self._adapt_rules = bool(adapt_rules)
        self._use_dossier = bool(use_dossier)
        # raw protocol lines accumulated per battle tag (real-time)
        self._proto_log: Dict[str, List[str]] = {}
        # Level C / A3: per-battle within-game opponent belief, keyed by battle tag like _proto_log (the
        # LiveStateEncoder is a per-PLAYER singleton — its BeliefState is GLOBAL across battles — so the
        # within-game belief CANNOT live on it). Created lazily per tag in _match_belief_for, reset in
        # _battle_finished_callback. Empty + unused unless use_match_belief is on.
        self._use_match_belief = bool(use_match_belief)
        # Level C / B1: lazily-built offline encoder + search config; per-battle search/fallback diagnostics.
        self._use_search = bool(use_search)
        self._search_encoder = None
        self._search_cfg = None
        self._search_counts = {"search": 0, "fallback": 0}
        if self._use_search:
            from v_dance.encoders.state_encoder import VodStateEncoder
            from v_dance.play.search import SearchConfig
            self._search_encoder = VodStateEncoder()
            self._search_cfg = SearchConfig()
        self._match_belief: Dict[str, "MatchBelief"] = {}
        # Phase-3 serve-time MEMORY CARRY (frame-stacked, no hidden state): with a
        # memory model (checkpoint memory_dim>0) every decision runs on the battle's
        # per-tag frame ring buffer [(turn, decision_type, state_vec), …] via
        # forward_with_memory; a stateless model never populates this (zero cost).
        # Frames key on (turn, decision_type) so poke-env's same-turn re-calls
        # (rejected orders) REPLACE the frame instead of stacking duplicates —
        # mirroring the trajectory collector's rejected-resample discipline.
        # Reset per battle in _battle_finished_callback.
        self._mem_frames: Dict[str, list] = {}
        # A3 idempotency: the last turn whose prev-turn events were fed into each battle's MatchBelief.
        # poke-env re-invokes choose_move for the SAME turn on a rejected/re-requested order, so without
        # this guard the same damage/speed events would be appended (and the Bayesian product double-
        # counted) on every retry. Reset in _battle_finished_callback. (audit 2026-06-30)
        self._belief_fed_turn: Dict[str, int] = {}
        # A3 diagnostics: cumulative belief-feed counts (dmg_used_def / dmg_used_off / spd_used / skips…)
        # over this player's lifetime — surfaced by the self-play runner so a run can CONFIRM the belief
        # actually fired (an empty count means belief-ON == belief-OFF and the A/B is null).
        self._belief_stats: "Counter[str]" = Counter()
        # ⚠ MUST NOT be named ``_save_replays`` — poke-env's ``Player`` uses that exact attribute as
        # its OWN native-replay flag, which ``_create_battle`` reads to stamp every battle and
        # ``AbstractBattle._finish_battle`` then uses to dump an UNCATEGORIZED ``./replays/<user> -
        # <tag>.html``. We pop the kwarg (so poke-env's flag stays False) and keep OUR flag under a
        # DISTINCT name, so poke-env's native dump never fires and only OUR structured
        # ``<live_dir>/<tag>.html`` (see _battle_finished_callback) is written.
        self._save_html_replays = bool(save_replays)
        self._replay_dir = replay_dir
        self._replay_label = replay_label
        self._replay_copy_dir = replay_copy_dir
        self._live = None
        if live_dir:
            from v_dance.selfplay.status import LiveBattles
            self._live = LiveBattles(live_dir, min_interval=0.5)
        # per-(battle,turn) actions already tried, so a DETERMINISTIC policy
        # doesn't loop forever when Showdown reports a choice "[Unavailable]"
        # (e.g. a trapped switch / disabled move our approximate mask allows).
        self._tried_actions: dict = {}
        # T3.3: per-(battle,turn) count of forced-aware partial-order builds (recharge/Struggle on one
        # slot + a free partner). If that partial order keeps getting REJECTED (a free-slot mask desync),
        # poke-env re-calls choose_move with the same (battle,turn) and the branch would rebuild the
        # same rejected order forever — this counter escalates it to /choose default after N, mirroring
        # the retry path's _MAX_TURN_RETRIES escape (the _active_empty_mask branch has no other guard).
        self._forced_aware_attempts: dict = {}
        # per-source decision tally (model / retry / forced_switch / model_error …)
        # so a run can show, at a glance, that the AI — not randomness — is driving.
        self._source_counts: "Counter[str]" = Counter()
        # battle tags already counted as a forfeit (so a server re-request before our /forfeit
        # is processed does not inflate the per-battle forfeit tally — fs-monitor #5).
        self._forfeited_tags: set = set()
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

    def _memory_stack(self, battle, state_vec, decision_type: str):
        """The model input for THIS decision: the raw ``state_vec`` for a
        stateless model, or the battle's frame-stack ``(T, STATE_DIM)`` (this
        decision appended last) for a memory model — the Phase-3 serve-time
        memory carry.  Recording paths keep the single frame; only the model
        input is stacked."""
        mdl = getattr(self, "_model", None)
        if state_vec is None or mdl is None or not int(getattr(mdl, "memory_dim", 0) or 0):
            return state_vec
        try:
            import numpy as _np
            tag = _norm_tag(getattr(battle, "battle_tag", None))
            frames = self._mem_frames.setdefault(tag, [])
            key = (int(getattr(battle, "turn", 0) or 0), decision_type)
            vec = _np.asarray(state_vec, dtype=_np.float32)
            if frames and frames[-1][0][:2] == key:
                if _np.array_equal(frames[-1][1], vec):
                    frames[-1] = (frames[-1][0], vec)   # same-decision re-call → replace, don't stack
                else:
                    # Same (turn, type) but the STATE moved on — e.g. a SECOND
                    # forced replacement after the first resolved. That is a
                    # distinct decision: sub-index it so it never overwrites the
                    # first one's frame (training sees both as separate examples).
                    frames.append(((key[0], key[1], frames[-1][0][2] + 1), vec))
            else:
                frames.append(((key[0], key[1], 0), vec))
            # Cap at the TRAINED window (model_io stamps _sequence_len), not just
            # max_mem_len — ages beyond the trained window would read untrained
            # positional rows (audit 2026-07-02).
            maxlen = int(getattr(mdl, "max_mem_len", 64) or 64)
            trained = int(getattr(mdl, "_sequence_len", 0) or 0)
            if trained > 1:
                maxlen = min(maxlen, trained)
            if len(frames) > maxlen:
                del frames[: len(frames) - maxlen]
            return _np.stack([f[1] for f in frames])
        except Exception:
            log.debug("memory-stack build failed — single-frame fallback", exc_info=True)
            return state_vec

    # ── Decision: encode with the spliced opponent snapshot ─────────────────────
    def choose_move(self, battle: DoubleBattle):
        """Same control flow as the root base, but the normal-turn state vector
        is built with the gap-#6 opponent splice."""
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        self._report_active(battle)        # #18 spectate feed (no-op unless live_dir set)
        if any(battle.force_switch):
            return self._handle_force_switch(battle)

        # ONE parse of the public-log prefix → the gap-#6 opponent snapshot AND the just-resolved
        # previous turn (the A3 belief feed), so a normal turn parses the prefix once.
        opp_snapshot, prev_turn = self._reconstruct_decision(battle)
        self._feed_match_belief(battle, prev_turn)                      # A3: feed prev-turn damage+speed (no-op unless on)
        # S1 L2b: cross-game dossier warm-start FIRST (weakest evidence tier), then the
        # within-game match belief refines on top; the encoder's belief narrowers run last.
        # getattr: test stubs drive choose_move without running __init__ (adapt_rules pattern).
        if getattr(self, "_use_dossier", False):
            from v_dance.play.opponent_dossier import apply_dossier
            opp_snapshot = apply_dossier(opp_snapshot, battle)
        opp_snapshot = self._apply_match_belief(opp_snapshot, battle)   # A3: within-game belief (no-op unless on)
        state_vec = self._encoder.encode(battle, opp_snapshot=opp_snapshot)
        # Phase-3 memory carry: decisions run on the frame-stack for a memory model
        # (no-op single frame otherwise). Recording below keeps the RAW state_vec.
        state_in = self._memory_stack(battle, state_vec, "turn")
        # gap-#6 reconstructed opponent slot occupancy → lets the codec target
        # opp_a vs opp_b DELIBERATELY when a same-species illusion makes poke-env
        # lose a foe slot (#15).  None when no reconstruction (legacy targeting).
        _oa = (opp_snapshot or {}).get("opp_active") or {}
        opp_present_recon = ({0: bool(_oa.get("opp_a")), 1: bool(_oa.get("opp_b"))}
                             if opp_snapshot else None)
        if getattr(self, "_use_search", False):
            # Search is single-frame by design (its leaf evaluator scores synthetic
            # successor states) — a memory model under search keeps stateless turns.
            action_s0, action_s1, source = self._search_select(battle, state_vec)
        else:
            action_s0, action_s1, source = self._select_actions(battle, state_in)

        # ── Forced-move escape (no representable legal action) ──────────────
        # If an ACTIVE, non-fainted slot has an EMPTY legal mask, its only usable order is one
        # the 16-action codec can't express — Struggle (all real moves unusable), a recharge /
        # 2-turn move's forced continuation, or a move whose live id != mon.moves. Passing such
        # a slot is ILLEGAL ("must make a move/switch") and the model has NO real choice, so
        # resolve the whole turn with /choose default (Showdown plays the forced move) instead
        # of the retry-storm + rejected Pass. Normal turns (non-empty mask) are unaffected.
        if self._active_empty_mask(battle):
            # An active slot is forced into a synthetic move (recharge/Struggle) the codec can't express.
            # Its empty mask makes it unrecordable for RL → drop the step. But if a PARTNER slot has a
            # real choice, PRESERVE its model decision (forced move + partner order) instead of collapsing
            # the WHOLE turn to /choose default (which plays the free slot randomly = policy-destructive).
            self._discard_rl_decision(battle, "turn")
            # T3.3 loop-guard: poke-env re-calls choose_move with the SAME (battle,turn) when Showdown
            # REJECTS our order (a rare free-slot mask desync). This branch had no escalation — unlike the
            # retry path and the forceSwitch backstop — so a rejected partial order could rebuild forever
            # (a retry-storm bounded only by the stall watchdog). After _MAX_TURN_RETRIES rebuilds for the
            # same (battle,turn), fall back to /choose default (always accepted; model not driving).
            fa_key = (battle.battle_tag, battle.turn)
            fa_n = self._forced_aware_attempts.get(fa_key, 0)
            self._forced_aware_attempts[fa_key] = fa_n + 1
            if len(self._forced_aware_attempts) > 512:        # bound over a long run
                self._forced_aware_attempts = {fa_key: fa_n + 1}
            if fa_n >= _MAX_TURN_RETRIES:
                log.warning("Turn %d [%s] forced-aware order rejected %d× — '/choose default' "
                            "(forced-move + free-slot mask desync; model not driving).",
                            battle.turn, battle.battle_tag, fa_n)
                self._source_counts["retry_default"] += 1
                return DefaultBattleOrder()
            g0, g1 = self._select_gimmicks(battle, state_in, action_s0, action_s1)
            order = self._forced_aware_double_order(battle, action_s0, action_s1, g0, g1,
                                                    opp_present_recon)
            self._source_counts["forced_default" if isinstance(order, DefaultBattleOrder)
                                else "forced_partial"] += 1
            return order

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
            # Rejection diagnostic (2026-07-12, Imprison/Protect incident): dump the
            # request-derived usable lists + our masks so the NEXT desync is root-causable
            # from the log alone (requests aren't otherwise recorded). Serve-only, no
            # behavior change; guarded — diagnostics must never break the retry path.
            try:
                av = battle.available_moves
                usable = [[m.id for m in (av[s] if s < len(av) else [])] for s in (0, 1)]
                masks = [build_legal_action_mask(battle, s) for s in (0, 1)]
                log.warning(
                    "  rejection diagnostic [%s]: rejected=(%s,%s) usable_moves=%s "
                    "mask_legal=%s tried=%s",
                    battle.battle_tag, action_s0, action_s1, usable,
                    [[i for i, ok in enumerate(mk) if ok] for mk in masks],
                    {0: sorted(tried[0]), 1: sorted(tried[1])},
                )
            except Exception:                              # noqa: BLE001
                log.debug("rejection diagnostic failed (non-fatal)", exc_info=True)
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
            g0, g1 = self._select_gimmicks(battle, state_in, action_s0, action_s1)

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

    def _search_select(self, battle: DoubleBattle, state_vec):
        """Level C / B1 — belief-weighted expectimax → (a0, a1, "search"). Builds the FULL both-sides snapshot
        (one prefix parse), runs the search over white_box_sim + the value head, decodes the winning joint to
        serve indices, and VALIDATES them against the live legal masks. ANY failure / illegal action → fall back
        to the policy argmax (search can never crash or play an illegal order mid-battle)."""
        try:
            from v_dance.play.search import search_with_model, parse_label, enrich_snapshot
            from v_dance.play.vgc_base import build_legal_action_mask
            log_str, own_role = self._assemble_log(battle)
            full_snap = reconstruct_full_for_decision(log_str, own_role, battle.turn) if log_str else None
            if not full_snap or not (full_snap.get("our_active")):
                raise ValueError("no full snapshot")
            # CRITICAL: the raw _snapshot_state has NO stats_estimate/belief (those come from fill_blanks in the
            # export path) → white_box_sim would deal ~zero damage. Enrich both sides before the search.
            enrich_snapshot(full_snap, getattr(self._encoder, "belief", None))
            best_label, _act, _ranked = search_with_model(
                full_snap, self._model, self._search_encoder, cfg=self._search_cfg,
                device=self._device, turn=battle.turn)
            if not best_label:
                raise ValueError("search returned no action")
            idx = parse_label(best_label)
            a0, a1 = idx.get("our_a"), idx.get("our_b")
            m0, m1 = build_legal_action_mask(battle, 0), build_legal_action_mask(battle, 1)
            if a0 is not None and not (a0 < len(m0) and m0[a0]):
                raise ValueError(f"a0={a0} illegal at serve")
            if a1 is not None and not (a1 < len(m1) and m1[a1]):
                raise ValueError(f"a1={a1} illegal at serve")
            self._search_counts["search"] += 1
            return (a0 if a0 is not None else 0, a1 if a1 is not None else 0, "search")
        except Exception as exc:
            self._search_counts["fallback"] += 1
            log.debug("B1 search fell back to policy argmax (%s)", exc)
            return self._select_actions(battle, state_vec)

    def _reconstruct_decision(self, battle: DoubleBattle):
        """``(opp_snapshot start-of-turn, prev_turn T-1)`` from a SINGLE parse of the public log, for a
        normal-turn decision — shared by the gap-#6 splice and the A3 belief feed. ``(None, None)`` on
        any failure (degrade to poke-env's opponent view, no feed)."""
        try:
            log_str, own_role = self._assemble_log(battle)
            if not log_str:
                return None, None
            return reconstruct_for_decision(log_str, own_role, battle.turn)
        except Exception:
            log.debug("decision reconstruct failed (degrading to poke-env)", exc_info=True)
            return None, None

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

    # ── Level C / A3: within-game opponent belief (MatchBelief) pre-enrichment ──
    def _match_belief_for(self, battle: DoubleBattle) -> Optional["MatchBelief"]:
        """The per-battle ``MatchBelief`` for this battle tag (lazily created on first
        use), or ``None`` when MatchBelief is off or the encoder carries no
        ``BeliefState``.

        The ``LiveStateEncoder`` is a per-PLAYER singleton (its ``belief`` is shared
        across battles), so the within-game opponent belief lives HERE — keyed by
        battle tag like ``_proto_log`` — and is reset in ``_battle_finished_callback``.
        """
        if not getattr(self, "_use_match_belief", False):
            return None
        belief = getattr(getattr(self, "_encoder", None), "belief", None)
        if belief is None:
            return None
        tag = _norm_tag(getattr(battle, "battle_tag", None))
        if not tag:
            return None
        store = self._match_belief
        mb = store.get(tag)
        if mb is None:
            from v_dance.parser.match_belief import MatchBelief
            mb = store[tag] = MatchBelief(belief, level=getattr(self._encoder, "level", 50))
        return mb

    def _apply_match_belief(
        self, opp_snapshot: Optional[dict], battle: DoubleBattle
    ) -> Optional[dict]:
        """PRE-ENRICH the reconstructed opponent snapshot with this battle's
        ``MatchBelief`` so the gap-#6 splice carries the WITHIN-GAME-narrowed opponent
        stats/spreads (accumulated reveals + A2 damage + A3 speed) instead of the
        static Pikalytics prior.  Returns the snapshot UNCHANGED — a true no-op — when
        MatchBelief is off or no snapshot was reconstructed (degrade to static belief).

        Each opponent mon is enriched via ``MatchBelief.enrich_mon``, which also
        ACCUMULATES that mon's reveals (moves/item/ability) across turns; the
        encoder's static ``_enrich_opp_snapshot`` then NO-OPs on these mons (they
        already carry a ``stats_estimate``) and splices the sharpened bytes — so the
        encoder itself is UNCHANGED.  A mon with no Pikalytics data keeps no estimate
        and still falls through to the encoder's static path, exactly as today.

        Guarded so a belief hiccup can never break a live turn (→ static belief).
        Identity (Zoroark/Ditto) is gated inside ``enrich_mon``/``ingest_mon`` via
        ``identity_reliable`` — a disguised/transformed mon's reveals are skipped.
        """
        if opp_snapshot is None:
            return opp_snapshot
        mb = self._match_belief_for(battle)
        if mb is None:
            return opp_snapshot
        try:
            opp_active = opp_snapshot.get("opp_active") or {}
            mons = [opp_active.get("opp_a"), opp_active.get("opp_b")]
            mons += list(opp_snapshot.get("opp_bench") or [])
            for m in mons:
                if not m:
                    continue
                try:
                    mb.enrich_mon(m)
                except Exception:
                    log.debug("MatchBelief enrich_mon failed (non-fatal)", exc_info=True)
        except Exception:
            log.debug("MatchBelief pre-enrich failed (degrading to static belief)", exc_info=True)
        return opp_snapshot

    def _feed_match_belief(self, battle: DoubleBattle, prev_turn: Optional[dict] = None) -> None:
        """Feed the JUST-RESOLVED previous turn's events (``prev_turn``, from the shared single parse in
        ``_reconstruct_decision``) into this battle's ``MatchBelief`` so the within-game opponent belief
        sharpens before this turn is encoded — DAMAGE (A2) via ``feed_damage_constraints`` and SPEED
        (A3) via ``feed_speed_constraints``.  A no-op when MatchBelief is off or there is no prior turn.
        Fully guarded — a feed hiccup must never break a live turn (the belief just stays one behind)."""
        if not getattr(self, "_use_match_belief", False) or not prev_turn:
            return
        mb = self._match_belief_for(battle)
        if mb is None:
            return
        try:
            own_role = getattr(battle, "player_role", None)
            if not own_role:
                return
            # Idempotency: feed each (battle, turn) ONCE — a rejected-order re-entry re-invokes
            # choose_move for the same turn with the same prev_turn, which would otherwise double-count
            # this turn's evidence into the (append-only) constraint product. (audit 2026-06-30)
            tag = _norm_tag(getattr(battle, "battle_tag", None))
            turn = getattr(battle, "turn", 0) or 0
            if not tag or self._belief_fed_turn.get(tag) == turn:
                return
            self._belief_fed_turn[tag] = turn
            from v_dance.play.live_belief_feed import (
                feed_damage_constraints, feed_speed_constraints,
            )
            belief = getattr(self._encoder, "belief", None)
            level = getattr(self._encoder, "level", 50)
            ds = feed_damage_constraints(
                mb, belief, prev_turn, own_role, self._our_stats_by_species(battle), level=level,
                damage_loglik_fn=lambda **kw: self._damage_loglik(battle, **kw))
            ss = feed_speed_constraints(
                mb, prev_turn, own_role, self._speed_ctx(battle, prev_turn))
            bs = getattr(self, "_belief_stats", None)
            if bs is not None:
                for _k, _v in (ds or {}).items():
                    bs["dmg_" + _k] += int(_v or 0)
                for _k, _v in (ss or {}).items():
                    bs["spd_" + _k] += int(_v or 0)
        except Exception:
            log.debug("MatchBelief per-turn feed failed (non-fatal)", exc_info=True)

    @staticmethod
    def _our_stats_by_species(battle: DoubleBattle) -> Dict[str, dict]:
        """``{norm_species: {hp,atk,def,spa,spd,spe}}`` for OUR team from the live poke-env battle —
        our exact stats (known from ``|request|``), the known side of every damage constraint.  Keyed by
        both the current species and the base species (so a mega forme resolves).  Skips any mon whose
        stats are not fully numeric."""
        out: Dict[str, dict] = {}
        for m in (getattr(battle, "team", None) or {}).values():
            st = getattr(m, "stats", None) or {}
            if not st or any(not isinstance(st.get(k), (int, float))
                             for k in ("hp", "atk", "def", "spa", "spd", "spe")):
                continue
            d = {k: st.get(k) for k in ("hp", "atk", "def", "spa", "spd", "spe")}
            for key in (getattr(m, "species", None), getattr(m, "base_species", None)):
                if key:
                    out.setdefault(norm_species(key), d)
        return out

    def _damage_loglik(self, battle: DoubleBattle, *, direction, opp_key, our_key, opp_species,
                       our_species, opp_snapshot_mon, move, crit, obs_frac):
        """A2 v2: resolve the live poke-env mons for a damage event and compute the per-spread poke-env
        log-likelihood (item + ability marginalised). ``direction='off'`` (opp attacker → narrow the
        opp's offence) / ``'def'`` (opp defender → narrow the opp's bulk). ``opp_key`` is the OPP's
        active slot, ``our_key`` ours. Returns ``{spread_key: loglik}`` or ``None`` (→ the feeder's
        analytic fallback). BOTH mons are species-matched to the event: the OPP's slot may have changed
        since the hit, and so may OURS (our mon took the hit / dealt it on T-1 then switched or fainted
        before the T decision) — a mismatched OUR mon would make the poke-env calc use the wrong
        defender/attacker + max-HP denominator and corrupt the narrowing (audit 2026-06-30). The opp mon
        is restored inside the calc helper. Guarded — a calc hiccup must never break a turn."""
        try:
            opp_active = list(getattr(battle, "opponent_active_pokemon", None) or [])
            our_active = list(getattr(battle, "active_pokemon", None) or [])
            oi = 0 if str(opp_key).endswith("a") else 1
            ui = 0 if str(our_key).endswith("a") else 1
            opp_mon = opp_active[oi] if oi < len(opp_active) else None
            our_mon = our_active[ui] if ui < len(our_active) else None
            if opp_mon is None or our_mon is None:
                return None
            live_sp = getattr(opp_mon, "base_species", None) or getattr(opp_mon, "species", "") or ""
            if norm_species(live_sp) != norm_species(opp_species or ""):
                return None                  # the slot's live OPP mon isn't the one in the event (a switch since)
            our_live_sp = getattr(our_mon, "base_species", None) or getattr(our_mon, "species", "") or ""
            if our_species and norm_species(our_live_sp) != norm_species(our_species):
                return None                  # OUR slot's live mon isn't the one in the event → wrong reference
            from v_dance.play.pokeenv_damage import damage_loglik
            return damage_loglik(
                battle, opp_mon=opp_mon, our_mon=our_mon, direction=direction, move_name=move,
                is_critical=crit, obs_frac=obs_frac, belief=getattr(self._encoder, "belief", None),
                opp_snapshot_mon=opp_snapshot_mon, level=getattr(self._encoder, "level", 50))
        except Exception:
            log.debug("poke-env damage loglik failed (→ analytic)", exc_info=True)
            return None

    def _speed_ctx(self, battle: DoubleBattle, prev_turn: dict) -> dict:
        """Assemble the live SPEED context for ``feed_speed_constraints`` from the poke-env battle:
        our active mons' EXACT effective speed (the threshold, via the encoder's ``_live_effective_speed``),
        the opp active mons' KNOWN speed multipliers (boost/Tailwind/paralysis/revealed-Scarf/weather-
        ability — divided out so the bound is in opp BASE-Spe units) + whether the opp's item+ability are
        revealed (σ width), Trick Room, and a forced-order hint (Quash/After-You/Quick-Claw/Instruct)."""
        enc = self._encoder
        try:
            weather = next((w.name for w in battle.weather), None)
        except Exception:
            weather = None
        fields = getattr(battle, "fields", {}) or {}
        trick_room = any(getattr(f, "name", "") == "TRICK_ROOM" for f in fields)
        magic_room = any(getattr(f, "name", "") == "MAGIC_ROOM" for f in fields)
        our_tw = any(getattr(k, "name", "") == "TAILWIND" and v
                     for k, v in (getattr(battle, "side_conditions", {}) or {}).items())
        opp_tw = any(getattr(k, "name", "") == "TAILWIND" and v
                     for k, v in (getattr(battle, "opponent_side_conditions", {}) or {}).items())
        try:
            our_active = list(battle.active_pokemon)
        except Exception:
            our_active = [None, None]
        try:
            opp_active = list(battle.opponent_active_pokemon)
        except Exception:
            opp_active = [None, None]

        our: Dict[str, dict] = {}
        for i, key in ((0, "our_a"), (1, "our_b")):
            m = our_active[i] if i < len(our_active) else None
            if m is None:
                continue
            try:
                eff = enc._live_effective_speed(m, True, our_tw, weather, magic_room)[0]
            except Exception:
                eff = 0.0
            our[key] = {"eff_speed": eff,
                        "ability": norm_species(getattr(m, "ability", None) or "") or None,
                        "hp_frac": (m.current_hp_fraction if getattr(m, "revealed", False) else 1.0)}
        opp: Dict[str, dict] = {}
        for i, key in ((0, "opp_a"), (1, "opp_b")):
            m = opp_active[i] if i < len(opp_active) else None
            if m is not None:
                opp[key] = self._opp_speed_mult_known(m, opp_tw, weather)
        return {"trick_room": trick_room,
                "order_forced": self._order_forced_hint(battle, prev_turn),
                "our": our, "opp": opp}

    @staticmethod
    def _opp_speed_mult_known(m, opp_tw: bool, weather) -> dict:
        """``{speed_mult_known, context_known, ability, hp_frac}`` for an opp active mon — the product of
        its KNOWN speed multipliers (boost stage · Tailwind · paralysis · a REVEALED Choice Scarf ·
        weather-speed ability) and whether its item+ability are both revealed (else a possible hidden
        Scarf → wider σ). Only revealed context is applied, so the bound never over-claims."""
        mult = 1.0
        stage = (getattr(m, "boosts", None) or {}).get("spe", 0) or 0
        mult *= ((2 + stage) / 2.0) if stage >= 0 else (2.0 / (2 - stage))
        if opp_tw:
            mult *= 2.0
        if getattr(getattr(m, "status", None), "name", None) == "PAR":
            mult *= 0.5                       # ignore opp Quick Feet (unrevealed) → conservative
        item = norm_species(getattr(m, "item", None) or "")
        ability = norm_species(getattr(m, "ability", None) or "")
        item_known = bool(item) and item != "unknownitem"
        if item_known and item == "choicescarf":
            mult *= 1.5
        if weather and ability and weather in _WEATHER_SPEED_ABILITY.get(ability, ()):
            mult *= 2.0
        return {"speed_mult_known": mult,
                "context_known": item_known and bool(ability),
                "ability": ability or None,
                # the CURRENT-board species at this slot, so the speed feeder can skip a read whose
                # prev-turn mover != the current occupant (a switch/faint+replacement). (audit 2026-06-30)
                "species": norm_species(getattr(m, "base_species", None)
                                        or getattr(m, "species", "") or "") or None,
                "hp_frac": (m.current_hp_fraction if getattr(m, "revealed", False) else 1.0)}

    def _order_forced_hint(self, battle: DoubleBattle, prev_turn: dict) -> bool:
        """True when the previous turn's order was (or may have been) decided by something OTHER than
        speed — a forced/extra move out of speed order: Quash / After You / Quick Claw / Instruct /
        Dancer. Detected from the raw proto segment of that turn (a repeated move is also caught in
        ``feed_speed_constraints``). Conservative — a hit just SKIPS the speed read, never corrupts."""
        try:
            seg = self._prev_turn_proto_segment(battle).lower()
        except Exception:
            return False
        return any(tok in seg for tok in
                   ("after you", "afteryou", "quash", "quick claw", "quickclaw", "instruct", "dancer"))

    def _prev_turn_proto_segment(self, battle: DoubleBattle) -> str:
        """The raw protocol lines of the PREVIOUS turn (between ``|turn|T-1`` and ``|turn|T``) from the
        per-battle capture buffer, joined — for the forced-order scan. Empty string if unavailable."""
        lines = self._proto_log.get(_norm_tag(getattr(battle, "battle_tag", None))) or []
        turn = getattr(battle, "turn", 0) or 0
        start, end = f"|turn|{turn - 1}", f"|turn|{turn}"
        seg: List[str] = []
        collecting = False
        for ln in lines:
            s = ln.strip()
            if s == start:
                collecting = True
                continue
            if s == end:
                break
            if collecting:
                seg.append(ln)
        return "\n".join(seg)

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
            # Distinguish the two escape outcomes for the fs-monitor: a FORFEIT (battle abandoned
            # after too many forced-switch handlings) vs a /choose default (Showdown resolves the
            # replacement). Both mean the model did NOT drive this decision. A forfeit can be
            # RE-REQUESTED by the server before it processes our /forfeit (re-entering this path for
            # the SAME battle), so count forfeit ONCE per battle tag — a forfeited-battle tally, not
            # a forfeit-re-request tally; the /choose default escape has no such re-entry inflation.
            if isinstance(escape, ForfeitBattleOrder):
                seen = getattr(self, "_forfeited_tags", None)
                if seen is None:
                    seen = self._forfeited_tags = set()
                tag = _norm_tag(getattr(battle, "battle_tag", None))
                if tag not in seen:
                    seen.add(tag)
                    self._source_counts["forfeit"] += 1
            else:
                self._source_counts["forced_switch_escape"] += 1
            self._discard_rl_decision(battle, "replacement")   # escape executes, not the model
            return escape

        state_vec = None
        opp_snapshot = None
        try:
            opp_snapshot = self._build_opp_snapshot_current(battle)
            if getattr(self, "_use_dossier", False):        # S1 L2b: same tier order as choose_move
                from v_dance.play.opponent_dossier import apply_dossier
                opp_snapshot = apply_dossier(opp_snapshot, battle)
            opp_snapshot = self._apply_match_belief(opp_snapshot, battle)   # A3: within-game belief (no-op unless on)
            state_vec = self._encoder.encode(battle, opp_snapshot=opp_snapshot)
        except Exception:
            log.debug("forceSwitch encode failed (non-fatal)", exc_info=True)

        repl = None
        if state_vec is not None:
            try:
                # Phase-3 memory carry: replacement decisions are frames too (they
                # are trajectory steps in the 1b dataset); recording keeps the raw vec.
                state_in = self._memory_stack(battle, state_vec, "replacement")
                repl = self._select_replacement_actions(battle, state_in)
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

    # ── #18 multi-battle spectate (base; works for collection AND eval players) ───
    def _report_active(self, battle) -> None:
        """Publish this battle's room + growing |-log to ``<live_dir>/<tag>.json`` so the
        dashboard can show several concurrent battles. No-op unless ``live_dir`` was wired."""
        live = getattr(self, "_live", None)        # tolerate __new__-built instances (tests)
        if live is None:
            return
        try:
            players = getattr(battle, "players", None) or []
            tag = battle.battle_tag
            live.update(
                tag, turn=getattr(battle, "turn", 0) or 0,
                log=self._proto_log.get(_norm_tag(tag)) or [],
                p1=players[0] if len(players) > 0 else getattr(self, "username", "p1"),
                p2=players[1] if len(players) > 1 else "opponent")
        except Exception:
            log.debug("live-battle report failed (non-fatal)", exc_info=True)

    def _battle_finished_callback(self, battle):
        """On finish (#18): when ``save_replays`` is on, SAVE a real Showdown **HTML** replay
        (``<live_dir>/<tag>.html`` via poke-env's ``battle.save_replay`` — a file you can open in a
        browser); else just DELETE the live-feed file. Either way the transient live JSON (the
        dashboard's LIVE feed only) is dropped. Guarded; then chains to the base."""
        live = getattr(self, "_live", None)        # tolerate __new__-built instances (tests)
        if live is not None:
            try:
                tag = battle.battle_tag
                if getattr(self, "_save_html_replays", False):
                    live.save_html_replay(tag, battle,    # playable HTML + drop the live JSON
                                          out_dir=getattr(self, "_replay_dir", None),
                                          label=getattr(self, "_replay_label", None),
                                          copy_dir=getattr(self, "_replay_copy_dir", None))
                else:
                    live.remove(tag)
            except Exception:
                log.debug("live-battle replay save failed (non-fatal)", exc_info=True)
        # audit (leak): reclaim this battle's per-tag splice buffers on finish — the whole reconstructed
        # protocol log + last Showdown error were never pruned (unlike the root base's _tp_decision pop),
        # so a finished battle's hundreds of log lines stayed resident for the player's lifetime. Mirror the
        # _tp_decision discipline; guarded so teardown never raises. (The browser host also pops _proto_log.)
        try:
            _t = _norm_tag(getattr(battle, "battle_tag", None))
            if isinstance(getattr(self, "_proto_log", None), dict):
                self._proto_log.pop(_t, None)
            if isinstance(getattr(self, "_last_error", None), dict):
                self._last_error.pop(_t, None)
            if isinstance(getattr(self, "_match_belief", None), dict):   # A3: drop this battle's within-game belief
                self._match_belief.pop(_t, None)
            if isinstance(getattr(self, "_belief_fed_turn", None), dict):
                self._belief_fed_turn.pop(_t, None)
            if isinstance(getattr(self, "_mem_frames", None), dict):     # Phase-3: drop this battle's frame stack
                self._mem_frames.pop(_t, None)
        except Exception:
            log.debug("proto-log prune on finish failed (non-fatal)", exc_info=True)
        return super()._battle_finished_callback(battle)

    def reset_battle_state(self, tag: str) -> None:
        """Forget every per-battle buffer for ``tag`` WITHOUT the finish bookkeeping (no replay
        save, no RL finalize). Used by the browser host's ``forget_battle`` on a link reconnect
        (2026-09-01): the server replays the whole battle log, so the splice / belief / memory /
        retry buffers must start empty or they would double-count. ``_tp_decision`` is kept on
        purpose (the team-preview choice already happened server-side)."""
        t = _norm_tag(tag)
        for name in ("_proto_log", "_last_error", "_match_belief", "_belief_fed_turn",
                     "_mem_frames", "_fs_battle_count", "_fs_attempts", "_tried_actions"):
            d = getattr(self, name, None)
            if not isinstance(d, dict):
                continue
            d.pop(t, None)                              # plain tag-keyed entries
            for k in [k for k in d if isinstance(k, tuple) and k and k[0] == t]:
                d.pop(k, None)                          # (tag, turn, …)-keyed entries

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
