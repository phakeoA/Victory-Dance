"""
local_battle/player.py  —  Neural-network VGC player (gap-#6 opponent splice)
=============================================================================
Identical to the repo-root ``player.py`` VGCPlayer, EXCEPT it inherits the
gap-#6 splicing base (``live_vgc_base.SplicingVGCPlayerBase``) instead of the
plain root base.  That single swap makes every turn encode the opponent side
from the public protocol via the training ``vod_parser`` (real-time), so the
live bot is immune to poke-env's duplicate-species illusion merge.

Action / network contract is unchanged — see the root player.py docstring.

    from local_battle.player import VGCPlayer
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Path bootstrap (local_battle FIRST; see live_vgc_base for rationale) ──────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
from poke_env.battle import DoubleBattle

# Gap-#6 spliced base (local) + the pure helpers (root, reused unchanged).
from v_dance.play.live_vgc_base import SplicingVGCPlayerBase as VGCPlayerBase
from v_dance.play.vgc_base import (
    _heuristic_team_order,
    build_legal_action_mask,
    build_replacement_mask,
    build_gimmick_legal_mask,
    random_legal_action,
    VGC_TEAM_SIZE,
)
from v_dance.encoders.state_encoder import (
    ACTIONS_PER_SLOT, STATE_DIM, SWITCH_OFFSET, GIMMICK_NONE, GIMMICK_MEGA,
)
import v_dance.play.model_io as _M   # dict-checkpoint load + mask-aware logit decode (#13)

log = logging.getLogger(__name__)

# ── Optional torch import ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    log.warning("PyTorch not found — VGCPlayer will always use random fallback.")


class VGCPlayer(VGCPlayerBase):
    """
    VGC player driven by a trained PyTorch model, with the gap-#6 opponent
    splice (inherited from SplicingVGCPlayerBase).

    If model_path is None (or PyTorch is unavailable), action selection falls
    back to uniform random — identical behaviour to RandomVGCPlayer.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        team_chooser_path: Optional[Path] = None,
        replay_path: Optional[Path] = None,
        device: str = "cpu",
        temperature: float = 0.0,
        top_p: float = 1.0,
        sample_seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(replay_path=replay_path, **kwargs)

        self._device       = device
        self._model        = None
        self._model_heads  = None
        self._team_chooser = None
        self._tc_vocab     = None
        self._tc_cfg       = None
        # Team-preview decision tally: did the TP NET drive the bring-4/leads, or
        # did we fall back to the first-N heuristic?  Surfaced per run so the TP
        # net's live behaviour is measurable (#4 — it was never exercised before).
        self._tp_source: Counter = Counter()
        # Value-head readout (#2): the model's per-turn win-probability estimate,
        # logged + traced so the value head is OBSERVABLE live (it does not drive
        # action selection — VGC has no doubles forward sim).
        self._last_value: Optional[float] = None
        self._value_trace: List[Tuple[int, float]] = []

        # Serve-side action sampling (TIER-4).  temperature<=0 → deterministic
        # masked argmax (the default, unchanged behaviour); >0 → temperature /
        # top-p nucleus sampling over the legal actions.  Forced replacement stays
        # deterministic (argmax).
        self._temperature = float(temperature)
        self._top_p       = float(top_p)
        self._rng = (np.random.default_rng(sample_seed)
                     if self._temperature > 0.0 else None)
        # #9 + #9b: whether to SAMPLE the model's stochastic sub-decisions (gimmick AND forced-replacement)
        # at self._temperature instead of argmax. Default False (eval/serve = deterministic). Self-play
        # collection sets it True so the PLAYED decision matches its tau-recorded behaviour log-prob (else it
        # is logged as tau-sampled but chosen greedily → off-policy gradient + zero exploration on that head).
        self._collect_sample = False
        # #10/#11: when recording (self-play), stash the per-slot legal mask the model was ACTUALLY sampled
        # under — including the cross-slot switch/replacement dedup (slot 1's colliding switch removed) —
        # keyed by (battle_tag, decision_type), so _record_rl_decision logs the true behaviour mask instead
        # of rebuilding an un-deduped one (which over-states slot 1's legal set on collision turns).
        self._record_masks = False
        self._sampling_masks: dict = {}

        # ── Load battle model (dict checkpoint → reconstruct + load_state_dict) ─
        if model_path is not None and _TORCH_AVAILABLE:
            try:
                self._model, self._model_heads = _M.load_bc_policy(model_path, device)
                log.info("VGCPlayer: loaded battle model from %s (heads=%s)",
                         model_path, self._model_heads)
            except Exception as exc:
                log.error(
                    "VGCPlayer: failed to load battle model from %s (%s) "
                    "— falling back to random.",
                    model_path, exc,
                )
                self._model = None

        # ── Load team-chooser model ───────────────────────────────────────────
        if team_chooser_path is not None and _TORCH_AVAILABLE:
            try:
                self._team_chooser, self._tc_vocab, self._tc_cfg = \
                    _M.load_team_chooser(team_chooser_path, device)
                log.info("VGCPlayer: loaded team-chooser from %s (vocab=%d)",
                         team_chooser_path, len(self._tc_vocab or {}))
            except Exception as exc:
                log.error(
                    "VGCPlayer: failed to load team-chooser from %s (%s) "
                    "— using heuristic.",
                    team_chooser_path, exc,
                )
                self._team_chooser = None

        log.info(
            "VGCPlayer ready (gap-#6 splice) | battle_model=%s | team_chooser=%s | replay=%s",
            f"loaded ({model_path})" if self._model else "none (random fallback)",
            f"loaded ({team_chooser_path})" if self._team_chooser else "none (heuristic)",
            replay_path or Path("artifacts/replay_buffer/replay.jsonl"),
        )

    # ── Action selection ──────────────────────────────────────────────────────

    @staticmethod
    def _first_legal(battle: DoubleBattle, slot: int) -> int:
        """First legal action for a slot (DETERMINISTIC — no randomness), used only
        on a hard model failure so the battle can continue without masquerading a
        random move as the AI.  Returns 0 (→ Pass via _safe_order) if none."""
        mask = build_legal_action_mask(battle, slot)
        for i, ok in enumerate(mask):
            if ok:
                return i
        return 0

    def _select_actions(
        self,
        battle: DoubleBattle,
        state_vec: np.ndarray,
    ) -> Tuple[int, int, str]:
        """Run the two-head BC policy and return (action_s0, action_s1, source).

        NO SILENT RANDOM: the model drives every live slot.  bc_action_indices is a
        masked argmax per head over build_legal_action_mask, so its index is already
        legal.  A None means that slot has NO legal action (an empty / fainted active
        slot) → it passes via _safe_order; that is the model's decision, not random,
        so the turn stays labelled "model".  A misconfigured player (no model) or a
        genuine inference exception is surfaced LOUDLY with a distinct source label
        and a deterministic legal action — never a hidden random move."""
        if self._model is None or not _TORCH_AVAILABLE:
            log.error("VGCPlayer has NO model loaded — cannot drive this turn "
                      "(NOT playing random; check the checkpoint path).")
            return self._first_legal(battle, 0), self._first_legal(battle, 1), "no_model"

        # ── Value-head readout (#2): log the win-probability for this board ──────
        if _M.value_trained(self._model):
            try:
                wp = _M.value_logit(self._model, state_vec, self._device)
                if wp is not None:
                    self._last_value = wp
                    self._value_trace.append((battle.turn, round(wp, 3)))
                    if len(self._value_trace) > 1000:   # audit (leak): bound this write-only serve trace
                        del self._value_trace[:-1000]   # (play_vs_human_browser runs one player indefinitely)
                    log.info("Turn %d [%s] value head win-prob: %.3f  (%s)",
                             battle.turn, battle.battle_tag, wp,
                             "winning" if wp >= 0.55 else "losing" if wp <= 0.45 else "even")
            except Exception as exc:        # never let the readout break a turn
                log.debug("value readout failed (%s)", exc)

        try:
            mask0 = build_legal_action_mask(battle, 0)
            mask1 = build_legal_action_mask(battle, 1)
            # B-L1 adapt-rules (default OFF): serve-time pattern tilt — a logit bias, never a
            # mask edit, so the model still chooses. A bias failure must never cost the turn.
            bias0 = bias1 = None
            if getattr(self, "_adapt_rules", False):
                try:
                    from v_dance.play.adapt_rules import action_biases
                    bias0, bias1 = action_biases(self, battle)
                except Exception:
                    log.debug("adapt-rules bias failed (non-fatal)", exc_info=True)
            a0, a1 = _M.bc_action_indices(
                self._model, self._model_heads, state_vec, mask0, mask1, self._device,
                temperature=self._temperature, top_p=self._top_p, rng=self._rng,
                bias0=bias0, bias1=bias1,
            )
            # ── Cross-slot SWITCH dedup (doubles "can only switch in once") ──────
            # The two heads pick independently, so both active slots can choose the
            # SAME bench mon to switch into — Showdown rejects the second order.
            # Re-decode slot 1 with the colliding switch masked out → the model's
            # best NON-colliding legal action (preserves intent; no retry storm).
            if (a0 is not None and a0 == a1 and a0 >= SWITCH_OFFSET
                    and a0 < len(mask1) and mask1[a0]):
                mask1 = list(mask1)
                mask1[a0] = False
                _, a1 = _M.bc_action_indices(
                    self._model, self._model_heads, state_vec, mask0, mask1, self._device,
                    temperature=self._temperature, top_p=self._top_p, rng=self._rng,
                    bias0=bias0, bias1=bias1,
                )
            # #10: stash the masks the model was sampled under (mask1 carries the cross-slot switch dedup)
            # so the recorded behaviour mask matches the sampling distribution.
            if getattr(self, "_record_masks", False):
                self._sampling_masks[(battle.battle_tag, "turn")] = (mask0, mask1)
            # None ⟺ all-zero mask ⟺ empty/fainted slot → 0 (passes via _safe_order).
            return (a0 if a0 is not None else 0,
                    a1 if a1 is not None else 0,
                    "model")

        except Exception as exc:
            # A real inference failure is a BUG we want to SEE, not hide behind a
            # random move.  Log loudly and fall back DETERMINISTICALLY.
            log.error("MODEL INFERENCE FAILED (%s) — using a deterministic legal "
                      "action (NOT random) so the failure stays visible.",
                      exc, exc_info=True)
            return self._first_legal(battle, 0), self._first_legal(battle, 1), "model_error"

    def _select_gimmicks(self, battle, state_vec, a0, a1):
        """Per-slot mega decision from the model's gimmick head.

        GIMMICK_NONE unless ALL of: the gimmick head is actually trained (a
        pre-gimmick checkpoint loaded non-strictly has an untrained head we must
        not act on), the slot's chosen action is a MOVE (switches/replacements
        never gimmick), and the masked-argmax over the slot's gimmick legal mask
        (byte-parity with training) picks mega.  ``action_to_order`` then applies
        the item-aware ``battle.can_mega_evolve`` gate, so an illegal mega is never
        sent — this only decides WHETHER the model wants to mega."""
        if (self._model is None or not _TORCH_AVAILABLE
                or not _M.gimmick_trained(self._model)):
            return GIMMICK_NONE, GIMMICK_NONE
        try:
            glog = _M.gimmick_logits(self._model, self._model_heads, state_vec, self._device)
            if glog is None:
                return GIMMICK_NONE, GIMMICK_NONE
            out = []
            for slot, action in ((0, a0), (1, a1)):
                if action is None or action >= SWITCH_OFFSET:
                    out.append(GIMMICK_NONE)        # switch / no action never megas
                    continue
                gmask = build_gimmick_legal_mask(battle, slot)
                # #18: an empty/fainted (action-less) slot is passed in as a fabricated move 0 by
                # _select_actions (None->0); force GIMMICK_NONE explicitly when it has no legal gimmick so
                # it can never read the head or participate in (and steal) the cross-slot dedup below.
                if not any(gmask):
                    out.append(GIMMICK_NONE)
                    continue
                # #9: self-play samples at tau (matches the recorded behaviour log-prob + explores mega);
                # eval/serve uses deterministic argmax. masked_sample(temperature<=0) == masked_argmax.
                g = (_M.masked_sample(glog[slot], gmask, temperature=self._temperature, rng=self._rng)
                     if getattr(self, "_collect_sample", False)
                     else _M.masked_argmax(glog[slot], gmask))
                out.append(g if g is not None else GIMMICK_NONE)
            # Cross-slot mega dedup: Showdown allows only ONE Mega Evolution per
            # BATTLE (so at most one per turn).  The two gimmick heads decide
            # independently and build_gimmick_legal_mask only blocks mega once the
            # team has mega'd on a PRIOR turn — so both active slots can pick mega the
            # SAME turn → Showdown rejects ("can only Mega-Evolve once per battle") →
            # retry.  Keep the slot with the stronger mega preference (the mega−none
            # logit margin) and drop the other to GIMMICK_NONE (the analogue of the
            # cross-slot SWITCH dedup in _select_actions).
            # v11 Phase D: generalize the dedup to ANY SAME non-none gimmick — both slots picking mega OR
            # both picking tera is illegal (each is once-per-battle), so drop the weaker-margin slot. A
            # slot0-mega + slot1-tera turn is LEGAL (different gimmicks, DoubleBattleOrder.join_orders only
            # filters same-gimmick pairs) and is NOT dropped here.
            if out[0] != GIMMICK_NONE and out[0] == out[1]:
                try:
                    margin0 = float(glog[0][out[0]]) - float(glog[0][GIMMICK_NONE])
                    margin1 = float(glog[1][out[1]]) - float(glog[1][GIMMICK_NONE])
                except Exception:
                    margin0, margin1 = 1.0, 0.0      # any error → keep slot 0's gimmick
                out[1 if margin0 >= margin1 else 0] = GIMMICK_NONE
            return out[0], out[1]
        except Exception as exc:
            log.warning("gimmick selection failed (%s) — no gimmick.", exc)
            return GIMMICK_NONE, GIMMICK_NONE

    # ── Forced replacement (post-faint) — model-driven ─────────────────────────

    def _select_replacement_actions(self, battle: DoubleBattle, state_vec: np.ndarray):
        """Model-driven post-faint replacement.

        For each slot that must switch (``battle.force_switch[slot]``), masked-argmax
        the MATCHING head (slot 0 → our_a, slot 1 → our_b) over a switch-only
        replacement mask (build_replacement_mask), deduping the chosen bench mon
        across slots when both fainted.  This is exactly how training's
        ``decision_type='replacement'`` transitions are labelled (the fainted slot's
        head, switch-only mask), so the heads are in-distribution for it.

        Returns ``(a0, a1, "forced_switch_model")`` — a switch index for each fainted
        slot that has a brought replacement, ``None`` (→ Pass) for a fainted slot whose
        bench is EXHAUSTED (a genuine double faint with fewer survivors than fainted
        slots — "Pattern A") AND for a non-switching slot.  Passing the empty slot IS
        the model's decision (the only legal action), not a fallback, so the whole
        post-faint turn stays MODEL-DRIVEN — Showdown accepts switch+Pass.  Returns
        ``None`` only to defer to the inherited random picker when the MODEL itself is
        unavailable / errors (so a genuine inference failure still has a safety net,
        and the forced-switch loop-guard escape remains the backstop for a rejected
        order)."""
        if self._model is None or not _TORCH_AVAILABLE:
            return None
        try:
            force = list(getattr(battle, "force_switch", []) or [])
            l0, l1 = _M.head_logits(self._model, self._model_heads, state_vec, self._device)
            logits = (l0, l1)
            out: List[Optional[int]] = [None, None]
            rep_masks: List = [None, None]   # #11: per-slot mask actually argmaxed under (with dedup)
            taken: set = set()      # bench indices already assigned (dedupe slots)
            forced_any = False
            for slot in (0, 1):
                if slot >= len(force) or not force[slot]:
                    continue        # not switching → out[slot] stays None → Pass
                forced_any = True
                mask = build_replacement_mask(battle, slot)
                for i in taken:
                    mask[SWITCH_OFFSET + i] = False
                rep_masks[slot] = mask
                # #9b: sample the replacement at tau in self-play (match the recorded log-prob + explore),
                # deterministic argmax in eval/serve. masked_sample(temperature<=0) == masked_argmax.
                a = (_M.masked_sample(logits[slot], mask, temperature=self._temperature, rng=self._rng)
                     if getattr(self, "_collect_sample", False)
                     else _M.masked_argmax(logits[slot], mask))
                if a is None:
                    # Forced slot but the brought bench is exhausted (Pattern A): PASS
                    # this slot (out[slot] stays None).  This is the model's decision,
                    # not a random fallback — so we DON'T bail the whole turn (which
                    # would also throw away a valid pick on the OTHER fainted slot).
                    continue
                out[slot] = a
                taken.add(a - SWITCH_OFFSET)
            if not forced_any:
                return None         # nothing to replace → defer (defensive; caller gates on any(force))
            if getattr(self, "_record_masks", False):
                self._sampling_masks[(battle.battle_tag, "replacement")] = (rep_masks[0], rep_masks[1])
            return out[0], out[1], "forced_switch_model"
        except Exception as exc:
            log.warning("Model replacement selection failed (%s) — using random.", exc)
            return None

    # ── Team chooser ──────────────────────────────────────────────────────────

    def _choose_team_order(self, battle: DoubleBattle, team: list, n: int) -> List[int]:
        """Score the matchup with the team-preview model (our + opponent rosters)
        and return n roster indices, LEADS FIRST.  Falls back to the first-N
        heuristic if the model is absent or anything goes wrong.

        Logs (INFO) whether the TP NET drove or it fell back, with the chosen
        bring/leads, and tallies it in ``self._tp_source`` so a run can report how
        often the net actually drove team preview (#4)."""
        if self._team_chooser is None or not _TORCH_AVAILABLE:
            self._tp_source["heuristic"] += 1
            log.info("Team-preview [%s]: no chooser model — HEURISTIC (first-%d).",
                     battle.battle_tag, n)
            return _heuristic_team_order(battle)[:n]

        try:
            our_species = [getattr(m, "species", None) for m in team]
            opp_team = list(getattr(battle, "teampreview_opponent_team", None)
                            or getattr(battle, "opponent_team", {}).values())
            opp_species = [getattr(m, "species", None) for m in opp_team]

            # An SBDA chooser (feature_schema=tpfeat-*) needs the SAME Pikalytics belief
            # its synergy/teammate features were trained on; the legacy 46-dim net does
            # not (and must not pay the belief-load cost).  Resolve the active-format
            # belief lazily, warn (never silently) if it is unavailable.
            belief = None
            if _M.uses_tp_features(self._tc_cfg):
                from v_dance.play.vgc_base import _default_belief
                belief = _default_belief()
                if belief is None:
                    log.warning("Team-preview [%s]: SBDA chooser but NO belief available — "
                                "synergy/usage features degrade to typing-only.",
                                battle.battle_tag)

            order = _M.team_order(
                self._team_chooser, self._tc_vocab, self._tc_cfg,
                our_species, opp_species, n, self._device, belief=belief,
            )
            valid = [i for i in order if 0 <= i < len(team)]
            if valid and len(set(valid)) == len(valid):
                # pad with unchosen roster slots if the model returned < n
                used = set(valid)
                for i in range(len(team)):
                    if len(valid) >= n:
                        break
                    if i not in used:
                        valid.append(i)
                        used.add(i)
                picks = valid[:n]
                lead_k = int((self._tc_cfg or {}).get("lead_k", 2))
                brought = [getattr(team[i], "species", "?") for i in picks]
                self._tp_source["model"] += 1
                log.info("Team-preview [%s] NET drove: bring=%s leads=%s (vs opp %s)",
                         battle.battle_tag, brought, brought[:lead_k],
                         [s for s in opp_species if s][:6])
                return picks
            log.warning("Team-preview [%s] NET produced invalid indices %s — "
                        "FALLING BACK to heuristic.", battle.battle_tag, order)
        except Exception as exc:
            log.warning("Team-preview [%s] NET inference FAILED (%s) — FALLING BACK "
                        "to heuristic.", battle.battle_tag, exc)

        self._tp_source["heuristic"] += 1
        return _heuristic_team_order(battle)[:n]
