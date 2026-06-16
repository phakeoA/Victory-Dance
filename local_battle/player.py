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
for _p in (str(_REPO_ROOT / "data" / "scripts"), str(_REPO_ROOT), str(_HERE)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from poke_env.battle import DoubleBattle

# Gap-#6 spliced base (local) + the pure helpers (root, reused unchanged).
from live_vgc_base import SplicingVGCPlayerBase as VGCPlayerBase
from vgc_base import (
    _heuristic_team_order,
    build_legal_action_mask,
    build_replacement_mask,
    build_gimmick_legal_mask,
    random_legal_action,
    VGC_TEAM_SIZE,
)
from state_encoder import ACTIONS_PER_SLOT, STATE_DIM, SWITCH_OFFSET, GIMMICK_NONE
import model_io as _M   # dict-checkpoint load + mask-aware logit decode (#13)

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

        # Serve-side action sampling (TIER-4).  temperature<=0 → deterministic
        # masked argmax (the default, unchanged behaviour); >0 → temperature /
        # top-p nucleus sampling over the legal actions.  Gimmick + forced
        # replacement stay deterministic (argmax).
        self._temperature = float(temperature)
        self._top_p       = float(top_p)
        self._rng = (np.random.default_rng(sample_seed)
                     if self._temperature > 0.0 else None)

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
            replay_path or Path("replay_buffer/replay.jsonl"),
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

        try:
            mask0 = build_legal_action_mask(battle, 0)
            mask1 = build_legal_action_mask(battle, 1)
            a0, a1 = _M.bc_action_indices(
                self._model, self._model_heads, state_vec, mask0, mask1, self._device,
                temperature=self._temperature, top_p=self._top_p, rng=self._rng,
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
                )
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
                g = _M.masked_argmax(glog[slot], gmask)
                out.append(g if g is not None else GIMMICK_NONE)
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

        Returns ``(a0, a1, "forced_switch_model")`` or ``None`` to fall back to the
        inherited random picker (no model, or no legal model replacement)."""
        if self._model is None or not _TORCH_AVAILABLE:
            return None
        try:
            force = list(getattr(battle, "force_switch", []) or [])
            l0, l1 = _M.head_logits(self._model, self._model_heads, state_vec, self._device)
            logits = (l0, l1)
            out: List[Optional[int]] = [None, None]
            taken: set = set()      # bench indices already assigned (dedupe slots)
            for slot in (0, 1):
                if slot >= len(force) or not force[slot]:
                    continue
                mask = build_replacement_mask(battle, slot)
                for i in taken:
                    mask[SWITCH_OFFSET + i] = False
                a = _M.masked_argmax(logits[slot], mask)
                if a is None:
                    return None     # no legal model replacement → random fallback
                out[slot] = a
                taken.add(a - SWITCH_OFFSET)
            if out[0] is None and out[1] is None:
                return None
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

            order = _M.team_order(
                self._team_chooser, self._tc_vocab, self._tc_cfg,
                our_species, opp_species, n, self._device,
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
