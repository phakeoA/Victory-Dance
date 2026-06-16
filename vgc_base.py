"""
vgc_base.py  —  Shared foundation for VGC Pokémon Showdown players
==================================================================
Contains:
  • ReplayBuffer          — JSON-lines per-turn transition recorder
  • Pure action helpers   — legal mask, action→order, random action
  • Teampreview heuristic — first-N roster order (TODO: type-advantage)
  • VGCPlayerBase         — abstract poke-env Player subclass

Concrete subclasses must implement _select_actions().

  RandomVGCPlayer  (random_player.py) — uniform random legal action
  VGCPlayer        (player.py)        — neural-network driven

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action encoding (matches state_encoder.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    0–11  →  move_idx (0–3)  ×  target (0=opp0, 1=opp1, 2=ally)
    12–15 →  switch to bench slot (0–3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Replay buffer format  (one JSON object per line)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "battle_id"  : str,
  "turn"       : int,
  "state"      : [float, ...],   # STATE_DIM floats
  "action_s0"  : int,            # action taken for slot 0
  "action_s1"  : int,            # action taken for slot 1
  "source"     : "model"|"random",
  "outcome"    : 1|0|-1|null     # 1=win 0=loss -1=draw; null until battle ends
}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import logging
import random
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from poke_env.player import Player
from poke_env.player.battle_order import (
    DefaultBattleOrder, DoubleBattleOrder, ForfeitBattleOrder,
    PassBattleOrder, SingleBattleOrder,
)

# A single battle should never need more than a handful of forced-switch
# handlings; far beyond that means Showdown is rejecting every order we send and
# re-requesting forever (a hang).  After this many, FORFEIT to break the loop so
# the battle ends cleanly (a loss) and the caller/gauntlet can continue.
_FS_FORFEIT_AT = 20
from poke_env.battle import DoubleBattle, Move, Pokemon

# state_encoder lives at data/scripts/ — add that directory to sys.path
import sys as _sys
from pathlib import Path as _Path
_SCRIPTS_DIR = _Path(__file__).resolve().parent / 'data' / 'scripts'
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))

from state_encoder import (
    MOVE_TARGET_PAIRS,
    SWITCH_OFFSET,
    ACTIONS_PER_SLOT,
    STATE_DIM,
    NUM_MOVES,
    # Shared move-target-kind tables — the SAME ones the training action mask
    # (build_action_mask) uses, so the live serve mask is identical to training.
    _move_target_kind,
    _CHOOSABLE_SINGLE,
    _ALLY_KINDS,
    # Gimmick (mega) codec — the SAME constants + capability check the training
    # gimmick mask (build_gimmick_mask) uses, so the serve gimmick mask matches.
    GIMMICK_DIM,
    GIMMICK_NONE,
    GIMMICK_MEGA,
    _species_is_mega_capable,
)
from live_state_encoder import LiveStateEncoder, own_bench_mons, team_has_megaed_live

log = logging.getLogger(__name__)

# ── Default Pikalytics belief (shared, loaded once) ───────────────────────────
# The trained nets were fit on belief-ENRICHED state vectors (opponent mons carry
# Pikalytics distribution est-stats + predicted move slots).  The live encoder
# must be given the SAME BeliefState or those opponent features sit at zero at
# serve time — a train/serve mismatch that degrades the policy.
_BELIEF_SINGLETON = None
_BELIEF_LOADED = False


def _default_belief():
    """Load (once) the default Pikalytics BeliefState, or None if unavailable."""
    global _BELIEF_SINGLETON, _BELIEF_LOADED
    if _BELIEF_LOADED:
        return _BELIEF_SINGLETON
    _BELIEF_LOADED = True
    try:
        from belief_state import BeliefState, _DEFAULT_PIKALYTICS_PATH
        if _DEFAULT_PIKALYTICS_PATH.exists():
            _BELIEF_SINGLETON = BeliefState(_DEFAULT_PIKALYTICS_PATH)
            log.info("Loaded default BeliefState from %s", _DEFAULT_PIKALYTICS_PATH)
        else:
            log.warning("Pikalytics file missing (%s) — opponent est-stats will be "
                        "zeroed (train/serve mismatch).", _DEFAULT_PIKALYTICS_PATH)
    except Exception as exc:  # pragma: no cover - degrade gracefully
        log.warning("Could not load default BeliefState (%s) — opponent est-stats "
                    "will be zeroed.", exc)
    return _BELIEF_SINGLETON

# ── Target constants (mirror state_encoder action space) ──────────────────────
_TARGET_OPP0 = 0
_TARGET_OPP1 = 1
_TARGET_ALLY = 2

# VGC: choose 4 leads from a 6-mon roster
VGC_TEAM_SIZE = 4


# ══════════════════════════════════════════════════════════════════════════════
# Replay buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Appends transitions to a JSON-lines file.
    Each line is one turn.  On battle end, outcome is written back to all
    turns from that battle by rewriting only the relevant lines.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._battle_line_indices: dict[str, list[int]] = {}
        self._line_count = _count_lines(path)

    def record(
        self,
        battle_id: str,
        turn: int,
        state: np.ndarray,
        action_s0: int,
        action_s1: int,
        source: str,
    ) -> None:
        entry = {
            "battle_id": battle_id,
            "turn":      turn,
            "state":     state.tolist(),
            "action_s0": action_s0,
            "action_s1": action_s1,
            "source":    source,
            "outcome":   None,
        }
        self._handle.write(json.dumps(entry) + "\n")
        self._handle.flush()
        self._battle_line_indices.setdefault(battle_id, []).append(self._line_count)
        self._line_count += 1

    def finalise(self, battle_id: str, outcome: int) -> None:
        """Back-fill outcome (1=win, 0=loss, -1=draw) for all turns of a battle."""
        indices = self._battle_line_indices.pop(battle_id, [])
        if not indices:
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for idx in indices:
                if idx < len(lines):
                    obj = json.loads(lines[idx])
                    obj["outcome"] = outcome
                    lines[idx] = json.dumps(obj)
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            log.warning("ReplayBuffer.finalise failed for %s: %s", battle_id, exc)

    def close(self) -> None:
        self._handle.close()


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ══════════════════════════════════════════════════════════════════════════════
# Teampreview heuristic
# ══════════════════════════════════════════════════════════════════════════════

def _heuristic_team_order(battle: DoubleBattle) -> List[int]:
    """
    Return a list of 0-based indices into battle.teampreview_team.
    Currently picks the first VGC_TEAM_SIZE mons in roster order.
    TODO: replace with type-advantage scoring once confirmed working.
    """
    team = list(battle.teampreview_team)
    n    = min(VGC_TEAM_SIZE, len(team))
    log.debug("Teampreview: first-%d roster order → %s", n, [team[i].species for i in range(n)])
    return list(range(n))


# ══════════════════════════════════════════════════════════════════════════════
# Action helpers  (pure functions, no player state)
# ══════════════════════════════════════════════════════════════════════════════

def build_legal_action_mask(battle: DoubleBattle, slot: int) -> list[bool]:
    """
    Return a 16-element bool list marking which actions are legal for `slot`.
    slot 0 = own active mon 0,  slot 1 = own active mon 1.

    Parity with training (state_encoder.build_action_mask): move-target buckets
    are restricted by the move's DEX TARGET KIND — a spread/self/field move
    exposes only its canonical bucket 0, an ally move only bucket 2, a single-
    target move only the present-foe buckets (+ ally for non-adjacentFoe).  The
    one serve-time addition is the PP check (the offline parser doesn't track PP,
    but a 0-PP move is genuinely illegal live).  Switch bench uses the same
    _is_real_mon filter as the live encoder, so switch index i refers to the
    SAME mon the model saw in encoder bench slot 4+i (drops broken-illusion
    phantoms).
    """
    mask = [False] * ACTIONS_PER_SLOT

    try:
        active = battle.active_pokemon
    except ValueError:
        return mask

    mon: Optional[Pokemon] = active[slot] if slot < len(active) else None
    if mon is None or mon.fainted:
        return mask

    # ── Move-target actions (0–11), keyed by DEX target kind (training parity) ──
    available_moves = list(mon.moves.values())[:NUM_MOVES]
    opp_active  = battle.opponent_active_pokemon
    opp0_alive  = len(opp_active) > 0 and opp_active[0] is not None
    opp1_alive  = len(opp_active) > 1 and opp_active[1] is not None
    ally_slot   = 1 - slot
    ally_alive  = (
        len(active) > ally_slot
        and active[ally_slot] is not None
        and not active[ally_slot].fainted
    )

    # Serve-time move legality: poke-env's available_moves[slot] is Showdown's
    # authoritative usable-move list — it drops Disabled / Choice-locked / Taunted
    # / Encored / consecutive-Protect / Fake-Out-after-turn-1 moves.  Gate on it
    # the SAME way the switch branch gates on available_switches[slot]; without it
    # the mask marks a disabled move legal, the model argmaxes it, and Showdown
    # rejects the order ("[Unavailable choice] Can't move: X is disabled") → the
    # player thrashes to a random fallback (the un-brought-switch bug's twin).
    # Like the PP guard this is SERVE-ONLY (the offline parser has no disabled
    # concept) and only ever REMOVES an action no player could take, so train/serve
    # legality stays consistent.  Skipped when poke-env exposes no list (empty →
    # request-less harness / non-move request) so the move mask never goes empty
    # spuriously; matched by Move.id (poke-env may hold distinct Move instances).
    try:
        _av = battle.available_moves
        usable_ids = {m.id for m in _av[slot]} if slot < len(_av) else set()
    except (ValueError, AttributeError, IndexError, TypeError):
        usable_ids = set()

    for m_idx, move in enumerate(available_moves):
        if move.current_pp == 0:        # serve-time only: 0-PP move is illegal
            continue
        if usable_ids and move.id not in usable_ids:   # Showdown-disabled / locked
            continue
        kind = _move_target_kind(move.id)
        if kind in _ALLY_KINDS:
            if kind == "adjacentAllyOrSelf" or ally_alive:
                mask[m_idx * 3 + _TARGET_ALLY] = True
        elif kind in _CHOOSABLE_SINGLE:
            buckets = [b for b, alive in ((_TARGET_OPP0, opp0_alive),
                                          (_TARGET_OPP1, opp1_alive)) if alive]
            # Ally bucket (2) is kept legal for any choosable-single move that can
            # face an ally (normal/any).  Targeting your OWN ally with a DAMAGING
            # move is intentionally allowed — it is a real tactic (activating an
            # ally's Justified / Anger Point / Berserk / Weakness Policy, Beat Up,
            # etc.) and, crucially, the offline training mask
            # (state_encoder.build_action_mask) marks it legal too, so blocking it
            # here would break train/serve parity.  A nonsensical self-attack is a
            # MODEL-QUALITY issue (rare: ~1/13164 corpus argmaxes), to be improved
            # by training/data — not by removing a legal action.
            if kind != "adjacentFoe" and ally_alive:
                buckets.append(_TARGET_ALLY)
            for b in (buckets or [_TARGET_OPP0]):   # no foe present → bucket 0
                mask[m_idx * 3 + b] = True
        else:
            # self / spread / field / unknown → canonical bucket 0 only,
            # unconditionally (these moves are always usable).
            mask[m_idx * 3 + _TARGET_OPP0] = True

    # ── Switch actions (12–15) ────────────────────────────────────────────────
    # own_bench_mons is the SAME brought-only bench the live encoder writes into
    # bench slots 4..7, so switch index i ↔ encoder bench slot 4+i.  It excludes
    # the un-brought 2-of-6 roster mons (VGC brings 4) that Showdown would reject
    # as a switch target.  Legality is then keyed PER SLOT by poke-env's
    # available_switches, so a trapped active correctly exposes no switches (and
    # a switch we DO mark legal is one Showdown will accept — no retry storm).
    bench = own_bench_mons(battle)
    try:
        avail = battle.available_switches
        slot_switchable = set(avail[slot]) if slot < len(avail) else set()
    except (ValueError, AttributeError, IndexError, TypeError):
        slot_switchable = set()
    for bench_idx, mon_b in enumerate(bench[:4]):
        if mon_b in slot_switchable:
            mask[SWITCH_OFFSET + bench_idx] = True

    return mask


def build_replacement_mask(battle: DoubleBattle, slot: int) -> list[bool]:
    """Switch-only legal mask for a FORCED replacement of ``slot`` (post-faint).

    ``build_legal_action_mask`` returns an all-zero row for a fainted/empty active
    slot (the mon is gone), so it can't drive a replacement.  This is the
    replacement analogue used by the model-driven forced-switch path: it marks
    switch index 12+i legal iff ``own_bench_mons(battle)[i]`` is one of Showdown's
    available replacements for that slot (``battle.available_switches[slot]``).
    Index i ↔ encoder bench slot 4+i, so the policy's switch logits line up with
    the bench the model saw — exactly the switch-only mask training's
    ``decision_type='replacement'`` transitions carried for the fainted slot.
    """
    mask = [False] * ACTIONS_PER_SLOT
    bench = own_bench_mons(battle)
    try:
        avail = battle.available_switches
        legal = set(avail[slot]) if slot < len(avail) else set()
    except (ValueError, AttributeError, IndexError, TypeError):
        legal = set()
    for i, mon_b in enumerate(bench[:4]):
        if mon_b in legal:
            mask[SWITCH_OFFSET + i] = True
    return mask


def build_gimmick_legal_mask(battle: DoubleBattle, slot: int) -> list[bool]:
    """Serve-time gimmick legality for ``slot``: ``[none_legal, mega_legal]``.

    BYTE-PARITY with training (state_encoder.build_gimmick_mask): bucket 0 (none)
    is legal for any present, non-fainted active slot; bucket 1 (mega) is legal
    iff the active mon is mega-CAPABLE — the SAME ``_species_is_mega_capable`` dex
    check the exporter used — AND no own mon has used mega this game.  An empty or
    fainted active slot gets an all-False row, mirroring build_legal_action_mask.

    This mask is capability-based (matching the offline data, where the mega STONE
    is unknown at decision time, so an item-gated mask would drop ~99% of real
    mega labels).  The FINAL order in action_to_order is the item-aware gate
    (battle.can_mega_evolve), so an illegal mega is never sent to Showdown.
    """
    row = [False] * GIMMICK_DIM
    try:
        active = battle.active_pokemon
    except ValueError:
        return row
    mon: Optional[Pokemon] = active[slot] if slot < len(active) else None
    if mon is None or mon.fainted:
        return row
    row[GIMMICK_NONE] = True
    base = getattr(mon, "base_species", None) or getattr(mon, "species", "")
    if _species_is_mega_capable(base) and not team_has_megaed_live(battle):
        row[GIMMICK_MEGA] = True
    return row


# Diagnostic counter: how many times the gap-#6 reconstruction let the codec aim
# a foe slot poke-env had merged away under a same-species illusion (#15).  Read
# by the illusion-targeting harness; harmless (a plain int) in production.
_ILLUSION_DELIBERATE_TARGETS = 0


def _live_can_mega(battle: DoubleBattle, slot: int) -> bool:
    """Authoritative serve-time mega legality for ``slot`` from poke-env's
    ``battle.can_mega_evolve`` — item- AND team-aware (False once the stone is
    gone or the team has already mega'd).  This is the FINAL safety gate so an
    illegal mega order is never sent, independent of the (capability-based)
    gimmick mask the model chose under.  Robust to the bool vs per-slot-list
    shapes across poke-env versions; any error → no mega."""
    try:
        cme = battle.can_mega_evolve
    except (ValueError, AttributeError):
        return False
    if isinstance(cme, (list, tuple)):
        return bool(cme[slot]) if slot < len(cme) else False
    return bool(cme)


def action_to_order(action: int, battle: DoubleBattle, slot: int,
                    gimmick: int = GIMMICK_NONE,
                    opp_present_recon: Optional[dict] = None) -> Optional[SingleBattleOrder]:
    """
    Convert an integer action (0–15) into a SingleBattleOrder for the given slot.
    Returns None if the action cannot be executed (caller should fall back).

    ``gimmick`` (GIMMICK_NONE / GIMMICK_MEGA) is the decoded gimmick decision for
    this slot.  A move is ordered with ``mega=True`` ONLY when the gimmick is mega
    AND ``battle.can_mega_evolve[slot]`` confirms it is currently legal — otherwise
    the plain move is ordered, so a rejected /choose is never emitted.  Switches
    never gimmick.  Tera / Z-move / Dynamax are never set (not in this format).

    ``opp_present_recon`` ({0: bool, 1: bool}) is the gap-#6 RECONSTRUCTED opponent
    slot occupancy.  During a same-species Zoroark illusion poke-env can MERGE the
    two foes and lose one of its target slots; the reconstruction still knows both
    slots are occupied, so when poke-env can't resolve the model's chosen opp_a/
    opp_b foe we target that slot's FIXED Showdown position — making mid-illusion
    targeting DELIBERATE (#15) instead of collapsing to the only visible foe.
    None ⇒ the legacy behaviour (no reconstruction available).

    Uses Player.create_order (static) with an integer move_target from
    DoubleBattle.to_showdown_target(), as required by poke-env's doubles API.
    """
    try:
        active = battle.active_pokemon
    except ValueError:
        return None

    mon: Optional[Pokemon] = active[slot] if slot < len(active) else None
    if mon is None:
        return None

    opp_active = battle.opponent_active_pokemon

    if action < SWITCH_OFFSET:
        # Mega is a checkbox on the chosen move — applied only when the model
        # picked it AND poke-env confirms it is legal right now (item + team).
        do_mega = (gimmick == GIMMICK_MEGA) and _live_can_mega(battle, slot)
        move_idx, target_code = MOVE_TARGET_PAIRS[action]
        moves = list(mon.moves.values())[:NUM_MOVES]
        if move_idx >= len(moves):
            return None
        move = moves[move_idx]
        # Serve-time: a Showdown-disabled / Choice-locked / Taunted move cannot be
        # ordered — fall back cleanly instead of emitting a rejected /choose order
        # (mirrors the build_legal_action_mask available_moves gate; Move.id match).
        try:
            _av = battle.available_moves
            _usable = {m.id for m in _av[slot]} if slot < len(_av) else set()
        except (ValueError, AttributeError, IndexError, TypeError):
            _usable = set()
        if _usable and move.id not in _usable:
            return None

        # Foe slots are positional (opp_a/opp_b).
        opp_present = [p for p in opp_active if p is not None]
        if target_code in (_TARGET_OPP0, _TARGET_OPP1):
            bucket = 0 if target_code == _TARGET_OPP0 else 1
            direct = opp_active[bucket] if (len(opp_active) > bucket and opp_active[bucket]) \
                else None
            if direct is not None:
                # poke-env has THIS exact foe → deliberate target by mon.
                return Player.create_order(
                    move, mega=do_mega,
                    move_target=battle.to_showdown_target(move, direct))

            # poke-env lost this exact slot to a same-species illusion merge.  If
            # the gap-#6 reconstruction confirms the slot is occupied and the move
            # is single-target, honour the model's opp_a/opp_b choice via the FIXED
            # Showdown position (opp_a→1, opp_b→2) — DELIBERATE mid-illusion targeting
            # (#15).  Showdown accepts a position target on an occupied slot.
            if (opp_present_recon and opp_present_recon.get(bucket)
                    and _move_target_kind(move.id) in _CHOOSABLE_SINGLE):
                global _ILLUSION_DELIBERATE_TARGETS
                _ILLUSION_DELIBERATE_TARGETS += 1
                return Player.create_order(
                    move, mega=do_mega,
                    move_target=(battle.OPPONENT_1_POSITION if bucket == 0
                                 else battle.OPPONENT_2_POSITION))

            # Fallbacks (UNCHANGED): the only visible foe; then a single-target move
            # with NO visible foe is ordered at the first opp slot (Showdown auto-
            # redirects to the only living foe); spread/self/field need no target.
            if opp_present:
                return Player.create_order(
                    move, mega=do_mega,
                    move_target=battle.to_showdown_target(move, opp_present[0]))
            if _move_target_kind(move.id) in _CHOOSABLE_SINGLE:
                return Player.create_order(move, mega=do_mega,
                                           move_target=battle.OPPONENT_1_POSITION)
            return Player.create_order(move, mega=do_mega)   # spread/self/field

        # ── ally (bucket 2) ──────────────────────────────────────────────────
        ally_slot = 1 - slot
        ally_mon = active[ally_slot] if ally_slot < len(active) else None
        if ally_mon is None:
            return None   # no ally to target → fall back (don't send a bad order)
        # poke-env's to_showdown_target returns EMPTY (0) for adjacentAlly moves
        # (Helping Hand / Coaching / Decorate / …) — Showdown then rejects the
        # order ("X needs a target").  The ally's OWN-side position is required;
        # supply it directly (same value to_showdown_target gives for a
        # normal/any move aimed at the ally, e.g. a Justified self-hit).
        pos = (battle.POKEMON_1_POSITION if ally_slot == 0
               else battle.POKEMON_2_POSITION)
        return Player.create_order(move, mega=do_mega, move_target=pos)

    else:
        bench_idx = action - SWITCH_OFFSET
        # SAME brought-only bench as the encoder + the mask, so switch index i
        # decodes to the mon in encoder bench slot 4+i (and never an un-brought
        # roster mon Showdown would reject).
        bench = own_bench_mons(battle)
        if bench_idx >= len(bench):
            return None
        return Player.create_order(bench[bench_idx])


def random_legal_action(battle: DoubleBattle, slot: int) -> int:
    """Pick a uniformly random legal action for the slot."""
    mask = build_legal_action_mask(battle, slot)
    legal = [i for i, ok in enumerate(mask) if ok]
    return random.choice(legal) if legal else 0


# ══════════════════════════════════════════════════════════════════════════════
# Abstract base player
# ══════════════════════════════════════════════════════════════════════════════

class VGCPlayerBase(Player):
    """
    Abstract poke-env Player for VGC doubles.

    Handles teampreview, forceSwitch, turn routing, replay recording, and
    order construction.  Subclasses only need to implement _select_actions().

    Parameters
    ----------
    replay_path : Path or None
        Where to write the JSON-lines replay buffer.
        Defaults to replay_buffer/replay.jsonl
    **kwargs
        Forwarded to poke_env.player.Player.
    """

    def __init__(self, replay_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        # Give the live encoder the SAME BeliefState the training data was
        # enriched with, so opponent est-stats + predicted move slots are
        # populated at serve time (matching the net's training distribution).
        self._encoder = LiveStateEncoder(belief=_default_belief())
        _rp = replay_path or Path("replay_buffer/replay.jsonl")
        self._replay  = ReplayBuffer(_rp)
        # Per-(battle,turn,force_switch) handling count — used to detect a forced-
        # switch request poke-env re-sends because Showdown REJECTED our last order
        # (an infinite-loop / battle-hang risk).  See _force_switch_escape.
        self._fs_attempts: dict = {}
        # Per-battle TOTAL forced-switch handlings (catches a loop even when the
        # turn/force_switch shifts each iteration, which the per-key counter misses).
        self._fs_battle_count: dict = {}

    # ── Subclass contract ─────────────────────────────────────────────────────

    @abstractmethod
    def _select_actions(
        self,
        battle: DoubleBattle,
        state_vec: np.ndarray,
    ) -> Tuple[int, int, str]:
        """
        Given the encoded battle state, return (action_s0, action_s1, source).
        source is a short string label, e.g. "model" or "random".
        Both actions must be in [0, ACTIONS_PER_SLOT).
        """

    def _select_gimmicks(
        self,
        battle: DoubleBattle,
        state_vec: np.ndarray,
        action_s0: int,
        action_s1: int,
    ) -> Tuple[int, int]:
        """Per-slot gimmick (mega) decision for the chosen actions: a pair of
        GIMMICK_* buckets passed alongside each action into ``_safe_order``.

        Base behaviour: NEVER gimmick (the random / heuristic players don't mega).
        The model player overrides this to masked-argmax its gimmick head over the
        per-slot gimmick legal mask (and forces no-gimmick on a switch)."""
        return GIMMICK_NONE, GIMMICK_NONE

    # ── poke-env entry points ─────────────────────────────────────────────────

    def teampreview(self, battle: DoubleBattle) -> str:
        """
        Called once per battle during teampreview.
        Returns a Showdown /team command, e.g. '/team 1234'.

        This format doesn't send |poke| messages so teampreview_team is
        always empty — falls back to battle.team directly.
        """
        team = list(battle.teampreview_team) or list(battle.team.values())
        max_size = battle.max_team_size if battle.max_team_size else VGC_TEAM_SIZE
        n = min(VGC_TEAM_SIZE, len(team), max_size)

        order = self._choose_team_order(battle, team, n)

        # Pad with unchosen mons if needed (edge-case guard)
        used = set(order)
        for i in range(len(team)):
            if len(order) >= n:
                break
            if i not in used:
                order.append(i)

        showdown_order = "".join(str(i + 1) for i in order)
        species_names  = [team[i].species for i in order]
        log.info(
            "Teampreview [%s] → /team %s  (%s)",
            battle.battle_tag, showdown_order, ", ".join(species_names),
        )
        return f"/team {showdown_order}"

    def choose_move(self, battle: DoubleBattle):
        """
        Called by poke-env every turn.  Handles three distinct request types:

        1. forceSwitch — one or both active slots fainted; send bench replacements.
                         Slots that did NOT faint receive PassBattleOrder.
        2. Normal turn — encode state, call _select_actions, record to replay.
        3. Non-doubles — shouldn't happen, falls back to choose_random_move.
        """
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        if any(battle.force_switch):
            return self._handle_force_switch(battle)

        state_vec = self._encoder.encode(battle)
        action_s0, action_s1, source = self._select_actions(battle, state_vec)
        g0, g1 = self._select_gimmicks(battle, state_vec, action_s0, action_s1)

        log.debug(
            "Turn %d [%s] a0=%d a1=%d src=%s g0=%d g1=%d",
            battle.turn, battle.battle_tag, action_s0, action_s1, source, g0, g1,
        )

        self._replay.record(
            battle_id = battle.battle_tag,
            turn      = battle.turn,
            state     = state_vec,
            action_s0 = action_s0,
            action_s1 = action_s1,
            source    = source,
        )

        order_s0 = self._safe_order(action_s0, battle, slot=0, gimmick=g0)
        order_s1 = self._safe_order(action_s1, battle, slot=1, gimmick=g1)
        return DoubleBattleOrder(order_s0, order_s1)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _choose_team_order(self, battle: DoubleBattle, team: list, n: int) -> List[int]:
        """
        Return up to n 0-based team indices for teampreview.
        Base implementation uses the first-N heuristic.
        Override in subclasses that have a team-chooser model.
        """
        return _heuristic_team_order(battle)[:n]

    def _force_switch_escape(self, battle: DoubleBattle):
        """Break a forced-switch loop, or None to handle the request normally.

        poke-env re-calls the forceSwitch handler with the same request when
        Showdown REJECTED our last order; a deterministic handler then rebuilds the
        same rejected order forever (the warning flood that hangs a battle).  Two
        layered escapes guarantee termination regardless of why it loops:

          * FORFEIT once a single battle has needed > ``_FS_FORFEIT_AT`` forced-
            switch handlings — a per-BATTLE counter, so it trips even if the turn /
            force_switch shifts each iteration (which a per-request counter misses).
            The battle ends as a loss and the run continues.
          * Before that, on the 3rd IDENTICAL (battle,turn,force_switch) handling,
            hand the request to Showdown's ``/choose default`` (server-resolved,
            always accepted) in case that alone clears it.

        A normal game needs only a few forced-switch handlings, so neither
        false-positives on real play."""
        tag = getattr(battle, "battle_tag", "?")
        c = self._fs_battle_count.get(tag, 0) + 1
        self._fs_battle_count[tag] = c
        if len(self._fs_battle_count) > 256:        # bound over a long run
            self._fs_battle_count = {tag: c}
        if c > _FS_FORFEIT_AT:
            log.error("forceSwitch [%s] handled %d× this battle — FORFEITING to "
                      "break a hang (run with -v / --spectate to diagnose).", tag, c)
            return ForfeitBattleOrder()
        try:
            key = (tag, getattr(battle, "turn", None),
                   tuple(bool(x) for x in (battle.force_switch or [])))
        except Exception:  # pragma: no cover - defensive
            return None
        n = self._fs_attempts.get(key, 0)
        self._fs_attempts[key] = n + 1
        if len(self._fs_attempts) > 512:
            self._fs_attempts = {key: n + 1}
        if n >= 2:                                   # 3rd identical attempt → rejected
            log.warning("forceSwitch [%s] re-requested %d× (order rejected) — sending "
                        "/choose default.", tag, n + 1)
            return DefaultBattleOrder()
        return None

    def _log_force_switch_state(self, battle: DoubleBattle, slot: int) -> None:
        """Dump the forced-switch board state so a 'no available switch' stall can
        be diagnosed: the request, poke-env's per-slot available switches, and the
        whole team with active/fainted/HP flags (an empty available list with a
        clearly-switchable bench mon ⇒ a poke-env desync, e.g. Ditto/Zoroark)."""
        try:
            avail = getattr(battle, "available_switches", []) or []
            avail_s = [[getattr(m, "species", "?") for m in (avail[s] if s < len(avail) else [])]
                       for s in range(2)]
            roster = []
            for mon in (getattr(battle, "team", {}) or {}).values():
                roster.append(
                    f"{getattr(mon, 'species', '?')}"
                    f"(act={getattr(mon, 'active', None)},"
                    f"fnt={getattr(mon, 'fainted', None)},"
                    f"hp={getattr(mon, 'current_hp_fraction', None)})")
            log.debug("forceSwitch DIAG [%s] turn=%s slot=%s force=%s "
                      "available_switches=%s team=[%s]",
                      getattr(battle, "battle_tag", "?"), getattr(battle, "turn", None),
                      slot, list(getattr(battle, "force_switch", []) or []),
                      avail_s, ", ".join(roster))
        except Exception:  # pragma: no cover - diagnostics must never throw
            log.debug("force-switch diag failed", exc_info=True)

    def _handle_force_switch(self, battle: DoubleBattle):
        """Handle a forceSwitch request (loop-guarded — see _force_switch_escape)."""
        escape = self._force_switch_escape(battle)
        if escape is not None:
            return escape
        return self._build_force_switch_order(battle)

    def _build_force_switch_order(self, battle: DoubleBattle):
        """Build the actual forceSwitch order (no loop guard — the caller guards).

        force_switch[i] == True  → slot fainted; pick a bench replacement.
        force_switch[i] == False → slot still alive; send PassBattleOrder.

        battle.available_switches is List[List[Pokemon]], one list per slot,
        already filtered by poke-env to exclude fainted / already-active mons.
        If a slot that MUST switch has NO available replacement, a hand-built Pass
        is exactly what Showdown rejects (→ it re-requests → loop); hand the whole
        request to Showdown's ``/choose default`` instead (always accepted)."""
        force    = battle.force_switch
        switches = battle.available_switches

        orders: list        = []
        used_switches: set  = set()

        for slot in range(2):
            if slot >= len(force) or not force[slot]:
                orders.append(PassBattleOrder())
            else:
                candidates = [
                    p for p in (switches[slot] if slot < len(switches) else [])
                    if id(p) not in used_switches
                ]
                if candidates:
                    chosen = random.choice(candidates)
                    used_switches.add(id(chosen))
                    orders.append(Player.create_order(chosen))
                else:
                    # A KNOWN, HANDLED poke-env desync (stale available_switches at an
                    # endgame double-faint): /choose default lets Showdown pick the real
                    # mon.  Quiet by default now that it's diagnosed — run -v to see it.
                    log.debug("forceSwitch slot %d: no available switch — '/choose "
                              "default' (handled poke-env desync).", slot)
                    self._log_force_switch_state(battle, slot)
                    return DefaultBattleOrder()

        order_s0 = orders[0] if len(orders) > 0 else PassBattleOrder()
        order_s1 = orders[1] if len(orders) > 1 else PassBattleOrder()

        log.debug(
            "forceSwitch [%s] force=%s → %s / %s",
            battle.battle_tag, force, order_s0.message, order_s1.message,
        )
        return DoubleBattleOrder(order_s0, order_s1)

    def _safe_order(self, action: int, battle: DoubleBattle, slot: int,
                    gimmick: int = GIMMICK_NONE,
                    opp_present_recon: Optional[dict] = None) -> SingleBattleOrder:
        """
        Convert action int → SingleBattleOrder, with two fallback levels:
          1. Try the given action (with the model's ``gimmick`` decision and the
             gap-#6 reconstructed opponent occupancy for deliberate targeting).
          2. Try a fresh random legal action (no gimmick — an emergency legal
             order, never speculatively mega'd).
          3. PassBattleOrder (slot has nothing to do this turn).
        DoubleBattleOrder never accepts None, so Pass is the correct no-op.
        """
        order = action_to_order(action, battle, slot, gimmick, opp_present_recon)
        if order is not None:
            return order
        # The chosen action was undecodable (e.g. a single-target move with an ally
        # present but no foe poke-env can see during a Zoroark illusion).  Try EVERY
        # other legal action — a switch always decodes — before giving up with Pass,
        # so we never Pass a slot that actually has a usable order (Showdown rejects
        # "Can't pass: your X must make a move or switch").
        log.debug("_safe_order: action %d slot %d → None, scanning legal actions.", action, slot)
        mask = build_legal_action_mask(battle, slot)
        for a, ok in enumerate(mask):
            if ok and a != action:
                order = action_to_order(a, battle, slot,
                                        opp_present_recon=opp_present_recon)
                if order is not None:
                    return order
        log.debug("_safe_order: slot %d has no decodable legal action — sending Pass.", slot)
        return PassBattleOrder()

    def _battle_finished_callback(self, battle: DoubleBattle) -> None:
        """Called by poke-env when a battle ends — back-fills outcomes in replay."""
        if battle.won:
            outcome = 1
        elif battle.lost:
            outcome = 0
        else:
            outcome = -1
        self._replay.finalise(battle.battle_tag, outcome)
        log.info(
            "Battle %s finished — outcome: %s  (W/L/D so far: %d/%d/%d)",
            battle.battle_tag,
            {1: "WIN", 0: "LOSS", -1: "DRAW"}[outcome],
            self.n_won_battles,
            self.n_lost_battles,
            self.n_tied_battles,
        )

    def close(self) -> None:
        """Flush and close the replay buffer."""
        self._replay.close()
