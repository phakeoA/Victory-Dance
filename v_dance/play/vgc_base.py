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

from v_dance.encoders.state_encoder import (
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
    GIMMICK_TERA,
    _species_is_mega_capable,
)
from v_dance.encoders.live_state_encoder import (
    LiveStateEncoder, own_bench_mons, own_active_move_list, own_switch_slot,
    team_has_megaed_live, team_has_teraed_live,
)

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
        from v_dance.parser.belief_state import BeliefState
        from v_dance.formats import pikalytics_path_for, default_format
        # Resolve the ACTIVE format's belief FRESH (not belief_state's import-time
        # constant) so an entry point's --format / set_active_format takes effect
        # before the first battle.
        path = pikalytics_path_for(default_format())
        if path and path.exists():
            _BELIEF_SINGLETON = BeliefState(path)
            log.info("Loaded BeliefState for %s from %s", default_format(), path)
        else:
            log.warning("Pikalytics file missing for %s — opponent est-stats will be "
                        "zeroed (train/serve mismatch).", default_format())
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


def prune_replay_buffer(buffer_dir, keep: int = 200) -> int:
    """Cap the BC-era diagnostic replay buffer to its ``keep`` most-recently-modified ``.jsonl``
    files, deleting the older ones. Returns the count deleted.

    Each live player streams a per-turn BC-style trace (state/action/source/outcome) to its own
    ``<username>.jsonl`` here; the usernames are uid-keyed, so a long run accumulates THOUSANDS of
    files that nothing cleans up. The self-play RL training does NOT read these (it uses a separate
    trajectory store — see ``selfplay/store.py``), so trimming old ones is safe. ``keep`` <= 0 keeps
    ALL. Crash-proof: a file that can't be deleted (an open handle on Windows = a battle still
    writing it) is skipped, so call this at GEN BOUNDARIES when no battle is mid-write (the current
    generation's files may transiently exceed ``keep`` until the next boundary prune)."""
    if not keep or keep <= 0:
        return 0
    d = Path(buffer_dir)
    if not d.is_dir():
        return 0
    files = []
    for f in d.glob("*.jsonl"):
        try:
            files.append((f.stat().st_mtime, f))
        except OSError:
            pass
    files.sort(key=lambda t: t[0], reverse=True)          # newest first
    deleted = 0
    for _mt, f in files[int(keep):]:                       # everything past the newest `keep`
        try:
            f.unlink()
            deleted += 1
        except OSError:                                    # open handle / perms -> skip, try next time
            pass
    return deleted


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

def _norm_move_id(mid) -> str:
    """A punctuation/case-insensitive move id, so a Choice-locked move whose
    ``available_moves`` id differs only cosmetically from its ``mon.moves`` entry still
    matches (the "live move-id != mon.moves" drop)."""
    return "".join(c for c in str(mid or "").lower() if c.isalnum())


def _move_usable_predicate(battle: DoubleBattle, slot: int):
    """Return ``(predicate, have_authoritative)``. ``predicate(move) -> bool`` decides whether
    ``move`` is usable THIS turn per Showdown's ``available_moves[slot]`` — which is AUTHORITATIVE
    (it already excludes Disabled / Choice-locked-out / Taunted / Encored / 0-PP moves). Matches by
    exact id, then NORMALISED id, then object identity, so a Choice-locked move whose
    ``available_moves`` id — or poke-env's stale ``current_pp`` — differs from its ``mon.moves``
    entry is NOT wrongly dropped (that drop left the slot move-less, collapsing the turn to
    ``/choose default`` and DISCARDING the training step — the Choice-Scarf bug). When Showdown
    exposes NO list (request-less harness / non-move request), falls back to the local
    ``current_pp > 0`` guard so a genuinely 0-PP move is still illegal and the mask never goes
    empty spuriously."""
    try:
        av = battle.available_moves
        usable = list(av[slot]) if (slot is not None and slot < len(av)) else []
    except (ValueError, AttributeError, IndexError, TypeError):
        usable = []
    if not usable:
        return (lambda mv: getattr(mv, "current_pp", 1) != 0), False
    ids = {m.id for m in usable}
    norm = {_norm_move_id(m.id) for m in usable}
    objs = {id(m) for m in usable}

    def _pred(mv) -> bool:
        return (mv.id in ids or _norm_move_id(mv.id) in norm or id(mv) in objs)

    return _pred, True


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
    # own_active_move_list is mon.moves normally, but the request-authoritative real
    # moves when an own-side Illusion leaves mon.moves empty (#1b) — so the mask
    # offers the true moves (matching the encoder) instead of collapsing to a switch-
    # only / empty row that forces /choose default.
    available_moves = own_active_move_list(battle, slot, mon)
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
    _usable, _ = _move_usable_predicate(battle, slot)

    for m_idx, move in enumerate(available_moves):
        if not _usable(move):           # Showdown-authoritative (id / norm-id / identity); 0-PP
            continue                    # only when Showdown exposes no list (see predicate)
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
    # bench slots 4..7 (request-authoritative — #11b), so switch index i ↔ encoder
    # bench slot 4+i, and it excludes the un-brought 2-of-6.  A normal-turn switch
    # is illegal ONLY when the active mon is TRAPPED.  That trap signal lives in
    # ``battle.trapped`` (poke-env fills it position-indexed straight from the
    # request's active[].trapped), so an Illusion's object-model desync can't
    # corrupt it — unlike the per-mon active/fainted MEMBERSHIP of
    # ``available_switches``, which the Illusion DOES corrupt (dropping a healthy
    # bench mon).  So we gate the request-authoritative bench on ``not trapped``
    # (#11c) rather than available_switches membership — robust to the same desync
    # the forced-replacement path handles, and closer to the offline training mask
    # (which offers every living bench switch).  ``maybe_trapped`` still offers
    # switches (Showdown resolves), matching available_switches.  Reviving (Revival
    # Blessing switches in a FAINTED mon) is the one case the living-bench codec
    # can't express → offer no switch (the prior behaviour: living ∩ fainted = ∅).
    bench = own_bench_mons(battle)
    if not getattr(battle, "reviving", False) and not _slot_trapped(battle, slot):
        for bench_idx in range(min(len(bench), 4)):
            mask[SWITCH_OFFSET + bench_idx] = True

    return mask


def _slot_trapped(battle: DoubleBattle, slot: int) -> bool:
    """Whether the active mon in ``slot`` is DEFINITELY trapped (cannot switch).

    Read from ``battle.trapped`` — poke-env fills it per active-slot position
    directly from the request's ``active[].trapped`` field, so an Illusion's
    object-model desync cannot corrupt it (unlike ``available_switches``' per-mon
    membership).  ``maybe_trapped`` is intentionally NOT treated as trapped
    (Showdown resolves an uncertain trap), matching poke-env's available_switches,
    which gates on ``trapped`` only.  Any error → not trapped (offer switches)."""
    try:
        tr = battle.trapped
        return bool(tr[slot]) if slot is not None and slot < len(tr) else False
    except (AttributeError, IndexError, TypeError):
        return False


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

    poke-env ``available_switches`` DESYNC RECOVERY (#11): poke-env builds
    available_switches from each team mon's ``active`` / ``fainted`` flags, which go
    stale after an Imposter (Ditto) / Illusion (Zoroark) turn — so a healthy, brought
    bench mon is DROPPED from the list.  Critically this happens on a DOUBLE FAINT:
    the gauntlet showed ``available_switches=[['ditto'],['ditto']]`` while basculegion
    + zoroark + ditto were ALL healthy — poke-env offered 1 of 3, boxing the model
    into one switch for two fainted slots → ``/choose default``.

    A post-faint replacement is NEVER trapped, so EVERY living, brought bench mon
    (``own_bench_mons`` — request-authoritative, #11b) is a legal replacement.  We
    therefore IGNORE ``available_switches`` here entirely and offer the full own bench
    by POSITION; cross-slot dedup (the caller) picks distinct mons, and
    ``_replacement_order`` builds the ``/switch`` from the same ``own_bench_mons``
    list, so Showdown accepts each.  The un-brought 2-of-6 are already excluded by
    ``own_bench_mons``, so this never offers an un-switchable mon.  (Reviving / Revival
    Blessing is a MOVE that brings in a FAINTED ally, not a post-faint switch, so it
    does not reach this mask — own_bench_mons is living-only.)
    """
    mask = [False] * ACTIONS_PER_SLOT
    # Only a slot that MUST switch has replacement actions.
    force = getattr(battle, "force_switch", None) or []
    if slot >= len(force) or not force[slot]:
        return mask
    bench = own_bench_mons(battle)
    for i in range(min(len(bench), 4)):
        mask[SWITCH_OFFSET + i] = True
    return mask


def build_gimmick_legal_mask(battle: DoubleBattle, slot: int) -> list[bool]:
    """Serve-time gimmick legality for ``slot``: ``[none_legal, mega_legal, tera_legal]`` (v11 Phase D).

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
    if not team_has_teraed_live(battle):        # v11 Phase D: no species/item gate (every mon can tera)
        row[GIMMICK_TERA] = True
    return row


def _live_can_tera(battle: DoubleBattle, slot: int) -> bool:
    """v11 Phase D: authoritative serve-time tera legality for ``slot`` from poke-env's ``battle.can_tera``
    (per-active-slot bool, re-derived each request → already once-per-game-aware). The FINAL safety gate so
    an illegal tera order is never sent, independent of the (capability-based) gimmick mask the model chose
    under. Robust to the bool vs per-slot-list shapes across poke-env versions; any error → no tera.
    (Tera is mod-disabled in Champions → can_tera is always falsy today; forward-compat with _live_can_mega.)"""
    try:
        ct = battle.can_tera
    except (ValueError, AttributeError):
        return False
    if isinstance(ct, (list, tuple)):
        return bool(ct[slot]) if slot < len(ct) else False
    return bool(ct)


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
                    opp_present_recon: Optional[dict] = None,
                    taken_switch_targets: Optional[set] = None) -> Optional[SingleBattleOrder]:
    """
    Convert an integer action (0–15) into a SingleBattleOrder for the given slot.
    Returns None if the action cannot be executed (caller should fall back).

    ``taken_switch_targets`` is the set of ``/choose switch …`` commands the OTHER
    active slot is already using this turn (see ``_switch_order_target``).  A switch
    whose resolved command is in that set is REFUSED (returns None) so the caller
    falls back to a different legal action — this is the doubles "[Invalid choice]
    Can't switch: The Pokémon in slot N can only switch in once" fix.  Because both
    slots decode switch index 12+i to the SAME benched mon (own_bench_mons is
    slot-independent), two slots can otherwise emit the same ``/choose switch`` —
    EITHER directly (both pick a switch index, e.g. the retry perturbation) OR
    indirectly (both move actions decode to None under an Illusion and each
    independently substitutes the first legal switch in ``_safe_order``'s fallback).

    ``gimmick`` (GIMMICK_NONE / GIMMICK_MEGA / GIMMICK_TERA) is the decoded gimmick decision for this
    slot.  A move is ordered with ``mega=True`` (resp. ``terastallize=True``) ONLY when the gimmick is mega
    (resp. tera) AND ``battle.can_mega_evolve[slot]`` (resp. ``battle.can_tera[slot]``) confirms it is
    currently legal — otherwise the plain move is ordered, so a rejected /choose is never emitted. do_mega
    and do_tera are mutually exclusive (one int gimmick).  Switches never gimmick.  ⚠ v11 Phase D: tera
    ordering is WIRED but INERT — the Champions mod hard-disables tera (canTerastallize → null) so can_tera
    is always falsy; the capability is forward-compat.  Z-move / Dynamax are not in this format.

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
        do_tera = (gimmick == GIMMICK_TERA) and _live_can_tera(battle, slot)   # v11 Phase D (mutually excl.)
        move_idx, target_code = MOVE_TARGET_PAIRS[action]
        # SAME list as the encoder + the mask (own_active_move_list): mon.moves
        # normally, the request's real moves when illusion leaves mon.moves empty
        # (#1b), so move_idx decodes to the move the model actually saw + chose.
        moves = own_active_move_list(battle, slot, mon)
        if move_idx >= len(moves):
            return None
        move = moves[move_idx]
        # Serve-time: a Showdown-disabled / Taunted move cannot be ordered — fall back cleanly
        # instead of emitting a rejected /choose. Uses the SAME Showdown-authoritative predicate
        # as build_legal_action_mask (id / normalised id / identity), so a Choice-locked move the
        # mask allowed isn't then rejected here for a cosmetic id difference / stale current_pp
        # (which would re-introduce the /choose default collapse the mask fix removed).
        _usable_pred, _have_av = _move_usable_predicate(battle, slot)
        if _have_av and not _usable_pred(move):
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
                    move, mega=do_mega, terastallize=do_tera,
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
                    move, mega=do_mega, terastallize=do_tera,
                    move_target=(battle.OPPONENT_1_POSITION if bucket == 0
                                 else battle.OPPONENT_2_POSITION))

            # Fallbacks (UNCHANGED): the only visible foe; then a single-target move
            # with NO visible foe is ordered at the first opp slot (Showdown auto-
            # redirects to the only living foe); spread/self/field need no target.
            if opp_present:
                return Player.create_order(
                    move, mega=do_mega, terastallize=do_tera,
                    move_target=battle.to_showdown_target(move, opp_present[0]))
            if _move_target_kind(move.id) in _CHOOSABLE_SINGLE:
                return Player.create_order(move, mega=do_mega, terastallize=do_tera,
                                           move_target=battle.OPPONENT_1_POSITION)
            return Player.create_order(move, mega=do_mega, terastallize=do_tera)   # spread/self/field

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
        return Player.create_order(move, mega=do_mega, terastallize=do_tera, move_target=pos)

    else:
        bench_idx = action - SWITCH_OFFSET
        # SAME brought-only bench as the encoder + the mask, so switch index i
        # decodes to the mon in encoder bench slot 4+i (and never an un-brought
        # roster mon Showdown would reject).
        bench = own_bench_mons(battle)
        if bench_idx >= len(bench):
            return None
        order = build_switch_order(battle, bench[bench_idx])
        # Cross-slot dedup: never emit a switch the OTHER active slot is already
        # using this turn ("slot N can only switch in once").  Compared by the
        # resolved /choose-switch command (robust to the Illusion name-vs-slot
        # ambiguity); the caller substitutes a different legal action.
        if taken_switch_targets and _switch_order_target(order) in taken_switch_targets:
            return None
        _log_switch_choice(battle, slot, bench[bench_idx], "normal")
        return order


def build_switch_order(battle: DoubleBattle, mon) -> SingleBattleOrder:
    """A ``/switch`` order for ``mon`` by unambiguous team SLOT (its position in the
    request's side.pokemon) when a live request is available, else by species name.

    Slot-based ordering is THE fix for the 'can't switch to an active Pokémon'
    rejection under Illusion / Imposter (see live_state_encoder.own_switch_slot): a
    name-based ``/choose switch Charizard`` can match an ACTIVE mon disguised as that
    species, but ``/choose switch 3`` targets the exact (benched) team member.  No
    request (offline / replay parity path) → name-based create_order, unchanged."""
    slot = own_switch_slot(battle, mon)
    if slot is not None:
        return SingleBattleOrder(f"/choose switch {slot}")
    return Player.create_order(mon)


def _switch_order_target(order) -> Optional[str]:
    """The normalised ``/choose switch …`` command an order represents, or None if
    the order is not a switch.  Two slots whose orders yield the SAME string are
    switching in the SAME team member — the doubles "can only switch in once"
    collision.  Comparing the command string is identical for the slot-based live
    order ("/choose switch 3") and the name-based offline order ("/choose switch
    Zoroark"), so it needs no mon resolution and is robust to the Illusion
    name-vs-slot ambiguity."""
    if order is None:
        return None
    msg = getattr(order, "message", None)
    if isinstance(msg, str):
        s = msg.strip()
        if s.startswith("/choose switch ") or s.startswith("/switch "):
            return s
    return None


def _log_switch_choice(battle: DoubleBattle, slot: int, chosen, ctx: str) -> None:
    """DIAG: dump a chosen SWITCH so a 'can't switch to an active Pokémon' rejection
    can be traced — the chosen mon vs the live active set (by object id) vs the
    request's own-side flags vs poke-env's available_switches.  Debug-only; never
    throws."""
    if not log.isEnabledFor(logging.DEBUG):
        return
    try:
        from v_dance.encoders.live_state_encoder import (_request_own_state, own_bench_mons,
                                         _active_mon_set)
        act = []
        try:
            for m in (battle.active_pokemon or []):
                act.append("None" if m is None else
                           f"{getattr(m,'species','?')}#{id(m)%10000}"
                           f"(act={getattr(m,'active',None)},fnt={getattr(m,'fainted',None)})")
        except Exception:
            act = ["err"]
        try:
            av = battle.available_switches
            avs = [[getattr(m, "species", "?") for m in (av[s] if s < len(av) else [])]
                   for s in range(2)]
        except Exception:
            avs = "err"
        bench = [f"{getattr(m,'species','?')}#{id(m)%10000}" for m in own_bench_mons(battle)]
        log.debug("switchDIAG [%s] turn=%s slot=%s ctx=%s chosen=%s#%s in_active_set=%s "
                  "active=%s own_bench=%s available_switches=%s request=%s",
                  getattr(battle, "battle_tag", "?"), getattr(battle, "turn", None),
                  slot, ctx, getattr(chosen, "species", "?"), id(chosen) % 10000,
                  chosen in _active_mon_set(battle), act, bench, avs,
                  _request_own_state(battle))
    except Exception:
        log.debug("switchDIAG failed", exc_info=True)


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
        Defaults to artifacts/replay_buffer/replay.jsonl
    **kwargs
        Forwarded to poke_env.player.Player.
    """

    def __init__(self, replay_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        # Give the live encoder the SAME BeliefState the training data was
        # enriched with, so opponent est-stats + predicted move slots are
        # populated at serve time (matching the net's training distribution).
        self._encoder = LiveStateEncoder(belief=_default_belief())
        _rp = replay_path or Path("artifacts/replay_buffer/replay.jsonl")
        self._replay  = ReplayBuffer(_rp)
        # Per-(battle,turn,force_switch) handling count — used to detect a forced-
        # switch request poke-env re-sends because Showdown REJECTED our last order
        # (an infinite-loop / battle-hang risk).  See _force_switch_escape.
        self._fs_attempts: dict = {}
        # Per-battle TOTAL forced-switch handlings (catches a loop even when the
        # turn/force_switch shifts each iteration, which the per-key counter misses).
        self._fs_battle_count: dict = {}
        # 15b.0: the ACTUAL team-preview decision submitted this battle, keyed by
        # battle_tag -> {bring, leads, own_team, opp_team}. Captured in teampreview()
        # so the self-play collector can attach the TRUE TP decision + opponent roster
        # to the trajectory (sec 14 outcome-driven TP); popped in _battle_finished_callback.
        self._tp_decision: dict = {}

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
        # Phase-2 z: stamp the model's own-team archetype id (once per player —
        # the serve-time nearest-centroid lookup; no-op for non-z checkpoints).
        self._ensure_archetype(team)
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

        # 15b.0: record the ACTUAL submitted decision + rosters seen. ``order`` is the
        # final (padded) leads-first submission, so bring = the whole order and leads =
        # its first two (Showdown leads positions 1-2 in doubles). own_team is captured
        # HERE so the recorded bring indices index the SAME roster they were chosen
        # against. Best-effort — bookkeeping must never disturb the returned /team order.
        try:
            store = getattr(self, "_tp_decision", None)
            if store is None:
                store = self._tp_decision = {}
            opp_src = (getattr(battle, "teampreview_opponent_team", None)
                       or getattr(battle, "opponent_team", None) or [])
            opp_mons = list(opp_src.values()) if isinstance(opp_src, dict) else list(opp_src)
            store[battle.battle_tag] = {
                "bring": [int(i) for i in order],
                "leads": [int(i) for i in order[:2]],
                "own_team": [getattr(team[i], "species", None) for i in range(len(team))],
                "opp_team": [sp for sp in (getattr(m, "species", None) for m in opp_mons) if sp],
            }
            if len(store) > 256:                     # bound over a long run (defensive)
                self._tp_decision = {battle.battle_tag: store[battle.battle_tag]}
        except Exception:
            log.debug("teampreview decision capture failed (non-fatal)", exc_info=True)

        return f"/team {showdown_order}"

    def _ensure_archetype(self, team) -> None:
        """Phase-2 z serve lookup: nearest-centroid archetype of OUR OWN team,
        computed ONCE per player instance (the team is fixed per player today;
        a Phase-4 TeamBuilder would need per-battle stamping) and set as the
        model's default archetype id — every forward then reads the right z
        embedding with zero call-site changes.  No-op for non-z checkpoints;
        a failed lookup serves the UNKNOWN embedding and logs loudly."""
        mdl = getattr(self, "_model", None)
        if (mdl is None or not int(getattr(mdl, "n_archetypes", 0) or 0)
                or getattr(self, "_archetype_stamped", False)):
            return
        self._archetype_stamped = True                 # one attempt per instance
        try:
            art = getattr(mdl, "_z_artifact", None)
            if not art:
                raise ValueError(
                    "z checkpoint has no embedded z_artifact (retrain with "
                    "--z-archetypes to stamp the centroid table)")
            from v_dance.datatools.team_archetypes import assign_team_sheet
            mons = [{
                "species": getattr(m, "species", None),
                "item": getattr(m, "item", None),
                "ability": getattr(m, "ability", None),
                "moves": list(getattr(m, "moves", {}) or {}),
                "nature": None,
            } for m in team]
            aid, dist = assign_team_sheet(mons, art)
            mdl.set_default_archetype(aid)
            log.info("Phase-2 z: own team → archetype %d (centroid distance %.2f)",
                     aid, dist)
        except Exception:
            log.warning(
                "Phase-2 z: own-team archetype lookup FAILED — serving with the "
                "UNKNOWN embedding (unconditioned play)", exc_info=True)

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

        # audit (#10 + round-4 regression fix): break a REJECTION LOOP. A deterministic root-base player (the
        # scripted gauntlet/collection opponents) rebuilds the SAME order when Showdown rejects it as an
        # unavailable choice (a rare mask desync), and poke-env re-calls choose_move with the unchanged battle
        # — with no escape the battle wedged until the chunk-stall watchdog abandoned the WHOLE chunk. After
        # repeated rebuilds for the same (battle,turn), escape to /choose default (always accepted). This runs
        # UNCONDITIONALLY (covers BOTH the normal-turn AND the empty-mask/forced-move partial-order paths — the
        # empty-mask _forced_aware_double_order is just as deterministic and could loop the same way). No-op on
        # a normal turn (called once → counter stays 1). The SplicingVGCPlayerBase fully overrides choose_move
        # with its own retry guard, so this is root-only.
        _rb = getattr(self, "_turn_rebuilds", None)
        if _rb is None:
            _rb = {}
            self._turn_rebuilds = _rb
        _k = (battle.battle_tag, battle.turn)
        if _k not in _rb:
            _rb.clear()                       # new turn → forget old keys (bounds the dict)
        _rb[_k] = _rb.get(_k, 0) + 1
        if _rb[_k] > 10:                      # mirrors live_vgc_base._MAX_TURN_RETRIES
            log.warning("Turn %d [%s] order rebuilt %d× without finishing — '/choose default' "
                        "(deterministic opponent mask desync; breaking the rejection loop).",
                        battle.turn, battle.battle_tag, _rb[_k])
            return DefaultBattleOrder()

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

        # Forced-move turn: an active slot with an EMPTY legal mask can only play a move the 16-action
        # codec can't express (Struggle / recharge / a forced 2-turn continuation). If a PARTNER slot
        # still has a real choice, preserve its decision (forced move + partner order); only when EVERY
        # active slot is forced does the model have nothing to drive → /choose default. (Was a blanket
        # whole-turn /choose default that discarded the free partner's decision.)
        if self._active_empty_mask(battle):
            return self._forced_aware_double_order(battle, action_s0, action_s1, g0, g1)
        order_s0 = self._safe_order(action_s0, battle, slot=0, gimmick=g0)
        # Slot-1 must not switch in the same mon slot-0 is switching in ("slot N can
        # only switch in once") — thread slot-0's switch command so slot-1's decode
        # AND fallback scan both avoid it (covers random/scripted players too).
        taken = _switch_order_target(order_s0)
        order_s1 = self._safe_order(action_s1, battle, slot=1, gimmick=g1,
                                    taken_switch_targets={taken} if taken else None)
        # _safe_order returns None when an ACTIVE, non-fainted slot has no representable
        # legal order (its only codec action — a switch — collides with the ally's,
        # leaving only an inexpressible forced move).  Passing it is illegal, so resolve
        # the whole turn via /choose default — the cross-slot variant of the empty-mask
        # escape above, which the per-slot mask can't see until both orders are built.
        if order_s0 is None or order_s1 is None:
            return DefaultBattleOrder()
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
            # #11 diag: what OUR fix actually sees — the request-authoritative state
            # and the resulting own bench (the set build_replacement_mask offers).
            try:
                from v_dance.encoders.live_state_encoder import _request_own_state, own_bench_mons
                req = _request_own_state(battle)
                req_s = ("None" if req is None
                         else {k.split(": ")[-1]: ("act" if a else "") + ("fnt" if f else "")
                               for k, (a, f) in req.items()})
                bench_s = [getattr(m, "species", "?") for m in own_bench_mons(battle)]
                team_keys = list((getattr(battle, "team", {}) or {}).keys())
            except Exception as _e:
                req_s, bench_s, team_keys = f"err:{_e}", "err", "err"
            log.debug("forceSwitch DIAG [%s] turn=%s slot=%s force=%s "
                      "available_switches=%s own_bench_mons=%s request_state=%s "
                      "team_keys=%s team=[%s]",
                      getattr(battle, "battle_tag", "?"), getattr(battle, "turn", None),
                      slot, list(getattr(battle, "force_switch", []) or []),
                      avail_s, bench_s, req_s, team_keys, ", ".join(roster))
        except Exception:  # pragma: no cover - diagnostics must never throw
            log.debug("force-switch diag failed", exc_info=True)

    def _handle_force_switch(self, battle: DoubleBattle):
        """Handle a forceSwitch request (loop-guarded — see _force_switch_escape)."""
        escape = self._force_switch_escape(battle)
        if escape is not None:
            # #4-ext: record a backstop-FORFEIT tag for the ROOT lineage too (scripted/random opponents that
            # don't use the Splicing handler), so the self-play recorder can FALLBACK-discard a battle the
            # OPPONENT forfeited instead of crediting us a spurious +1. Mirrors SplicingVGCPlayerBase's
            # populate; inline the '>'-strip (== _norm_tag) to avoid a circular import from live_vgc_base.
            if isinstance(escape, ForfeitBattleOrder):
                seen = getattr(self, "_forfeited_tags", None)
                if seen is None:
                    seen = self._forfeited_tags = set()
                _t = str(getattr(battle, "battle_tag", "") or "").strip().lstrip(">").strip()
                if _t not in seen:
                    seen.add(_t)
                    _sc = getattr(self, "_source_counts", None)   # root scripted/random players lack it
                    if _sc is not None:
                        _sc["forfeit"] += 1
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
                    orders.append(build_switch_order(battle, chosen))
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
                    opp_present_recon: Optional[dict] = None,
                    taken_switch_targets: Optional[set] = None) -> Optional[SingleBattleOrder]:
        """
        Convert action int → SingleBattleOrder, with these fallback levels:
          1. Try the given action (with the model's ``gimmick`` decision and the
             gap-#6 reconstructed opponent occupancy for deliberate targeting).
          2. Scan EVERY other legal action for one the codec can express.
          3a. PassBattleOrder — the correct no-op for a FAINTED / empty slot.
          3b. ``None`` — an ACTIVE, non-fainted slot for which NO legal action
              decodes (its only codec action is a switch the OTHER slot is taking,
              leaving only a forced move the 16-action codec can't express, e.g.
              Struggle).  Passing such a slot is ILLEGAL ("must make a move/switch"),
              so the caller resolves the whole turn via ``/choose default`` instead.
        DoubleBattleOrder never accepts None, so the callers convert the 3b None to
        a DefaultBattleOrder (the same escape ``_active_empty_mask`` uses).

        ``taken_switch_targets`` (the OTHER slot's ``/choose switch`` command, if it
        switched) is threaded into BOTH the primary decode AND the level-2 scan so
        neither can synthesize a switch the other slot already uses — closing the
        "can only switch in once" collision for the perturbation path AND the
        under-illusion case where two move actions both fall back to the same switch.
        """
        order = action_to_order(action, battle, slot, gimmick, opp_present_recon,
                                taken_switch_targets=taken_switch_targets)
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
                                        opp_present_recon=opp_present_recon,
                                        taken_switch_targets=taken_switch_targets)
                if order is not None:
                    return order
        # Passing an ACTIVE, non-fainted mon is ALWAYS illegal ("Can't pass: your X must
        # make a move or switch"). If we reach here for such a slot the mask/codec is out of
        # sync with Showdown — log WHY (move/switch availability) so it can be fixed, not
        # just papered over by the retry→/choose default escape.
        try:
            _active = battle.active_pokemon
            _mon = _active[slot] if slot < len(_active) else None
        except (ValueError, IndexError):
            _mon = None
        if _mon is not None and not getattr(_mon, "fainted", False):
            try:
                _av = battle.available_moves
                _avs = battle.available_switches
                # exact move ids + PP so a RESIDUAL drop (id that isn't id/norm/identity-equal to
                # mon.moves, or a truly-forced Struggle/recharge) is debuggable from the log.
                _own = [(getattr(m, "id", "?"), getattr(m, "current_pp", "?"))
                        for m in own_active_move_list(battle, slot, _mon)]
                _use = [getattr(m, "id", "?") for m in (_av[slot] if _av and slot < len(_av) else [])]
                _diag = (f"species={getattr(_mon, 'species', '?')} "
                         f"mon.moves={len(getattr(_mon, 'moves', {}) or {})} "
                         f"avail_moves[{slot}]={len(_av[slot]) if _av and slot < len(_av) else 'NA'} "
                         f"avail_switches[{slot}]={len(_avs[slot]) if _avs and slot < len(_avs) else 'NA'} "
                         f"mask_legal={sum(1 for x in mask if x)} own={_own} usable={_use}")
            except Exception:
                _diag = "(diag unavailable)"
            # An ACTIVE, non-fainted slot whose ONLY codec-expressible legal action is
            # unusable this turn — typically its sole legal switch collides with the
            # ally's (one bench mon both slots want), leaving only a forced move
            # (Struggle / recharge / a live move-id != mon.moves) the 16-action codec
            # can't express.  Passing it is ILLEGAL ("must make a move/switch"); return
            # None so the caller resolves the whole turn with /choose default instead of
            # sending a Pass Showdown rejects (the retry-storm the user's live run hit).
            log.warning("_safe_order: slot %d has NO representable legal order for an ACTIVE "
                        "mon (forced move / cross-slot switch collision) — resolving the turn "
                        "via /choose default. %s", slot, _diag)
            return None
        log.debug("_safe_order: slot %d has no decodable legal action — sending Pass.", slot)
        return PassBattleOrder()

    @staticmethod
    def _active_empty_mask(battle: DoubleBattle) -> bool:
        """True if some ACTIVE, non-fainted slot has an EMPTY legal-action mask — its only
        usable order is one the 16-action codec can't express (Struggle / recharge / a forced
        2-turn continuation, or a live move-id that != mon.moves). Passing such a slot is
        illegal (and loops in the retry-less root path), so the caller resolves the turn with
        /choose default. False for normal turns. Lives on the ROOT base so the model, scripted,
        and gauntlet players all share it."""
        try:
            active = battle.active_pokemon
        except (ValueError, AttributeError):
            return False
        for slot in range(min(2, len(active))):
            mon = active[slot] if slot < len(active) else None
            if mon is None or getattr(mon, "fainted", False):
                continue
            if not any(build_legal_action_mask(battle, slot)):
                return True
        return False

    @staticmethod
    def _any_active_free(battle: DoubleBattle) -> bool:
        """True if some ACTIVE, non-fainted slot has a NON-empty 16-action legal mask — a real model
        decision worth preserving. False ⇒ every active slot is forced into a synthetic move (recharge/
        Struggle) the codec can't express, so the model has NO decision and /choose default is correct."""
        try:
            active = battle.active_pokemon
        except (ValueError, AttributeError):
            return False
        for slot in range(min(2, len(active))):
            mon = active[slot] if slot < len(active) else None
            if mon is None or getattr(mon, "fainted", False):
                continue
            if any(build_legal_action_mask(battle, slot)):
                return True
        return False

    @staticmethod
    def _forced_single_order(battle: DoubleBattle, slot: int) -> Optional[SingleBattleOrder]:
        """The forced SYNTHETIC move (recharge after Hyper Beam/Giga Impact/…, or Struggle) for an
        ACTIVE slot whose 16-action codec mask is empty: order ``available_moves[slot][0]`` directly
        (recharge/Struggle auto-target — no codec slot, no move_target). Returns None when
        ``available_moves[slot]`` is empty (Commander/Dondozo — genuinely no action) or the slot is
        empty/fainted, so the caller falls back to /choose default for that degenerate case."""
        try:
            active = battle.active_pokemon
            mon = active[slot] if slot < len(active) else None
        except (ValueError, IndexError):
            return None
        if mon is None or getattr(mon, "fainted", False):
            return None
        try:
            av = battle.available_moves
            moves = av[slot] if av and slot < len(av) else []
        except (ValueError, IndexError):
            moves = []
        if not moves:
            return None
        return Player.create_order(moves[0])

    def _forced_aware_double_order(self, battle: DoubleBattle, action_s0: int, action_s1: int,
                                   g0: int = GIMMICK_NONE, g1: int = GIMMICK_NONE,
                                   opp_present_recon: Optional[dict] = None):
        """Build the turn's order when ≥1 active slot is FORCED into a synthetic move (recharge/Struggle)
        the codec can't express, WHILE a partner slot has a genuine choice — PRESERVING the partner's
        model decision instead of collapsing the WHOLE turn to /choose default (which plays the free slot
        randomly). Per slot: a free slot → the model's ``_safe_order``; an empty-mask forced slot → its
        ``_forced_single_order``. Returns DefaultBattleOrder when EVERY active slot is forced (no model
        decision) or a forced slot has no expressible action (Commander) / a free slot's only order
        collides — the proven whole-turn escape. ``g0``/``g1`` apply the model's gimmick to a free slot."""
        if not self._any_active_free(battle):
            return DefaultBattleOrder()          # every active slot forced → no decision → proven escape
        o0 = self._safe_order(action_s0, battle, slot=0, gimmick=g0,
                              opp_present_recon=opp_present_recon)
        taken = _switch_order_target(o0)
        o1 = self._safe_order(action_s1, battle, slot=1, gimmick=g1,
                              opp_present_recon=opp_present_recon,
                              taken_switch_targets={taken} if taken else None)
        if o0 is None:
            o0 = self._forced_single_order(battle, 0)
        if o1 is None:
            o1 = self._forced_single_order(battle, 1)
        if o0 is None or o1 is None:
            return DefaultBattleOrder()           # a slot has NO expressible action → proven escape
        return DoubleBattleOrder(o0, o1)

    def _battle_finished_callback(self, battle: DoubleBattle) -> None:
        """Called by poke-env when a battle ends — back-fills outcomes in replay."""
        # 15b.0: drop the captured TP decision for ALL players (the SelfPlay recorder
        # reads it BEFORE calling super(); non-recording players just need the cleanup,
        # so the per-battle dict can't grow unbounded over a long run).
        try:
            getattr(self, "_tp_decision", {}).pop(battle.battle_tag, None)
        except Exception:
            pass
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
        """Flush and close the replay buffer, then EVICT this player's poke-env logger.

        poke-env's ``PSClient._create_logger`` does ``logging.getLogger(self.username)`` +
        ``addHandler(StreamHandler())`` for EVERY player, and the stdlib interns every logger by
        name in the process-global ``logging.Logger.manager.loggerDict`` forever — gc never frees
        it. Because the 22d gen-salted account names make every player's username SINGLE-USE
        (``BC{gen}x{uid}`` / ``LG0x{gen}x{uid}`` / …, never reconnected after the chunk), a fresh
        Logger + StreamHandler + Formatter leaks per player per generation → tens of thousands of
        unreclaimable objects over a long run, in every collection/eval worker. Removing the
        handlers + popping the logger reclaims it. Fully guarded — teardown must never raise."""
        self._replay.close()
        try:
            import logging
            name = self.ps_client.username
            lg = logging.getLogger(name)
            for h in list(lg.handlers):
                lg.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
            logging.Logger.manager.loggerDict.pop(name, None)
        except Exception:
            pass
