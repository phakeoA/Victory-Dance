"""Action codec: legality masks, gimmick mask, move-slot permutation, opp-perspective.

AUTO-SPLIT from state_encoder.py (v17 refactor) — PURE verbatim moves; the
frozen tensor layout + parity are unchanged.  state_encoder.py re-exports every
name moved here, so all existing imports keep working.
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Sequence
from v_dance.encoders.mechanic_tags import NUM_ABILITY_TAGS
from v_dance.parser.vod_parser.pokedex import get_pokedex, norm_species
from v_dance.encoders.encoder_layout import (ACTIONS_PER_SLOT, BENCH_SLOTS, GIMMICK_DIM, GIMMICK_MEGA, GIMMICK_NONE, GIMMICK_TERA, ITEM_BLOCK_V9, MOVE_FEATURES, NUM_BOOSTS, NUM_MOVES, NUM_STATUS, NUM_TYPES, POKEMON_FEATURES, SWITCH_OFFSET, WEIGHT_FEATURES)
from v_dance.encoders.battle_mechanics import (_gen9_moves, _get_moves_data, pp_max)



# ══════════════════════════════════════════════════════════════════════════════
# Action codec — string actions ⇄ fixed indices + legality masks
# (see ACTION SPACE in the module docstring for the frozen conventions)
# ══════════════════════════════════════════════════════════════════════════════

def move_slots_for_mon(mon: Optional[dict]) -> list[tuple[str, float]]:
    """
    Canonical ≤4-slot move list for one snapshot mon dict, as
    ``[(move_name, confidence), …]``.

    This is THE single source of truth shared by the feature encoder
    (_write_pokemon_json) and the action codec (action_to_index /
    build_action_mask): policy action ``m*3+t`` always refers to the move
    written into feature slot ``m`` of the same snapshot.

    Order: known_moves (exact/injected) wins over revealed_moves, then the
    remaining slots are padded with belief moves_predicted carrying p(usage)
    as confidence.
    """
    if not mon:
        return []
    slots: list[tuple[str, float]] = []
    for mv in (mon.get("known_moves") or mon.get("revealed_moves") or []):
        if mv:
            slots.append((mv, 1.0))
    belief = mon.get("belief") or {}
    for predicted in belief.get("moves_predicted") or []:
        if len(slots) >= NUM_MOVES:
            break
        slots.append((predicted["name"], float(predicted.get("p") or 0.0)))
    return slots[:NUM_MOVES]


# Showdown dex target kinds with a player-choosable single target
_CHOOSABLE_SINGLE = {"normal", "any", "adjacentFoe"}
# Kinds that always point at our own side (bucket 2)
_ALLY_KINDS = {"adjacentAlly", "adjacentAllyOrSelf"}


def _move_target_kind(move_name: Optional[str]) -> Optional[str]:
    """data/moves.json 'target' field for a move ('normal', 'self', …)."""
    if not move_name:
        return None
    data = _get_moves_data().get(norm_species(move_name))
    return (data or {}).get("target")


def _target_bucket(
    move_name: Optional[str],
    target_slot: Optional[str],
    opp_present: Optional[dict] = None,
) -> int:
    """Resolve a logged move action to its target bucket (0/1/2).

    ``opp_present`` ({0: bool, 1: bool} for opp_a / opp_b presence at decision
    time) lets a single-target move whose target Showdown did not log — it omits
    the target when auto-targeting the ONLY legal foe — resolve to the foe that
    is actually on the field, instead of defaulting to a possibly-empty opp_a
    slot.  Without this the index disagrees with build_action_mask (which marks
    only the present foe legal), producing an action that is illegal under its
    own mask.  Passing None preserves the legacy "trust the logged slot" path.
    """
    kind = _move_target_kind(move_name)
    if kind in _ALLY_KINDS:
        return 2
    if kind in _CHOOSABLE_SINGLE or kind is None:
        # kind None = move missing from moves.json → trust the logged slot
        if target_slot == "opp_a" and (opp_present is None or opp_present.get(0)):
            return 0
        if target_slot == "opp_b" and (opp_present is None or opp_present.get(1)):
            return 1
        if target_slot in ("our_a", "our_b"):
            return 2
        # No usable foe target logged (auto-targeted), or the logged foe slot is
        # empty at decision time (redirect) → point at whichever single foe is
        # actually present, matching Showdown auto-targeting and the mask.
        if opp_present is not None:
            if opp_present.get(0) and not opp_present.get(1):
                return 0
            if opp_present.get(1) and not opp_present.get(0):
                return 1
        return 0          # ambiguous (both/neither present) → canonical bucket
    return 0              # non-choosable (self / spread / field / …)


def _living_bench(snap: dict) -> list[dict]:
    """Our bench in encoder order: living mons only, capped at BENCH_SLOTS."""
    return [m for m in (snap.get("our_bench") or [])
            if not m.get("is_fainted")][:BENCH_SLOTS]


def _species_matches(mon: dict, species: Optional[str]) -> bool:
    if not species:
        return False
    want = norm_species(species)
    return want in (
        norm_species(mon.get("species") or ""),
        norm_species(mon.get("base_species") or ""),
    )


def action_to_index(
    action: dict,
    mon: Optional[dict],
    snap: Optional[dict] = None,
) -> Optional[int]:
    """
    Encode one perspective-relative action dict (transitions.py our_actions
    entry) into the fixed 0–15 index space.

    ``mon``  — the acting mon's dict from snap["our_active"][action["slot"]]
               (needed for move actions; switches only need ``snap``).
    ``snap`` — the decision-time snapshot (state_before_actions), used for
               the bench ordering of switch actions.

    Returns None when the action cannot be expressed: move not among the
    mon's 4 encoded slots, or switch target not on the living bench (e.g.
    mid-turn forced replacement by a mon that was active at turn start).
    """
    kind = action.get("action")
    if kind == "switch":
        for i, bench_mon in enumerate(_living_bench(snap or {})):
            if _species_matches(bench_mon, action.get("species")):
                return SWITCH_OFFSET + i
        return None
    if kind == "move":
        want = norm_species(action.get("move") or "")
        if not want:
            return None
        opp_active = (snap or {}).get("opp_active") or {}
        opp_present = {0: "opp_a" in opp_active, 1: "opp_b" in opp_active}
        for m_idx, (name, _conf) in enumerate(move_slots_for_mon(mon)):
            if norm_species(name) == want:
                return m_idx * 3 + _target_bucket(
                    name, action.get("target_slot"), opp_present
                )
        return None
    return None


def action_to_gimmick(action: dict) -> int:
    """Map one our_actions entry to its gimmick bucket (0=none, 1=mega, 2=tera).

    The parser stamps ``mega=True`` / ``tera=True`` (v11 Phase D) onto the chosen move for the slot that
    mega-evolved / terastallized that turn (replay_parser._extract_actions).  Switches/replacements never
    gimmick → GIMMICK_NONE.  Mega takes precedence (a mon cannot do both on one move; mirrors the
    SingleBattleOrder mega>tera elif).
    """
    if action.get("mega"):
        return GIMMICK_MEGA
    if action.get("tera"):
        return GIMMICK_TERA
    return GIMMICK_NONE


def index_to_action(idx: int) -> dict:
    """Decode a 0–15 action index into its structural meaning."""
    if 0 <= idx < SWITCH_OFFSET:
        m_idx, bucket = divmod(idx, 3)
        return {
            "kind": "move",
            "move_slot": m_idx,
            "target_bucket": bucket,
            "target": ("opp_a", "opp_b", "ally")[bucket],
        }
    if SWITCH_OFFSET <= idx < ACTIONS_PER_SLOT:
        return {"kind": "switch", "bench_slot": idx - SWITCH_OFFSET}
    raise ValueError(f"action index out of range: {idx}")


# ── Move-slot permutation (training augmentation, task #22) ──────────────────
# The BC net is ~96% move-ORDER sensitive: it leans on a move's slot POSITION (a
# "slot 0 = main move" prior baked in by reveal-order training) instead of the
# move's FEATURES — and at serve the own moves arrive in poke-env REQUEST order,
# so the prior is misapplied.  Train-only fix: randomly permute a mon's 4 move
# sub-blocks AND the matching action label so the net must read move features,
# not position.  These helpers are the single source for WHERE the move blocks
# live (so the dataset augmentation can never drift from the encoder layout).
#
# Within a POKEMON_FEATURES block the 4 move sub-blocks start here (after hp,
# both types, base+est stats, stats-known, mega, tera, status, boosts):
_WEIGHT_REL = 1 + 2 * NUM_TYPES + 6 + 6 + 1 + 1 + 1 + NUM_STATUS + NUM_BOOSTS  # 70 (v9 weight feature)
_MOVE_BLOCK_REL = _WEIGHT_REL + WEIGHT_FEATURES  # 71 (v9: moves start after weight)

# v9 within-mon-row offsets of the IDENTITY-INDEX columns (the model extracts + embeds these; never a
# Linear input). Computed from the block sizes so encoder + model can't drift.
_ITEM_BLOCK_REL    = _MOVE_BLOCK_REL + NUM_MOVES * MOVE_FEATURES   # 279
ITEM_ID_REL        = _ITEM_BLOCK_REL                              # item identity index
_ABILITY_BLOCK_REL = _ITEM_BLOCK_REL + ITEM_BLOCK_V9              # 304
ABILITY_ID_REL     = _ABILITY_BLOCK_REL                           # ability identity index
ABILITY_KNOWN_REL  = _ABILITY_BLOCK_REL + 1 + NUM_ABILITY_TAGS    # ability known/confidence (conf-scales the emb)
# per-move identity index = 2nd-to-last float in each MOVE_FEATURES block (before is_known)
MOVE_ID_RELS       = tuple(_MOVE_BLOCK_REL + m * MOVE_FEATURES + (MOVE_FEATURES - 2) for m in range(NUM_MOVES))
# Own ACTIVE mons are state blocks 0 (our_a) and 1 (our_b); opp/bench moves have
# no action head, so they are never permuted.


def own_active_move_base(slot: int) -> int:
    """Absolute index of own active ``slot``'s (0=our_a, 1=our_b) first move
    feature in an encoded state vector."""
    return slot * POKEMON_FEATURES + _MOVE_BLOCK_REL


def permute_move_slots(vec: np.ndarray, slot: int, perm: Sequence[int]) -> None:
    """In place: reorder the 4 move sub-blocks of own active ``slot`` so the new
    move-slot ``i`` holds the OLD move-slot ``perm[i]``.  ``perm`` is a
    permutation of ``range(NUM_MOVES)``; all other features are left untouched."""
    base = own_active_move_base(slot)
    blocks = [vec[base + m * MOVE_FEATURES: base + (m + 1) * MOVE_FEATURES].copy()
              for m in range(NUM_MOVES)]
    for i, src in enumerate(perm):
        vec[base + i * MOVE_FEATURES: base + (i + 1) * MOVE_FEATURES] = blocks[src]


def permute_action_index(idx: Optional[int], perm: Sequence[int]) -> Optional[int]:
    """Remap an action index under move-slot permutation ``perm`` (``perm[i]`` =
    the old move-slot now at position ``i``).  Switch indices (>=SWITCH_OFFSET),
    None and out-of-range pass through unchanged; a move index ``m*3+bucket`` maps
    to ``j*3+bucket`` where ``perm[j] == m``."""
    if idx is None or idx < 0 or idx >= SWITCH_OFFSET:
        return idx
    m, bucket = divmod(idx, 3)
    return list(perm).index(m) * 3 + bucket


def permute_action_mask_row(row: Sequence, perm: Sequence[int]) -> list:
    """Remap a 16-wide action mask row under ``perm``: new move block ``j`` takes
    old move block ``perm[j]``; switch entries (>=SWITCH_OFFSET) are unchanged."""
    out = list(row)
    for j in range(NUM_MOVES):
        src = perm[j]
        for b in range(3):
            out[j * 3 + b] = row[src * 3 + b]
    return out


# v11 exact-mask: items / move-ids the conservative legality drop keys on.
_CHOICE_ITEMS    = frozenset({"choiceband", "choicescarf", "choicespecs"})
_FIRST_TURN_ONLY = frozenset({"fakeout", "firstimpression", "matblock"})


def _certainly_illegal_move_slots(mon: dict, field: dict) -> set:
    """Slot indices into move_slots_for_mon(mon) that the OFFLINE snapshot PROVES illegal at decision time.
    CONSERVATIVE-SUPERSET: the actually-taken action is NEVER in this set — we drop only on certainty, and
    the Encore/known-Choice locks fail OPEN (keep all moves) when the locked move can't be mapped to a slot.
    Drops: 0-PP (PP-Up-inflated denom), Gravity-grounded moves, non-first-turn Fake-Out-class, named Disable,
    Taunt/known-Assault-Vest status moves, and the Encore / KNOWN-Choice single-move lock. Switch-trapping,
    belief-Choice and Imprison are NOT modelled (still a superset there). MASK-only; no encoder/layout impact."""
    vol = mon.get("volatiles") or {}
    item = "" if mon.get("item_consumed") else norm_species(mon.get("known_item") or "")
    pp_used = mon.get("move_pp_used") or {}
    first_turn = bool(vol.get("first_turn"))
    taunt = bool(vol.get("taunt"))
    gravity_on = (field.get("gravity_turns_remaining") or 0) > 0
    # v11 P5: Magic Room (ignoringItem) lifts ALL item restrictions → the Assault Vest status-ban and the
    # Choice move-lock no longer apply. Skipping them keeps the mask a conservative SUPERSET (more permissive,
    # never drops the taken action) and matches poke-env available_moves under Magic Room.
    magic_room = (field.get("magic_room_turns_remaining") or 0) > 0
    av = (item == "assaultvest") and not magic_room
    disabled = norm_species(vol.get("disabled_move") or "") if vol.get("disabled_move") else ""
    encore = norm_species(vol.get("encore_move") or "") if vol.get("encore_move") else ""

    slot_ids = [norm_species(n) for n, _ in move_slots_for_mon(mon)]
    # Single-move lock target (Encore first, else a KNOWN Choice item) — fail-open unless it maps to a slot.
    locked = None
    if encore and encore in slot_ids:
        locked = encore
    elif item in _CHOICE_ITEMS and not magic_room:    # v11 P5: Magic Room lifts the Choice lock
        stint = mon.get("stint_moves") or []
        cand = norm_species(stint[0]) if stint else ""
        if cand and cand in slot_ids:
            locked = cand

    illegal = set()
    for i, mid in enumerate(slot_ids):
        data = _gen9_moves().get(mid) or {}
        mx = pp_max(data.get("pp"))
        if mx and pp_used.get(mid, 0) >= mx:                       # 0-PP (inflated denom → never the taken move)
            illegal.add(i); continue
        if gravity_on and (data.get("flags") or {}).get("gravity"):  # Gravity-grounded move
            illegal.add(i); continue
        if mid in _FIRST_TURN_ONLY and not first_turn:             # Fake Out / First Impression / Mat Block
            illegal.add(i); continue
        if (taunt or av) and (data.get("category") or "").lower() == "status":   # Taunt / Assault Vest
            illegal.add(i); continue
        if disabled and mid == disabled:                          # Disable: the named move
            illegal.add(i); continue
        if locked and mid != locked:                              # Encore / known-Choice: keep only the lock
            illegal.add(i)
    return illegal


def build_action_mask(snap: dict) -> dict[str, list[int]]:
    """
    Decision-time legality mask for one state_before_actions snapshot:
    ``{"our_a": [16×0/1], "our_b": [16×0/1]}``.

    Legal = the agent could have selected it at the START of the turn:
      · every named move slot, restricted to its reachable target buckets
        (occupied opposing slots / present ally; non-choosable moves expose
        only their canonical bucket 0; no legal foe → bucket-0 fallback,
        mirroring Showdown's auto-targeting),
      · one switch index per living bench mon.
    An empty/fainted active slot has an all-zero row.  v11 exact-mask: moves
    the snapshot PROVES illegal (0-PP, Gravity-grounded, non-first-turn Fake-Out
    class, named Disable, Taunt/known-AV status, Encore / KNOWN-Choice lock) are
    dropped CONSERVATIVELY (never the taken action).  Switch-trapping, belief-
    Choice and Imprison remain UN-modelled (still a superset there).
    """
    our_active = snap.get("our_active") or {}
    opp_active = snap.get("opp_active") or {}
    n_bench    = len(_living_bench(snap))
    opp_present = {0: "opp_a" in opp_active, 1: "opp_b" in opp_active}
    field = snap.get("field") or {}

    mask: dict[str, list[int]] = {}
    for slot in ("our_a", "our_b"):
        row = [0] * ACTIONS_PER_SLOT
        mon = our_active.get(slot)
        if mon and not mon.get("is_fainted"):
            ally_slot = "our_b" if slot == "our_a" else "our_a"
            ally_present = ally_slot in our_active
            illegal = _certainly_illegal_move_slots(mon, field)   # v11 exact-mask
            for m_idx, (name, _conf) in enumerate(move_slots_for_mon(mon)):
                if m_idx in illegal:
                    continue                                       # certainly illegal → no target buckets
                kind = _move_target_kind(name)
                if kind in _ALLY_KINDS:
                    if kind == "adjacentAllyOrSelf" or ally_present:
                        row[m_idx * 3 + 2] = 1
                elif kind in _CHOOSABLE_SINGLE:
                    buckets = [b for b in (0, 1) if opp_present[b]]
                    if kind != "adjacentFoe" and ally_present:
                        buckets.append(2)
                    for b in (buckets or [0]):
                        row[m_idx * 3 + b] = 1
                else:
                    # self / spread / field / unknown → canonical bucket only
                    row[m_idx * 3] = 1
            for i in range(n_bench):
                row[SWITCH_OFFSET + i] = 1
        mask[slot] = row
    return mask


def _species_is_mega_capable(species: Optional[str]) -> bool:
    """True iff this (base) species has a mega forme in the dex.

    We gate the gimmick mask on mega-CAPABILITY (species) rather than on holding
    a mega stone (item), because at the moment a mon mega-evolves its stone is
    revealed DURING that turn — the decision-time snapshot (state_before_actions)
    does NOT yet carry the item for ~99% of real megas.  An item-gated mask would
    mark the actual mega label illegal and annotate_transition_actions would drop
    it, leaving the head nothing to learn.  Capability is the stable, offline-
    computable signal that keeps every real mega label legal; the LIVE serve path
    additionally gates the emitted order on ``battle.can_mega_evolve`` (true item-
    aware legality), so an illegal mega order is never sent.
    """
    dex = get_pokedex()
    return bool(dex.mega_formes_for(species)) if dex else False


def _own_team_has_megaed(snap: dict) -> bool:
    """Has any own mon already mega-evolved this game?  A team megas at most once
    per game, so the moment any own active/bench mon shows ``is_mega`` the gimmick
    is spent and mega is no longer a legal choice for either slot."""
    mons = list((snap.get("our_active") or {}).values()) + list(snap.get("our_bench") or [])
    return any(m.get("is_mega") for m in mons)


def _own_team_has_teraed(snap: dict) -> bool:
    """v11 Phase D: has any own mon already terastallized this game?  Tera is once-per-game per side
    (like mega), so the moment any own active/bench mon shows ``is_terastallized`` the tera gimmick is
    spent.  (Tera is mod-disabled in Champions → always False today; forward-compat with the mega twin.)"""
    mons = list((snap.get("our_active") or {}).values()) + list(snap.get("our_bench") or [])
    return any(m.get("is_terastallized") for m in mons)


def build_gimmick_mask(snap: dict) -> dict[str, list[int]]:
    """
    Decision-time gimmick legality for one state_before_actions snapshot:
    ``{"our_a": [3×0/1], "our_b": [3×0/1]}`` over (none, mega, tera).

    bucket 0 (none) is legal for any acting (present, non-fainted) slot — not gimmicking is always allowed.
    bucket 1 (mega) is legal iff the acting mon is mega-capable AND no own mon has used mega this game.
    bucket 2 (tera, v11 Phase D) is legal iff no own mon has tera'd this game — EVERY mon can tera (no
    species/item gate; the LIVE serve path additionally gates on battle.can_tera). An empty/fainted active
    slot gets an all-zero row, mirroring build_action_mask.
    """
    our_active  = snap.get("our_active") or {}
    team_megaed = _own_team_has_megaed(snap)
    team_teraed = _own_team_has_teraed(snap)      # v11 Phase D

    mask: dict[str, list[int]] = {}
    for slot in ("our_a", "our_b"):
        row = [0] * GIMMICK_DIM
        mon = our_active.get(slot)
        if mon and not mon.get("is_fainted"):
            row[GIMMICK_NONE] = 1
            base = mon.get("base_species") or mon.get("species")
            if not team_megaed and _species_is_mega_capable(base):
                row[GIMMICK_MEGA] = 1
            if not team_teraed:                    # v11 Phase D: no species/item gate (every mon can tera)
                row[GIMMICK_TERA] = 1
        mask[slot] = row
    return mask


def annotate_transition_actions(transition: dict) -> dict:
    """
    Stamp the action-space view onto one transitions.py dict, in place:

      · each our_actions entry gains ``action_index`` (0–15 or None) and
        ``gimmick_index`` (0=none / 1=mega / None),
      · ``action_mask`` becomes the build_action_mask() of
        state_before_actions (the decision-time state),
      · ``gimmick_mask`` becomes the build_gimmick_mask() of the same state.

    Pure-python and deterministic given the stored snapshot (the belief
    padding that orders move slots is baked into the same JSON), so exports
    can carry it without the state_vector being encoded.

    Invariant guaranteed here: every non-null ``action_index`` is legal under
    the same transition's ``action_mask``, and every non-null ``gimmick_index``
    is legal under its ``gimmick_mask``.  An action that cannot be resolved to a
    mask-legal index is stamped None (never a wrong index); its gimmick is then
    None too (a gimmick is a checkbox on an action we could not encode).  A
    decoded action whose gimmick bucket is not legal (a mega by a mon the mask
    says cannot mega) likewise drops the gimmick to None — never a wrong label.
    """
    snap = transition.get("state_before_actions") or {}
    our_active = snap.get("our_active") or {}
    mask = build_action_mask(snap)
    gimmick_mask = build_gimmick_mask(snap)
    for act in transition.get("our_actions") or []:
        slot = act.get("slot")
        idx = action_to_index(act, our_active.get(slot), snap)
        if idx is not None:
            row = mask.get(slot) or []
            if idx >= len(row) or row[idx] != 1:
                idx = None  # not expressible in the turn-start frame → unencoded
        act["action_index"] = idx

        # Gimmick (mega) is a checkbox on the chosen action — only meaningful
        # when that action is itself encodable and the gimmick bucket is legal.
        if idx is None:
            g = None
        else:
            g = action_to_gimmick(act)
            grow = gimmick_mask.get(slot) or []
            if g >= len(grow) or grow[g] != 1:
                g = None  # gimmick illegal under its mask (never on real data)
        act["gimmick_index"] = g

    transition["action_mask"] = mask
    transition["gimmick_mask"] = gimmick_mask
    return transition


# ── Opponent-perspective action codec (auxiliary head, task #9) ──────────────
# The aux opponent head predicts the OPPONENT's action from the same (our-
# perspective) state vector — a representation-shaping signal so the trunk
# captures opponent threats (e.g. for board-dependent mega timing).  It is
# TRAIN-ONLY (never served), so it lives entirely offline.
#
# Rather than fork the battle-tested codec, we FLIP the snapshot (swap our<->opp
# actives/benches + relabel slot keys) and reuse build_action_mask /
# action_to_index unchanged: from the opponent's seat, OUR mons are the foes and
# the opp bench holds the switch-ins.
OPP_HEADS: tuple[str, str] = ("opp_a", "opp_b")


def _relabel_slots(d: Optional[dict], frm: str, to: str) -> dict:
    return {(k.replace(frm, to, 1) if isinstance(k, str) else k): v
            for k, v in (d or {}).items()}


def _flip_perspective(snap: dict) -> dict:
    """A snapshot seen from the OPPONENT's seat: our<->opp actives/benches swapped
    and the active-slot keys relabeled (opp_a->our_a, our_a->opp_a, …).  Only the
    fields the action codec reads are populated."""
    return {
        "our_active": _relabel_slots(snap.get("opp_active"), "opp_", "our_"),
        "opp_active": _relabel_slots(snap.get("our_active"), "our_", "opp_"),
        "our_bench": snap.get("opp_bench") or [],
        "opp_bench": snap.get("our_bench") or [],
        # #07: carry the global FIELD through the flip. build_action_mask's exact-legality drops keyed
        # on field (Gravity-grounded moves, Magic-Room) read snap["field"]; omitting it left the opp_a/
        # opp_b aux-head mask a looser superset on Gravity/Magic-Room turns. Field is side-symmetric
        # (global), so it carries verbatim — no relabel.
        "field": snap.get("field"),
    }


def _flip_action_slots(action: dict) -> dict:
    """Swap our_<->opp_ in an action's ``slot`` / ``target_slot`` so an opp action
    reads correctly under the flipped snapshot."""
    out = dict(action)
    for field in ("slot", "target_slot"):
        v = out.get(field)
        if isinstance(v, str):
            if v.startswith("opp_"):
                out[field] = "our_" + v[4:]
            elif v.startswith("our_"):
                out[field] = "opp_" + v[4:]
    return out


def build_opp_action_mask(snap: dict) -> dict[str, list[int]]:
    """Decision-time legality mask for the OPPONENT's active slots:
    ``{"opp_a": [16], "opp_b": [16]}`` — build_action_mask over the flipped snap."""
    m = build_action_mask(_flip_perspective(snap))
    return {"opp_a": m.get("our_a") or [0] * ACTIONS_PER_SLOT,
            "opp_b": m.get("our_b") or [0] * ACTIONS_PER_SLOT}


def opp_action_to_index(action: dict, snap: dict) -> Optional[int]:
    """Encode one opp_actions_actual entry into the 0–15 codec from the
    opponent's perspective (our mons are its foes, opp bench its switch-ins)."""
    flipped = _flip_perspective(snap)
    fa = _flip_action_slots(action)
    mon = (flipped.get("our_active") or {}).get(fa.get("slot"))
    return action_to_index(fa, mon, flipped)


def annotate_opp_actions(transition: dict) -> dict:
    """Stamp ``opp_action_index`` on each opp_actions_actual entry + the
    ``opp_action_mask`` onto a transition, in place — the auxiliary opponent-head
    target/mask.  Same invariant as annotate_transition_actions: a non-null
    opp_action_index is always legal under opp_action_mask (else stamped None)."""
    snap = transition.get("state_before_actions") or {}
    opp_mask = build_opp_action_mask(snap)
    for act in transition.get("opp_actions_actual") or []:
        slot = act.get("slot")
        idx = opp_action_to_index(act, snap)
        if idx is not None:
            row = opp_mask.get(slot) or []
            if idx >= len(row) or row[idx] != 1:
                idx = None
        act["opp_action_index"] = idx
    transition["opp_action_mask"] = opp_mask
    return transition
