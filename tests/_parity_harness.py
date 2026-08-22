"""Train/serve parity harness (NOT a test module — leading underscore so
pytest does not collect it).

Feeds ONE Showdown replay log to BOTH encode paths so they can be diffed
feature-block by feature-block:

  (a) the LIVE path   — a poke-env DoubleBattle reconstructed by replaying the
      protocol messages through its own parser (no |request| lines exist in a
      replay, so the player_role side is built from public switch/poke events
      exactly like the opponent side), then StateEncoder.encode(battle);
  (b) the OFFLINE path — the existing vod_parser, then
      StateEncoder.encode_snapshot(state_before_actions[perspective]).

Alignment: both are sampled at every ``|turn|N|`` boundary.  Processing the
log up to and including ``|turn|N|`` leaves the live battle at the START of
turn N, which is exactly what the offline ``state_before_actions`` of turn N
captures.

IMPORTANT scope note (see test_encoder_parity for the full writeup):
  * The OPPONENT side (opp active + opp bench) and the GLOBAL block are built
    purely from public log info on both paths, so they are directly
    comparable.  This is where the mechanical encoder gaps live.
  * The OWN side is NOT faithfully reproducible from a replay alone: the real
    live bot also receives private |request| messages (exact own stats, full
    brought-team roster, own-side illusion correction) that a replay does not
    contain.  Own-side blocks therefore carry inherent, documented divergences
    and are excluded from strict parity assertions.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

from poke_env.battle import DoubleBattle

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser, extract_log_from_html
from v_dance.encoders.state_encoder import (
    VodStateEncoder, POKEMON_FEATURES, ACTIVE_SLOTS, BENCH_SLOTS,
    MOVE_FEATURES, NUM_MOVES, STATE_DIM, OPP_BENCH_SLOTS,
    NUM_WEATHER, NUM_FIELDS, NUM_SIDE_CONDS,
    NUM_ITEM_EFFECTS, NUM_ABILITY_EFFECTS, ITEM_FEATURES, ABILITY_FEATURES,
    _MOVE_BLOCK_REL, ITEM_BLOCK_V9, ABILITY_BLOCK_V9, NUM_ITEM_TAGS, NUM_ABILITY_TAGS,  # v9
)
from v_dance.encoders.live_state_encoder import LiveStateEncoder, opp_snapshot_from_log_prefix

# Replay-log message types poke-env's parse_message does not handle (they are
# consumed by the Player layer, not the battle) — safe to skip when driving a
# battle straight from a replay.
_SKIP = {"t:", "uhtmlchange", "win", "tie"}

_LOGGER = logging.getLogger("parity_harness")
_LOGGER.disabled = True

# ── Per-slot feature offsets (within one POKEMON_FEATURES block) ───────────────
OFF_HP      = 0
OFF_TYPE1   = 1
OFF_TYPE2   = 1 + 20
OFF_BASE    = 1 + 20 + 20            # 41  base stats (6)
OFF_EST     = OFF_BASE + 6           # 47  est stats (6)  ← belief-dependent
OFF_KNOWN   = OFF_EST + 6            # 53  stats_known    ← belief-dependent
OFF_MEGA    = OFF_KNOWN + 1          # 54
OFF_TERA    = OFF_MEGA + 1           # 55
OFF_STATUS  = OFF_TERA + 1           # 56  status one-hot (7)
OFF_BOOST   = OFF_STATUS + 7         # 63  boosts (7)
OFF_MOVES   = _MOVE_BLOCK_REL        # 71 (v9: moves start after the per-mon weight feature)
# v9 item/ability blocks (identity index + tags + known) sit right after the move block
OFF_ITEM        = OFF_MOVES + NUM_MOVES * MOVE_FEATURES   # 279  item identity index
OFF_ITEM_KNOWN  = OFF_ITEM + 1 + NUM_ITEM_TAGS            # item_known (after id + tags)
OFF_ABILITY     = OFF_ITEM + ITEM_BLOCK_V9               # 304  ability identity index
OFF_ABIL_KNOWN  = OFF_ABILITY + 1 + NUM_ABILITY_TAGS     # ability_known
OFF_ACTIVE  = POKEMON_FEATURES - 4   # 140
OFF_REV     = POKEMON_FEATURES - 3   # 141
OFF_FNT     = POKEMON_FEATURES - 2   # 142
OFF_TRANS   = POKEMON_FEATURES - 1   # 143

# pp_fraction position WITHIN one MOVE_FEATURES block (gap #3/#7):
#   0 base_power, 1 type, 2 category, 3 priority, 4 accuracy, 5 pp_fraction,
#   6 is_protect, 7 STAB, 8 is_spread (gap #6), 9 is_known
OFF_MOVE_PP = 5
OFF_MOVE_SPREAD = 8


OFF_MOVE_KNOWN = MOVE_FEATURES - 1   # is_known position within a move block


def move_pp_offsets(slot_idx: int) -> list:
    """Absolute vector indices of the pp_fraction feature for each of a slot's
    NUM_MOVES move blocks (gap #3 parity target)."""
    b = slot_base(slot_idx) + OFF_MOVES
    return [b + m * MOVE_FEATURES + OFF_MOVE_PP for m in range(NUM_MOVES)]


def move_known_offsets(slot_idx: int) -> list:
    """Absolute vector indices of the is_known feature for each move block — a
    move block is populated iff its is_known feature is non-zero (an empty
    move slot is all-zero on both paths)."""
    b = slot_base(slot_idx) + OFF_MOVES
    return [b + m * MOVE_FEATURES + OFF_MOVE_KNOWN for m in range(NUM_MOVES)]

# Slot index ranges in the full vector
ACTIVE_OWN  = (0, 1)
ACTIVE_OPP  = (2, 3)
OWN_BENCH   = tuple(range(ACTIVE_SLOTS, ACTIVE_SLOTS + BENCH_SLOTS))          # 4..7
OPP_BENCH   = tuple(range(ACTIVE_SLOTS + BENCH_SLOTS,
                          ACTIVE_SLOTS + 2 * BENCH_SLOTS))                    # 8..11
ALL_SLOTS   = (*ACTIVE_OWN, *ACTIVE_OPP, *OWN_BENCH, *OPP_BENCH)              # 0..11

# Team-count globals (gap #4): the LAST 4 features of the vector, each stored as
# min(count, 4) / 4.0 — multiply by 4 to recover the integer count.
G_OWN_LIVE = STATE_DIM - 4
G_OPP_LIVE = STATE_DIM - 3
G_OWN_FNT  = STATE_DIM - 2
G_OPP_FNT  = STATE_DIM - 1
TEAM_COUNT_GLOBALS = {
    "own_live": G_OWN_LIVE, "opp_live": G_OPP_LIVE,
    "own_fnt":  G_OWN_FNT,  "opp_fnt":  G_OPP_FNT,
}


def team_count(vec: np.ndarray, name: str) -> int:
    """Recover the integer team count for one of the 4 globals from its
    min(count,4)/4 encoding."""
    return round(float(vec[TEAM_COUNT_GLOBALS[name]]) * 4)


# Global side-condition blocks (gap #5 binary presence): absolute index ranges.
_GLOBAL_BASE = (ACTIVE_SLOTS + BENCH_SLOTS + OPP_BENCH_SLOTS) * POKEMON_FEATURES
OUR_SIDE_SC = (_GLOBAL_BASE + NUM_WEATHER + NUM_FIELDS,
               _GLOBAL_BASE + NUM_WEATHER + NUM_FIELDS + NUM_SIDE_CONDS)
OPP_SIDE_SC = (OUR_SIDE_SC[1], OUR_SIDE_SC[1] + NUM_SIDE_CONDS)


def slot_base(slot_idx: int) -> int:
    return slot_idx * POKEMON_FEATURES


def player_usernames(log: str) -> Dict[str, str]:
    """Map {"p1": username, "p2": username} from the replay's |player| lines."""
    names: Dict[str, str] = {}
    for ln in log.split("\n"):
        if ln.startswith("|player|"):
            parts = ln.split("|")
            if len(parts) >= 4 and parts[2] in ("p1", "p2") and parts[3]:
                names[parts[2]] = parts[3]
    return names


def drive_live_battle(log: str, username: str) -> DoubleBattle:
    """Replay the protocol log through a fresh DoubleBattle (no requests)."""
    battle = DoubleBattle("parity-tag", username, _LOGGER, gen=9)
    for ln in log.split("\n"):
        if not ln.startswith("|"):
            continue
        parts = ln.split("|")
        if len(parts) < 2 or parts[1] in _SKIP:
            continue
        try:
            battle.parse_message(parts)
        except Exception:        # replay-only artefacts poke-env can't parse
            pass
    return battle


def drive_live_battle_to_turn(log: str, username: str, turn: int) -> DoubleBattle:
    """Replay the protocol log into a fresh DoubleBattle, stopping right after
    the ``|turn|{turn}`` line (battle is at the START of that turn)."""
    battle = DoubleBattle("parity-tag", username, _LOGGER, gen=9)
    for ln in log.split("\n"):
        if not ln.startswith("|"):
            continue
        parts = ln.split("|")
        if len(parts) < 2 or parts[1] in _SKIP:
            continue
        try:
            battle.parse_message(parts)
        except Exception:
            pass
        if parts[1] == "turn" and int(parts[2]) == turn:
            break
    return battle


def live_vectors_per_turn(
    log: str, username: str, encoder=None, splice: bool = False
) -> Dict[int, np.ndarray]:
    """encode(battle) sampled at each |turn| boundary, keyed by turn number.

    ``splice=True`` exercises the gap-#6 opponent reconstruction: the opponent
    side is rebuilt from the public log prefix via vod_parser (the same path
    training uses) and spliced into the live vector — the production-honest live
    path.  ``splice=False`` (default) is the raw poke-env opponent view (kept so
    diagnostics can still measure poke-env's duplicate-species illusion merge).
    """
    encoder = encoder or LiveStateEncoder()
    battle = DoubleBattle("parity-tag", username, _LOGGER, gen=9)
    own_role = None
    if splice:
        for role, user in player_usernames(log).items():
            if user == username:
                own_role = role
                break
    out: Dict[int, np.ndarray] = {}
    for ln in log.split("\n"):
        if not ln.startswith("|"):
            continue
        parts = ln.split("|")
        if len(parts) < 2 or parts[1] in _SKIP:
            continue
        try:
            battle.parse_message(parts)
        except Exception:
            pass
        if parts[1] == "turn":
            turn = int(parts[2])
            opp_snap = None
            if splice and own_role is not None:
                opp_snap = opp_snapshot_from_log_prefix(log, own_role, turn)
            out[turn] = encoder.encode(battle, opp_snapshot=opp_snap).copy()
    return out


def offline_vectors_per_turn(
    log: str, perspective: str, encoder=None
) -> Dict[int, np.ndarray]:
    """encode_snapshot(state_before_actions[perspective]) per turn.

    Uses the RAW parser output (no belief enrichment / own-side retrofit) so the
    comparison isolates the encoder's mechanical handling of public state.
    """
    encoder = encoder or VodStateEncoder()
    parser = ShowdownReplayParser(log, our_player=perspective)
    res = parser.parse()
    out: Dict[int, np.ndarray] = {}
    for turn in res["turns"]:
        snap = turn["state_before_actions"][perspective]
        out[turn["turn"]] = encoder.encode_snapshot(snap, turn=turn["turn"])
    return out


# ── Block extraction helpers ──────────────────────────────────────────────────
def structural_slot(vec: np.ndarray, slot_idx: int) -> np.ndarray:
    """The species-identity + slot-flag features of one mon slot:
    hp, both type one-hots, base stats, and the 4 trailing flags
    (is_active / is_revealed / is_fainted / is_transformed).

    Excludes est_stats + stats_known (belief-dependent) and the move block
    (reveal-order dependent) so the comparison isolates *which mon is in the
    slot* and its hard flags — the surface gap #1/#2/#4 fixes target.
    """
    b = slot_base(slot_idx)
    return np.concatenate([
        vec[b + OFF_HP:    b + OFF_BASE + 6],   # hp + types + base stats
        vec[b + OFF_ACTIVE:b + OFF_TRANS + 1],  # the 4 slot flags
    ])


def identity_slot(vec: np.ndarray, slot_idx: int) -> np.ndarray:
    """Type one-hots + base stats only — the deterministic species fingerprint
    (no HP, so robust to %/real HP-scale differences between paths)."""
    b = slot_base(slot_idx)
    return vec[b + OFF_TYPE1:b + OFF_BASE + 6]


def itemabil_block(vec: np.ndarray, slot_idx: int) -> np.ndarray:
    """The item+ability effect block (item effects + item_known + ability effects
    + ability_known) of one mon slot — the gap-#5 features, isolated for parity
    diffs (structural_slot/diff_blocks deliberately skip this region)."""
    b = slot_base(slot_idx) + OFF_ITEM
    return vec[b:b + ITEM_BLOCK_V9 + ABILITY_BLOCK_V9]


def slot_is_empty(vec: np.ndarray, slot_idx: int) -> bool:
    b = slot_base(slot_idx)
    return bool(np.all(vec[b:b + POKEMON_FEATURES] == 0.0))


def diff_blocks(
    live: np.ndarray, offline: np.ndarray, slots: Tuple[int, ...]
) -> Dict[int, Dict[str, np.ndarray]]:
    """Return {slot_idx: {'live':..., 'offline':...}} for slots whose
    structural features differ (atol 1e-4 on identity, exact on flags)."""
    bad: Dict[int, Dict[str, np.ndarray]] = {}
    for s in slots:
        ls, os_ = structural_slot(live, s), structural_slot(offline, s)
        if not np.allclose(ls, os_, atol=1e-4):
            bad[s] = {"live": ls, "offline": os_}
    return bad
