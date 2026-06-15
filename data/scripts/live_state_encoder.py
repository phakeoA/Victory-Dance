"""
live_state_encoder.py
=====================
LIVE encoder: a poke-env DoubleBattle → (STATE_DIM,) float32 vector, plus
own-side resolution (.env username + team-file species → which side is ours).

The FROZEN tensor layout and every shared constant live in state_encoder.py
(the VOD-parsing module); this module imports them, so the LIVE vector and the
OFFLINE vector are guaranteed to use byte-identical feature indices.  poke-env
is required for this module's encode(battle); the offline path never imports it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Bootstrap: sibling imports work when run/imported from anywhere.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from vod_parser.pokedex import norm_species
from vod_parser.team_sheet import parse_showdown_team, base_species as _paste_base_species
from belief_state import BeliefState, STAT_ORDER

# The FROZEN layout is owned by the VOD module — import it so live == offline.
from state_encoder import (
    _TYPE_IDX, _STATUS_IDX, _WEATHER_IDX, _FIELD_IDX, _SC_IDX,
    NUM_TYPES, NUM_STATUS, NUM_WEATHER, NUM_FIELDS, NUM_SIDE_CONDS,
    NUM_BOOSTS, NUM_MOVES, MOVE_FEATURES,
    POKEMON_FEATURES, STATE_DIM, ACTION_DIM,
    ACTIVE_SLOTS, BENCH_SLOTS, OPP_BENCH_SLOTS,
    _BOOST_KEYS, _EST_STAT_NORM,
    VodStateEncoder,
)

# ── Opponent byte ranges in the frozen layout (slots 0,1=own active; 2,3=opp
# active; 4-7=own bench; 8-11=opp bench).  Used by the gap-#6 opponent splice
# (encode(..., opp_snapshot=...)): the live bot reconstructs the OPPONENT side
# from the public protocol log with the SAME vod_parser the training data uses,
# so train==serve for the opp side even when poke-env's species-keyed opponent
# model merges a duplicate-species illusion (Zoroark disguised as a teammate).
_OPP_ACTIVE_LO = (ACTIVE_SLOTS - 2) * POKEMON_FEATURES        # slot 2
_OPP_ACTIVE_HI = ACTIVE_SLOTS * POKEMON_FEATURES              # end of slot 3
_OPP_BENCH_LO  = (ACTIVE_SLOTS + BENCH_SLOTS) * POKEMON_FEATURES          # slot 8
_OPP_BENCH_HI  = (ACTIVE_SLOTS + BENCH_SLOTS + OPP_BENCH_SLOTS) * POKEMON_FEATURES  # end slot 11
_G_OPP_LIVE = STATE_DIM - 3
_G_OPP_FNT  = STATE_DIM - 1


# ── poke-env is optional (offline training machines don't need it) ────────────
try:  # poke-env ≥ 0.8 layout
    from poke_env.battle import (
        DoubleBattle, Pokemon, Move, PokemonType, Status, Weather, Field,
        SideCondition, MoveCategory,
    )
    _HAS_POKE_ENV = True
except ImportError:
    try:  # older poke-env layout
        from poke_env.environment import (
            DoubleBattle, Pokemon, Move, PokemonType, Status, Weather, Field,
            SideCondition, MoveCategory,
        )
        _HAS_POKE_ENV = True
    except ImportError:
        DoubleBattle = Pokemon = Move = None  # type: ignore
        PokemonType = Status = Weather = Field = SideCondition = MoveCategory = None  # type: ignore
        _HAS_POKE_ENV = False

# poke-env's pokedex + id normaliser (for the gap-#5 mega-forme detector).
try:
    from poke_env.data import GenData
    from poke_env.data.normalize import to_id_str
except ImportError:  # pragma: no cover
    GenData = None  # type: ignore
    to_id_str = None  # type: ignore

# Mega-forme dex-id suffixes (Charizard/Mewtwo use X/Y; everything else plain).
_MEGA_SUFFIXES = ("mega", "megax", "megay")


# poke-env member names that differ from the frozen canonical names
_LIVE_NAME_ALIASES = {"SNOWSCAPE": "SNOW", "MATBLOCK": "MAT_BLOCK"}


def _enum_to_idx(enum_cls, idx_map: dict[str, int]) -> dict:
    """Map live poke-env enum members onto frozen indices by member name."""
    out = {}
    for member in enum_cls:
        name = _LIVE_NAME_ALIASES.get(member.name, member.name)
        i = idx_map.get(name)
        if i is not None:
            out[member] = i
    return out


# ══════════════════════════════════════════════════════════════════════════════
class LiveStateEncoder:
    """
    Live encoder: encodes a poke-env DoubleBattle into a (STATE_DIM,)
    float32 vector identical in layout to VodStateEncoder.encode_snapshot.

        enc = LiveStateEncoder(belief=BeliefState("pikalytics_regma.json"),
                               own_username="...", own_team=team_paste)
        vec = enc.encode(battle)                  # shape (STATE_DIM,)
    """

    def __init__(
        self,
        belief: Optional[BeliefState] = None,
        level: int = 50,
        own_username: Optional[str] = None,
        own_team=None,
    ):
        self.state_dim  = STATE_DIM
        self.action_dim = ACTION_DIM
        self.belief     = belief   # opponent stat estimates on the live path
        self.level      = level

        # Offline encoder used for the gap-#6 opponent splice: when encode() is
        # given an opp_snapshot (the opponent side reconstructed from the public
        # protocol log via vod_parser, the SAME path training uses), the opp byte
        # ranges are produced by this encoder so live == train for the opponent
        # even when poke-env merges a duplicate-species illusion.
        self._vod_enc = VodStateEncoder(belief=belief, level=level)

        # Session identity for live own-side resolution (see resolve_own_role).
        # own_team may be a Showdown team paste (str) or a species list; a
        # missing username OR team means "no own side" (Type B) → resolves None.
        self.own_username = own_username
        if isinstance(own_team, str):
            self.own_team_species: Optional[List[str]] = team_species_from_paste(own_team)
        elif own_team:
            self.own_team_species = list(own_team)
        else:
            self.own_team_species = None

        # Live-path enum→index lookups (poke-env members mapped by name onto
        # the frozen orderings; only built when poke-env is importable)
        if _HAS_POKE_ENV:
            self._type_idx    = _enum_to_idx(PokemonType, _TYPE_IDX)
            self._status_idx  = _enum_to_idx(Status, _STATUS_IDX)
            self._weather_idx = _enum_to_idx(Weather, _WEATHER_IDX)
            self._field_idx   = _enum_to_idx(Field, _FIELD_IDX)
            self._sc_idx      = _enum_to_idx(SideCondition, _SC_IDX)
            # Gen-9 pokedex for the mega-forme detector (see _is_mega_forme).
            self._pokedex = GenData.from_gen(9).pokedex if GenData else {}

    # ── Own-side resolution from a live battle ──────────────────────────────────
    def resolve_own_role_from_battle(self, battle: "DoubleBattle") -> Optional[str]:
        """Resolve which role is the bot's own side for a live poke-env battle,
        using the encoder's configured ``own_username`` + ``own_team_species``.

        Reads player usernames from ``battle._players`` and per-role teampreview
        rosters from poke-env's role-relative views.  When actively playing this
        CONFIRMS poke-env's ``player_role``; for an observed/spectated battle the
        future driver should feed per-role rosters (e.g. via ``own_role_from_log``
        on the raw protocol) and set ``battle.player_role`` before encoding —
        ``encode()`` itself still reads own/opp through poke-env's player_role.

        Returns None when no own side can be determined (no team configured →
        Type B, no match, or ambiguous).
        """
        usernames: Dict[str, str] = {}
        for p in getattr(battle, "_players", None) or []:
            role, user = p.get("player"), p.get("username")
            if role in ("p1", "p2") and user:
                usernames[role] = user

        rosters: Dict[str, List[str]] = {}
        pr = getattr(battle, "player_role", None)
        opr = getattr(battle, "opponent_role", None)
        if pr:
            own = list(getattr(battle, "teampreview_team", None) or []) \
                or list(getattr(battle, "team", {}).values())
            rosters[pr] = [m.species for m in own]
        if opr:
            opp = list(getattr(battle, "teampreview_opponent_team", None) or []) \
                or list(getattr(battle, "opponent_team", {}).values())
            rosters[opr] = [m.species for m in opp]

        return resolve_own_role(
            usernames, rosters, self.own_username, self.own_team_species
        )

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE PATH — poke-env DoubleBattle
    # ══════════════════════════════════════════════════════════════════════════
    def encode(
        self, battle: "DoubleBattle", opp_snapshot: Optional[dict] = None
    ) -> np.ndarray:
        """
        Return a float32 vector of shape (STATE_DIM,) for the given battle.
        Safe to call at any point during a battle, including teampreview.

        ``opp_snapshot`` (gap #6): the OPPONENT side of a vod_parser perspective
        snapshot (``opp_active`` / ``opp_bench`` keys), reconstructed from the
        public protocol log with the SAME parser the training data uses — see
        ``opp_snapshot_from_log_prefix``.  When provided, the opponent byte
        ranges (active slots 2,3 + bench 8-11 + opp_live/opp_fnt globals) are
        re-derived from it via the offline encoder and spliced in, so the live
        opponent view matches training even when poke-env's species-keyed model
        merges a duplicate-species illusion (Zoroark disguised as a teammate).
        The OWN side + globals stay poke-env-derived (the own side carries
        private |request| data a replay/log-reconstruction lacks).  When omitted,
        encode() degrades to poke-env's opponent view (the prior behaviour).
        """
        if not _HAS_POKE_ENV:
            raise RuntimeError(
                "poke-env is not installed — the live encode(battle) path is "
                "unavailable. Use encode_snapshot()/encode_transition() for "
                "parsed VOD JSON instead."
            )
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        cursor = 0

        # ── [A] Active slots ─────────────────────────────────────────────────
        # active_pokemon / opponent_active_pokemon are List[Optional[Pokemon]]
        # of length 2; None means that slot is empty (fainted / not yet sent).
        try:
            own_active = battle.active_pokemon          # List[Optional[Pokemon]]
        except ValueError:
            own_active = [None, None]
        try:
            opp_active = battle.opponent_active_pokemon
        except ValueError:
            opp_active = [None, None]

        own_active_set = {p for p in own_active if p is not None}

        for mon in list(own_active):
            self._write_pokemon(vec, cursor, mon, is_active=True, is_own=True)
            cursor += POKEMON_FEATURES
        for mon in list(opp_active):
            self._write_pokemon(vec, cursor, mon, is_active=True, is_own=False)
            cursor += POKEMON_FEATURES

        # ── [B] Bench slots ──────────────────────────────────────────────────
        # _is_real_mon drops broken-illusion phantoms (gap #4): when an OWN
        # Zoroark illusion breaks, poke-env leaves the disguise species in
        # battle.team revealed=True but with max_hp==0 — it would otherwise fill
        # a bench slot the offline parser leaves empty (it resolves the illusion
        # cleanly). This mirrors the opp side's _is_seen phantom filter.
        bench = [
            p for p in battle.team.values()
            if p not in own_active_set and not p.fainted and self._is_real_mon(p)
        ][:BENCH_SLOTS]

        for i in range(BENCH_SLOTS):
            mon = bench[i] if i < len(bench) else None
            self._write_pokemon(vec, cursor, mon, is_active=False, is_own=True)
            cursor += POKEMON_FEATURES

        # ── [B2] Opponent bench (layout-v2): the opponent's FULL teampreview
        # roster minus its active mons, ordered seen-alive → seen-fainted →
        # unseen stubs — matching encode_snapshot's offline ordering exactly.
        # The old implementation used only battle.opponent_team (REVEALED mons),
        # so early-game these 4 slots sat empty while the offline path filled
        # them with unseen teampreview stubs: a large train/serve gap.
        # _opp_bench_mons rebuilds the same roster view from
        # battle.teampreview_opponent_team (the public |poke| roster).
        opp_active_set = {p for p in opp_active if p is not None}
        opp_bench = self._opp_bench_mons(battle, opp_active_set)
        for i in range(OPP_BENCH_SLOTS):
            mon = opp_bench[i] if i < len(opp_bench) else None
            self._write_pokemon(vec, cursor, mon, is_active=False, is_own=False)
            cursor += POKEMON_FEATURES

        # ── [C] Global features ──────────────────────────────────────────────

        # Weather: Dict[Weather, int] — multi-hot (normally one at a time)
        for w in battle.weather:
            if w in self._weather_idx:
                vec[cursor + self._weather_idx[w]] = 1.0
        cursor += NUM_WEATHER

        # Fields: Dict[Field, int]
        trick_room_active = False
        for f in battle.fields:
            if f.name == "TRICK_ROOM":
                trick_room_active = True
            if f in self._field_idx:
                vec[cursor + self._field_idx[f]] = 1.0
        cursor += NUM_FIELDS

        # Side conditions — BINARY presence (1.0 = active), gap #5.  poke-env's
        # value is a layer count (stackable) OR the turn a condition started
        # (non-stackable) — NOT turns remaining — so its magnitude is not
        # comparable to the offline path.  Both paths therefore encode a presence
        # bit.  (NB: the offline parser currently tracks only tailwind + screens,
        # so entry-hazard slots still diverge — a separate parser-coverage gap.)
        for sc, val in battle.side_conditions.items():
            if sc in self._sc_idx and val:
                vec[cursor + self._sc_idx[sc]] = 1.0
        cursor += NUM_SIDE_CONDS

        # Opponent side conditions
        for sc, val in battle.opponent_side_conditions.items():
            if sc in self._sc_idx and val:
                vec[cursor + self._sc_idx[sc]] = 1.0
        cursor += NUM_SIDE_CONDS

        # Turn (cap at 60 for normalisation)
        vec[cursor] = min(battle.turn, 60) / 60.0
        cursor += 1

        # Trick Room explicit flag (already in fields but surfaced separately
        # because it flips speed priority — critical strategic signal)
        vec[cursor] = 1.0 if trick_room_active else 0.0
        cursor += 1

        # ── Team counts (layout-v2): living-bench + fainted per side ──────────
        # Opp counts use SEEN (revealed, non-active) mons only — the information
        # asymmetry the offline path encodes (opp_seen_alive / opp_seen_fainted).
        # Own counts iterate REAL team members only: _is_real_mon drops the
        # broken-illusion phantom (max_hp==0) that would otherwise inflate
        # own_live by one after an own Zoroark's disguise drops (gap #4).
        own_team = [p for p in battle.team.values() if self._is_real_mon(p)]
        own_live = sum(1 for p in own_team if p not in own_active_set and not p.fainted)
        own_fnt  = sum(1 for p in own_team if p.fainted)
        opp_active_bases = {self._base_species(p) for p in opp_active_set}
        opp_seen = [
            p for p in getattr(battle, "opponent_team", {}).values()
            if self._is_seen(p)
            and self._base_species(p) not in opp_active_bases
        ]
        opp_live = sum(1 for p in opp_seen if not p.fainted)
        opp_fnt  = sum(1 for p in opp_seen if p.fainted)
        vec[cursor] = min(own_live, 4) / 4.0; cursor += 1
        vec[cursor] = min(opp_live, 4) / 4.0; cursor += 1
        vec[cursor] = min(own_fnt, 4) / 4.0;  cursor += 1
        vec[cursor] = min(opp_fnt, 4) / 4.0;  cursor += 1

        assert cursor == STATE_DIM, (
            f"LiveStateEncoder cursor mismatch: wrote {cursor}, expected {STATE_DIM}"
        )

        # ── [D] Gap #6 opponent splice ───────────────────────────────────────
        # Replace the poke-env-derived opponent bytes with the vod_parser
        # reconstruction (same code as training), so the live opp view is
        # immune to poke-env's duplicate-species illusion merge.
        if opp_snapshot is not None:
            self._splice_opponent(vec, opp_snapshot, turn=getattr(battle, "turn", 0))

        return vec

    def _splice_opponent(self, vec: np.ndarray, opp_snapshot: dict, turn: int) -> None:
        """Overwrite the opponent byte ranges of ``vec`` with the offline
        encoding of ``opp_snapshot`` (gap #6).  Only the opponent-owned features
        are touched: active slots 2,3, bench slots 8-11, and the opp_live /
        opp_fnt team-count globals.  The own side, globals, weather/field and
        opp side-conditions are left as the poke-env-derived live values."""
        off = self._vod_enc.encode_snapshot(opp_snapshot, turn=turn)
        vec[_OPP_ACTIVE_LO:_OPP_ACTIVE_HI] = off[_OPP_ACTIVE_LO:_OPP_ACTIVE_HI]
        vec[_OPP_BENCH_LO:_OPP_BENCH_HI]   = off[_OPP_BENCH_LO:_OPP_BENCH_HI]
        vec[_G_OPP_LIVE] = off[_G_OPP_LIVE]
        vec[_G_OPP_FNT]  = off[_G_OPP_FNT]

    # ── Opponent roster reconstruction (live) ──────────────────────────────────
    @staticmethod
    def _base_species(mon: "Pokemon") -> str:
        """Base-forme species id for matching across formes / teampreview stubs
        (e.g. Charizard-Mega-Y → 'charizard', Zoroark-Hisui → 'zoroark')."""
        return (getattr(mon, "base_species", None) or getattr(mon, "species", "") or "")

    def _is_mega_forme(self, mon: "Pokemon") -> bool:
        """Whether the mon is CURRENTLY mega-evolved (gap #5).

        poke-env keeps ``mon.species`` at the base forme even after a mega, but it
        DOES rewrite the mon's ``base_stats`` from the mega pokedex entry on
        ``|detailschange|``/``|-mega|``.  So a mon is mega iff its current base
        stats equal those of a ``<base>mega``/``megax``/``megay`` dex entry — a
        precise, deterministic test that catches real megas (Charizard-Mega-Y,
        Aerodactyl-Mega, …) yet never trips on a non-mega whose NAME contains the
        substring 'mega' (Meganium, Yanmega).  Mirrors the offline is_mega flag."""
        dex = getattr(self, "_pokedex", None)
        if not dex or to_id_str is None:
            return False
        # A transformed mon (Ditto/Imposter) copies its target's forme — incl. a
        # mega target's stats — but is NOT itself mega-evolved.  The offline
        # parser keeps is_mega=False for it, so skip the stat match here to match
        # (the copied stats/types are still encoded via base_stats/types).
        if getattr(mon, "transformed", False):
            return False
        base = to_id_str(self._base_species(mon))
        if not base:
            return False
        cur = mon.base_stats
        if not cur:
            return False
        for suf in _MEGA_SUFFIXES:
            entry = dex.get(base + suf)
            stats = entry.get("baseStats") if entry else None
            if stats and dict(stats) == dict(cur):
                return True
        return False

    @staticmethod
    def _is_real_mon(mon: "Pokemon") -> bool:
        """Whether a mon in our OWN team is a genuine member (not a phantom).

        The own-side analogue of ``_is_seen`` for the team-count globals and the
        own-bench slots (gap #4).  When an OWN Zoroark's illusion breaks, poke-env
        leaves the disguise species in ``battle.team`` flagged revealed=True but
        with ``max_hp == 0`` — a phantom that was never a real mon.  A genuine own
        mon always has ``max_hp > 0``, whether it has already been seen on the
        field OR is brought-but-unentered (its real stats/HP arrive via the
        private ``|request|``).  Unlike ``_is_seen`` we therefore do NOT require
        ``revealed``, so brought-but-unentered mons still count as real."""
        return bool(getattr(mon, "max_hp", 0))

    @staticmethod
    def _is_seen(mon: "Pokemon") -> bool:
        """Whether an opponent mon has been genuinely observed on the field.

        ``revealed`` alone is not enough: when a Zoroark illusion breaks,
        poke-env keeps the DISGUISE mon (the apparent species) in opponent_team
        flagged revealed=True but with no HP (max_hp 0) — a phantom that was
        never really seen.  The offline parser relabels the disguise to the true
        Zoroark and keeps the apparent species UNSEEN, so we mirror that by
        requiring a real max_hp here (every genuinely-seen mon, fainted ones
        included, has its max_hp set from a switch-in)."""
        return bool(getattr(mon, "revealed", False)) and bool(getattr(mon, "max_hp", 0))

    def _opp_bench_mons(
        self, battle: "DoubleBattle", opp_active_set: set
    ) -> list:
        """The opponent's non-active roster as the offline path orders it:
        seen-alive → seen-fainted → unseen teampreview stubs (roster order
        within each group), capped at OPP_BENCH_SLOTS.

        Mirrors encode_snapshot's [B2] logic: the roster is the full public
        teampreview list (battle.teampreview_opponent_team); a roster slot is
        "seen" once a same-base-species mon has been revealed
        (battle.opponent_team), otherwise it is an unseen stub.  Active mons are
        excluded by base species.
        """
        roster = list(getattr(battle, "teampreview_opponent_team", None) or [])
        opp_active_bases = {self._base_species(p) for p in opp_active_set}

        # Seen (revealed + real HP), non-active opp mons keyed by base species.
        # _is_seen drops broken-illusion phantoms so the disguise's apparent
        # species stays an unseen roster stub (matching the offline parser).
        revealed: dict = {}
        for p in getattr(battle, "opponent_team", {}).values():
            if not self._is_seen(p):
                continue
            b = self._base_species(p)
            if b not in opp_active_bases:
                revealed.setdefault(b, p)

        if not roster:
            # No teampreview info (shouldn't happen in a real battle) — fall
            # back to the revealed-only view: alive switch-ins first.
            alive = [p for p in revealed.values() if not p.fainted]
            dead  = [p for p in revealed.values() if p.fainted]
            return (alive + dead)[:OPP_BENCH_SLOTS]

        seen_alive: list = []
        seen_fainted: list = []
        unseen: list = []
        used: set = set()
        for stub in roster:
            b = self._base_species(stub)
            if b in opp_active_bases or b in used:
                continue
            used.add(b)
            mon = revealed.get(b)
            if mon is not None:
                (seen_fainted if mon.fainted else seen_alive).append(mon)
            else:
                unseen.append(stub)
        return (seen_alive + seen_fainted + unseen)[:OPP_BENCH_SLOTS]

    # ── Pokémon encoder (live) ────────────────────────────────────────────────
    def _write_pokemon(
        self,
        vec: np.ndarray,
        start: int,
        mon: Optional["Pokemon"],
        is_active: bool,
        is_own: bool,
    ) -> None:
        """Write POKEMON_FEATURES floats into vec starting at `start`."""
        # Empty / unrevealed slot → all zeros (already zeroed by np.zeros)
        if mon is None:
            return

        i = start

        # hp_frac — current_hp_fraction returns 0 if current_hp is falsy.
        # Unrevealed roster stubs (opp teampreview mons not yet sent out) carry
        # no HP yet; the offline path encodes such unseen mons at full HP (1.0),
        # so mirror that placeholder rather than emitting 0 for a healthy mon
        # we simply haven't seen.
        vec[i] = mon.current_hp_fraction if mon.revealed else 1.0
        i += 1

        # type1 one-hot
        if mon.type_1 in self._type_idx:
            vec[i + self._type_idx[mon.type_1]] = 1.0
        i += NUM_TYPES

        # type2 one-hot (None if single-type)
        if mon.type_2 is not None and mon.type_2 in self._type_idx:
            vec[i + self._type_idx[mon.type_2]] = 1.0
        i += NUM_TYPES

        # Base stats: hp atk def spa spd spe (normalised by 255)
        base_stats = mon.base_stats
        for key in STAT_ORDER:
            vec[i] = base_stats.get(key, 0) / 255.0
            i += 1

        # Est in-battle stats + confidence flag (belief integration):
        #   own mon  → exact stats from the request (stats_known = 1.0)
        #   opp mon  → Pikalytics-weighted expectation   (stats_known = 0.5)
        est, known = self._live_est_stats(mon, is_own)
        for key in STAT_ORDER:
            vec[i] = (est.get(key) or 0) / _EST_STAT_NORM if est else 0.0
            i += 1
        vec[i] = known
        i += 1

        # Mega (gap #5): the old `"mega" in mon.species` test was doubly broken —
        # it false-POSITIVED on species whose name merely contains the substring
        # (Meganium, Yanmega) and false-NEGATIVED real megas, because poke-env's
        # mega_evolve() keeps mon.species at the BASE forme (store_species=False).
        # _is_mega_forme instead checks whether the mon's CURRENT base stats match
        # a mega forme of its base species (poke-env DOES update base_stats on
        # |detailschange|/|-mega|), matching the offline parser's is_mega flag.
        vec[i] = 1.0 if self._is_mega_forme(mon) else 0.0
        i += 1

        # Terastallized
        vec[i] = 1.0 if mon.is_terastallized else 0.0
        i += 1

        # Status one-hot (None = healthy)
        if mon.status is not None and mon.status in self._status_idx:
            vec[i + self._status_idx[mon.status]] = 1.0
        i += NUM_STATUS

        # Boosts (Dict[str, int], values in [-6, +6])
        boosts = mon.boosts
        for j, key in enumerate(_BOOST_KEYS):
            vec[i + j] = boosts.get(key, 0) / 6.0
        i += NUM_BOOSTS

        # Moves (up to 4; remaining slots stay zero = unknown)
        move_list = list(mon.moves.values())[:NUM_MOVES]
        for m_idx in range(NUM_MOVES):
            if m_idx < len(move_list):
                self._write_move(vec, i, move_list[m_idx], mon)
            i += MOVE_FEATURES

        # is_active slot flag
        vec[i] = 1.0 if is_active else 0.0
        i += 1

        # is_revealed (own mons always 1; opp mons 1 once sent out / identified)
        vec[i] = 1.0 if mon.revealed else 0.0
        i += 1

        # is_fainted (layout-v2): explicit KO flag for the opp-bench / counting
        vec[i] = 1.0 if mon.fainted else 0.0
        i += 1

        # is_transformed (layout-v2). poke-env exposes this as `mon.transformed`
        # (True while _transform_moves is set); the previous `is_transformed`
        # attribute does not exist, so this flag was ALWAYS 0 on the live path.
        # The copied forme's typing + base stats are already encoded above for
        # free: when transformed, mon.type_1/type_2 return _temporary_types and
        # mon.base_stats returns _temporary_base_stats (the COPY's), matching
        # encode_snapshot's dex-of-transformed_into Solution A.  (Copied MOVES
        # are a separate, intentionally-untracked nuance — the parser gates them
        # out, so move features still diverge on a live transform.)
        vec[i] = 1.0 if getattr(mon, "transformed", False) else 0.0
        i += 1

    def _live_est_stats(
        self, mon: "Pokemon", is_own: bool
    ) -> tuple[Optional[dict], float]:
        """(stats dict | None, stats_known flag) for the live path."""
        if is_own:
            stats = getattr(mon, "stats", None) or {}
            if all(stats.get(k) for k in ("atk", "def", "spa", "spd", "spe")):
                est = dict(stats)
                if not est.get("hp"):
                    est["hp"] = mon.max_hp
                return est, 1.0
        if self.belief is not None:
            est = self.belief.expected_stats_weighted(
                mon.species, mon.base_stats, level=self.level
            )
            if est:
                return est, 0.5
        return None, 0.0

    # ── Move encoder (live) ───────────────────────────────────────────────────
    def _write_move(
        self,
        vec: np.ndarray,
        start: int,
        move: "Move",
        user: "Pokemon",
    ) -> None:
        """Write MOVE_FEATURES floats into vec starting at `start`."""
        i = start

        # Base power (cap at 250 since a few moves are absurdly high)
        vec[i] = min(move.base_power, 250) / 150.0
        i += 1

        # Move type as ordinal index (compact; type embeddings learned by net)
        if move.type in self._type_idx:
            vec[i] = self._type_idx[move.type] / (NUM_TYPES - 1)
        i += 1

        # Category: 0.0=physical, 0.5=special, 1.0=status
        cat = move.category
        if cat == MoveCategory.PHYSICAL:
            vec[i] = 0.0
        elif cat == MoveCategory.SPECIAL:
            vec[i] = 0.5
        else:
            vec[i] = 1.0
        i += 1

        # Priority (range roughly -7 to +5 in practice)
        vec[i] = move.priority / 7.0
        i += 1

        # Accuracy (already 0–1 float; always-hit moves return 1.0)
        vec[i] = move.accuracy
        i += 1

        # PP fraction — pinned to 1.0 to MATCH the offline path (gap #3).
        # encode_snapshot's _write_move_json always emits 1.0 because the VOD
        # parser does not track PP yet. Emitting the real live PP here
        # (move.current_pp / move.max_pp) would be a train/serve mismatch: the
        # net only ever saw PP==1.0 in training, so a depleted-PP value at serve
        # time is an out-of-distribution signal on a feature it learned to
        # ignore. Keep both paths constant until the parser learns to count
        # move uses per mon (the TODO in state_encoder._write_move_json).
        vec[i] = 1.0
        i += 1

        # Protect-family move flag (high strategic value in doubles)
        vec[i] = 1.0 if move.is_protect_move else 0.0
        i += 1

        # STAB: move type in user's current types (accounts for tera)
        vec[i] = 1.0 if move.type in user.types else 0.0
        i += 1

        # Known flag (always 1 here — zero-slot means unknown)
        vec[i] = 1.0
        i += 1


# ══════════════════════════════════════════════════════════════════════════════
# Own-side resolution (live) — "which side is the bot's own side?"
#
# The bot is configured (its .env) with a Showdown username and a team file for
# the session.  A side is the bot's OWN side iff BOTH hold:
#   • that player's username matches the configured username, AND
#   • every species in the configured team is present in that player's
#     teampreview roster (compared as BASE formes, so a mega matches its base).
# This is the live analogue of the offline team_sheet.detect_our_side, but adds
# the username gate.  It also encodes nuance #1 directly: with NO team (or no
# username) configured there is no own side (a Type B view), so it returns None
# and the caller should treat BOTH sides as opponent/distribution.
# ══════════════════════════════════════════════════════════════════════════════

_USERNAME_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _norm_username(name: Optional[str]) -> str:
    """Showdown-style username id: lowercase, alphanumerics only
    ('Victorious Dancing' → 'victoriousdancing')."""
    return _USERNAME_NON_ALNUM.sub("", (name or "").lower())


def _norm_dex(species: Optional[str]) -> str:
    """Normalised BASE-forme species id for roster matching (mega → base)."""
    return norm_species(_paste_base_species(species or "")) if species else ""


def team_species_from_paste(text: str) -> List[str]:
    """Species (base formes) parsed from a Showdown team paste."""
    return [m["species"] for m in parse_showdown_team(text or "") if m.get("species")]


def resolve_own_role(
    player_usernames: Dict[str, Optional[str]],
    teampreview_species: Dict[str, List[str]],
    own_username: Optional[str],
    own_team_species: Optional[List[str]],
) -> Optional[str]:
    """Return the bot's own role ("p1"/"p2") or None.

    Requires BOTH a username match and a full team-roster match.  Returns None
    when nothing matches, when no username/team is configured (Type B), or when
    the match is ambiguous (e.g. mirror self-play with identical username AND
    team on both sides — which side is "ours" is then undecidable here).
    """
    want_user = _norm_username(own_username)
    want_team = {_norm_dex(s) for s in (own_team_species or []) if s}
    want_team.discard("")
    if not want_user or not want_team:
        return None
    matches: List[str] = []
    for role in ("p1", "p2"):
        if _norm_username(player_usernames.get(role)) != want_user:
            continue
        have = {_norm_dex(s) for s in (teampreview_species.get(role) or [])}
        if want_team <= have:
            matches.append(role)
    return matches[0] if len(matches) == 1 else None


def parse_log_players_and_rosters(log: str):
    """(usernames, rosters) keyed by role from a Showdown protocol log's
    |player| and |poke| lines.  Robust for both playing and spectating (it does
    not depend on poke-env's player_role-relative split)."""
    usernames: Dict[str, str] = {}
    rosters: Dict[str, List[str]] = {"p1": [], "p2": []}
    for ln in (log or "").split("\n"):
        if ln.startswith("|player|"):
            parts = ln.split("|")
            if len(parts) >= 4 and parts[2] in ("p1", "p2") and parts[3]:
                usernames[parts[2]] = parts[3]
        elif ln.startswith("|poke|"):
            parts = ln.split("|")
            if len(parts) >= 4 and parts[2] in ("p1", "p2"):
                sp = parts[3].split(",")[0].strip()
                if sp:
                    rosters[parts[2]].append(sp)
    return usernames, rosters


def own_role_from_log(
    log: str, own_username: Optional[str], own_team_species: Optional[List[str]]
) -> Optional[str]:
    """Resolve the bot's own role straight from a raw protocol log (the data a
    spectating/observing driver has): parses |player| + |poke| then applies
    resolve_own_role."""
    usernames, rosters = parse_log_players_and_rosters(log)
    return resolve_own_role(usernames, rosters, own_username, own_team_species)


# ══════════════════════════════════════════════════════════════════════════════
# Gap #6: real-time opponent reconstruction for the live splice
# ══════════════════════════════════════════════════════════════════════════════
def opp_snapshot_from_log_prefix(
    log: str, own_role: str, turn: int
) -> Optional[dict]:
    """Reconstruct the OPPONENT side as of the START of ``turn`` from the public
    protocol log, using the SAME ``vod_parser`` the training data uses.

    REAL-TIME / production-honest: the log is truncated to the prefix up to and
    including ``|turn|{turn}`` BEFORE parsing, so the illusion resolver only sees
    information a live bot would have at that moment (no future |replace| can
    leak backwards into the snapshot).  Returns the perspective snapshot for
    ``own_role`` (whose ``opp_active`` / ``opp_bench`` describe the opponent), or
    None if the prefix has no such turn yet.

    Pass the result to ``LiveStateEncoder.encode(battle, opp_snapshot=...)`` to
    make the live opponent view immune to poke-env's duplicate-species illusion
    merge (gap #6 sub-cause 2).
    """
    from vod_parser.replay_parser import ShowdownReplayParser

    marker = f"|turn|{turn}"
    prefix: List[str] = []
    seen = False
    for ln in (log or "").split("\n"):
        prefix.append(ln)
        if ln.strip() == marker:
            seen = True
            break
    if not seen:
        return None

    parser = ShowdownReplayParser("\n".join(prefix), our_player=own_role)
    parser.parse()
    # _state_before is the start-of-current-turn snapshot captured when the
    # |turn| line was handled; the prefix ends exactly there.
    state_before = getattr(parser, "_state_before", None)
    if not state_before:
        return None
    return state_before.get(own_role)
