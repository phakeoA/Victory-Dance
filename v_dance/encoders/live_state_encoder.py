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
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Bootstrap: sibling imports work when run/imported from anywhere.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
from v_dance.parser.vod_parser.pokedex import norm_species
from v_dance.parser.vod_parser.team_sheet import parse_showdown_team, base_species as _paste_base_species
from v_dance.parser.belief_state import BeliefState, STAT_ORDER

# The FROZEN layout is owned by the VOD module — import it so live == offline.
from v_dance.encoders.state_encoder import (
    _TYPE_IDX, _STATUS_IDX, _WEATHER_IDX, _FIELD_IDX, _SC_IDX,
    NUM_TYPES, NUM_STATUS, NUM_WEATHER, NUM_FIELDS, NUM_SIDE_CONDS,
    NUM_BOOSTS, NUM_MOVES, MOVE_FEATURES,
    POKEMON_FEATURES, STATE_DIM, ACTION_DIM,
    ACTIVE_SLOTS, BENCH_SLOTS, OPP_BENCH_SLOTS,
    _BOOST_KEYS, _EST_STAT_NORM,
    dex_unique_ability,
    is_spread_target, _type_eff_signed_immune, _type_mult, _damage_band, _moves_first,
    _situational_damage_mult, _WEATHER_SPEED_ABILITY, field_duration_scalars,
    _ability_damage_mult, _DEF_HIT_MOVES, _GRASSY_WEAKENED, _ability_trapped, _is_grounded, _ground_immune, _move_immune,
    _item_active,                                   # v11 P5: Magic Room item-suppression gate (shared)
    _expected_crit_mult,                            # v11 N4: expected-crit band multiplier (shared)
    _immunity_neg_ctx, _move_hit_range,
    champ_bp, champ_type, champ_acc_raw, _canon,   # v11 N1: Champions move overrides (shared, parity)
    _disguise_intact,                              # v11 B1: Mimikyu Disguise intact-block (shared, parity)
    _accuracy_modifiers, _move_always_hit, _move_is_ohko, _per_enemy_hit_chance,  # v11 B2/B2b (shared, parity)
    _TYPE_BOOST_TYPE, _RESIST_BERRY_TYPE, _BAND_ITEM_MULT,   # v11 B3: item band mults (shared, parity)
    VodStateEncoder,
    # v9 (B1-mechanics): mechanic substrate (re-exported from state_encoder's namespace)
    NUM_MOVE_TAGS, NUM_ABILITY_TAGS, NUM_ITEM_TAGS, VOLATILE_FEATURES,
    _PROTECT_COUNTER_CAP,
    move_tag_indices, ability_tag_indices, item_tag_indices, ABILITY_SUPPRESSED_IDX,
    ability_index, move_index, item_index,
)
from v_dance.encoders import damage_mechanics as _DMG
from v_dance.parser.vod_parser.battle_models import volatile_flags


def _live_eff_types(mon) -> list:
    """A poke-env mon's CURRENT effective type NAMES (upper) for the type-eff cross: tera overrides
    (ACTIVE in Champions), else a runtime TYPECHANGE, else its dex types. Mirrors the offline
    _effective_types precedence (tera > typechange > transform > dex). ⚠ v11 N2/D7: poke-env's attribute
    is `is_terastallized` (the old `terastallized` was a NONEXISTENT attr → getattr always False → live
    used PRE-tera typing on every post-tera turn). ⚠ v11 P2: NO functional change needed for typechange —
    poke-env's mon.type_1/type_2 properties (pokemon.py:1360-1394) already fold _temporary_types
    (Protean/Libero/Soak/Burn Up/Reflect Type) at exactly the 2nd precedence (tera > _temporary_types >
    dex), AND tera forces type_2=None. So reading type_1/type_2 below mirrors the offline runtime_types
    branch by construction; the offline parser captures the SAME typechange protocol line that drives
    _temporary_types live."""
    if mon is None:
        return []
    if getattr(mon, "is_terastallized", False) and getattr(mon, "tera_type", None):
        return [mon.tera_type.name]
    return [t.name for t in (mon.type_1, mon.type_2) if t is not None]

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


# ── Reusable live mega-forme detection (gap #5) ──────────────────────────────
# Shared by the encoder (LiveStateEncoder._is_mega_forme) AND the serve codec
# (vgc_base.build_gimmick_legal_mask / action_to_order) so train and serve use
# ONE detection.  poke-env keeps ``mon.species`` at the base forme after a mega
# but rewrites ``base_stats`` from the mega dex entry on ``|detailschange|``, so
# a mon is mega iff its current base stats equal a ``<base>mega/megax/megay`` dex
# entry.  A transformed Ditto copies a mega's stats but is NOT itself mega.
_LIVE_POKEDEX_CACHE = None


def _default_live_pokedex() -> dict:
    """Cached poke-env raw gen-9 pokedex ({} when poke-env is unavailable)."""
    global _LIVE_POKEDEX_CACHE
    if _LIVE_POKEDEX_CACHE is None and GenData is not None:
        _LIVE_POKEDEX_CACHE = GenData.from_gen(9).pokedex
    return _LIVE_POKEDEX_CACHE or {}


def _base_species_of(mon) -> str:
    """Base-forme species id for forme/teampreview matching."""
    return (getattr(mon, "base_species", None) or getattr(mon, "species", "") or "")


def is_mega_forme_live(mon, dex=None) -> bool:
    """Whether ``mon`` is CURRENTLY mega-evolved (gap #5), as a free function so
    the encoder and the serve codec share one detection."""
    if dex is None:
        dex = _default_live_pokedex()
    if not dex or to_id_str is None:
        return False
    if getattr(mon, "transformed", False):
        return False
    base = to_id_str(_base_species_of(mon))
    if not base:
        return False
    cur = getattr(mon, "base_stats", None)
    if not cur:
        return False
    for suf in _MEGA_SUFFIXES:
        entry = dex.get(base + suf)
        stats = entry.get("baseStats") if entry else None
        if stats and dict(stats) == dict(cur):
            return True
    return False


def team_has_megaed_live(battle) -> bool:
    """True iff any own mon has already mega-evolved this game (a team megas at
    most once per game).  Live analogue of state_encoder._own_team_has_megaed,
    which reads the offline ``is_mega`` flag."""
    team = getattr(battle, "team", None) or {}
    dex = _default_live_pokedex()
    return any(is_mega_forme_live(m, dex) for m in team.values())


def team_has_teraed_live(battle) -> bool:
    """v11 Phase D: True iff any own mon has already terastallized this game (tera is once-per-game per
    side). Live analogue of state_encoder._own_team_has_teraed (offline ``is_terastallized``); reads
    poke-env Pokemon.is_terastallized over battle.team. (Tera is mod-disabled in Champions → always
    False today; forward-compat with the mega twin.)"""
    team = getattr(battle, "team", None) or {}
    return any(getattr(m, "is_terastallized", False) for m in team.values())


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
            # The offline parser tracks ONLY Tailwind + the 3 screens, so the corpus NEVER sets the
            # hazard / Safeguard / Mist side-condition channels (constant 0.0). Restrict the live emit
            # to the SAME channels (by index) so train==serve — emitting a live hazard bit the net
            # only ever saw as 0.0 is an out-of-distribution feed on the un-spliced SC globals.
            self._offline_sc_idx = frozenset(
                _SC_IDX[n] for n in ("TAILWIND", "REFLECT", "LIGHT_SCREEN", "AURORA_VEIL",
                                     "STEALTH_ROCK", "SPIKES", "TOXIC_SPIKES", "STICKY_WEB",   # v11 C.1
                                     "SAFEGUARD", "MIST", "LUCKY_CHANT")                       # v11 C.2b
                if n in _SC_IDX
            )
            # v11 C.2e: the FIELD block had NO offline-tracked guard (unlike the SC block) — the offline
            # encoder only ever sets the 4 terrains + TRICK_ROOM + (now) GRAVITY, but poke-env's
            # battle.fields can also carry MAGIC_ROOM / WONDER_ROOM / HEAL_BLOCK / FAIRY_LOCK /
            # NEUTRALIZING_GAS, which the live encoder used to emit as a spurious 1.0 (train/serve drift).
            # Restrict the live field emit to the SAME channels the offline parser populates.
            self._offline_field_idx = frozenset(
                _FIELD_IDX[n] for n in ("ELECTRIC_TERRAIN", "GRASSY_TERRAIN", "MISTY_TERRAIN",
                                        "PSYCHIC_TERRAIN", "TRICK_ROOM", "GRAVITY",
                                        "MAGIC_ROOM", "WONDER_ROOM")        # v11 P5
                if n in _FIELD_IDX
            )
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

        # B1.2b: global field mods (weather/terrain) for the damage situational multiplier.
        _weather = next((w.name for w in battle.weather), None)
        _terrain = next((f.name for f in battle.fields if f.name.endswith("_TERRAIN")), None)
        field_mods = (_weather, _terrain)

        # B1/#25: enemy-active DEFENDER PROFILES (types/stats/HP/grounded/screens) — own mons attack the
        # opp actives & vice-versa; is_own picks the right est-stat belief + side screens for each side.
        _oa, _wa = list(opp_active), list(own_active)

        # v11 B.1b: per-side tailwind + Trick Room resolved ONCE (mirror offline) — feed the move-block
        # who-moves-first channel (defender eff_speed + attacker att_ctx.eff_speed) and the GLOBAL block.
        _own_tw = any(getattr(k, "name", "") == "TAILWIND" and v
                      for k, v in battle.side_conditions.items())
        _opp_tw = any(getattr(k, "name", "") == "TAILWIND" and v
                      for k, v in battle.opponent_side_conditions.items())
        _tr = any(getattr(f, "name", "") == "TRICK_ROOM" for f in battle.fields)
        _gravity = any(getattr(f, "name", "") == "GRAVITY" for f in battle.fields)   # v11 C.2e
        _magic_room = any(getattr(f, "name", "") == "MAGIC_ROOM" for f in battle.fields)     # v11 P5
        _wonder_room = any(getattr(f, "name", "") == "WONDER_ROOM" for f in battle.fields)   # v11 P5

        def _prof(mon, is_own):
            if mon is None:
                return None
            est = self._live_est_stats(mon, is_own)[0] or {}
            scr = battle.side_conditions if is_own else battle.opponent_side_conditions
            phys_scr = any(getattr(k, "name", "") in ("REFLECT", "AURORA_VEIL") and v for k, v in scr.items())
            spec_scr = any(getattr(k, "name", "") in ("LIGHT_SCREEN", "AURORA_VEIL") and v for k, v in scr.items())
            types = _live_eff_types(mon)
            _ab = self._live_ability(mon, is_own)[0]
            # v11 C.2c: grounding volatiles from poke-env effects (byte-parity with offline volatile_flags).
            _eff = {getattr(e, "name", str(e)).lower().replace("_", "")
                    for e in (getattr(mon, "effects", {}) or {})}
            _it = _item_active(self._live_item(mon, is_own)[0], _magic_room,   # v11 P5: MR
                               klutz=_ab == "klutz",                          # v11 Klutz: suppress held item
                               embargo="embargo" in _eff)                     # v11 Embargo volatile
            _pvf = volatile_flags(_eff)
            _levit, _fg = _pvf["levitating"], _pvf["force_grounded"]
            _fg_g = _fg or _gravity            # v11 C.2e: Gravity grounds everything (isGrounded-first)
            grounded = _is_grounded(types, _ab, _it, _levit, _fg_g)
            return {"types": types,
                    "def": est.get("def"), "spd": est.get("spd"),
                    "hp": est.get("hp") or getattr(mon, "max_hp", None),
                    "hp_frac": (mon.current_hp_fraction if mon.revealed else 1.0),
                    # v11 A.1: the ACTIVE ability (belief-aware) for damage-band immunity — parity twin
                    # of the offline _defender_profile's resolve_active_ability_json.
                    "ability": _ab,
                    "grounded": grounded, "screen_phys": phys_scr, "screen_spec": spec_scr,
                    # v11 C.2c: Ground-move immunity (Air Balloon / Magnet Rise / Telekinesis) + Tar Shot.
                    "ground_immune": _ground_immune(_it, _levit, _fg_g), "tar_shot": _pvf["tar_shot"],
                    # v11 C.2d: type-immunity NEGATION inputs (parity twin of the offline profile).
                    "force_grounded": _fg_g or _it == "ironball", "ring_target": _it == "ringtarget",
                    "foresight": _pvf["foresight"], "miracleeye": _pvf["miracleeye"],
                    # v11 B1: intact Mimikyu Disguise (ability + full HP; the busted species is invisible to
                    # poke-env so we key on hp_frac, parity twin of the offline _defender_profile).
                    "intact_disguise": _disguise_intact(_ab, mon.current_hp_fraction if mon.revealed else 1.0),
                    # v11 B.1b: resolved in-battle speed for the move-block who-moves-first channel.
                    "eff_speed": self._live_effective_speed(mon, is_own,
                                                            _own_tw if is_own else _opp_tw,
                                                            _weather, _magic_room)[0],
                    # v11 B.2: target Atk (Foul Play), raw weight (Low Kick/Heavy Slam), +boost sum
                    # (Punishment) — parity twin of the offline _defender_profile.
                    "atk": est.get("atk"),
                    # v11 A4: poke-env mon.weight is mega/forme/transform-aware (it keeps mon.species at the
                    # BASE forme post-mega but DOES update _weightkg); offline mutates species → species_weight
                    # is correct there. Read mon.weight to restore parity, falling back to species_weight.
                    "weight": (getattr(mon, "weight", None)
                               or _DMG.species_weight(getattr(mon, "species", None)) or 0.0),
                    "pos_boosts": sum(v for v in (mon.boosts or {}).values() if v > 0),
                    # v11 B2b: defender-side accuracy/evasion inputs (parity twin of the offline profile).
                    "eva_stage": (mon.boosts or {}).get("evasion", 0),
                    "confused": _pvf["confused"],
                    "no_guard": _ab == "noguard",
                    "evasion_item": _it in ("brightpowder", "laxincense"),
                    # v11 B3: defensive items (parity twin of the offline _defender_profile).
                    "assault_vest": _it == "assaultvest",
                    "evio": _it == "eviolite" and _DMG.is_nfe(getattr(mon, "species", None)),
                    "resist_berry": _RESIST_BERRY_TYPE.get(_it)}

        own_enemy = [_prof(_oa[0] if len(_oa) > 0 else None, False),
                     _prof(_oa[1] if len(_oa) > 1 else None, False)]
        opp_enemy = [_prof(_wa[0] if len(_wa) > 0 else None, True),
                     _prof(_wa[1] if len(_wa) > 1 else None, True)]

        own_active_set = {p for p in own_active if p is not None}

        # v11 B.2 gap-fix (Last Respects): per-side fainted count for the move's BP (50+50×faints). SAME
        # logic as the parity-proven team-count block below (own brought/request fainted; opp seen-fainted —
        # opp actives are alive at decision time, so the active-exclusion there is a no-op for the count).
        _own_req = _request_own_state(battle)
        _own_fnt = (sum(1 for a, f in _own_req.values() if f) if _own_req is not None
                    else sum(1 for p in brought_team_mons(battle) if p.fainted))
        _opp_fnt = sum(1 for p in getattr(battle, "opponent_team", {}).values()
                       if self._is_seen(p) and p.fainted)
        # v11 C.4 (Architecture B): the OWN-side Rage Fist hit count rides the SAME offline parse that
        # builds opp_snapshot (state_before[own_role].our_active carries times_attacked from to_dict) — so
        # there is NO second live filter to drift. Index our_a→slot 0, our_b→slot 1 (the own-active loop's
        # enumerate order). Bench own = 0 (reset-on-switch-in convention); opp byte ranges are spliced over.
        _oa_snap = (opp_snapshot or {}).get("our_active") or {}
        _own_ta = {0: (_oa_snap.get("our_a") or {}).get("times_attacked", 0),
                   1: (_oa_snap.get("our_b") or {}).get("times_attacked", 0)}

        _own_list = list(own_active)
        for slot, mon in enumerate(_own_list):
            # #1b: under an own-side Illusion the active object's moves can be empty
            # while the |request| holds the real ones — encode those (request-
            # authoritative) so the move features match the mask + the offline parser.
            mv = own_active_move_list(battle, slot, mon) if mon is not None else None
            # v11 Victory Star: the ACTIVE ally's ability (other slot) — boosts this mon's accuracy.
            _ally = _own_list[1 - slot] if len(_own_list) > 1 else None
            _ally_ab = self._live_ability(_ally, True)[0] if _ally is not None else None
            self._write_pokemon(vec, cursor, mon, is_active=True, is_own=True,
                                move_override=mv, enemy_defenders=own_enemy, field_mods=field_mods,
                                side_tailwind=_own_tw, trick_room=_tr, gravity=_gravity,
                                fainted_allies=_own_fnt, times_attacked=_own_ta.get(slot, 0),
                                magic_room=_magic_room, wonder_room=_wonder_room, ally_ability=_ally_ab)
            cursor += POKEMON_FEATURES
        _opp_list = list(opp_active)
        for slot, mon in enumerate(_opp_list):
            _ally = _opp_list[1 - slot] if len(_opp_list) > 1 else None
            _ally_ab = self._live_ability(_ally, False)[0] if _ally is not None else None
            self._write_pokemon(vec, cursor, mon, is_active=True, is_own=False, enemy_defenders=opp_enemy,
                                field_mods=field_mods, side_tailwind=_opp_tw, trick_room=_tr, gravity=_gravity,
                                fainted_allies=_opp_fnt,
                                magic_room=_magic_room, wonder_room=_wonder_room, ally_ability=_ally_ab)
            cursor += POKEMON_FEATURES

        # ── [B] Bench slots ──────────────────────────────────────────────────
        # own_bench_mons returns the BROUGHT (4-of-6), non-active, non-fainted
        # own mons in stable team order.  VGC team-preview brings only 4 of the 6
        # roster mons, but poke-env keeps all 6 in battle.team; the 2 un-brought
        # ones can never enter play, so listing them here both desyncs the bench
        # from the offline parser (which only ever sees the brought 4) AND makes
        # the switch action codec emit un-switchable mons (Showdown rejects the
        # order).  It also drops broken-illusion phantoms (max_hp==0, gap #4).
        bench = own_bench_mons(battle)[:BENCH_SLOTS]

        for i in range(BENCH_SLOTS):
            mon = bench[i] if i < len(bench) else None
            self._write_pokemon(vec, cursor, mon, is_active=False, is_own=True, enemy_defenders=own_enemy,
                                field_mods=field_mods, side_tailwind=_own_tw, trick_room=_tr, gravity=_gravity,
                                fainted_allies=_own_fnt,
                                magic_room=_magic_room, wonder_room=_wonder_room)
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
            self._write_pokemon(vec, cursor, mon, is_active=False, is_own=False, enemy_defenders=opp_enemy,
                                field_mods=field_mods, side_tailwind=_opp_tw, trick_room=_tr, gravity=_gravity,
                                fainted_allies=_opp_fnt,
                                magic_room=_magic_room, wonder_room=_wonder_room)
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
            idx = self._field_idx.get(f)
            if idx is not None and idx in self._offline_field_idx:   # v11 C.2e: offline-tracked only
                vec[cursor + idx] = 1.0
        cursor += NUM_FIELDS

        # Side conditions. poke-env's value is a LAYER COUNT (stackable: Spikes/Toxic Spikes) OR the turn
        # a condition started (non-stackable) — NOT turns remaining. v11 C.1: Spikes/Toxic Spikes encode
        # the normalised layer count (parity with the offline parser's layer count); everything else is a
        # presence bit. RESTRICTED to the offline-tracked channels (_offline_sc_idx = Tailwind + 3 screens
        # + 4 hazards): emitting an un-tracked SC live would feed the net 1.0 on a channel it only saw as
        # 0.0 in training (these SC globals are NOT corrected by the gap-#6 opponent splice).
        for sc, val in battle.side_conditions.items():
            idx = self._sc_idx.get(sc)
            if val and idx in self._offline_sc_idx:
                vec[cursor + idx] = self._sc_value(sc, val)
        cursor += NUM_SIDE_CONDS

        # Opponent side conditions (same offline-tracked restriction)
        for sc, val in battle.opponent_side_conditions.items():
            idx = self._sc_idx.get(sc)
            if val and idx in self._offline_sc_idx:
                vec[cursor + idx] = self._sc_value(sc, val)
        cursor += NUM_SIDE_CONDS

        # Turn (cap at 60 for normalisation)
        vec[cursor] = min(battle.turn, 60) / 60.0
        cursor += 1

        # Trick Room explicit flag (already in fields but surfaced separately
        # because it flips speed priority — critical strategic signal)
        vec[cursor] = 1.0 if trick_room_active else 0.0
        cursor += 1

        # ── B1.4 turn-order block (mirror offline): 4 effective speeds + 4 moves-first margins + conf.
        # (_own_tw/_opp_tw resolved once above for B.1b.)
        _sp = {  # v11 E4-fix: pass _magic_room (parity with offline) so the global turn-order speed suppresses
                 # Choice Scarf under Magic Room; Klutz/Embargo are gated inside _live_effective_speed.
            "our_a": self._live_effective_speed(own_active[0] if len(own_active) > 0 else None, True, _own_tw, _weather, _magic_room),
            "our_b": self._live_effective_speed(own_active[1] if len(own_active) > 1 else None, True, _own_tw, _weather, _magic_room),
            "opp_a": self._live_effective_speed(opp_active[0] if len(opp_active) > 0 else None, False, _opp_tw, _weather, _magic_room),
            "opp_b": self._live_effective_speed(opp_active[1] if len(opp_active) > 1 else None, False, _opp_tw, _weather, _magic_room),
        }
        for _k in ("our_a", "our_b", "opp_a", "opp_b"):       # 4 effective-speed channels (/600 clamped)
            vec[cursor] = min(_sp[_k][0] / 600.0, 1.0)
            cursor += 1
        for _ok in ("our_a", "our_b"):                         # 4 our×opp pairs × [margin, confidence]
            for _pk in ("opp_a", "opp_b"):
                vec[cursor] = _moves_first(_sp[_ok][0], _sp[_pk][0], trick_room_active)
                vec[cursor + 1] = _sp[_ok][1] * _sp[_pk][1]
                cursor += 2

        # ── Field-duration block (11, v10): turns-active AGE per condition. SOURCED from the gap-#6
        # vod_parser RECONSTRUCTION (opp_snapshot) when available — poke-env REFRESHES battle.weather to
        # the current turn on every ``[upkeep]`` line, so its value is unusable as a weather start-turn.
        # The reconstruction counts turns-active with the SAME parser as training → byte-identical age.
        # Fallback (no reconstruction, a diagnostic-only path): screens/tailwind from poke-env's STABLE
        # start-turns; weather/terrain age 0 (poke-env can't supply it).
        _turn = battle.turn or 0
        _fd_field = (opp_snapshot or {}).get("field")
        _fd_sc = (opp_snapshot or {}).get("side_conditions")
        if _fd_field is not None and _fd_sc is not None:
            _tr_rem = _fd_field.get("trick_room_turns_remaining") or 0
            _fd_sides = []
            for _sk in ("our_side", "opp_side"):
                _s = _fd_sc.get(_sk) or {}
                _tw_rem = _s.get("tailwind_turns_remaining") or 0
                _fd_sides.append({"tailwind_age": (4 - _tw_rem) if _tw_rem > 0 else 0,
                                  "screens": _s.get("screens")})
            _fd = field_duration_scalars(_fd_field.get("weather_turns_active"),
                                         _fd_field.get("terrain_turns_active"),
                                         (5 - _tr_rem) if _tr_rem > 0 else 0, _fd_sides)
        else:
            def _age(start):
                return max(0, _turn - start) if isinstance(start, int) else 0

            def _side_ages(side_conditions):
                tw_age, screens = 0, {}
                for _sc, _v in side_conditions.items():
                    _nm = getattr(_sc, "name", "")
                    if _nm == "TAILWIND":
                        tw_age = _age(_v)
                    elif _nm == "REFLECT":
                        screens["reflect"] = _age(_v)
                    elif _nm == "LIGHT_SCREEN":
                        screens["light_screen"] = _age(_v)
                    elif _nm == "AURORA_VEIL":
                        screens["aurora_veil"] = _age(_v)
                return {"tailwind_age": tw_age, "screens": screens}

            _tr_age = next((_age(v) for f, v in battle.fields.items()
                            if getattr(f, "name", "") == "TRICK_ROOM"), 0)
            # Terrain carries a STABLE poke-env start-turn in battle.fields (like Trick Room), so
            # derive its age here instead of zeroing it — closing the terrain half of the fallback
            # gap for free.  Weather has NO stable start-turn (poke-env refreshes battle.weather each
            # upkeep), so it stays 0 on this reconstruction-failure-only fallback path.
            _terrain_age = next((_age(v) for f, v in battle.fields.items()
                                 if getattr(f, "name", "").endswith("_TERRAIN")), 0)
            _fd = field_duration_scalars(
                0, _terrain_age, _tr_age,
                [_side_ages(battle.side_conditions), _side_ages(battle.opponent_side_conditions)])
        for _v in _fd:
            vec[cursor] = _v
            cursor += 1

        # ── Team counts (layout-v2): living-bench + fainted per side ──────────
        # Opp counts use SEEN (revealed, non-active) mons only — the information
        # asymmetry the offline path encodes (opp_seen_alive / opp_seen_fainted).
        # Own counts iterate the BROUGHT team only (brought_team_mons): the
        # un-brought 2-of-6 roster mons in battle.team would otherwise inflate
        # own_live by up to 2 vs the offline parser (which only sees the brought
        # 4).  _is_real_mon (inside the helper) also drops the broken-illusion
        # phantom (max_hp==0) that would inflate own_live after an own Zoroark's
        # disguise drops (gap #4).
        # Own counts: authoritative from the in-battle request when available (#11b
        # — an Illusion can corrupt poke-env's per-mon fainted/active FLAGS, but the
        # brought-only request carries the truth), else poke-env's flags (offline /
        # replay-driven path).
        own_req = _request_own_state(battle)
        if own_req is not None:
            own_live = sum(1 for a, f in own_req.values() if not a and not f)
            own_fnt  = sum(1 for a, f in own_req.values() if f)
        else:
            own_team = brought_team_mons(battle)
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
        # immune to poke-env's duplicate-species illusion merge.  Belief-enrich
        # the opponent mons FIRST (distribution est-stats + predicted move slots)
        # so the spliced bytes match the (belief-enriched) TRAINING data instead
        # of leaving those features zeroed at serve time — see _enrich_opp_snapshot.
        if opp_snapshot is not None:
            opp_snapshot = self._enrich_opp_snapshot(opp_snapshot)
            self._splice_opponent(vec, opp_snapshot, turn=getattr(battle, "turn", 0))

        return vec

    def _enrich_opp_snapshot(self, opp_snapshot: dict) -> dict:
        """Belief-enrich the OPPONENT mons of a reconstructed snapshot so the
        spliced opp bytes carry the same Pikalytics distribution est-stats
        (``stats_estimate``) and predicted move-slot padding (``belief``) that the
        TRAINING data was exported with (Type B fill_blanks → opponent =
        distribution).  Without this the gap-#6 splice produces opp bytes with the
        est-stat block and the unrevealed move slots all ZERO — a large train/serve
        mismatch on every opponent mon (the net was trained with those populated).

        Idempotent (skips a mon that already carries an estimate) and a no-op when
        the encoder has no BeliefState.  Mutates the ephemeral per-turn snapshot in
        place (it is rebuilt every decision) and returns it.
        """
        if self.belief is None or not opp_snapshot:
            return opp_snapshot
        try:
            from v_dance.parser.belief_state import _enrich_mon_belief
        except Exception:  # pragma: no cover
            return opp_snapshot
        opp_active = opp_snapshot.get("opp_active") or {}
        mons = [opp_active.get("opp_a"), opp_active.get("opp_b")]
        mons += list(opp_snapshot.get("opp_bench") or [])
        warned: set = set()
        for m in mons:
            if m and not m.get("stats_estimate"):
                try:
                    # top_k=5 matches belief_state.fill_blanks' default (the value
                    # the training export used), so serve == train byte-for-byte.
                    _enrich_mon_belief(m, self.belief, 5, [], warned, level=self.level)
                except Exception:  # pragma: no cover - never break a live turn
                    pass
        return opp_snapshot

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
        substring 'mega' (Meganium, Yanmega).  Mirrors the offline is_mega flag.

        Delegates to the module-level ``is_mega_forme_live`` so the encoder and
        the serve codec (vgc_base) share ONE detection."""
        return is_mega_forme_live(mon, getattr(self, "_pokedex", None))

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

    @staticmethod
    def _sc_value(sc, val) -> float:
        """v11 C.1: a side-condition channel value. Spikes/Toxic Spikes encode the normalised LAYER count
        (poke-env val IS the layer count for these) — parity with the offline parser's layer count;
        everything else (screens/Tailwind/SR/Web) is a presence bit."""
        name = getattr(sc, "name", "")
        if name == "SPIKES":
            return min(val, 3) / 3.0
        if name == "TOXIC_SPIKES":
            return min(val, 2) / 2.0
        return 1.0

    # ── Pokémon encoder (live) ────────────────────────────────────────────────
    def _live_effective_speed(self, mon, is_own, side_tailwind, weather=None, magic_room=False):
        """(speed, known) — mirror of the offline _effective_speed: est speed folding the spe boost
        stage · paralysis ×0.5 · Choice Scarf ×1.5 · Tailwind ×2 · weather-speed ability ×2 (B1.2b)."""
        if mon is None:
            return 0.0, 0.0
        est, known = self._live_est_stats(mon, is_own)
        spe = (est or {}).get("spe")
        if not spe:
            return 0.0, 0.0
        stage = (mon.boosts or {}).get("spe", 0) if getattr(mon, "boosts", None) else 0
        spe *= ((2 + stage) / 2.0) if stage >= 0 else (2.0 / (2 - stage))
        # v11 B.1 (#4): Quick Feet ×1.5 on ANY status, negating the para ×0.5 (parity with offline).
        _statused = getattr(mon, "status", None) is not None
        if self._live_ability(mon, is_own)[0] == "quickfeet" and _statused:
            spe *= 1.5
        elif getattr(getattr(mon, "status", None), "name", None) == "PAR":
            spe *= 0.5
        # Resolve item/ability through the BELIEF-aware helpers — NOT raw poke-env attrs — so this
        # mirrors the offline _effective_speed (resolve_item_json / resolve_ability_json), which for an
        # unrevealed OPPONENT falls back to the Pikalytics top prior. Reading raw mon.item/mon.ability
        # here (unknown_item / None for an unrevealed opp) silently dropped the Choice Scarf ×1.5 and
        # weather-speed ×2 that TRAINING applied → a 1.5-2× serve-only who-moves-first divergence.
        _eff_ids = {getattr(e, "name", str(e)).lower().replace("_", "")
                    for e in (getattr(mon, "effects", {}) or {})}
        if _item_active(self._live_item(mon, is_own)[0], magic_room,          # v11 P5: MR suppresses Scarf
                        klutz=self._live_ability(mon, is_own)[0] == "klutz",  # v11 Klutz
                        embargo="embargo" in _eff_ids) == "choicescarf":      # v11 Embargo
            spe *= 1.5
        if side_tailwind:
            spe *= 2.0
        if weather and weather in _WEATHER_SPEED_ABILITY.get(self._live_ability(mon, is_own)[0], ()):
            spe *= 2.0
        # v11 B.1 (#4): Protosynthesis/Quark Drive ×1.5 when boosting SPEED — gated on the SPE-suffixed
        # poke-env effect (parity with offline volatile_flags.paradox_speed: identical normalised id).
        if _eff_ids & {"protosynthesisspe", "quarkdrivespe"}:
            spe *= 1.5
        return float(spe), known

    def _write_pokemon(
        self,
        vec: np.ndarray,
        start: int,
        mon: Optional["Pokemon"],
        is_active: bool,
        is_own: bool,
        move_override: Optional[list] = None,
        enemy_defenders: Optional[list] = None,
        field_mods: tuple = (None, None),
        side_tailwind: bool = False,
        trick_room: bool = False,
        gravity: bool = False,
        fainted_allies: int = 0,
        times_attacked: int = 0,
        magic_room: bool = False,
        wonder_room: bool = False,
        ally_ability: Optional[str] = None,
    ) -> None:
        """Write POKEMON_FEATURES floats into vec starting at `start`.

        ``move_override`` (own active mons only) supplies the move list when poke-env's
        ``mon.moves`` is an Illusion-stale empty (#1b — see own_active_move_list); None
        ⇒ the default ``mon.moves`` ordering, so every other slot is unchanged.
        ``gravity`` (v11 C.2e) force-grounds this mon (Arena Trap) + boosts move accuracy.
        ``fainted_allies`` (v11 B.2 gap-fix) = this mon's side faint count → Last Respects BP."""
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

        # v9: normalized species weight (weight-based moves + Heavy/Light Metal). v11 A4: read poke-env's
        # mega/forme/transform-aware mon.weight (mon.species stays BASE post-mega → species_weight would read
        # the base weight). This _wt feeds BOTH the weight channel below AND att_ctx.weight (Low Kick/Heavy
        # Slam variable-BP) — fix once here, both consumers corrected. species_weight is the fallback.
        _wt = getattr(mon, "weight", None) or _DMG.species_weight(getattr(mon, "species", None)) or 0.0
        vec[i] = min(_wt / 500.0, 1.0)
        i += 1

        # v9: current ability + volatile state from poke-env (Effect enum names normalise to the parser's
        # volatile ids -> volatile_flags gives the SAME booleans as offline).
        abil_id, abil_known = self._live_ability(mon, is_own)
        _eff_ids = {getattr(e, "name", str(e)).lower().replace("_", "")
                    for e in (getattr(mon, "effects", {}) or {})}
        vf = volatile_flags(_eff_ids)

        # Moves (up to 4; remaining slots stay zero = unknown)
        move_list = (move_override if move_override is not None
                     else list(mon.moves.values())[:NUM_MOVES])
        # B1.2/B1.2b attacker context: A stat + burn + Life-Orb/Choice item.
        _est = self._live_est_stats(mon, is_own)[0] or {}
        _it = _item_active(getattr(mon, "item", None), magic_room,   # v11 P5: Magic Room
                           klutz=abil_id == "klutz",                 # v11 Klutz: suppress held item
                           embargo="embargo" in _eff_ids)            # v11 Embargo volatile
        att_ctx = {"atk": _est.get("atk"), "spa": _est.get("spa"),
                   "burned": getattr(getattr(mon, "status", None), "name", None) == "BRN",
                   "statused": getattr(mon, "status", None) is not None,   # v11 A.2: Guts on ANY status
                   "life_orb": _it == "lifeorb", "choice": _it in ("choiceband", "choicespecs"),
                   # v11 B.1b: this mon's resolved speed + Trick Room for the who-moves-first channel.
                   "eff_speed": self._live_effective_speed(mon, is_own, side_tailwind,
                                                           field_mods[0] if field_mods else None,
                                                           magic_room)[0],
                   "trick_room": trick_room,
                   "wonder_room": wonder_room,             # v11 P5: swaps Def<->SpD in the band
                   # v11 B.2: damage-band R1 context (parity twin of the offline att_ctx).
                   "def": _est.get("def"), "weight": _wt,
                   "hp_frac": (mon.current_hp_fraction if mon.revealed else 1.0),
                   "pos_boosts": sum(v for v in (mon.boosts or {}).values() if v > 0),
                   "fainted_allies": fainted_allies,     # v11 B.2 gap-fix: Last Respects BP
                   "times_attacked": times_attacked,     # v11 C.4: Rage Fist BP
                   "loaded_dice": _it == "loadeddice",    # v11 A1: 2-5 moves hit 4-5
                   "acc_stage": (mon.boosts or {}).get("accuracy", 0),  # v11 B2: attacker accuracy stage
                   "wide_lens": _it == "widelens",        # v11 B2: Wide Lens ×1.1 accuracy
                   "expert_belt": _it == "expertbelt",    # v11 B3: ×1.2 super-effective
                   "type_boost": _TYPE_BOOST_TYPE.get(_it),  # v11 B3: ×1.2 matching-type move
                   "scope_lens": _it in ("scopelens", "razorclaw"),  # v11 N4: +1 crit stage (P5: gated via _it)
                   # v11: Victory Star ×1.1 accuracy — the holder OR its active ally has the ability.
                   "victory_star": abil_id == "victorystar" or ally_ability == "victorystar"}
        for m_idx in range(NUM_MOVES):
            if m_idx < len(move_list):
                self._write_move(vec, i, move_list[m_idx], mon, enemy_defenders, att_ctx,
                                 field_mods, ability_id=abil_id, gravity=gravity)
            i += MOVE_FEATURES

        # ── v9 ITEM block: identity index + effect-tag multi-hot + known ──
        # v11 P5: under Magic Room an ACTIVE mon's item is HELD but non-functional → keep identity + known,
        # ZERO the EFFECT tags (parity twin of the offline path).
        item_id, item_known = self._live_item(mon, is_own)
        vec[i] = float(item_index(item_id))
        i += 1
        for idx in item_tag_indices(_item_active(item_id, magic_room and is_active,
                                                 klutz=abil_id == "klutz", embargo="embargo" in _eff_ids)):
            vec[i + idx] = 1.0
        i += NUM_ITEM_TAGS
        vec[i] = item_known
        i += 1

        # ── v9 ABILITY block: identity index + effect-tag multi-hot + known. Gastro Acid suppresses the
        # functional tags but keeps the identity (+ sets ability_suppressed). ──
        vec[i] = float(ability_index(abil_id))
        i += 1
        if "gastroacid" in _eff_ids:
            vec[i + ABILITY_SUPPRESSED_IDX] = 1.0
        else:
            for idx in ability_tag_indices(abil_id):
                vec[i + idx] = 1.0
        i += NUM_ABILITY_TAGS
        vec[i] = abil_known
        i += 1

        # ── v9 VOLATILE block (byte-parity with offline via volatile_flags) ──
        # v11 B.3: ability-trapping (Shadow Tag / Arena Trap / Magnet Pull on an opposing active) zeroes
        # can_switch — parity twin of the offline computation (same shared _ability_trapped logic).
        _mtypes = _live_eff_types(mon)
        _item_id = self._live_item(mon, is_own)[0]
        _mab = self._live_ability(mon, is_own)[0]
        _trap = _ability_trapped(
            "GHOST" in _mtypes, "STEEL" in _mtypes,
            _is_grounded(_mtypes, _mab, _item_id, vf["levitating"],
                         vf["force_grounded"] or gravity),    # v11 C.2c grounding + C.2e Gravity
            _mab == "shadowtag", _item_id == "shedshell",
            [d.get("ability") for d in (enemy_defenders or []) if d])
        vec[i] = 1.0 if vf["rooted"] else 0.0
        vec[i + 1] = 1.0 if vf["trapped"] else 0.0
        vec[i + 2] = 1.0 if vf["has_substitute"] else 0.0
        vec[i + 3] = 1.0 if vf["move_restricted"] else 0.0
        vec[i + 4] = 0.0 if (vf["trapped"] or vf["rooted"] or _trap) else 1.0
        vec[i + 5] = 1.0 if vf["locked_action"] else 0.0
        # v11 A.3: consecutive-Protect counter read DIRECTLY from poke-env (pokemon.py:1236) — NOT via
        # volatile_flags (a set→bool helper that cannot carry an int). Same clamp/divisor as offline.
        vec[i + 6] = min(getattr(mon, "protect_counter", 0) or 0, _PROTECT_COUNTER_CAP) / _PROTECT_COUNTER_CAP
        # v11 C.2/C.2b: read straight off vf (pure id-set functions → byte-parity with offline).
        vec[i + 7] = 1.0 if vf["residual_damage"] else 0.0
        vec[i + 8] = 1.0 if vf["confused"] else 0.0
        vec[i + 9] = vf["perish_norm"]
        vec[i + 10] = 1.0 if vf["drowsy"] else 0.0
        # v11 C.3: first_turn read DIRECTLY from poke-env (pokemon.py:1018, first_turn = _active_turns == 1)
        # — NOT via vf (a set→bool helper with no access to _active_turns), mirroring the A.3 protect_counter
        # precedent. getattr default False covers stub/unrevealed mons (active_turns 0 → False).
        vec[i + 11] = 1.0 if getattr(mon, "first_turn", False) else 0.0
        i += VOLATILE_FEATURES

        # v11 Phase D (D9): tera_type one-hot (NUM_TYPES) — parity twin of the offline write. REVEALED tera
        # type only (same condition as _live_eff_types): set iff is_terastallized AND tera_type. ⚠ Tera is
        # mod-disabled → PERMANENTLY ZERO today (can_tera always falsy); forward-compat.
        if getattr(mon, "is_terastallized", False) and getattr(mon, "tera_type", None):
            _tt = _TYPE_IDX.get(_canon(mon.tera_type.name))
            if _tt is not None:
                vec[i + _tt] = 1.0
        i += NUM_TYPES

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

    # ── Item / ability resolution (live, gap #5) ──────────────────────────────
    def _live_item(self, mon: "Pokemon", is_own: bool) -> tuple[str, float]:
        """(normalised item id, confidence) for a live mon — parity twin of
        resolve_item_json.  A mega'd mon is treated as holding a mega stone
        (matching the offline ``known_item="mega stone"`` stamp).  A concrete
        poke-env item (incl. ``''`` = confirmed itemless after Knock Off) is
        confidence 1.0; an unknown item falls back to the BeliefState top item
        at 0.5 (opponent), else unknown."""
        if self._is_mega_forme(mon):
            return "megastone", 1.0
        it = getattr(mon, "item", None)
        if it is not None and it != "unknown_item":
            return norm_species(it), 1.0     # '' (itemless) → has_item stays 0
        if self.belief is not None:
            top = self.belief.top_item(mon.species)
            if top:
                return norm_species(top), 0.5
        return "", 0.0

    def _live_ability(self, mon: "Pokemon", is_own: bool) -> tuple[str, float]:
        """(normalised ability id, confidence) for a live mon — parity twin of
        resolve_active_ability_json.  v9: a mega'd mon uses poke-env's ``mon.ability`` (the FIXED mega
        forme ability, e.g. Mega Gengar -> Shadow Tag) at 1.0, matching the offline resolve_active path,
        so the v9 mechanic tags (trapping / parental_bond / …) fire for megas.  Otherwise a revealed
        ``mon.ability`` is 1.0, else belief top at 0.5."""
        ab = getattr(mon, "ability", None)
        if ab:
            return norm_species(ab), 1.0
        # Single-ability species → publicly known (parity with resolve_ability_json
        # when poke-env has not yet populated mon.ability for an unseen stub).
        uniq = dex_unique_ability(self._base_species(mon))
        if uniq:
            return norm_species(uniq), 1.0
        if self.belief is not None:
            top = self.belief.top_ability(mon.species)
            if top:
                return norm_species(top), 0.5
        return "", 0.0

    # ── Move encoder (live) ───────────────────────────────────────────────────
    def _write_move(
        self,
        vec: np.ndarray,
        start: int,
        move: "Move",
        user: "Pokemon",
        enemy_defenders: Optional[list] = None,
        att_ctx: Optional[dict] = None,
        field_mods: tuple = (None, None),
        ability_id: Optional[str] = None,
        gravity: bool = False,
    ) -> None:
        """Write MOVE_FEATURES floats into vec starting at `start`. ``enemy_defenders`` = defender
        profiles; ``att_ctx`` = attacker stats/burn/item; ``field_mods`` = (weather, terrain);
        ``ability_id`` = user's current ability (v9 effective-type change). ``gravity`` (v11 C.2e)
        boosts numeric move accuracy ×6840/4096 (the Gravity-Hypnosis payoff)."""
        i = start
        _mid = getattr(move, "id", None)           # v11 N1: move-id key (Champions overrides + the type-eff cross)

        # Base power (cap at 250 since a few moves are absurdly high). v11 N1: Champions BP override.
        vec[i] = min(champ_bp(_mid, move.base_power), 250) / 150.0
        i += 1

        # Move type as ordinal index — v9 EFFECTIVE type (Normalize/Pixilate/-ate/Liquid Voice).
        _mt = move.type.name if getattr(move, "type", None) is not None else ""
        _mt = _canon(champ_type(_mid, _mt))        # v11 N1: Champions type override (snaptrap Grass→Steel)
        _mt = _DMG.effective_move_type(_mid, _mt, ability_id)
        _user_types = {t.name for t in user.types if t}
        if _mt in _TYPE_IDX:
            vec[i] = _TYPE_IDX[_mt] / (NUM_TYPES - 1)
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

        # Accuracy (already 0–1 float; always-hit moves return 1.0). v11 N1: a Champions accuracy override is
        # in RAW moves.ts form (percent int / True) → normalize to live's 0–1 float so both encoders emit equal.
        _acc_ov = champ_acc_raw(_mid, None)
        acc = move.accuracy if _acc_ov is None else (1.0 if _acc_ov is True else _acc_ov / 100.0)
        if gravity and acc < 1.0:        # v11 C.2e: Gravity ×6840/4096 on numeric accuracy, cap 1.0
            acc = min(acc * (6840.0 / 4096.0), 1.0)
        _base_acc = acc                  # v11 B2b: post-Gravity numeric base for the per-enemy hit-chance
        if not _move_always_hit(_mid):   # v11 B2: attacker-side accuracy mods (skip always-hit moves)
            _ac2 = att_ctx or {}
            acc = _accuracy_modifiers(acc, ability_id=ability_id,
                                      is_physical=move.category == MoveCategory.PHYSICAL,
                                      is_ohko=_move_is_ohko(_mid),
                                      acc_stage=_ac2.get("acc_stage", 0), wide_lens=_ac2.get("wide_lens", False),
                                      victory_star=_ac2.get("victory_star", False))   # v11 Victory Star
        vec[i] = acc
        i += 1

        # PP fraction (gap #7): real remaining PP / max.  The VOD parser now counts
        # per-mon self-selected move uses (move_pp_used) and the offline encoder
        # derives the same fraction with max = base·8//5 — which equals poke-env's
        # Move.max_pp — so this matches training (a replay's |move| count == the
        # parser's use-count == poke-env's current_pp decrement; Pressure aside).
        mx = getattr(move, "max_pp", 0) or 0
        vec[i] = (move.current_pp / mx) if mx else 1.0
        i += 1

        # Protect-family move flag (high strategic value in doubles)
        vec[i] = 1.0 if move.is_protect_move else 0.0
        i += 1

        # STAB: EFFECTIVE move type in user's current types (accounts for tera + -ate)
        vec[i] = 1.0 if _mt in _user_types else 0.0
        i += 1

        # is_spread (gap #6): hits both foes — poke-env Move.target enum, mapped to
        # the same id as the offline data/moves.json target by is_spread_target.
        _spread = is_spread_target(getattr(move, "target", None))
        vec[i] = 1.0 if _spread else 0.0
        i += 1

        # B1.1 type-eff cross: signed multiplier + immune vs each of the 2 enemy actives. 4 channels.
        # (_mt = the EFFECTIVE move type, computed above; _mid computed at the top of the writer.)
        _neg = _immunity_neg_ctx(enemy_defenders, ability_id, _mid)   # v11 C.2d (None per slot = legacy)
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            if d and _move_immune(_mt, d, ability_id, _mid):
                signed, immune = -1.0, 1.0          # ability 0× (A.1/A.1b) OR Ground-immune (C.2c)
            else:
                signed, immune = _type_eff_signed_immune(_mt, (d.get("types") if d else []), _neg[e])
                if d and _mt == "FIRE" and immune == 0.0 and d.get("tar_shot"):
                    signed = float(np.clip(signed + 0.5, -1.0, 1.0))   # v11 C.2c: Tar Shot +1 step (×2)
            vec[i] = signed
            vec[i + 1] = immune
            i += 2

        # B1.2 damage band + B1.2b situational mods: [min,max] roll as a fraction of each enemy active's
        # CURRENT HP. 4 channels.
        _phys = move.category == MoveCategory.PHYSICAL
        _ac = att_ctx or {}
        _A = _ac.get("atk") if _phys else _ac.get("spa")
        _stab = _mt in _user_types
        _weather, _terrain = field_mods if field_mods else (None, None)
        _hits_def = _phys or _mid in _DEF_HIT_MOVES                 # v11 B.2 stat-override (Psyshock family)
        _raw_bp = champ_bp(_mid, getattr(move, "base_power", 0))   # v11 N1: Champions BP override (band + variable-BP)
        # v11 A1: multi-hit band scaling (parity twin of offline — shared _move_hit_range + same Skill Link/
        # Loaded Dice adjustments). A 2-5 move's min = min-hits × low-roll, max = max-hits × high.
        _hmin, _hmax = _move_hit_range(_mid)
        if ability_id == "skilllink" and _hmin != _hmax:
            _hmin = _hmax
        elif _ac.get("loaded_dice") and (_hmin, _hmax) == (2, 5):
            _hmin = 4
        # v11 N4: expected-crit multiplier (parity twin of offline — defender-independent, computed once).
        _crit = _expected_crit_mult(_mid, ability_id, _ac.get("scope_lens"))
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            if d and not _move_immune(_mt, d, ability_id, _mid):
                _tmult = _type_mult(_mt, d.get("types") or [], _neg[e])   # v11 C.2d negation
                if _mt == "FIRE" and d.get("tar_shot"):
                    _tmult *= 2.0                                  # v11 C.2c: Tar Shot Fire ×2
                _abm = _ability_damage_mult(                       # v11 A.2 ability damage multipliers
                    _mid, ability_id, d.get("ability"), eff_move_type=_mt, is_stab=_stab,
                    is_physical=_phys, type_mult=_tmult, hp_frac=d.get("hp_frac") or 0.0,
                    att_burned=bool(_ac.get("burned")), att_statused=bool(_ac.get("statused")),
                    is_spread=_spread,                             # v11 A2: Parental Bond eligibility
                    fainted_allies=_ac.get("fainted_allies", 0),  # v11 A3: Supreme Overlord faint-count
                    defender_intact_disguise=d.get("intact_disguise", False),  # v11 B1: Disguise block
                    weather=_weather)                             # v11 G5/G6: Sand Force / Solar Power
                _sit = _situational_damage_mult(_mt, _phys, _weather, _terrain, d,
                                                _ac.get("burned"), _ac.get("life_orb"), _ac.get("choice"),
                                                hits_def=_hits_def,
                                                grassy_eq=_mid in _GRASSY_WEAKENED) * _abm  # v11 G7
                # v11 B3 attacker item band mults + B3b defender resist berry (parity twin of the offline writer).
                if _ac.get("type_boost") == _mt:
                    _sit *= _BAND_ITEM_MULT
                if _ac.get("expert_belt") and _tmult > 1.0:
                    _sit *= _BAND_ITEM_MULT
                _rb = d.get("resist_berry")
                if _rb == _mt and (_mt == "NORMAL" or _tmult > 1.0):
                    _sit *= 0.5
                _sit *= _crit                                   # v11 N4: expected-crit EV (band crit-blind otherwise)
                _bp = _DMG.variable_base_power(                     # v11 B.2 variable base power
                    _mid, _raw_bp,
                    attacker_weight=_ac.get("weight"), target_weight=d.get("weight"),
                    attacker_hp_frac=_ac.get("hp_frac"),
                    attacker_speed=_ac.get("eff_speed"), target_speed=d.get("eff_speed"),
                    attacker_pos_boosts=_ac.get("pos_boosts", 0), target_pos_boosts=d.get("pos_boosts", 0),
                    fainted_allies=_ac.get("fainted_allies", 0),   # v11 B.2 gap-fix: Last Respects
                    times_hit=_ac.get("times_attacked", 0))        # v11 C.4: Rage Fist
                _Aov = (_ac.get("def") if _mid == "bodypress"      # v11 B.2 offensive-stat override
                        else d.get("atk") if _mid == "foulplay" else _A)
                _wr = bool(_ac.get("wonder_room"))   # v11 P5: Wonder Room swaps the DEFENSIVE stat (Def<->SpD)
                dmin, dmax = _damage_band(
                    _bp, _Aov, (d.get("def") if (_hits_def ^ _wr) else d.get("spd")),
                    d.get("hp"), d.get("hp_frac"), _tmult,
                    _stab, _spread, _sit, hits_min=_hmin, hits_max=_hmax)   # v11 A1: multi-hit
            else:
                dmin, dmax = 0.0, 0.0                # empty slot OR defender-ability immunity → 0 damage
            vec[i] = dmin
            vec[i + 1] = dmax
            i += 2

        # v11 B.1b: priority-aware who-moves-first vs each enemy active (2 channels; parity with offline).
        _att_spd = _ac.get("eff_speed") or 0.0
        _prio = getattr(move, "priority", 0) or 0
        _tr = bool(_ac.get("trick_room"))
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            if d:
                vec[i] = _moves_first(_att_spd, d.get("eff_speed") or 0.0, _tr, prio_delta=_prio)
            i += 1

        # B1.3 move intrinsics (per-move, public, from poke-env Move) — parity with the offline
        # moves.json flags. contact · recoil · drain · multihit-count/5.
        _flags = (getattr(move, "entry", None) or {}).get("flags") or {}
        vec[i] = 1.0 if _flags.get("contact") else 0.0
        vec[i + 1] = 1.0 if getattr(move, "recoil", 0) else 0.0
        vec[i + 2] = 1.0 if getattr(move, "drain", 0) else 0.0
        _nh = getattr(move, "n_hit", None)
        _mh_max = (_nh[1] if isinstance(_nh, (list, tuple)) and len(_nh) > 1 and _nh[1] > 1 else 0)
        vec[i + 3] = min(_mh_max, 5) / 5.0
        i += 4

        # v11 B2b: per-enemy realized HIT CHANCE vs each enemy active (parity twin of the offline writer).
        _ah_b2b = _move_always_hit(_mid)
        _oh_b2b = _move_is_ohko(_mid)
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            if d is None:
                vec[i] = 0.0
            elif _ah_b2b:
                vec[i] = 1.0
            else:
                vec[i] = _per_enemy_hit_chance(
                    _base_acc, attacker_ability=ability_id, acc_stage=_ac.get("acc_stage", 0),
                    wide_lens=_ac.get("wide_lens", False), is_physical=_phys, is_ohko=_oh_b2b,
                    defender=d, move_id=_mid, weather=_weather,
                    victory_star=_ac.get("victory_star", False))   # v11 Victory Star
            i += 1

        # v9: move effect-tags + identity index, BEFORE the trailing is_known
        _mid = getattr(move, "id", None)
        for idx in move_tag_indices(_mid):
            vec[i + idx] = 1.0
        i += NUM_MOVE_TAGS
        vec[i] = float(move_index(_mid))
        i += 1

        # Known flag (always 1 here — zero-slot means unknown)
        vec[i] = 1.0
        i += 1


# ══════════════════════════════════════════════════════════════════════════════
# Own brought-team / bench resolution  (SINGLE SOURCE OF TRUTH)
#
# VGC team-preview brings 4 of the 6 roster mons.  poke-env still keeps all 6 in
# ``battle.team`` (it knows the full roster from the team paste), but the 2
# un-brought mons can NEVER enter play — they are never active, never fainted,
# never revealed, and never appear in ``available_switches``.  Listing them as
# bench / switch targets has two bad effects:
#   1. the live own bench desyncs from the OFFLINE parser (which only ever sees
#      the brought 4) → a train/serve state mismatch, and
#   2. the action codec emits a switch to an un-brought mon → Showdown rejects
#      the order ("you do not have a Pokémon named X to switch to") → the live
#      player thrashes through retry/random fallbacks every turn.
#
# These helpers define the brought set ONCE so the encoder bench, the team-count
# globals, the legal-action mask, and ``action_to_order`` all agree on which mon
# occupies bench slot i / switch action 12+i (stable team order).
# ══════════════════════════════════════════════════════════════════════════════
def _active_mon_set(battle) -> set:
    """The set of currently-active own mons (handles poke-env's ValueError when
    the active list is momentarily unavailable)."""
    try:
        actives = battle.active_pokemon
    except (ValueError, AttributeError):
        actives = []
    return {m for m in (actives or []) if m is not None}


def switchable_union(battle) -> set:
    """Every mon Showdown currently lists as a legal switch-in across both active
    slots (``battle.available_switches`` is List[List[Pokemon]] per active slot)."""
    out: set = set()
    for slot_sw in (getattr(battle, "available_switches", None) or []):
        for m in (slot_sw or []):
            out.add(m)
    return out


def _is_brought(mon, active_set: set, switch_set: set) -> bool:
    """Whether ``mon`` is one of the BROUGHT 4-of-6.  A brought mon is (or has
    been) in play or is currently switchable; the un-brought roster mons are
    never any of these."""
    return (
        mon in active_set
        or bool(getattr(mon, "fainted", False))
        or bool(getattr(mon, "revealed", False))
        or mon in switch_set
    )


# ── Request-authoritative own-side state (#11b) ───────────────────────────────
# poke-env derives a mon's active/fainted from per-mon flags + a species/name-keyed
# object model, which a same-species Illusion (Zoroark disguised as a brought
# teammate) corrupts — dropping a healthy bench mon from available_switches (and so
# from the flag-based bench below).  Showdown's in-battle |request|, by contrast, is
# AUTHORITATIVE and uncorruptible: at team preview Showdown rebuilds side.pokemon to
# the picked mons only (sim/battle.ts 'team' action), so an in-battle request lists
# EXACTLY the brought team, each with its true ident + active flag + condition.  We
# read that directly, immune to the object-model desync.  poke-env still keeps all 6
# in battle.team (from the team-preview request), keyed by the same ident, so we map
# the request idents back to objects for order construction.
def _request_own_state(battle) -> Optional[Dict[str, tuple]]:
    """``{ident: (active: bool, fainted: bool)}`` for the OWN side from the live
    in-battle ``battle.last_request`` (brought-only, authoritative).  None when no
    usable request exists (e.g. the replay-driven parity path) so callers fall back
    to the flag-based heuristic and behaviour is unchanged offline."""
    req = getattr(battle, "last_request", None) or {}
    if not isinstance(req, dict) or req.get("teamPreview"):
        return None     # team-preview request lists all 6, not the brought team
    side = req.get("side") or {}
    pokemon = side.get("pokemon") if isinstance(side, dict) else None
    if not pokemon:
        return None
    out: Dict[str, tuple] = {}
    for e in pokemon:
        ident = e.get("ident")
        if not ident:
            continue
        cond = str(e.get("condition") or "").strip()
        # condition is "cur/max[ status]" or "0 fnt"; fainted iff it ends "fnt".
        out[ident] = (bool(e.get("active")), cond.endswith("fnt"))
    return out or None


def brought_team_mons(battle) -> list:
    """The BROUGHT own mons (~4 of 6) in stable team order, real mons only
    (broken-illusion phantoms with max_hp==0 excluded).

    Brought membership is taken from the authoritative in-battle |request| (which
    lists ONLY the brought team) when one is available — immune to the Illusion
    flag corruption — falling back to the flag/reveal heuristic offline."""
    req = _request_own_state(battle)
    if req is not None:
        return [m for ident, m in battle.team.items()
                if ident in req and LiveStateEncoder._is_real_mon(m)]
    active_set = _active_mon_set(battle)
    switch_set = switchable_union(battle)
    return [
        m for m in battle.team.values()
        if LiveStateEncoder._is_real_mon(m) and _is_brought(m, active_set, switch_set)
    ]


def own_bench_mons(battle) -> list:
    """Brought, non-active, non-fainted own mons — the mons that occupy encoder
    bench slots 4..7 and switch action indices 12..15 (stable team order; the
    un-brought 2-of-6 are excluded so a switch action always decodes to a mon
    Showdown will accept).

    The living/active determination is taken from the authoritative in-battle
    |request| when available (#11b — so an Illusion-corrupted active/fainted FLAG
    can no longer drop a healthy bench mon from the replacement set), falling back
    to poke-env's flags only when no request exists (replay-driven path)."""
    active_set = _active_mon_set(battle)
    req = _request_own_state(battle)
    if req is not None:
        # Cross-check the request's `active` flag against the LIVE field
        # (battle.active_pokemon): under a mid-turn-faint Illusion the request can
        # momentarily report an on-field mon as inactive, and offering it as a
        # switch-in makes Showdown reject the order ("can't switch to an active
        # Pokémon") → the forced-switch loop escape.  A mon whose OBJECT is on the
        # field is never a legal switch-in, so exclude it regardless of the flag.
        # This only ever drops a genuinely-active mon (active_pokemon is the field
        # truth, not the per-mon flag / available_switches the #11 desync corrupts),
        # so it can't re-drop a healthy bencher.
        return [m for ident, m in battle.team.items()
                if ident in req and not req[ident][0] and not req[ident][1]
                and m not in active_set
                and LiveStateEncoder._is_real_mon(m)]
    return [
        m for m in brought_team_mons(battle)
        if m not in active_set and not getattr(m, "fainted", False)
    ]


def own_switch_slot(battle, mon) -> Optional[int]:
    """Showdown's 1-based switch SLOT for ``mon`` — its position in the request's
    ``side.pokemon`` — so a switch can be ordered by SLOT (``/choose switch 3``)
    instead of by species name (``/choose switch Charizard``).

    THE root fix for the 'can't switch to an active Pokémon' rejection (#1b): under
    an Illusion (Zoroark) or Imposter (Ditto), an ACTIVE mon can be DISPLAYED as the
    same species as a benched mon; Showdown then resolves a name-based switch to the
    ACTIVE disguise and rejects it.  The team-slot index is assigned at team preview
    and never changes, so it identifies the exact (benched) team member regardless of
    what any mon is currently disguised/transformed as — AND it is immune to a stale
    ``last_request`` (the order of side.pokemon is fixed; only the active/condition
    fields churn).  Returns None when there is no usable request (offline / replay
    path) or the mon isn't found, so the caller falls back to the name-based order
    (unchanged offline behaviour)."""
    req = getattr(battle, "last_request", None) or {}
    if not isinstance(req, dict) or req.get("teamPreview"):
        return None
    side = req.get("side") or {}
    pokemon = side.get("pokemon") if isinstance(side, dict) else None
    if not pokemon:
        return None
    team = getattr(battle, "team", {}) or {}
    target_ident = next((ident for ident, m in team.items() if m is mon), None)
    if target_ident is None:
        return None
    for i, e in enumerate(pokemon):
        if e.get("ident") == target_ident:
            return i + 1
    return None


def own_active_move_list(battle, slot: int, mon) -> list:
    """The own ACTIVE mon's move list (``Move`` objects, <= NUM_MOVES) shared by the
    encoder, the action mask, and the order codec — normally ``mon.moves``, but
    request-authoritative under an own-side Illusion (#1b).

    poke-env attributes a disguise's |move| lines to the DISGUISE object, so a freshly
    sent-in Zoroark (disguised as a brought teammate) that hasn't moved yet leaves the
    active object's ``moves`` EMPTY: the encoder then writes zero move features, the
    action mask offers no move, and — if the slot also can't switch (trapped / no
    bench) — the turn collapses to ``/choose default`` (a wasted turn).  Showdown's
    private |request| always lists the TRUE active mon's usable moves, which poke-env
    exposes as ``battle.available_moves[slot]`` (already ``Move`` objects).  When
    ``mon.moves`` is empty we fall back to it so all three codec sites see the SAME
    real moves in ONE order — matching the offline parser, which reconstructs the true
    mon under illusion (so this moves the LIVE encoding TOWARD offline parity; the
    replay-driven path has no live ``available_moves`` and is unaffected).  Empty
    ``mon.moves`` is the only trigger, so every normal turn is byte-identical to the
    prior behaviour."""
    moves = list(mon.moves.values())[:NUM_MOVES] if mon is not None else []
    if moves:
        return moves
    try:
        av = battle.available_moves
    except (ValueError, AttributeError):
        return moves
    if (isinstance(av, (list, tuple)) and slot is not None
            and slot < len(av) and av[slot]):
        return list(av[slot])[:NUM_MOVES]
    return moves


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
    from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser

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


def opp_snapshot_current(log: str, own_role: str) -> Optional[dict]:
    """Reconstruct the OPPONENT side as of the END of the given log (the CURRENT
    board), for MID-TURN decisions where the start-of-turn snapshot is stale.

    A forced replacement (poke-env ``forceSwitch``) is decided AFTER a faint, in
    the middle of a turn — there is no new ``|turn|`` line, so
    ``opp_snapshot_from_log_prefix`` would return the pre-faint start-of-turn
    board.  This instead parses the WHOLE log-so-far (which a live bot has
    captured up to, but not including, its own pending replacement) and returns
    ``_snapshot_state`` — the post-faint board.  That is exactly the moment the
    OFFLINE parser captures a ``replacement`` decision's state (in
    ``_handle_switch``, just before the replacement switch mutates the slot), so
    the live replacement decision sees the same opponent composition training did.

    Works for turn boundaries too (there the end-of-log board == the start-of-turn
    board), so it is the general form of the prefix helper.  Returns None on
    failure / empty log.
    """
    from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser

    if not (log or "").strip():
        return None
    try:
        parser = ShowdownReplayParser(log, our_player=own_role)
        parser.parse()
        return parser._snapshot_state(own_role)
    except Exception:
        return None
