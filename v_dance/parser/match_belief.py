"""
match_belief.py — within-game opponent belief tracker (Level C, Component A1).
================================================================================

``MatchBelief`` wraps a static :class:`~v_dance.parser.belief_state.BeliefState`
(Pikalytics prior) and maintains a PERSISTENT, per-opponent-mon belief that is
narrowed by what the game reveals — the within-game analogue of the static prior.

It is the foundation of Level C's "reads the matchup" half (see
``docs/levelC_belief_search_design.md``).  A1 (this file) implements three things
the static per-turn ``belief_block`` does NOT:

  1. **Authoritative accumulation** — reveals (moves/item/ability/choice-lock) are
     accumulated across turns AND across switch-outs, keyed by mon identity, so a
     mon that leaves and returns keeps its narrowed belief.  (The offline
     ``fill_blanks`` path already rebuilds belief per-snapshot from the parser's
     cumulative ``revealed_moves``; ``MatchBelief`` makes that accumulation an
     explicit, live, testable object instead of a side effect of snapshot order.)

  2. **Move-category spread narrowing (NEW)** — observing the opponent use a
     STANDARD damaging move tells us which offensive stat it invests in, so EV
     spreads whose nature DROPS that stat become implausible and are down-weighted
     (never zeroed).  This sharpens ``expected_stats`` (which feed the encoder's
     est-stat block and, later, the search's damage bands).  It is deliberately
     CONSERVATIVE: moves whose damage does NOT scale with the user's nominal
     offensive stat (Body Press → Def, Foul Play → target Atk, fixed-damage,
     redirect) reveal NOTHING about the spread and are excluded — a naive
     "physical ⇒ Attack" rule mislabels e.g. a Body-Press Corviknight.

  3. **Drop-in enrichment** — :meth:`enrich_mon` mirrors
     ``belief_state._enrich_mon_belief`` exactly (same mega/lookup resolution,
     same ``mon['belief']`` / ``mon['stats_estimate']`` outputs) so wiring it into
     the live encoder in A3 is a one-line swap.

NOT in A1 (deferred, by design):
  * **A2** — damage-roll spread narrowing (invert ``_damage_band`` over candidate
    spreads from observed % damage; finally implements ``back_calculate_evs``).
  * **A3** — wiring into ``live_state_encoder`` + ``opp_action_prior`` for search.

This module is import-light and has NO side effects at import time.
"""
from __future__ import annotations

import math
from typing import Optional

from v_dance.parser.belief_state import (
    BeliefState,
    NATURE_BOOSTS,
    STAT_ORDER,
    calc_full_stats,
    dex_base_stats,
)
from v_dance.parser.vod_parser.pokedex import norm_species

# ── Tunables ────────────────────────────────────────────────────────────────
# Down-weight factor applied to an EV spread whose nature contradicts the
# observed offensive category.  STRICTLY > 0 so a spread is never eliminated
# (the conservative "never zero the truth" invariant): if EVERY spread is
# penalised equally the renormalised distribution is unchanged.
_CATEGORY_PENALTY = 0.15

# Revealed-item → spread COHERENCE prior (Level C, the 1 deferred B1-audit item; design
# docs/levelC_item_spread_reconditioning_design.md).  Pikalytics exposes only the MARGINAL item & spread
# (no joint P(spread|item)), so a usage-correlation heuristic would be speculative and could HURT.  Instead
# we exploit a DETERMINISTIC fact that needs no joint data: a Choice item's ×1.5 boost makes a nature that
# LOWERS that very stat competitively incoherent (you don't run a −Spe Choice Scarf, a −Atk Choice Band, or a
# −SpA Choice Specs).  We down-weight ONLY those incoherent spreads — the safe INVERSE of guessing a
# correlation up.  Renormalisation then lifts the survivors proportionally (no claim about which survivor is
# likelier).  The penalty is slightly stronger than _CATEGORY_PENALTY: a held item is more direct evidence
# than a single observed move.  Items that merely imply a DIRECTION (Assault Vest / Eviolite / Life Orb /
# type-boost) contradict no nature → deliberately EXCLUDED (their damage effect is already in A2).
_ITEM_NATURE_PENALTY = 0.12
_ITEM_BOOST_STAT = {                        # norm_species(item) → the stat the item boosts (== the stat a
    "choicescarf": "spe",                   # coherent spread must NOT lower
    "choiceband": "atk",
    "choicespecs": "spa",
}

# Paradox stat-reveal (Protosynthesis / Quark Drive, incl. Booster Energy): when the boost activates the
# boosted stat IS the mon's HIGHEST non-HP stat (Atk>Def>SpA>SpD>Spe tiebreak per the mechanic). A candidate
# spread whose argmax stat differs from the revealed one is (near-)impossible → strong down-weight. This is a
# RANK reveal (which stat is highest), orthogonal to the magnitude signals (damage/speed). Floored, never zeroed.
_BOOST_STAT_PENALTY = 0.10
_STAT_TIEBREAK = ("atk", "def", "spa", "spd", "spe")   # the paradox highest-stat tiebreak order

# A2 damage-roll narrowing (fully-fleshed): a per-spread Bayesian likelihood that the observed damage % came
# from that spread, modelled LOG-NORMAL around the spread's PREDICTED mean damage. The log-space stdev σ encodes
# the prediction uncertainty — and SHRINKS as the caller pins down context, so the SAME code narrows loosely
# offline (crit/mitigation unknown) and sharply live (poke-env exact calc + revealed item/ability/crit):
#   σ² = roll² + misc² + (crit² if crit unknown) + (mitigation² if defender item/ability unknown).
# Spreads are weighted by likelihood (floored → never zeroed), multiplied across events (Bayesian product).
_SIGMA_ROLL = 0.05     # the 0.85–1.0 damage roll, in log space
_SIGMA_CRIT = 0.08     # an unknown crit (~1/24 chance of x1.5)
_SIGMA_MITIG = 0.20    # an unknown defender resist-berry / Multiscale / screen (~x0.5)
_SIGMA_MISC = 0.05     # stat rounding, boosts, multi-target, situational slop
_SIGMA_LOG_DEFAULT = (_SIGMA_ROLL ** 2 + _SIGMA_CRIT ** 2
                      + _SIGMA_MITIG ** 2 + _SIGMA_MISC ** 2) ** 0.5   # ≈0.23 (offline, context unknown)
_LIKELIHOOD_FLOOR = 1e-3   # per-event likelihood floor (numerical safety over many events)
_P_FLOOR = 1e-3            # final per-spread probability floor (never zero the truth)

# A3 speed-tier narrowing: default logistic resolution (fraction of the base-Spe threshold) when the caller
# passes no explicit σ — a candidate within ~this fraction of the threshold is treated as a speed-tie ambiguity.
_SPEED_SIGMA_REL = 0.06


# Moves that are Physical/Special CATEGORY but whose damage does NOT scale with
# the user's nominal offensive stat — so observing them reveals nothing about the
# Atk-vs-SpA investment or nature.  Excluding them keeps the category narrowing
# conservative (a Body-Press / Foul-Play / Seismic-Toss user must NOT be pushed
# toward an Atk-boosting nature).  Note: Psyshock-class (Special, hits Def) DO
# scale with the user's SpA, so they are intentionally NOT here.
_STAT_INDEPENDENT_MOVES = frozenset({
    "bodypress",                         # uses Defense
    "foulplay",                          # uses the TARGET's Attack
    "photongeyser", "lightthatburnsthesky",  # higher of Atk/SpA — ambiguous
    # fixed / level / fractional damage (category varies, no stat scaling):
    "seismictoss", "nightshade", "psywave", "superfang", "naturesmadness",
    "ruination", "dragonrage", "sonicboom", "endeavor", "finalgambit",
    "guardianofalola",
    # damage-redirect counters (scale off damage taken, not the user's stat):
    "counter", "mirrorcoat", "metalburst",
})


def _move_category(move_name: Optional[str]) -> Optional[str]:
    """Lower-cased move category ('physical'|'special'|'status') from the pinned
    Gen-9 moves data, or None when the move is unknown.  Lazy import keeps the
    belief import path cheap (mirrors ``belief_state.is_mega_stone``)."""
    if not move_name:
        return None
    try:
        from v_dance.encoders.battle_mechanics import _gen9_moves
        data = _gen9_moves().get(norm_species(move_name)) or {}
    except Exception:  # pragma: no cover - data unavailable
        return None
    cat = (data.get("category") or "").lower()
    return cat or None


def _offensive_stat_for_move(move_name: Optional[str]) -> Optional[str]:
    """The user's offensive stat a STANDARD damaging move scales with:
    ``'atk'`` for a physical move, ``'spa'`` for a special move, else ``None``
    (status, or a stat-independent move in ``_STAT_INDEPENDENT_MOVES``)."""
    if not move_name:
        return None
    if norm_species(move_name) in _STAT_INDEPENDENT_MOVES:
        return None
    cat = _move_category(move_name)
    if cat == "physical":
        return "atk"
    if cat == "special":
        return "spa"
    return None


def identity_reliable(mon: Optional[dict]) -> bool:
    """False when a mon's APPARENT species/stats cannot be trusted for belief narrowing — a Zoroark Illusion
    (``illusion_active`` / ``disguise_species``) or a Transform / Imposter copy (``is_transformed`` /
    ``transformed_into``, incl. a Ditto that copied YOUR mon). The live feeder MUST skip every observe_* — and
    ``ingest_mon`` self-skips — for such a mon: its moves / damage / speed reflect the disguise or the copied
    mon, not its own EV spread, so attributing them to the apparent species would corrupt belief."""
    if not mon:
        return False
    if mon.get("illusion_active") or mon.get("disguise_species"):
        return False
    if mon.get("is_transformed") or mon.get("transformed_into"):
        return False
    return True


def _spread_key(spread: dict):
    """Stable identity for an EV spread (nature + actual EVs) — used to align externally-computed
    per-spread damage log-likelihoods (the A2 v2 live poke-env path) with the belief block's spreads
    across re-sorts."""
    return (spread.get("nature"), tuple(spread.get("evs_actual") or ()))


def damage_main_term(base_power, attacker_stat, defender_stat) -> float:
    """The +2-free, multiplier-free, roll-free MAIN damage term (HP points) of the L50 Gen-9 formula:
    ``22·bp·A/(D·50)`` (the part of ``_damage_band``'s ``base`` BEFORE the additive ``+2``). Pass its value at
    the REFERENCE to ``observe_damage_*`` (def: A=our attacker stat, D=ref defender stat; off: A=ref attacker
    stat, D=our defender stat) so the additive +2 is handled EXACTLY per candidate — damage is affine in A and
    in 1/D, NOT purely proportional, so scaling the whole prediction biases the +2. Returns 0.0 on bad input."""
    if not base_power or not attacker_stat or not defender_stat:
        return 0.0
    return (22.0 * float(base_power) * float(attacker_stat)) / (float(defender_stat) * 50.0)


# ── A3 speed-tier turn-order analyzer ─────────────────────────────────────────────
def _effective_priority(base_priority, category, move_type, ability, hp_frac) -> int:
    """Effective priority BRACKET = the move's base priority + the ability bumps that re-order WITHIN brackets:
    Prankster (+1 to STATUS moves) and Gale Wings (+1 to FLYING moves at FULL HP). (Triage / Mycelium Might /
    Stall etc. are omitted — ~0 prevalence in Champions; add here if needed.) Negative-priority moves need no
    special handling — their negative base priority IS their bracket."""
    p = int(base_priority or 0)
    a = norm_species(ability or "")
    cat = (category or "").lower()
    mt = (move_type or "").lower()
    if a == "prankster" and cat == "status":
        p += 1
    elif a == "galewings" and mt == "flying" and (hp_frac is None or hp_frac >= 1.0):
        p += 1
    return p


def analyze_speed_order(
    *,
    opp_moved_first: bool,
    our_eff_speed: Optional[float],
    our_move: Optional[dict],
    opp_move: Optional[dict],
    our_ability: Optional[str] = None,
    opp_ability: Optional[str] = None,
    our_hp_frac: Optional[float] = None,
    opp_hp_frac: Optional[float] = None,
    trick_room: bool = False,
    order_forced: bool = False,
    opp_speed_mult_known: float = 1.0,
    opp_context_known: bool = True,
) -> Optional[dict]:
    """Turn-order → a speed-tier observation (the kwargs for ``MatchBelief.observe_speed_bound``), or ``None``
    when the order does NOT reveal speed. Encodes the full priority physics: move-priority brackets, Prankster /
    Gale Wings bumps, Trick Room (flips the read), and forced/extra moves the caller flags via ``order_forced``
    — Quash / After You / Quick-Claw, and **Instruct / Dancer** (a forced REPEAT move is out of speed order, so
    the feeder sets ``order_forced`` for that turn). Identity (Zoroark/Ditto) is gated UPSTREAM by the feeder via
    :func:`identity_reliable` — do not feed a disguised/transformed mon's order here. ``our_move`` / ``opp_move`` = ``{priority, category,
    type}``. ``our_eff_speed`` = our mon's EXACT effective speed; ``opp_speed_mult_known`` = product of the opp's
    KNOWN speed multipliers (spe-boost stage, Tailwind, paralysis, a REVEALED Scarf, weather-ability) — the
    threshold is divided by it to express the bound in the opp's BASE-Spe units. ``opp_context_known`` False
    (item/ability unrevealed → a possible hidden Scarf) widens σ. Caller must pass MOVE-vs-MOVE turns only."""
    if (order_forced or not our_eff_speed or our_eff_speed <= 0
            or not our_move or not opp_move):
        return None
    eff_our = _effective_priority(our_move.get("priority"), our_move.get("category"),
                                  our_move.get("type"), our_ability, our_hp_frac)
    eff_opp = _effective_priority(opp_move.get("priority"), opp_move.get("category"),
                                  opp_move.get("type"), opp_ability, opp_hp_frac)
    if eff_our != eff_opp:
        return None                              # a priority bracket decided the order → no speed signal
    mult = opp_speed_mult_known if (opp_speed_mult_known and opp_speed_mult_known > 0) else 1.0
    threshold = our_eff_speed / mult
    faster = bool(opp_moved_first) != bool(trick_room)    # Trick Room reverses: moving first under TR ⇒ SLOWER
    sigma = threshold * (0.20 if not opp_context_known else _SPEED_SIGMA_REL)   # widen for a possible hidden Scarf
    return {"threshold_base_spe": threshold, "faster": faster, "sigma_spe": sigma}


class _MonObservations:
    """Accumulated within-game reveals for ONE opponent mon line."""

    __slots__ = ("revealed_moves", "_revealed_norm", "known_item", "known_ability",
                 "can_have_choice_item", "known_boosted_stat", "damage_constraints",
                 "speed_constraints", "_seen_keys")

    def __init__(self) -> None:
        self.revealed_moves: list[str] = []   # display names, insertion order, dedup
        self._revealed_norm: set[str] = set()
        self.known_item: Optional[str] = None
        self.known_ability: Optional[str] = None
        self.can_have_choice_item: Optional[bool] = None
        self.known_boosted_stat: Optional[str] = None   # paradox highest-stat reveal (atk/def/spa/spd/spe)
        self.damage_constraints: list[dict] = []   # A2: defensive damage-roll constraints
        self.speed_constraints: list[dict] = []    # A3: speed-tier bounds (base-Spe threshold + direction)
        self._seen_keys: set = set()               # internal idempotency: event keys already folded in

    def fresh_event(self, key: Optional[str]) -> bool:
        """True if ``key`` has NOT been folded in yet (and records it). None ⇒ always fresh (no dedup).
        Defence-in-depth against a re-fed observation double-counting the Bayesian product (audit 2026-06-30)."""
        if key is None:
            return True
        if key in self._seen_keys:
            return False
        self._seen_keys.add(key)
        return True

    def add_move(self, move: Optional[str]) -> None:
        if not move:
            return
        key = norm_species(move)
        if key and key not in self._revealed_norm:
            self._revealed_norm.add(key)
            self.revealed_moves.append(move)

    def offensive_categories(self) -> set[str]:
        """Subset of {'atk','spa'} the mon has demonstrably invested in
        (from STANDARD damaging moves only)."""
        cats: set[str] = set()
        for mv in self.revealed_moves:
            st = _offensive_stat_for_move(mv)
            if st:
                cats.add(st)
        return cats


class MatchBelief:
    """Within-game opponent belief: a static :class:`BeliefState` prior that is
    narrowed by accumulated reveals.  One instance per battle (call :meth:`reset`
    between games unless deliberately carrying belief over, e.g. Bo3).

    The public surface mirrors the belief_state functions it supersedes:
      * :meth:`observe_move` / :meth:`observe_item` / :meth:`observe_ability` /
        :meth:`observe_choice_constraint` — feed reveals in;
      * :meth:`ingest_mon` — pull reveals off a parsed/live snapshot mon dict;
      * :meth:`block_for` — the narrowed belief block (``belief_block`` shape);
      * :meth:`enrich_mon` — drop-in for ``belief_state._enrich_mon_belief``.
    """

    def __init__(self, belief: BeliefState, *, top_k: int = 5, level: int = 50):
        self._belief = belief
        self._top_k = top_k
        self._level = level
        self._mons: dict[str, _MonObservations] = {}

    # ── identity ──────────────────────────────────────────────────────────────
    @staticmethod
    def _key(species: Optional[str]) -> str:
        """Identity key for a mon line — base species, forme/mega-insensitive
        (Charizard-Mega-Y and Charizard share belief, like the Pikalytics prior)."""
        return norm_species(species or "")

    def _obs(self, species: Optional[str]) -> _MonObservations:
        k = self._key(species)
        mb = self._mons.get(k)
        if mb is None:
            mb = self._mons.setdefault(k, _MonObservations())
        return mb

    # ── observation API ─────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Forget all accumulated reveals (new battle)."""
        self._mons.clear()

    def observe_move(self, species: Optional[str], move: Optional[str]) -> None:
        self._obs(species).add_move(move)

    def observe_item(self, species: Optional[str], item: Optional[str]) -> None:
        if item:
            self._obs(species).known_item = item

    def observe_ability(self, species: Optional[str], ability: Optional[str]) -> None:
        if ability:
            self._obs(species).known_ability = ability

    def observe_boosted_stat(self, species: Optional[str], stat: Optional[str]) -> None:
        """Record a Protosynthesis / Quark-Drive boost reveal — the mon's HIGHEST non-HP stat. Monotone: the
        spread doesn't change mid-game, so once seen it stays. No-op for a missing/invalid stat."""
        if stat in _STAT_TIEBREAK:
            self._obs(species).known_boosted_stat = stat

    def observe_choice_constraint(self, species: Optional[str],
                                  can_have_choice_item: Optional[bool]) -> None:
        """Replay-derived Choice constraint (False = used 2 moves in one stay →
        not a Choice item).  Monotone: once proven False it stays False."""
        if can_have_choice_item is None:
            return
        mb = self._obs(species)
        if can_have_choice_item is False or mb.can_have_choice_item is None:
            mb.can_have_choice_item = can_have_choice_item

    def observe_damage_taken(
        self,
        species: Optional[str],
        *,
        mu_ref: Optional[float],
        ref_main: Optional[float],
        ref_def_stat: Optional[float],
        ref_hp: Optional[float],
        category: Optional[str],
        obs_frac_of_max: Optional[float],
        sigma_log: float = _SIGMA_LOG_DEFAULT,
        was_ko: bool = False,
        event_key: Optional[str] = None,
    ) -> None:
        """DEFENSIVE constraint (A2): the opp mon (`species`) TOOK ``obs_frac_of_max`` (= |hp_pct_delta|/100
        of its MAX hp) from a ``category`` ('physical'|'special') move whose attacker is KNOWN (ours). The
        CALLER predicts the MEAN damage fraction ``mu_ref`` against a REFERENCE defender with relevant
        defensive stat ``ref_def_stat`` (Def/physical, SpD/special) and max hp ``ref_hp``, plus ``ref_main`` =
        ``damage_main_term`` at the reference (so the additive +2 is handled exactly per candidate). Crit + known
        mitigation fold into ``mu_ref``; residual uncertainty into ``sigma_log``. MatchBelief scales the
        prediction to each candidate spread (main ∝ 1/Def, +2 constant; % ∝ 1/hp) and weights it by a log-normal
        likelihood. Live caller → poke-env ``calculate_damage``; offline → ``_damage_band``. FULL-HP NON-KO only."""
        if (category not in ("physical", "special") or not ref_def_stat or not ref_hp
                or not mu_ref or mu_ref <= 0 or ref_main is None or ref_main < 0
                or not obs_frac_of_max or obs_frac_of_max <= 0
                or not sigma_log or sigma_log <= 0
                or was_ko):                       # a KO censors the damage (obs is only a lower bound) — drop it
            return
        obs = self._obs(species)
        if not obs.fresh_event(event_key):        # idempotency: never fold the same event twice
            return
        obs.damage_constraints.append({
            "mode": "def",
            "stat": "def" if category == "physical" else "spd",
            "ref_stat": float(ref_def_stat),
            "ref_hp": float(ref_hp),
            "ref_main": float(ref_main),
            "mu_ref": float(mu_ref),
            "obs": float(obs_frac_of_max),
            "sigma": float(sigma_log),
        })

    def observe_damage_dealt(
        self,
        species: Optional[str],
        *,
        mu_ref: Optional[float],
        ref_main: Optional[float],
        ref_atk_stat: Optional[float],
        category: Optional[str],
        obs_frac_of_max: Optional[float],
        sigma_log: float = _SIGMA_LOG_DEFAULT,
        was_ko: bool = False,
        event_key: Optional[str] = None,
    ) -> None:
        """OFFENSIVE constraint (A2 v2): the opp mon (`species`) DEALT ``obs_frac_of_max`` (of OUR mon's MAX
        hp) with a ``category`` move whose defender is KNOWN (ours). The CALLER predicts the MEAN damage
        fraction ``mu_ref`` against OUR known defender for a REFERENCE attacker with relevant offensive stat
        ``ref_atk_stat`` (Atk/physical, SpA/special), plus ``ref_main`` = ``damage_main_term`` at the reference.
        MatchBelief scales it per candidate spread (main ∝ Atk, +2 constant; OUR hp cancels) and weights by a
        log-normal likelihood → narrows the opp's OFFENSIVE investment. Same gating as taken."""
        if (category not in ("physical", "special") or not ref_atk_stat
                or not mu_ref or mu_ref <= 0 or ref_main is None or ref_main < 0
                or not obs_frac_of_max or obs_frac_of_max <= 0
                or not sigma_log or sigma_log <= 0
                or was_ko):                       # a KO censors OUR mon's damage taken (lower bound only) — drop
            return
        obs = self._obs(species)
        if not obs.fresh_event(event_key):        # idempotency: never fold the same event twice
            return
        obs.damage_constraints.append({
            "mode": "off",
            "stat": "atk" if category == "physical" else "spa",
            "ref_stat": float(ref_atk_stat),
            "ref_main": float(ref_main),
            "mu_ref": float(mu_ref),
            "obs": float(obs_frac_of_max),
            "sigma": float(sigma_log),
        })

    def observe_damage_loglik(self, species: Optional[str], loglik_by_key: Optional[dict],
                              *, direction: str = "off", stat: Optional[str] = None) -> None:
        """Record a PRECOMPUTED per-spread LOG-likelihood (A2 v2): an external calculator (the live
        poke-env path, which marginalises over item hypotheses) computed ``ln L(spread)`` for each
        candidate spread of an observed damage event, keyed by :func:`_spread_key`. MatchBelief just
        multiplies these into the spread distribution at ``block_for`` time — staying calculator-agnostic
        (it never imports poke-env). No-op for an empty map.

        ``direction`` ('off' = opp DEALT it, narrows opp Atk/SpA; 'def' = opp TOOK it, narrows opp
        Def/SpD/HP) and ``stat`` (the offensive stat 'atk'/'spa' an 'off' event pins; None for 'def' or a
        stat-independent move) TAG what the loglik constrains. The move-category / item nature-coherence
        guards defer ONLY to an OFFENSIVE loglik on the SAME stat: a DEFENSIVE loglik weights spreads by
        Def/SpD/HP and says NOTHING about Atk/SpA, so it must not cancel that narrowing (audit 2026-06-30)."""
        if not loglik_by_key:
            return
        self._obs(species).damage_constraints.append(
            {"mode": "loglik", "direction": direction, "stat": stat, "loglik": dict(loglik_by_key)})

    def observe_speed_bound(
        self,
        species: Optional[str],
        *,
        threshold_base_spe: Optional[float],
        faster: bool,
        sigma_spe: Optional[float] = None,
        event_key: Optional[str] = None,
    ) -> None:
        """Record a SPEED-TIER constraint (A3): the opp mon (`species`) moved FASTER (``faster=True``) or
        SLOWER (``faster=False``) than OUR mon in the SAME effective priority bracket. ``threshold_base_spe``
        = our mon's EFFECTIVE speed divided by the opp's KNOWN speed multipliers (spe-boost stage, Tailwind,
        paralysis, a REVEALED Choice Scarf, weather-speed ability) → expressed in the opp's BASE-Spe units so
        it compares directly to each candidate spread's Spe. ``sigma_spe`` = absolute spe-point uncertainty
        (speed ties, stage rounding, UNKNOWN multipliers such as a possible un-revealed Scarf — pass a WIDER σ
        when the opp's item/ability is unrevealed); defaults to ``_SPEED_SIGMA_REL·threshold``.

        ⚠ The CALLER (the live turn-order analyzer) owns ALL the priority physics — it must call this ONLY when
        the order was SPEED-determined: same effective bracket after accounting for move priority, Prankster /
        Gale Wings (+1 to status / Flying), Quash / After You, negative-priority, and Trick Room (which the
        caller folds by FLIPPING ``faster``). When the brackets differ, the order says nothing about speed → do
        not call. MatchBelief then narrows spreads by a soft logistic on each candidate's Spe."""
        if not threshold_base_spe or threshold_base_spe <= 0:
            return
        # absolute floor of 2.0 base-Spe pts applies to a CALLER-supplied σ too (else slow mons get an
        # over-sharp bound from an analyzer σ that rounds tiny) — audit 2026-06-30.
        sig = (max(2.0, sigma_spe) if (sigma_spe and sigma_spe > 0)
               else max(2.0, _SPEED_SIGMA_REL * threshold_base_spe))
        obs = self._obs(species)
        if not obs.fresh_event(event_key):        # idempotency: never fold the same order-event twice
            return
        obs.speed_constraints.append({
            "threshold": float(threshold_base_spe),
            "faster": bool(faster),
            "sigma": float(sig),
        })

    def observe_speed_order(self, species: Optional[str], **ctx) -> None:
        """Convenience for the live wiring: run :func:`analyze_speed_order` on the turn context and, if it
        yields a speed observation (same effective bracket, MOVE-vs-MOVE, not order-forced), record it for
        ``species`` via :meth:`observe_speed_bound`. A no-op when the order does not reveal speed."""
        obs = analyze_speed_order(**ctx)
        if obs:
            self.observe_speed_bound(species, **obs)

    def ingest_mon(self, mon: Optional[dict]) -> None:
        """Accumulate every reveal carried by a snapshot mon dict (parser output
        or a live reconstruction).  Idempotent — re-ingesting the same snapshot
        adds nothing new."""
        if not mon or not identity_reliable(mon):
            return                       # skip a Zoroark Illusion / Ditto-Transform — see identity_reliable
        species = mon.get("base_species") or mon.get("species")
        for mv in mon.get("revealed_moves") or []:
            self.observe_move(species, mv)
        for mv in mon.get("known_moves") or []:   # exact/injected moves count as revealed
            self.observe_move(species, mv)
        # a CONSUMED item is no longer HELD → don't collapse the held-item belief to it (audit 2026-06-30)
        self.observe_item(species, mon.get("known_item") if not mon.get("item_consumed") else None)
        self.observe_ability(species, mon.get("known_ability"))
        self.observe_choice_constraint(species, mon.get("can_have_choice_item"))
        # paradox highest-stat reveal (Protosynthesis/Quark Drive) — a transient volatile the accumulating
        # belief keeps permanently once observed (the spread is fixed for the game).
        self.observe_boosted_stat(species, (mon.get("volatiles") or {}).get("paradox_boosted_stat"))

    # ── belief block ─────────────────────────────────────────────────────────────
    def offensive_categories(self, species: Optional[str]) -> set[str]:
        """Public read of the accumulated offensive-stat signal for a mon."""
        mb = self._mons.get(self._key(species))
        return mb.offensive_categories() if mb else set()

    def block_for(
        self,
        species: str,
        *,
        lookup_species: Optional[str] = None,
        ability_species: Optional[str] = None,
        stats_species: Optional[str] = None,
    ) -> Optional[dict]:
        """Narrowed belief block for ``species`` using its accumulated reveals.

        Same shape as ``BeliefState.belief_block`` (so it is a drop-in), with the
        spread distribution + ``expected_stats`` further narrowed by the observed
        offensive category.  Returns None when the species has no Pikalytics data.

        ``lookup_species`` — the Pikalytics key to pull distributions from (mega
        forme when confirmed); defaults to ``species``.
        ``ability_species`` — base forme for the ability distribution (Bug 8).
        ``stats_species`` — forme to source base stats from (mega base stats).
        """
        lookup = lookup_species or species
        mb = self._mons.get(self._key(species))
        block = self._belief.belief_block(
            lookup,
            top_k=self._top_k,
            revealed_moves=list(mb.revealed_moves) if mb else [],
            revealed_item=mb.known_item if mb else None,
            revealed_ability=mb.known_ability if mb else None,
            ability_species=ability_species,
            can_have_choice_item=mb.can_have_choice_item if mb else None,
            stats_species=stats_species or species,
            level=self._level,
        )
        if block is None:
            return None
        self._narrow_spreads_by_category(block, species, stats_species or species)
        self._narrow_spreads_by_item(block, species, stats_species or species)
        self._narrow_spreads_by_boost_stat(block, species, stats_species or species)
        self._narrow_spreads_by_damage(block, species, stats_species or species)
        self._narrow_spreads_by_speed(block, species, stats_species or species)
        return block

    def enrich_mon(self, mon: Optional[dict]) -> bool:
        """Drop-in for ``belief_state._enrich_mon_belief``: accumulate this mon's
        reveals, then write the narrowed ``belief`` + ``stats_estimate`` onto the
        mon dict in place.  Returns True if a block was written.

        Mirrors ``_enrich_mon_belief``'s mega/lookup resolution exactly so the
        encoded bytes match the training pipeline (modulo the category narrowing).
        """
        if not mon:
            return False
        self.ingest_mon(mon)
        species = mon.get("base_species") or mon.get("species")
        lookup = species if self._belief.known(species) else mon.get("species")
        ability_species = None
        # VOD-confirmed mega → the mega forme's own Pikalytics entry; abilities
        # stay base-forme context (Bug 8).
        if mon.get("is_mega") and mon.get("species") and self._belief.known(mon["species"]):
            lookup, ability_species = mon["species"], species
        if not lookup or not self._belief.known(lookup):
            return False
        block = self.block_for(
            species,
            lookup_species=lookup,
            ability_species=ability_species,
            stats_species=mon.get("species"),
        )
        if block is None:
            return False
        mon["belief"] = block
        mon["stats_estimate"] = {"mode": "distribution", "stats": block["expected_stats"]}
        return True

    # ── narrowing ────────────────────────────────────────────────────────────────
    def _narrow_spreads_by_category(self, block: dict, species: str,
                                    stats_species: Optional[str]) -> None:
        """In-place: down-weight EV spreads whose nature drops the observed
        offensive stat, renormalise, re-sort, and recompute ``expected_stats``.

        No-op unless the mon has revealed STANDARD damaging moves of EXACTLY one
        offensive category (mixed/none/cross-stat → no signal).  Conservative:
        every spread keeps p > 0 (never zeroed)."""
        spreads = block.get("spreads") or []
        if len(spreads) < 2:
            return
        offense = self.offensive_categories(species)
        if offense == {"atk"}:
            drop = "atk"
        elif offense == {"spa"}:
            drop = "spa"
        else:
            return   # mixed / none / stat-independent only → no narrowing

        # avoid double-counting correlated evidence: if OBSERVED DAMAGE already constrains this offensive stat
        # (an 'off' analytic constraint on the same stat, or an OFFENSIVE loglik on the same stat), the damage
        # likelihood already down-weights the stat-dropping natures — the move-category prior would re-apply it.
        # A DEFENSIVE loglik (narrows Def/SpD/HP) is direction-blind to Atk/SpA and must NOT trigger this guard
        # (audit 2026-06-30 — it was previously cancelling genuine offensive narrowing).
        mb = self._mons.get(self._key(species))
        if mb and any((c.get("mode") == "off" and c.get("stat") == drop)
                      or (c.get("mode") == "loglik" and c.get("direction") == "off" and c.get("stat") == drop)
                      for c in mb.damage_constraints):
            return

        changed = False
        for s in spreads:
            if NATURE_BOOSTS.get(s.get("nature"), ("", ""))[1] == drop:
                s["p"] = s["p"] * _CATEGORY_PENALTY
                changed = True
        if not changed:
            return   # no incompatible spread to penalise

        total = sum(s["p"] for s in spreads) or 1.0
        for s in spreads:
            s["p"] = round(s["p"] / total, 4)
        spreads.sort(key=lambda s: s["p"], reverse=True)

        # Recompute probability-weighted in-battle stats over the re-weighted spreads
        # (parity with BeliefState.expected_stats_weighted, using the new p's).
        base = dex_base_stats(stats_species) or dex_base_stats(block.get("species_key"))
        if not base:
            return
        acc = {st: 0.0 for st in STAT_ORDER}
        for s in spreads:
            stats = calc_full_stats(base, s["evs_actual"], s["nature"], level=self._level)
            for st in STAT_ORDER:
                acc[st] += s["p"] * stats[st]
        block["expected_stats"] = {st: round(v, 1) for st, v in acc.items()}

    def _narrow_spreads_by_item(self, block: dict, species: str,
                                stats_species: Optional[str]) -> None:
        """In-place: a revealed Choice Scarf/Band/Specs makes a spread whose nature LOWERS the item's
        boosted stat competitively incoherent → down-weight it (never zeroed), renormalise, re-sort,
        recompute ``expected_stats``.  A COHERENCE prior, not a P(spread|item) guess — it needs no joint
        usage data (see ``_ITEM_BOOST_STAT``).  No-op unless the revealed item is in ``_ITEM_BOOST_STAT``
        and the SAME stat is not already constrained by an observed move (category), damage (A2) or speed
        (A3) event — so this only ever ADDS signal a physics/observation channel hasn't already covered
        (mirrors the double-count guards in the other narrowers)."""
        mb = self._mons.get(self._key(species))
        if not mb or not mb.known_item:
            return
        boosted = _ITEM_BOOST_STAT.get(norm_species(mb.known_item))
        if not boosted:
            return    # item has no deterministic nature-coherence implication (AV/Eviolite/Life Orb/…)
        spreads = block.get("spreads") or []
        if len(spreads) < 2:
            return
        # double-count guards — defer to an already-fired channel on the SAME stat (audit-pattern parity):
        if boosted in ("atk", "spa"):
            if self.offensive_categories(species) == {boosted}:      # move-category prior ALREADY fired on this
                return                                               # stat (it only fires on a SINGLE-category
            #                                                          attacker; a MIXED {atk,spa} set bailed, so
            #                                                          the item signal is still genuinely new)
            if any((c.get("mode") == "off" and c.get("stat") == boosted)
                   or (c.get("mode") == "loglik" and c.get("direction") == "off" and c.get("stat") == boosted)
                   for c in mb.damage_constraints):                  # an OFFENSIVE damage event already
                return                                               # constrained it (a DEFENSIVE loglik does NOT)
        elif boosted == "spe" and mb.speed_constraints:              # A3 speed order already constrained Spe
            return

        changed = False
        for s in spreads:
            if NATURE_BOOSTS.get(s.get("nature"), ("", ""))[1] == boosted:
                s["p"] = s["p"] * _ITEM_NATURE_PENALTY
                changed = True
        if not changed:
            return   # no incoherent spread to penalise (single-role mon → near-no-op, by design)

        total = sum(s["p"] for s in spreads) or 1.0
        for s in spreads:
            s["p"] = round(s["p"] / total, 4)
        spreads.sort(key=lambda s: s["p"], reverse=True)

        base = dex_base_stats(stats_species) or dex_base_stats(block.get("species_key"))
        if not base:
            return
        acc = {st: 0.0 for st in STAT_ORDER}
        for s in spreads:
            stats = calc_full_stats(base, s["evs_actual"], s["nature"], level=self._level)
            for st in STAT_ORDER:
                acc[st] += s["p"] * stats[st]
        block["expected_stats"] = {st: round(v, 1) for st, v in acc.items()}

    def _narrow_spreads_by_boost_stat(self, block: dict, species: str,
                                      stats_species: Optional[str]) -> None:
        """In-place: a revealed Protosynthesis/Quark-Drive boost pins the mon's HIGHEST non-HP stat, so a
        candidate spread whose argmax stat (Atk>Def>SpA>SpD>Spe tiebreak) differs is (near-)impossible →
        down-weight it (never zeroed), renormalise, re-sort, recompute ``expected_stats``. A RANK reveal —
        orthogonal to the damage/speed magnitude signals, so no double-count guard. No-op without a reveal."""
        mb = self._mons.get(self._key(species))
        if not mb or not mb.known_boosted_stat:
            return
        revealed = mb.known_boosted_stat
        spreads = block.get("spreads") or []
        if len(spreads) < 2:
            return
        base = dex_base_stats(stats_species) or dex_base_stats(block.get("species_key"))
        if not base:
            return
        changed = False
        for s in spreads:
            full = calc_full_stats(base, s["evs_actual"], s["nature"], level=self._level)
            argmax = max(_STAT_TIEBREAK, key=lambda st: (full.get(st, 0), -_STAT_TIEBREAK.index(st)))
            if argmax != revealed:
                s["p"] = s["p"] * _BOOST_STAT_PENALTY
                changed = True
        if not changed:
            return   # every candidate's highest stat already matches the reveal → no-op

        total = sum(s["p"] for s in spreads) or 1.0
        for s in spreads:
            s["p"] = round(s["p"] / total, 4)
        spreads.sort(key=lambda s: s["p"], reverse=True)
        acc = {st: 0.0 for st in STAT_ORDER}
        for s in spreads:
            stats = calc_full_stats(base, s["evs_actual"], s["nature"], level=self._level)
            for st in STAT_ORDER:
                acc[st] += s["p"] * stats[st]
        block["expected_stats"] = {st: round(v, 1) for st, v in acc.items()}

    def _narrow_spreads_by_damage(self, block: dict, species: str,
                                  stats_species: Optional[str]) -> None:
        """In-place (A2): weight each EV spread by a LOG-NORMAL likelihood that the observed damage % came
        from it (mean = the spread's scaled predicted damage; σ = the caller's prediction uncertainty),
        multiply the likelihoods across constraints (Bayesian product), renormalise, re-sort, recompute
        expected_stats. Floored → never zeroes a spread. No-op without stored constraints. Handles both the
        DEFENSIVE (damage ∝ 1/Def, % ∝ 1/(Def·hp)) and OFFENSIVE (damage ∝ Atk) directions."""
        mb = self._mons.get(self._key(species))
        if not mb or not mb.damage_constraints:
            return
        spreads = block.get("spreads") or []
        if len(spreads) < 2:
            return
        base = dex_base_stats(stats_species) or dex_base_stats(block.get("species_key"))
        if not base:
            return
        log_floor = math.log(_LIKELIHOOD_FLOOR)
        log_w = [0.0] * len(spreads)        # accumulate log-likelihood (numerically stable over many events)
        fired = False
        for c in mb.damage_constraints:
            if c["mode"] == "loglik":       # A2 v2: a precomputed per-spread log-likelihood (poke-env path)
                ll = c["loglik"]
                # A MISSING spread key → the NEUTRAL value (log_floor), NOT 0.0: every real loglik the
                # producer stores is ≤ 0 (a mixture of exp(-½z²)≤1, floored at _LL_FLOOR == log_floor), so
                # defaulting to 0.0 would rank an un-evaluated spread as the MOST likely and invert the
                # narrowing (reachable when producer/consumer spread keys diverge). Floor → clean no-op for
                # the unmatched spreads instead. (audit 2026-06-30)
                for i, s in enumerate(spreads):
                    log_w[i] += max(ll.get(_spread_key(s), log_floor), log_floor)
                fired = True
                continue
            stat, sigma, mode = c["stat"], c["sigma"], c["mode"]
            ref_main, ref_stat, mu_ref = c["ref_main"], c["ref_stat"], c["mu_ref"]
            denom = ref_main + 2.0
            log_obs = math.log(c["obs"])
            for i, s in enumerate(spreads):
                full = calc_full_stats(base, s["evs_actual"], s["nature"], level=self._level)
                cand = full.get(stat)
                if not cand:
                    continue
                if mode == "def":
                    hp_cand = full.get("hp")
                    if not hp_cand:
                        continue
                    main_cand = ref_main * (ref_stat / cand)         # main ∝ 1/Def; the +2 stays ADDITIVE
                    mu = mu_ref * (c["ref_hp"] / hp_cand) * (main_cand + 2.0) / denom
                else:  # offensive: main ∝ Atk; OUR defender hp cancels (folded into mu_ref)
                    main_cand = ref_main * (cand / ref_stat)
                    mu = mu_ref * (main_cand + 2.0) / denom
                if mu <= 0:
                    continue
                z = (log_obs - math.log(mu)) / sigma
                log_w[i] += max(-0.5 * z * z, log_floor)             # per-event floor, in log-space
                fired = True
        if not fired:
            return
        m = max(log_w)                       # max-subtraction → no underflow; best spread weight = 1.0
        weights = [math.exp(lw - m) for lw in log_w]
        for i, s in enumerate(spreads):
            s["p"] = s["p"] * weights[i]
        total = sum(s["p"] for s in spreads) or 1.0
        floored = [max(s["p"] / total, _P_FLOOR) for s in spreads]   # floor → never zero the truth
        total2 = sum(floored)                                        # ≥ len·_P_FLOOR > 0 (no guard needed)
        for s, p in zip(spreads, floored):
            s["p"] = round(p / total2, 4)
        spreads.sort(key=lambda s: s["p"], reverse=True)
        acc = {st: 0.0 for st in STAT_ORDER}
        for s in spreads:
            stats = calc_full_stats(base, s["evs_actual"], s["nature"], level=self._level)
            for st in STAT_ORDER:
                acc[st] += s["p"] * stats[st]
        block["expected_stats"] = {st: round(v, 1) for st, v in acc.items()}

    def _narrow_spreads_by_speed(self, block: dict, species: str,
                                 stats_species: Optional[str]) -> None:
        """In-place (A3): weight each EV spread by a soft LOGISTIC on its base Spe vs an observed speed-tier
        bound — ``faster`` ⇒ the opp's Spe should exceed ``threshold`` (slow spreads down-weighted), ``slower``
        ⇒ below it. Accumulated in LOG-space (per-event floor; max-subtraction → no underflow), floored
        (never zeroed), renormalised, expected_stats recomputed. No-op without speed constraints."""
        mb = self._mons.get(self._key(species))
        if not mb or not mb.speed_constraints:
            return
        spreads = block.get("spreads") or []
        if len(spreads) < 2:
            return
        base = dex_base_stats(stats_species) or dex_base_stats(block.get("species_key"))
        if not base:
            return
        log_floor = math.log(_LIKELIHOOD_FLOOR)
        log_w = [0.0] * len(spreads)
        fired = False
        for c in mb.speed_constraints:
            T, faster, sig = c["threshold"], c["faster"], c["sigma"]
            for i, s in enumerate(spreads):
                spe = calc_full_stats(base, s["evs_actual"], s["nature"], level=self._level).get("spe")
                if not spe:
                    continue
                x = (spe - T) / sig if faster else (T - spe) / sig
                like = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))    # logistic, clamped for exp safety
                log_w[i] += max(math.log(like), log_floor)
                fired = True
        if not fired:
            return
        m = max(log_w)
        weights = [math.exp(lw - m) for lw in log_w]
        for i, s in enumerate(spreads):
            s["p"] = s["p"] * weights[i]
        total = sum(s["p"] for s in spreads) or 1.0
        floored = [max(s["p"] / total, _P_FLOOR) for s in spreads]
        total2 = sum(floored)
        for s, p in zip(spreads, floored):
            s["p"] = round(p / total2, 4)
        spreads.sort(key=lambda s: s["p"], reverse=True)
        acc = {st: 0.0 for st in STAT_ORDER}
        for s in spreads:
            stats = calc_full_stats(base, s["evs_actual"], s["nature"], level=self._level)
            for st in STAT_ORDER:
                acc[st] += s["p"] * stats[st]
        block["expected_stats"] = {st: round(v, 1) for st, v in acc.items()}
