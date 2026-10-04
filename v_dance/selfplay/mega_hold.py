"""MEGA-HOLD EXPLORATION for the self-play learner (2026-10-03, USER: "delaying tyranitar's mega is a standard play …
i wanted to design the model so it could learn it … during reinforcement learning but that never happened" → "build
all of this").

Why RL never found the play: the mega (gimmick) head is SAMPLED PER TURN, so "wait until their rain setter comes in"
needs several unlikely coins in a row — practically never tried (the 10-03 mega-hold probe: gen39 megas in ~82 % of
the ladder states where the standard play holds). This module makes the hold a GAME-LEVEL exploration choice, like
the picker's exploring decode:

  · at a battle's first gimmick decision the learner decides ONCE whether this is a HOLD game — probability
    ``p_weather`` when the opponent's six carries a weather setter that fights ours (a species whose abilities, or its
    mega's, set a weather our team does not set), else ``p`` — and a hold game draws a horizon k ∈ [kmin, kmax];
  · while held (battle turn < k and — ``end_on_weather`` — no weather that is not ours is up), every slot that
    SAMPLED its gimmick (a move, not a switch), may MEGA and holds a DELAY-RELEVANT mega is forced to GIMMICK_NONE. The
    hold ends for good at turn k or on the first turn their weather is up; the policy decides freely from then on
    (e.g. megas NOW, which re-sets sand for Mega Tyranitar).

DELAY-RELEVANT ONLY (2026-10-03, USER: "I'm afraid if the bot learns that, it will mess up with other megas. Such as
mega salamence who doesn't really need to delay mega"; then "redirection / immunities or type changing … raichu has
lighting rod … mega raichu does not have that ability. so if the bot predicts … electro shot, we wouldn't want it to
mega evolve"): a mega is held only when delaying it can pay — ``delay_reason`` (read off the mon's OWN stone + ability
and the project dex) is one of
  · ``field``   — its MEGA ability sets a weather / terrain when it evolves (Mega Tyranitar's Sand Stream re-sets sand,
                  Mega Charizard Y's Drought, Mega Raichu X's Electric Surge);
  · ``ability`` — its BASE ability guards against something and the mega loses it: redirection / absorption / type
                  immunity (Raichu's Lightning Rod vs Electro Shot, Storm Drain, Water / Volt Absorb, Flash Fire, Sap
                  Sipper, Levitate …), a move-category or status immunity (Bulletproof, Soundproof, Good as Gold,
                  Insomnia …), or a stat-drop / Intimidate guard (Metagross's Clear Body, Defiant, Inner Focus …);
  · ``type``    — the mega CHANGES TYPE (Charizard X: Fire/Flying → Fire/Dragon loses the Ground immunity; Golisopod
                  Bug/Water → Bug/Steel; Gyarados, Altaria, Ampharos, Meganium, Lopunny, Garchomp Z …).
Every other mega (Mega Salamence: Intimidate → Aerilate, same types) is never forced: its decisions stay the policy's
own samples, so the exploration gives the mega head no push to delay it. A guard metric watches for drift anyway (the
megatime drill's ``guard_first_mega`` — guard=auto = our team's megas with no delay reason — and the probe's
``guard:<mon>`` rows). The per-gen readout counts the forced steps by reason.

OFF-POLICY CORRECTION. A forced "none" is not the policy's sample. The step keeps the policy's OWN behaviour log-prob
(the recorder evaluates it on the executed action, so the PPO ratio starts at exactly 1 and stays in the usual trust
region) and carries an importance weight w = Π π_old(none | slot) over the forced slots — π at the collection
temperature over the same gimmick legal mask the recorder stores — i.e. the probability that the policy would have
held on its own. PPO multiplies that step's clipped surrogate by w (``Transition.is_weight``, ``rl.ppo``):
E_μ[w·g] = E_π[g], so the gradient stays unbiased while the hold branch is observed in every hold game instead of in
~π(none) of them. The moves of a forced step are the policy's own samples and share the joint weight (the surrogate
is joint) — a small loss of signal, not a bias. ``w_min`` (default 0 = exact) floors w: a deliberate, bounded bias that
lets forced steps count for more while π(none) is still tiny.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np

# poke-env Weather enum name → weather KIND (the field report's kinds)
WEATHER_ENUM_KIND = {"SANDSTORM": "sand", "RAINDANCE": "rain", "PRIMORDIALSEA": "rain", "SUNNYDAY": "sun",
                     "DESOLATELAND": "sun", "SNOWSCAPE": "snow", "SNOW": "snow", "HAIL": "snow", "DELTASTREAM": "wind"}


def _id(s) -> str:
    return "".join(c for c in str(s or "").lower() if c.isalnum())


@dataclass
class MegaHoldConfig:
    p: float = 0.0                 # P(hold game) when the opponent's six has no weather setter that fights ours
    p_weather: float = 0.0         # P(hold game) when it does
    kmin: int = 2                  # hold while battle turn < k, k ~ U{kmin..kmax} (k = 2 holds turn 1 only)
    kmax: int = 6
    end_on_weather: bool = True    # the hold also ends on the first turn a weather that is not ours is up
    w_min: float = 0.0             # floor on the importance weight (0 = exact; module docstring)

    def __post_init__(self):
        for name in ("p", "p_weather", "w_min"):
            v = float(getattr(self, name))
            if not (0.0 <= v <= 1.0) or v != v:
                raise ValueError(f"mega-hold {name} must be in [0, 1] (got {v})")
            setattr(self, name, v)
        self.kmin, self.kmax = int(self.kmin), int(self.kmax)
        if not (2 <= self.kmin <= self.kmax):
            raise ValueError(f"mega-hold needs 2 <= kmin <= kmax (got {self.kmin}, {self.kmax})")
        self.end_on_weather = bool(self.end_on_weather)

    @property
    def on(self) -> bool:
        return self.p > 0.0 or self.p_weather > 0.0

    def to_spec(self) -> dict:
        """Picklable primitives for mp_collect.ChunkSpec (the worker adds its own seeded rng)."""
        return asdict(self)

    def describe(self) -> str:
        return (f"P(hold game) {self.p:g} (field fight {self.p_weather:g}) · hold DELAY-RELEVANT megas only (field: a "
                f"weather / terrain mega ability · ability: a guarding base ability the mega loses — redirection / "
                f"immunity / stat-drop guard · type: the mega changes type) until turn k ∈ [{self.kmin}, {self.kmax}]"
                f"{' or their field is up' if self.end_on_weather else ''} · importance weight π(none)"
                f"{f' floored at {self.w_min:g}' if self.w_min > 0 else ' (exact)'}")


# ── weather setters ──────────────────────────────────────────────────────────
@lru_cache(maxsize=1024)
def species_weather_kinds(species: str) -> frozenset:
    """The weather kinds a species CAN set (any ability slot, or its mega's ability), from the project dex."""
    from v_dance.dex.pokedex import get_pokedex
    from v_dance.eval.field_control_report import ABILITY_WEATHER
    dx = get_pokedex()
    if dx is None or not species:
        return frozenset()
    try:
        ab = list(dx.abilities_for(species)) + [m.get("ability") for m in dx.mega_formes_for(species)]
    except Exception:
        return frozenset()
    return frozenset(ABILITY_WEATHER[_id(a)] for a in ab if _id(a) in ABILITY_WEATHER)


def team_weather_kinds(species: Iterable[str]) -> frozenset:
    out = set()
    for s in species:
        out |= species_weather_kinds(_id(s))
    return frozenset(out)


def weather_conflict(ours: Iterable[str], theirs: Iterable[str]) -> bool:
    """True when we can set a weather AND their six can set one we cannot."""
    o, t = team_weather_kinds(ours), team_weather_kinds(theirs)
    return bool(o) and bool(t - o)


def current_weather_kinds(battle) -> frozenset:
    w = getattr(battle, "weather", None) or {}
    kinds = set()
    for k in (w.keys() if isinstance(w, dict) else w):
        kinds.add(WEATHER_ENUM_KIND.get(getattr(k, "name", str(k)).upper(), _id(getattr(k, "name", k))))
    return frozenset(kinds)


# poke-env Field enum name → terrain KIND
TERRAIN_ENUM_KIND = {"ELECTRIC_TERRAIN": "electric", "GRASSY_TERRAIN": "grassy", "MISTY_TERRAIN": "misty",
                     "PSYCHIC_TERRAIN": "psychic"}


@lru_cache(maxsize=1024)
def species_terrain_kinds(species: str) -> frozenset:
    """The terrain kinds a species CAN set (any ability slot, or its mega's ability), from the project dex."""
    from v_dance.dex.pokedex import get_pokedex
    from v_dance.eval.field_control_report import ABILITY_TERRAIN
    dx = get_pokedex()
    if dx is None or not species:
        return frozenset()
    try:
        ab = list(dx.abilities_for(species)) + [m.get("ability") for m in dx.mega_formes_for(species)]
    except Exception:
        return frozenset()
    return frozenset(ABILITY_TERRAIN[_id(a)] for a in ab if _id(a) in ABILITY_TERRAIN)


def team_terrain_kinds(species: Iterable[str]) -> frozenset:
    out = set()
    for s in species:
        out |= species_terrain_kinds(_id(s))
    return frozenset(out)


def current_terrain_kinds(battle) -> frozenset:
    f = getattr(battle, "fields", None) or {}
    return frozenset(TERRAIN_ENUM_KIND[n] for n in (getattr(k, "name", str(k)).upper()
                                                    for k in (f.keys() if isinstance(f, dict) else f))
                     if n in TERRAIN_ENUM_KIND)


# BASE abilities a mega would LOSE that guard the mon against something a hold can dodge (module docstring: 'ability')
STAT_GUARD_ABILITIES = frozenset({"clearbody", "whitesmoke", "fullmetalbody", "hypercutter", "mirrorarmor", "defiant",
                                  "competitive", "innerfocus", "owntempo", "oblivious", "scrappy", "guarddog"})
IMMUNITY_ABILITIES = frozenset({
    # redirection / absorption / type immunity or resistance (USER: Raichu's Lightning Rod vs Electro Shot)
    "lightningrod", "stormdrain", "voltabsorb", "waterabsorb", "motordrive", "flashfire", "wellbakedbody", "sapsipper",
    "levitate", "eartheater", "dryskin", "windrider", "thickfat", "heatproof", "waterbubble", "purifyingsalt",
    # move-category immunities
    "bulletproof", "soundproof", "overcoat", "goodasgold", "magicbounce", "wonderguard", "armortail",
    "queenlymajesty", "dazzling", "telepathy",
    # status immunities
    "insomnia", "vitalspirit", "sweetveil", "limber", "immunity", "waterveil", "magmaarmor", "pastelveil",
    "leafguard", "comatose"})
GUARDING_BASE_ABILITIES = STAT_GUARD_ABILITIES | IMMUNITY_ABILITIES


@lru_cache(maxsize=1024)
def mega_info(species: str, item: str) -> Optional[tuple]:
    """``(mega ability id, mega types)`` for ``species`` mega-evolving with ``item`` (its stone, via the dex's
    requiredItem), or None (no stone / not this species' stone / no dex)."""
    from v_dance.dex.pokedex import get_pokedex
    dx = get_pokedex()
    if dx is None or not species or not item:
        return None
    try:
        for m in dx.mega_formes_for(species):
            e = dx.entry(m.get("forme")) or {}
            req = e.get("requiredItem")
            if req and _id(req) == _id(item):
                return _id(m.get("ability")), tuple(_id(t) for t in e.get("types") or ())
    except Exception:
        return None
    return None


def mega_ability_for(species: str, item: str) -> Optional[str]:
    info = mega_info(species, item)
    return info[0] if info else None


@lru_cache(maxsize=1024)
def _base_types(species: str) -> tuple:
    from v_dance.dex.pokedex import get_pokedex
    dx = get_pokedex()
    try:
        return tuple(_id(t) for t in ((dx.entry(species) or {}).get("types") or ())) if dx else ()
    except Exception:
        return ()


def field_domain(ability_id: str) -> Optional[str]:
    from v_dance.eval.field_control_report import ABILITY_TERRAIN, ABILITY_WEATHER
    if ability_id in ABILITY_WEATHER:
        return "weather"
    if ability_id in ABILITY_TERRAIN:
        return "terrain"
    return None


def delay_reason(mon) -> Optional[str]:
    """Why this mon's mega may be worth HOLDING — 'field' / 'ability' / 'type' (module docstring: DELAY-RELEVANT
    ONLY) — or None (never held). Unknown stone → None."""
    if mon is None:
        return None
    species = _id(getattr(mon, "base_species", None) or getattr(mon, "species", None))
    info = mega_info(species, _id(getattr(mon, "item", None)))
    if info is None:
        return None
    mab, mtypes = info
    if field_domain(mab):
        return "field"
    base = _id(getattr(mon, "ability", None))
    if base in GUARDING_BASE_ABILITIES and base != mab:
        return "ability"
    bt = _base_types(species)
    if bt and mtypes and set(bt) != set(mtypes):
        return "type"
    return None


def delay_relevant(mon) -> bool:
    """Is this mon's mega worth HOLDING (``delay_reason`` is not None)?"""
    return delay_reason(mon) is not None


def _species_of(mons) -> list:
    if isinstance(mons, dict):
        mons = mons.values()
    return [getattr(m, "species", None) for m in (mons or ()) if getattr(m, "species", None)]


# ── the per-battle decision + the per-turn force ─────────────────────────────
def field_mega_domains(team) -> frozenset:
    """The field domains ('weather' / 'terrain') our team's FIELD megas set (Tyranitar → weather, Raichu X →
    terrain) — the only domains where a foreign field is a FIGHT for the hold (a Grassy Surge is no reason to hold
    Mega Tyranitar)."""
    mons = list(team.values()) if isinstance(team, dict) else list(team or ())
    out = set()
    for m in mons:
        if delay_reason(m) == "field":
            info = mega_info(_id(getattr(m, "base_species", None) or getattr(m, "species", None)),
                             _id(getattr(m, "item", None)))
            d = field_domain(info[0]) if info else None
            if d:
                out.add(d)
    return frozenset(out)


def decide(battle, cfg: dict) -> dict:
    """The battle's one hold decision (``cfg`` = MegaHoldConfig.to_spec() + ``rng``). A FIGHT (→ ``p_weather``) = the
    opponent's six can set a weather / terrain kind ours does not, in a domain one of our FIELD megas sets."""
    team = getattr(battle, "team", None)
    ours = _species_of(team)
    theirs = _species_of(getattr(battle, "teampreview_opponent_team", None)
                         or getattr(battle, "opponent_team", None))
    domains = field_mega_domains(team)
    kinds_ours = {"weather": team_weather_kinds(ours), "terrain": team_terrain_kinds(ours)}
    kinds_theirs = {"weather": team_weather_kinds(theirs), "terrain": team_terrain_kinds(theirs)}
    conflict = any(kinds_ours[d] and (kinds_theirs[d] - kinds_ours[d]) for d in domains)
    p = float(cfg.get("p_weather", 0.0) if conflict else cfg.get("p", 0.0))
    rng = cfg["rng"]
    hold = bool(rng.random() < p)
    k = int(rng.integers(int(cfg.get("kmin", 2)), int(cfg.get("kmax", 6)) + 1)) if hold else 0
    return {"hold": hold, "k": k, "conflict": conflict, "ours": kinds_ours["weather"],
            "ours_terrain": kinds_ours["terrain"], "domains": domains, "ended": not hold, "end": None,
            "forced_steps": 0, "reasons": {}}


def their_field_is_up(battle, st: dict) -> Optional[str]:
    """The domain in which a field that is NOT ours is up — only the domains our field megas set — else None."""
    if "weather" in st.get("domains", ()) and (current_weather_kinds(battle) - st.get("ours", frozenset())):
        return "weather"
    if "terrain" in st.get("domains", ()) and (current_terrain_kinds(battle) - st.get("ours_terrain", frozenset())):
        return "terrain"
    return None


def p_none(logits, gmask, tau: float, none: int = 0) -> float:
    """π(none) of one slot's gimmick head: masked softmax at ``tau`` over ``gmask`` (policy_eval's definition)."""
    z = np.asarray(logits, dtype=np.float64).ravel() / float(tau)
    legal = np.asarray(list(gmask)[:len(z)], dtype=bool)
    if not legal.any() or not legal[none]:
        return 0.0
    z = np.where(legal, z, -np.inf)
    z = z - z[legal].max()
    p = np.exp(z)
    return float(p[none] / p.sum())


def apply(player, battle, glog, out: list, acts: Sequence, *, none: int, mega: int, switch_offset: int,
          tau: float, build_mask, relevant=None) -> Optional[float]:
    """Force the hold onto ``out`` (in place) when this is a held game and the hold is still on. Returns the step's
    importance weight, also stashed as ``player._mega_hold_w[(tag, "turn")] = (battle turn, w)`` for the recorder —
    the TURN travels with it so a stash whose step was discarded (a non-model order executed) can never weight a
    later turn's step (``stashed_weight``); None = nothing forced (the step is the policy's own)."""
    cfg = getattr(player, "_mega_hold", None)
    if not cfg or float(tau) <= 0.0:
        return None
    tag = getattr(battle, "battle_tag", None)
    states = player.__dict__.setdefault("_mega_hold_state", {})
    st = states.get(tag)
    if st is None:
        st = states[tag] = decide(battle, cfg)
        if len(states) > 512:                                  # a battle that never finishes stays bounded
            states.pop(next(iter(states)))
    if st["ended"]:
        return None
    turn = int(getattr(battle, "turn", 0) or 0)
    if turn >= st["k"]:
        st["ended"], st["end"] = True, "turn"
        return None
    if cfg.get("end_on_weather", True):
        dom = their_field_is_up(battle, st)            # their weather / terrain is up where our field mega fights
        if dom:
            st["ended"], st["end"] = True, dom
            return None
    w, forced = 1.0, 0
    relevant = relevant or delay_reason
    active = list(getattr(battle, "active_pokemon", None) or [])
    reasons = st.setdefault("reasons", {})
    for slot in (0, 1):
        a = acts[slot] if slot < len(acts) else None
        if a is None or a >= switch_offset:                    # a switch never megas — not a sampled gimmick
            continue
        gm = build_mask(battle, slot)
        if not gm or len(gm) <= mega or not gm[mega]:
            continue
        why = relevant(active[slot] if slot < len(active) else None)
        if not why:
            continue                                           # Mega Salamence & co: the policy's own call, always
        w *= p_none(glog[slot], gm, tau, none)
        out[slot] = none
        forced += 1
        label = why if isinstance(why, str) else "any"
        reasons[label] = reasons.get(label, 0) + 1
    if not forced:
        return None
    st["forced_steps"] += 1
    w = max(float(w), float(cfg.get("w_min", 0.0) or 0.0))
    player.__dict__.setdefault("_mega_hold_w", {})[(tag, "turn")] = (turn, w)
    return w


def stashed_weight(player, tag, decision_type: str, turn) -> float:
    """Pop the importance weight ``apply`` stashed for THIS turn's decision; 1.0 when there is none, or when the stash
    belongs to another turn (its own step was discarded — it must never weight this one)."""
    st = getattr(player, "_mega_hold_w", None)
    v = st.pop((tag, decision_type), None) if isinstance(st, dict) else None
    if not v or int(v[0]) != int(turn or 0):
        return 1.0
    return float(v[1])


def game_meta(player, tag) -> Optional[dict]:
    """Pop the battle's hold record for its trajectory meta (``sampling["mega_hold"]``); None = never decided."""
    st = (player.__dict__.get("_mega_hold_state") or {}).pop(tag, None)
    w = player.__dict__.get("_mega_hold_w")
    if isinstance(w, dict):
        for key in [k for k in w if k[0] == tag]:
            w.pop(key, None)
    if st is None:
        return None
    return {"held": bool(st["hold"]), "k": int(st["k"]), "conflict": bool(st["conflict"]), "end": st["end"],
            "forced_steps": int(st["forced_steps"]), "reasons": dict(st.get("reasons") or {})}


# ── the per-generation readout ───────────────────────────────────────────────
def summarize_trajectories(trajs) -> dict:
    """Hold games / forced steps / importance weights of one generation's trajectories (both perspectives counted)."""
    decided = held = conflict = held_conflict = forced = 0
    ws = []
    by_reason: dict = {}
    for t in trajs or ():
        mh = ((getattr(getattr(t, "meta", None), "sampling", None) or {}).get("mega_hold"))
        if mh:
            decided += 1
            held += int(bool(mh.get("held")))
            conflict += int(bool(mh.get("conflict")))
            held_conflict += int(bool(mh.get("held")) and bool(mh.get("conflict")))
            forced += int(mh.get("forced_steps") or 0)
            for r, n in (mh.get("reasons") or {}).items():
                by_reason[r] = by_reason.get(r, 0) + int(n)
        for s in getattr(t, "transitions", ()) or ():
            w = float(getattr(s, "is_weight", 1.0))
            if w != 1.0:
                ws.append(w)
    return {"trajs": len(trajs or ()), "decided": decided, "held": held, "conflict": conflict,
            "held_conflict": held_conflict, "forced_steps": forced, "weighted_steps": len(ws),
            "w_mean": (round(float(np.mean(ws)), 4) if ws else None), "w_min": (round(float(min(ws)), 4) if ws else None),
            "forced_by_reason": by_reason}


def format_stats(s: dict) -> str:
    wm = "—" if s.get("w_mean") is None else f"{s['w_mean']:.2f} (min {s['w_min']:.2f})"
    br = s.get("forced_by_reason") or {}
    why = (" (mega holds by reason: " + " · ".join(f"{k} {v}" for k, v in sorted(br.items())) + ")") if br else ""
    return (f"mega-hold exploration: held {s['held']}/{s['decided']} games ({s['held_conflict']}/{s['conflict']} with "
            f"a field fight) · forced steps {s['forced_steps']}{why} · importance weight mean {wm}")
