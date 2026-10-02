"""Resolve a mega forme from its STONE — the poke-env Z-mega overwrite fix (mega audit gap 1, 2026-10-02).

poke-env's ``|-mega|`` handler (abstract_battle.py, ``pokemon, megastone = event[2:4]``) hands
``Pokemon.mega_evolve`` the SPECIES field (event[3]), not the stone (event[4]); ``mega_evolve`` then loads
``<species>mega``.  The server sends ``|detailschange|`` with the TRUE forme first and ``|-mega|`` second
(pokemon-showdown data/mods/champions/scripts.ts), so for the Champions Z-megas the correct forme is
overwritten by the regular mega and never corrected for an opponent:

    Garchomp-Mega-Z  → read as Garchomp-Mega  (Dragon/Ground, Sand Force, Spe 92  — true: Dragon, Levitate, 151)
    Absol-Mega-Z     → read as Absol-Mega     (Dark, Magic Bounce               — true: Dark/Ghost, Sharpness)
    Lucario-Mega-Z   → read as Lucario-Mega   (Adaptability, Spe 112            — true: Aura Guard, 151)

Every move feature the bot scores against that opponent (type-eff, damage band, who-moves-first) read the
wrong forme.  The fix: the battle notes event[4] for the mon just before poke-env's handler runs, and the
patched ``mega_evolve`` loads the dex forme whose ``requiredItem`` IS that stone (narrowed to the mon's own
species when one stone serves several formes: Meowsticite, Tatsugirinite).  Without a stone it keeps a mega
forme the ``|detailschange|`` already applied, and only falls back to poke-env's own guess for a mon that is
not yet in any mega forme.

``install()`` patches the two poke-env methods once per process (idempotent).  ``vgc_base`` calls it at
import, so every player built on it — the ladder bot, self-play workers, eval opponents — reads megas the
same way.  Pure poke-env state: nothing here touches the encoders, so train/serve parity is unchanged
(training already read the true forme from the replay parser).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_INSTALLED = False
_ORIG_MEGA_EVOLVE = None
_ORIG_PARSE_MESSAGE = None
# id(Pokemon) -> the stone named by the |-mega| line being handled.  poke-env's Pokemon uses __slots__ (no
# attribute stash), and the entry is consumed by the mega_evolve call that runs synchronously right after.
_PENDING_STONE: Dict[int, str] = {}
_STONE_FORMES: Optional[Dict[str, List[str]]] = None


def _to_id(s) -> str:
    return "".join(c for c in str(s or "").lower() if c.isalnum())


def _pokedex() -> dict:
    from poke_env.data import GenData
    return GenData.from_gen(9).pokedex


def stone_formes() -> Dict[str, List[str]]:
    """stone id → the dex ids of every MEGA forme that requires it (cached; built from poke-env's dex, the
    same dex the patched ``_update_from_pokedex`` reads).  A forme counts as mega iff its ``forme`` contains
    'Mega' (Mega / Mega-X / Mega-Y / Mega-Z / M-Mega / F-Mega / Curly-Mega …)."""
    global _STONE_FORMES
    if _STONE_FORMES is None:
        out: Dict[str, List[str]] = {}
        for sid, e in _pokedex().items():
            if "Mega" not in (e.get("forme") or ""):
                continue
            stone = _to_id(e.get("requiredItem"))
            if stone:
                out.setdefault(stone, []).append(sid)
        _STONE_FORMES = {k: sorted(v) for k, v in out.items()}
    return _STONE_FORMES


def resolve_mega_forme(species: str, stone: str, *, base_stats=None, female: bool = False) -> Optional[str]:
    """The dex id of the mega forme ``stone`` turns ``species`` into, or None when the stone is not a mega
    stone (or names no forme of this species).

    One stone can serve several formes (Meowsticite → M-Mega / F-Mega; Tatsugirinite → Curly/Droopy/
    Stretchy): narrow by the mon's own species id (``meowsticf`` → ``meowsticfmega``), then by the base
    stats a ``|detailschange|`` already applied, then by gender."""
    cands = list(stone_formes().get(_to_id(stone), ()))
    if not cands:
        return None
    dex = _pokedex()
    sp = _to_id(species)
    base = _to_id((dex.get(sp) or {}).get("baseSpecies") or sp)
    cands = [c for c in cands if _to_id(dex[c].get("baseSpecies")) == base]
    if len(cands) <= 1:
        return cands[0] if cands else None
    pref = [c for c in cands if c.startswith(sp)]     # 'meowsticf' keeps only meowsticfmega
    if pref:
        cands = pref
    if len(cands) > 1 and base_stats:
        same = [c for c in cands if dict(dex[c].get("baseStats") or {}) == dict(base_stats)]
        if same:
            cands = same
    if len(cands) > 1:
        tag = "F-" if female else "M-"
        g = [c for c in cands if (dex[c].get("forme") or "").startswith(tag)]
        if g:
            cands = g
    return sorted(cands)[0]


def current_mega_forme(mon) -> Optional[str]:
    """The dex id of the mega forme ``mon`` ALREADY shows (its current base stats + typing equal a mega forme
    of its species), else None — the state a ``|detailschange|`` leaves before ``|-mega|`` arrives."""
    try:
        dex = _pokedex()
        base = _to_id(mon.base_species)
        cur = dict(getattr(mon, "_base_stats", None) or {})
        types = [t for t in (getattr(mon, "_type_1", None), getattr(mon, "_type_2", None)) if t is not None]
        tnames = [getattr(t, "name", str(t)).upper() for t in types]
    except Exception:
        return None
    if not cur:
        return None
    for sid, e in dex.items():
        if "Mega" not in (e.get("forme") or "") or _to_id(e.get("baseSpecies")) != base:
            continue
        if dict(e.get("baseStats") or {}) == cur and [t.upper() for t in e.get("types") or []] == tnames:
            return sid
    return None


def _female(mon) -> bool:
    try:
        return "FEMALE" in str(getattr(mon, "gender", "") or "").upper()
    except Exception:
        return False


def _patched_mega_evolve(self, stone):
    """Load the forme the STONE names; keep an already-applied mega forme; else poke-env's own guess."""
    real = _PENDING_STONE.pop(id(self), None)
    forme = None
    if real:
        try:
            forme = resolve_mega_forme(self.species, real, base_stats=getattr(self, "_base_stats", None),
                                       female=_female(self))
        except Exception:
            log.debug("mega forme resolve failed (non-fatal)", exc_info=True)
            forme = None
    if forme is None:
        forme = current_mega_forme(self)
    if forme is None:
        return _ORIG_MEGA_EVOLVE(self, stone)
    self.temporary_ability = None
    self._update_from_pokedex(forme, store_species=False)
    return None


def _patched_parse_message(self, split_message):
    try:
        if len(split_message) > 4 and split_message[1] == "-mega" and split_message[4]:
            _PENDING_STONE[id(self.get_pokemon(split_message[2]))] = split_message[4]
    except Exception:
        log.debug("mega stone capture failed (non-fatal)", exc_info=True)
    try:
        return _ORIG_PARSE_MESSAGE(self, split_message)
    finally:
        if len(split_message) > 1 and split_message[1] == "-mega":
            _PENDING_STONE.clear()        # never leak a stone into a later, unrelated mega_evolve call


def install() -> bool:
    """Patch poke-env once per process.  Returns True when the patch is (now) active."""
    global _INSTALLED, _ORIG_MEGA_EVOLVE, _ORIG_PARSE_MESSAGE
    if _INSTALLED:
        return True
    try:
        from poke_env.battle.abstract_battle import AbstractBattle
        from poke_env.battle.pokemon import Pokemon
    except ImportError:                    # offline training machine without poke-env
        return False
    _ORIG_MEGA_EVOLVE = Pokemon.mega_evolve
    _ORIG_PARSE_MESSAGE = AbstractBattle.parse_message
    Pokemon.mega_evolve = _patched_mega_evolve
    AbstractBattle.parse_message = _patched_parse_message
    _INSTALLED = True
    return True
