"""Live mega formes come from the STONE, for every mega in the dex (mega audit gaps 1 + 5, 2026-10-02).

poke-env's ``|-mega|`` handler passes the species, not the stone, and loaded ``<species>mega`` AFTER the
server's ``|detailschange|`` had applied the true forme — so an opponent Garchomp-Mega-Z read as
Garchomp-Mega (Dragon/Ground, Sand Force, Spe 92), Absol-Mega-Z lost its Ghost typing and Lucario-Mega-Z
its Aura Guard.  ``v_dance.play.mega_forme_fix`` (installed by ``vgc_base``) resolves the forme from the
stone; ``is_mega_forme_live`` now knows every mega forme (Mega-Z, M-Mega/F-Mega, Curly/Droopy/Stretchy-Mega).

The existing parity test (o6) passed WITH the bug, so this one feeds the real protocol order for every
mega forme, on both sides of the board.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("poke_env")

from poke_env.battle import DoubleBattle  # noqa: E402
from poke_env.data import GenData  # noqa: E402

import v_dance.play.vgc_base  # noqa: E402,F401  (installs the fix)
from v_dance.encoders.live_state_encoder import is_mega_forme_live, team_has_megaed_live  # noqa: E402
from v_dance.play import mega_forme_fix as F  # noqa: E402

_DEX = GenData.from_gen(9).pokedex
_MEGAS = sorted(k for k, v in _DEX.items() if "Mega" in (v.get("forme") or "") and v.get("requiredItem"))


def _id(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _pre_mega_name(e: dict) -> str:
    """The display forme a mon is in BEFORE it megas (Meowstic-F, Tatsugiri-Droopy, else the base)."""
    base, forme = e["baseSpecies"], e["forme"]
    pre = base
    if forme.startswith("F-"):
        pre = base + "-F"
    for pfx in ("Droopy-", "Stretchy-", "Original-"):
        if forme.startswith(pfx):
            pre = base + "-" + pfx[:-1]
    return pre if _id(pre) in _DEX else base


def _battle() -> DoubleBattle:
    b = DoubleBattle("battle-gen9championsvgc2026regmc-1", "me", logging.getLogger("t"), gen=9)
    b._player_role = "p1"
    return b


def _assert_forme(mon, e: dict) -> None:
    assert [t.name for t in mon.types if t is not None] == [t.upper() for t in e["types"]]
    assert dict(mon.base_stats) == dict(e["baseStats"])
    abilities = [a for a in e["abilities"].values() if a]
    if len(abilities) == 1:                      # a fixed mega ability (Tatsugiri's megas keep two)
        assert mon.ability == _id(abilities[0])
    assert is_mega_forme_live(mon)


def test_there_are_megas_to_check():
    assert len(_MEGAS) >= 82                      # every Reg M-C mega (82) is in poke-env's dex
    assert {"garchompmegaz", "absolmegaz", "lucariomegaz", "meowsticmmega", "golisopodmega"} <= set(_MEGAS)


@pytest.mark.parametrize("sid", _MEGAS)
@pytest.mark.parametrize("role", ["p1", "p2"])
def test_detailschange_then_mega_keeps_the_true_forme(sid, role):
    """The server's real order: |detailschange| (true forme) then |-mega| (species, stone)."""
    e = _DEX[sid]
    b = _battle()
    ident = f"{role}a: Mon"
    b.parse_message(["", "switch", ident, f"{_pre_mega_name(e)}, L50", "100/100"])
    b.parse_message(["", "detailschange", ident, f"{e['name']}, L50"])
    b.parse_message(["", "-mega", ident, e["baseSpecies"], e["requiredItem"]])
    _assert_forme(b.get_pokemon(ident), e)


@pytest.mark.parametrize("sid", ["garchompmegaz", "absolmegaz", "lucariomegaz", "charizardmegax",
                                 "charizardmegay", "meowsticmmega", "meowsticfmega", "golisopodmega"])
def test_mega_without_detailschange_resolves_from_the_stone(sid):
    """No |detailschange| at all: the stone alone names the forme (Charizardite X vs Y, Garchompite Z)."""
    e = _DEX[sid]
    b = _battle()
    b.parse_message(["", "switch", "p2a: Mon", f"{_pre_mega_name(e)}, L50", "100/100"])
    b.parse_message(["", "-mega", "p2a: Mon", e["baseSpecies"], e["requiredItem"]])
    _assert_forme(b.get_pokemon("p2a: Mon"), e)


def test_garchomp_z_reads_levitate_dragon_and_fast():
    """The audit's headline case, spelled out: Electric no longer 'immune', Ground no longer hits."""
    b = _battle()
    b.parse_message(["", "switch", "p2a: Garchomp", "Garchomp, L50", "100/100"])
    b.parse_message(["", "detailschange", "p2a: Garchomp", "Garchomp-Mega-Z, L50"])
    b.parse_message(["", "-mega", "p2a: Garchomp", "Garchomp", "Garchompite Z"])
    mon = b.get_pokemon("p2a: Garchomp")
    assert [t.name for t in mon.types if t is not None] == ["DRAGON"]
    assert mon.ability == "levitate"
    assert mon.base_stats["spe"] == _DEX["garchompmegaz"]["baseStats"]["spe"]
    assert b.opponent_used_mega_evolve


def test_own_z_mega_closes_the_partner_mega_option():
    """Gap 5: our own Garchomp-Mega-Z now counts as 'the team has mega'd'."""
    b = _battle()
    b.parse_message(["", "switch", "p1a: Garchomp", "Garchomp, L50", "100/100"])
    b.parse_message(["", "detailschange", "p1a: Garchomp", "Garchomp-Mega-Z, L50"])
    b.parse_message(["", "-mega", "p1a: Garchomp", "Garchomp", "Garchompite Z"])
    assert is_mega_forme_live(b.get_pokemon("p1a: Garchomp"))
    assert team_has_megaed_live(b)


def test_a_non_mega_is_not_flagged():
    b = _battle()
    b.parse_message(["", "switch", "p2a: Incineroar", "Incineroar, L50", "100/100"])
    b.parse_message(["", "switch", "p2b: Garchomp", "Garchomp, L50", "100/100"])
    assert not is_mega_forme_live(b.get_pokemon("p2a: Incineroar"))
    assert not is_mega_forme_live(b.get_pokemon("p2b: Garchomp"))
    assert not team_has_megaed_live(b)


def test_one_stone_two_formes_meowstic():
    assert F.resolve_mega_forme("meowsticf", "Meowsticite") == "meowsticfmega"
    assert F.resolve_mega_forme("meowstic", "Meowsticite") == "meowsticmmega"
    assert F.resolve_mega_forme("meowstic", "Meowsticite", female=True) == "meowsticfmega"
    assert F.resolve_mega_forme("garchomp", "Garchompite") == "garchompmega"
    assert F.resolve_mega_forme("garchomp", "Garchompite Z") == "garchompmegaz"
    assert F.resolve_mega_forme("garchomp", "Choice Scarf") is None
    assert F.resolve_mega_forme("garchomp", "Absolite Z") is None      # a stone of another species


def test_install_is_idempotent():
    from poke_env.battle.pokemon import Pokemon
    patched = Pokemon.mega_evolve
    assert F.install() is True
    assert Pokemon.mega_evolve is patched


def test_meowstic_and_tatsugiri_are_mega_capable():
    """The 'M-Mega' / 'Curly-Mega' names failed a startswith('Mega') check → our mega option was closed."""
    from v_dance.encoders.state_encoder import _species_is_mega_capable
    for sp in ("Meowstic", "Meowstic-F", "Tatsugiri", "Garchomp", "Golisopod"):
        assert _species_is_mega_capable(sp), sp
    assert not _species_is_mega_capable("Incineroar")      # (Meganium HAS a Champions mega)
