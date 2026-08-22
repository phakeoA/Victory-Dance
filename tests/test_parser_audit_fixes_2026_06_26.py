"""Regression tests for the 2026-06-26 parser audit fixes (#00 / #12 / #13).

These lock the corpus-quality fixes that drove the BC anchor from val 0.526 -> 0.549:
  #00  a forced post-faint replacement must NOT leak into a turn's our_actions as a
       phantom turn-start switch label (it stays a separate decision_type='replacement').
  #12  Heal Bell / Aromatherapy / -cureteam must cure BENCHED mons (bare 'pN: Nick' idents).
  #13  a Species-Clause-relabeled, still-disguised Zoroark must faint ITSELF, not its twin.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("poke_env")  # transitions/belief import path pulls poke_env in some envs

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
from v_dance.parser.vod_parser.battle_models import PokemonSlot
from v_dance.parser.vod_parser.transitions import replay_to_transitions

HEADER = """\
|player|p1|alice|101|1500
|player|p2|bob|102|1500
|gen|9
|tier|[Gen 9] Test Doubles
|poke|p1|Floette-Eternal, L50, F|
|poke|p1|Sneasler, L50, F|
|poke|p1|Incineroar, L50, F|
|poke|p1|Kingambit, L50, M|
|poke|p2|Aerodactyl, L50, M|
|poke|p2|Meganium, L50, M|
|poke|p2|Kingambit, L50, M|
|poke|p2|Rotom-Wash, L50|
|teamsize|p1|4
|teamsize|p2|4
|start
|switch|p1a: Floette|Floette-Eternal, L50, F|100/100
|switch|p1b: Sneasler|Sneasler, L50, F|100/100
|switch|p2a: Kingambit|Kingambit, L50, M|100/100
|switch|p2b: Aerodactyl|Aerodactyl, L50, M|100/100
"""


def _parse(body: str) -> ShowdownReplayParser:
    p = ShowdownReplayParser(HEADER + body, our_player="p1")
    p.parse()
    return p


def _find(parser, nickname):
    return next(m for m in parser.seen_mons.values() if m.nickname == nickname)


# ── #13: disguised-Zoroark faint guard (direct _faint_mon unit test) ───────────
def test_faint_mon_disguised_zoroark_faints_itself_not_twin():
    p = ShowdownReplayParser("", our_player="p1")
    zoro = PokemonSlot(species="Zoroark", nickname="Zoroark", player="p1", slot="a",
                       base_species="Zoroark", illusion_active=True, disguise_species="Incineroar")
    p.active_slots["p1a"] = zoro
    twin = PokemonSlot(species="Incineroar", nickname="Incineroar", player="p1", slot="b",
                       base_species="Incineroar")
    p.seen_mons["p1:Incineroar"] = twin
    p._faint_mon("p1a", "Incineroar")            # the |faint| line still shows the DISGUISE species
    assert zoro.is_fainted is True               # the actual Zoroark in the slot faints
    assert twin.is_fainted is False              # NOT the genuine teammate of the named species


def test_faint_mon_still_redirects_on_genuine_desync():
    # the guard must NOT weaken the original Illusion-desync redirect: a wrong slot pointer with NO
    # active disguise still re-targets the faint to the named species.
    p = ShowdownReplayParser("", our_player="p1")
    wrong = PokemonSlot(species="Pikachu", nickname="Pikachu", player="p1", slot="a",
                        base_species="Pikachu")
    p.active_slots["p1a"] = wrong
    real = PokemonSlot(species="Incineroar", nickname="Incineroar", player="p1", slot="b",
                       base_species="Incineroar")
    p.seen_mons["p1:Incineroar"] = real
    p._faint_mon("p1a", "Incineroar")
    assert real.is_fainted is True
    assert wrong.is_fainted is False


# ── #12: bench cure (Heal Bell / -cureteam) ───────────────────────────────────
_STATUS_THEN_BENCH = (
    "|turn|1\n"
    "|-status|p1a: Floette|psn\n"
    "|switch|p1a: Incineroar|Incineroar, L50, F|100/100\n"   # Floette -> bench, retains psn
)


def test_curestatus_clears_benched_mon():
    not_cured = _parse(_STATUS_THEN_BENCH + "|turn|2\n")
    cured = _parse(_STATUS_THEN_BENCH
                   + "|turn|2\n|-curestatus|p1: Floette|psn|[from] move: Heal Bell\n")
    assert _find(not_cured, "Floette").status == "psn"   # control: status persists on the bench
    assert _find(cured, "Floette").status is None        # #12: a bare-ident bench cure clears it


def test_cureteam_clears_benched_mon():
    cured = _parse(_STATUS_THEN_BENCH + "|turn|2\n|-cureteam|p1: Floette\n")
    assert _find(cured, "Floette").status is None        # #12: -cureteam clears the whole party


# ── #00: forced post-faint replacement excluded from turn-start our_actions ────
def test_forced_replacement_not_a_turn_start_action(tmp_path):
    # turn 1: p1a Floette is KO'd by p2 BEFORE it can act, then force-replaced by Incineroar.
    body = (
        "|turn|1\n"
        "|move|p2a: Kingambit|Sucker Punch|p1a: Floette\n"
        "|-damage|p1a: Floette|0 fnt\n"
        "|faint|p1a: Floette\n"
        "|move|p1b: Sneasler|Close Combat|p2a: Kingambit\n"
        "|-damage|p2a: Kingambit|50/100\n"
        "|switch|p1a: Incineroar|Incineroar, L50, F|100/100\n"   # FORCED post-faint replacement
        "|turn|2\n"
        "|move|p1a: Incineroar|Fake Out|p2a: Kingambit\n"
        "|move|p1b: Sneasler|Close Combat|p2a: Kingambit\n"
        "|faint|p2a: Kingambit\n"
        "|win|alice\n"
    )
    html = f'<script class="battle-log-data">{HEADER}{body}</script>'
    rp = tmp_path / "rep.html"
    rp.write_text(html, encoding="utf-8")

    transitions = replay_to_transitions(rp)
    t1 = [t for t in transitions if t["turn"] == 1 and t["perspective"] == "p1"
          and t.get("decision_type") == "turn"]
    assert t1, "expected a turn-1 p1 turn-decision transition"
    acts = t1[0]["our_actions"]
    # #00: the forced p1a (our_a) replacement switch must NOT appear as a turn-start action
    assert all(not (a["slot"] == "our_a" and a["action"] == "switch") for a in acts), acts
    # the genuine turn-start decision (Sneasler / our_b moved) is still present
    assert any(a["slot"] == "our_b" and a["action"] == "move" for a in acts), acts
    # ...and the replacement IS still captured as its own decision_type='replacement' transition
    repl = [t for t in transitions if t.get("decision_type") == "replacement"
            and t["perspective"] == "p1"]
    assert any(a["action"] == "switch" for t in repl for a in t["our_actions"]), repl
