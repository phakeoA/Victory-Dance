"""Matchup rules — narrow, named overrides for a known ladder leak (USER 2026-10-01).

The loss breakdown (``v_dance/eval/loss_breakdown.py``) found the Baltimore_Sand_Psy team's biggest leak:
Rillaboom is in 30 % of its games and it wins 37 % of them (Grassy Surge overwrites our Psychic Terrain; Grass hits
Tyranitar). When the picker DID bring Salamence into those games it won 52 % (21 games) vs 34 % without, and
Mega Salamence 7/12 vs Mega Tyranitar 46/135 — Mega Salamence's Flying-type Hyper Voice (Aerilate) + Flamethrower
beat Rillaboom whatever the terrain. Imitation can't learn "Salamence is THIS team's Rillaboom answer", so a rule
sidesteps the terrain war instead of trying to teach it. NOT vs Archaludon (it resists Flying: 26 % with Salamence
vs 42 % without).

A rule fires at TEAM PREVIEW (the picker still chooses the other three + the leads, constrained to sets that
contain ``bring``) and then holds the battle's one mega for ``mega``. Rules are opt-in PER BANDIT ARM
(``"matchup_rules": [...]`` in config/serve_bandit.json) so the ladder A/Bs them; no arm = no rule. Pure.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

RULES = {
    "rillaboom_salamence": {
        "when_opp_has": ("rillaboom",),
        "unless_opp_has": ("archaludon",),
        "bring": "salamence",
        "lead": True,           # it must be ON THE FIELD to mega: benched, it would only hold Tyranitar's mega hostage
        "mega": "salamence",
        "why": "Rillaboom leak (37 % in 30 % of games); Mega Salamence beats it whatever the terrain",
    },
}


def toid(s) -> str:
    return "".join(c for c in str(s or "").lower() if c.isalnum())


def active_rule(rule_names: Optional[Iterable[str]], our_species: Sequence, opp_species: Sequence) -> Optional[dict]:
    """The first named rule whose conditions hold for this preview (+ ``name`` and the roster index of ``bring``),
    else None. Our species must include ``bring``; the opponent must show one of ``when_opp_has`` and none of
    ``unless_opp_has`` (base-species ids: a preview never shows a mega forme)."""
    ours = [toid(s) for s in our_species]
    opp = {toid(s) for s in opp_species}
    for name in rule_names or ():
        r = RULES.get(name)
        if not r or r["bring"] not in ours:
            continue
        if opp & set(r["when_opp_has"]) and not opp & set(r["unless_opp_has"]):
            return {"name": name, **r, "bring_index": ours.index(r["bring"])}
    return None


def mega_override(rule: Optional[dict], team_species_alive: Iterable[str], slot_species: Sequence[Optional[str]],
                  slot_moves: Sequence[bool], slot_can_mega: Sequence[bool], picks: Sequence[int], *,
                  mega: int, none: int) -> list:
    """Apply the rule's mega to one turn's per-slot gimmick ``picks``. While the rule's mon is alive and not yet
    mega'd (``team_species_alive`` holds its un-mega'd id), it megas whenever it moves and can; every OTHER slot's
    mega is held. Once it has mega'd or fainted the rule is spent and ``picks`` pass through unchanged."""
    out = list(picks)
    if not rule or not rule.get("mega") or rule["mega"] not in {toid(s) for s in team_species_alive}:
        return out
    target = rule["mega"]
    for slot, sp in enumerate(slot_species):
        if toid(sp) == target:
            if slot_moves[slot] and slot_can_mega[slot]:
                out[slot] = mega
        elif out[slot] == mega:
            out[slot] = none
    return out
