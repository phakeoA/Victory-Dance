"""v11 Victory Star (Victini): ×1.1 accuracy for the holder OR its active ally.

Source: pokemon-showdown/data/abilities.ts victorystar — onAnyModifyAccuracy returns
chainModify([4506, 4096]) when ``source.isAlly(holder)`` (the holder itself counts as its own ally).
The encoder folds this into the shared accuracy helpers ``_accuracy_modifiers`` (the per-move accuracy
channel) and ``_per_enemy_hit_chance`` (the per-enemy hit-chance channels); the att_ctx ``victory_star``
flag = own-ability OR active-ally-ability == "victorystar", computed on BOTH paths → byte-parity.
0 Victini in-meta → forward-compat plumbing (encoder-only, no layout bump).
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    _per_enemy_hit_chance, _accuracy_modifiers,
)

VS = 4506 / 4096   # ≈1.10010


def _D(ability=None, eva_stage=0, confused=False, no_guard=False, evasion_item=False):
    return dict(ability=ability, eva_stage=eva_stage, confused=confused, no_guard=no_guard,
                evasion_item=evasion_item)


# ════════════════════════════ unit: the ×4506/4096 fold ════════════════════════════
def test_per_enemy_hit_chance_victory_star():
    base = 0.80
    no = _per_enemy_hit_chance(base, attacker_ability=None, acc_stage=0, wide_lens=False,
                               is_physical=True, is_ohko=False, defender=_D(), move_id="stoneedge",
                               weather=None, victory_star=False)
    yes = _per_enemy_hit_chance(base, attacker_ability=None, acc_stage=0, wide_lens=False,
                                is_physical=True, is_ohko=False, defender=_D(), move_id="stoneedge",
                                weather=None, victory_star=True)
    assert abs(no - 0.80) < 1e-9
    assert abs(yes - 0.80 * VS) < 1e-9


def test_accuracy_modifiers_victory_star():
    base = 0.80
    no = _accuracy_modifiers(base, ability_id=None, is_physical=True, is_ohko=False,
                             acc_stage=0, wide_lens=False, victory_star=False)
    yes = _accuracy_modifiers(base, ability_id=None, is_physical=True, is_ohko=False,
                              acc_stage=0, wide_lens=False, victory_star=True)
    assert abs(no - 0.80) < 1e-9
    assert abs(yes - 0.80 * VS) < 1e-9


def test_victory_star_stacks_with_wide_lens_and_clamps():
    # Wide Lens ×4505/4096 then Victory Star ×4506/4096; a 100% move clamps at 1.0.
    v = _accuracy_modifiers(0.70, ability_id=None, is_physical=True, is_ohko=False,
                            acc_stage=0, wide_lens=True, victory_star=True)
    assert abs(v - 0.70 * (4505 / 4096) * VS) < 1e-9
    full = _accuracy_modifiers(1.0, ability_id=None, is_physical=True, is_ohko=False,
                               acc_stage=0, wide_lens=False, victory_star=True)
    assert full == 1.0     # 100% × 1.1 clamped


def test_victory_star_ohko_unaffected():
    # OHKO moves bypass accuracy mods entirely (their fixed accuracy is a separate mechanic).
    v = _accuracy_modifiers(0.30, ability_id=None, is_physical=True, is_ohko=True,
                            acc_stage=0, wide_lens=False, victory_star=True)
    assert v == 0.30


# ════════════════════════════ e2e: ally + self threading ════════════════════════════
def _mon(species, ability, *, moves=()):
    return {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True, "is_fainted": False,
            "known_moves": list(moves), "revealed_moves": [], "boosts": {}, "status": None,
            "known_ability": ability, "volatiles": {},
            "stats_estimate": {"mode": "exact",
                               "stats": {"atk": 130, "spa": 100, "def": 100, "spd": 100, "hp": 300, "spe": 100}}}


def _hitchance(att, move, opp_a, our_b=None):
    snap = {"our_active": {"our_a": att, "our_b": our_b},
            "opp_active": {"opp_a": opp_a, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return float(vec[b + 23])     # hit-chance vs opp_a
    raise AssertionError(move)


def test_e2e_ally_victory_star_boosts_attacker():
    att = _mon("Garchomp", "Rough Skin", moves=["Stone Edge"])     # 80% accuracy, no Victory Star
    opp = _mon("Blissey", "Natural Cure")
    base = _hitchance(att, "Stone Edge", opp, our_b=None)
    assert abs(base - 0.80) < 1e-3
    # Ally is Victini with Victory Star → the attacker's hit chance is boosted ×4506/4096.
    victini = _mon("Victini", "Victory Star", moves=["V-create"])
    boosted = _hitchance(att, "Stone Edge", opp, our_b=victini)
    assert abs(boosted - 0.80 * VS) < 1e-3


def test_e2e_self_victory_star_boosts():
    # The holder itself benefits (isAlly includes self).
    victini = _mon("Victini", "Victory Star", moves=["Zen Headbutt"])   # Zen Headbutt = 90% accuracy
    opp = _mon("Blissey", "Natural Cure")
    boosted = _hitchance(victini, "Zen Headbutt", opp, our_b=None)
    assert abs(boosted - 0.90 * VS) < 1e-3


def test_e2e_no_victory_star_baseline():
    att = _mon("Garchomp", "Rough Skin", moves=["Stone Edge"])
    ally = _mon("Landorus-Therian", "Intimidate", moves=["Earthquake"])   # no Victory Star
    opp = _mon("Blissey", "Natural Cure")
    base = _hitchance(att, "Stone Edge", opp, our_b=ally)
    assert abs(base - 0.80) < 1e-3
