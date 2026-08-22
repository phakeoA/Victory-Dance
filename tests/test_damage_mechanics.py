"""B1-mechanics step 3: damage-correctness primitives (effective move type for Normalize/Pixilate,
species weight from the dex, variable base power for Body Press-family / Heavy Slam / Low Kick / etc.)."""
from __future__ import annotations

import pytest

from v_dance.encoders import damage_mechanics as DM
from v_dance.encoders.mechanic_vocab import _DEX

pytestmark = pytest.mark.skipif(not (_DEX / "moves.ts").exists(), reason="pinned dex not present")


def test_effective_move_type_ate_and_normalize():
    # -ate abilities convert NORMAL moves only
    assert DM.effective_move_type("tackle", "Normal", "pixilate") == "FAIRY"
    assert DM.effective_move_type("tackle", "Normal", "refrigerate") == "ICE"
    assert DM.effective_move_type("tackle", "Normal", "aerilate") == "FLYING"
    assert DM.effective_move_type("tackle", "Normal", "galvanize") == "ELECTRIC"
    assert DM.effective_move_type("tackle", "Normal", "dragonize") == "DRAGON"   # custom
    # a non-Normal move is unaffected by an -ate
    assert DM.effective_move_type("flamethrower", "Fire", "pixilate") == "FIRE"
    # Normalize makes EVERYTHING Normal
    assert DM.effective_move_type("flamethrower", "Fire", "normalize") == "NORMAL"
    # no / irrelevant ability -> unchanged
    assert DM.effective_move_type("tackle", "Normal", None) == "NORMAL"
    assert DM.effective_move_type("tackle", "Normal", "intimidate") == "NORMAL"


def test_effective_move_type_liquidvoice_sound_only():
    # Liquid Voice converts SOUND moves to Water; hypervoice/boomburst are sound
    assert "hypervoice" in DM._SOUND_MOVES
    assert DM.effective_move_type("hypervoice", "Normal", "liquidvoice") == "WATER"
    # a non-sound move is unaffected by Liquid Voice
    assert DM.effective_move_type("tackle", "Normal", "liquidvoice") == "NORMAL"


def test_species_weight_from_dex():
    assert len(DM.SPECIES_WEIGHT) > 800
    assert 85 <= DM.species_weight("Charizard") <= 95         # 90.5 kg
    assert DM.species_weight("Snorlax") >= 400                # 460 kg
    assert DM.species_weight("Gastly") < 1                    # 0.1 kg
    assert DM.species_weight("not_a_species") is None


def test_variable_base_power_weight():
    assert DM.variable_base_power("lowkick", 1, target_weight=210) == 120
    assert DM.variable_base_power("lowkick", 1, target_weight=5) == 20
    assert DM.variable_base_power("grassknot", 1, target_weight=60) == 80
    assert DM.variable_base_power("heavyslam", 1, attacker_weight=200, target_weight=10) == 120
    assert DM.variable_base_power("heavyslam", 1, attacker_weight=100, target_weight=90) == 40


def test_variable_base_power_speed_hp_boosts_counts():
    # Gyro Ball: slower user vs faster target -> high BP. Showdown: floor(25*tgt/usr)+1 (the +1 was missing).
    assert DM.variable_base_power("gyroball", 1, attacker_speed=50, target_speed=200) == 101
    assert DM.variable_base_power("gyroball", 1, attacker_speed=200, target_speed=50) == 7    # floor(25*50/200)+1
    assert DM.variable_base_power("gyroball", 1, attacker_speed=1, target_speed=999) == 150   # clamped to 150
    assert DM.variable_base_power("eruption", 1, attacker_hp_frac=1.0) == 150
    assert DM.variable_base_power("waterspout", 1, attacker_hp_frac=0.5) == 75
    assert DM.variable_base_power("storedpower", 1, attacker_pos_boosts=3) == 80
    assert DM.variable_base_power("punishment", 1, target_pos_boosts=10) == 200      # capped
    assert DM.variable_base_power("lastrespects", 1, fainted_allies=3) == 200
    assert DM.variable_base_power("ragefist", 1, times_hit=2) == 150


def test_variable_base_power_graceful_fallbacks():
    # a normal move keeps its static BP
    assert DM.variable_base_power("flamethrower", 90) == 90
    # a variable move with missing context falls back to static BP
    assert DM.variable_base_power("lowkick", 50, target_weight=None) == 50
