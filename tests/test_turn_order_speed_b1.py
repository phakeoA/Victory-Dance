"""v11 Phase B.1a (#4) — _effective_speed speed-ability corrections:
  · Protosynthesis/Quark Drive (incl. Booster Energy) ×1.5 when boosting SPEED (gated on the SPE-suffixed
    paradox volatile; byte-parity via volatile_flags.paradox_speed which poke-env mirrors).
  · Quick Feet ×1.5 on ANY status AND it NEGATES the paralysis ×0.5 (net ×1.5, NOT ×0.75).
  · Sticky Web is a no-op here (its −1 Spe is already a boost stage); weather-speed already handled.
Verified vs pokemon-showdown/data/abilities.ts (B.1 design workflow wf_1c1bdc85-206).
"""
from __future__ import annotations

import math

from v_dance.encoders.state_encoder import _effective_speed
from v_dance.parser.vod_parser.battle_models import volatile_flags

BASE_SPE = 100.0


def _mon(*, status=None, ability=None, spe_stage=0, paradox_spe=False):
    vol = {"paradox_speed": True} if paradox_spe else {}
    return {
        "species": "Flutter Mane", "base_species": "Flutter Mane",
        "known_ability": ability, "status": status,
        "boosts": {"spe": spe_stage},
        "volatiles": vol,
        "stats_estimate": {"mode": "exact", "stats": {"spe": BASE_SPE}},
    }


def _spe(mon, tailwind=False, weather=None):
    return _effective_speed(mon, tailwind, weather)[0]


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


# ── volatile_flags carries the paradox_speed key ──────────────────────────────
def test_volatile_flags_paradox_speed():
    assert volatile_flags({"protosynthesisspe"})["paradox_speed"] is True
    assert volatile_flags({"quarkdrivespe"})["paradox_speed"] is True
    assert volatile_flags({"protosynthesisatk"})["paradox_speed"] is False    # non-spe boost
    assert volatile_flags({"protosynthesis"})["paradox_speed"] is False       # bare id never appears
    assert volatile_flags(set())["paradox_speed"] is False


# ── paradox-spe ×1.5 ──────────────────────────────────────────────────────────
def test_paradox_spe_boost():
    assert _close(_spe(_mon(paradox_spe=True)), BASE_SPE * 1.5)
    assert _close(_spe(_mon(paradox_spe=False)), BASE_SPE)
    # a non-spe paradox boost (atk best stat) must NOT touch speed
    m = _mon()
    m["volatiles"] = volatile_flags({"protosynthesisatk"})
    assert _close(_spe(m), BASE_SPE)


# ── Quick Feet ×1.5 + paralysis negation ──────────────────────────────────────
def test_quick_feet_negates_paralysis():
    qf_par = _spe(_mon(status="par", ability="Quick Feet"))
    assert _close(qf_par, BASE_SPE * 1.5), "paralyzed Quick Feet must be ×1.5 (para cut negated)"
    assert not _close(qf_par, BASE_SPE * 0.75)   # would be the buggy stack
    assert not _close(qf_par, BASE_SPE * 0.5)


def test_quick_feet_on_any_status():
    for st in ("brn", "psn", "tox", "slp", "frz"):
        assert _close(_spe(_mon(status=st, ability="Quick Feet")), BASE_SPE * 1.5), st


def test_quick_feet_unstatused_is_neutral():
    assert _close(_spe(_mon(status=None, ability="Quick Feet")), BASE_SPE)


def test_paralyzed_non_quick_feet_still_halved():
    assert _close(_spe(_mon(status="par", ability="Protosynthesis")), BASE_SPE * 0.5)


# ── Sticky Web (boost stage) is already handled — no double count ──────────────
def test_sticky_web_minus1_is_just_the_boost_stage():
    # Sticky Web applies -1 spe as a boost stage; a -1 stage multiplier is 2/(2+1) = 2/3, nothing extra.
    assert _close(_spe(_mon(spe_stage=-1)), BASE_SPE * (2.0 / 3.0))


# ── composition ───────────────────────────────────────────────────────────────
def test_multiplier_composition():
    # paralyzed Quick Feet (×1.5) + Tailwind (×2) + paradox-spe (×1.5) = ×4.5
    m = _mon(status="par", ability="Quick Feet", paradox_spe=True)
    assert _close(_spe(m, tailwind=True), BASE_SPE * 1.5 * 2.0 * 1.5)


# ── T2.1: a mega uses its ACTIVE (forme) ability for weather-speed, not its base ability ──────
def _mega(*, mega_ability, pre_mega_ability, species="Swampert"):
    return {
        "species": f"{species}-Mega", "base_species": species,
        "is_mega": True, "pre_mega_ability": pre_mega_ability, "mega_ability": mega_ability,
        "known_ability": mega_ability, "status": None, "boosts": {"spe": 0},
        "volatiles": {},
        "stats_estimate": {"mode": "exact", "stats": {"spe": BASE_SPE}},
    }


def test_mega_active_speed_ability_under_weather():
    # Mega Swampert: base Torrent → mega Swift Swim. Under rain the ACTIVE (mega) ability drives the
    # ×2 weather-speed — matching the live path (poke-env mon.ability) and Showdown. Before the T2.1 fix
    # the offline encoder resolved the BASE (Torrent) ability here → no ×2 → a 2× offline↔live divergence.
    m = _mega(mega_ability="Swift Swim", pre_mega_ability="Torrent")
    assert _close(_spe(m, weather="RAINDANCE"), BASE_SPE * 2.0), "active Swift Swim → ×2 in rain"
    assert _close(_spe(m, weather=None), BASE_SPE)
    assert _close(_spe(m, weather="SUNNYDAY"), BASE_SPE)        # wrong weather → no boost


def test_mega_base_speed_ability_is_ignored():
    # Inverse: Mega Venusaur's BASE forme has Chlorophyll, but its mega ability is Thick Fat. Under sun
    # the ACTIVE (Thick Fat) ability must NOT grant the ×2 the base Chlorophyll would — active wins.
    m = _mega(mega_ability="Thick Fat", pre_mega_ability="Chlorophyll", species="Venusaur")
    assert _close(_spe(m, weather="SUNNYDAY"), BASE_SPE), "active Thick Fat → no sun-speed boost"


# ── #6: weather-speed abilities key on the RAW weather token both encoders receive ────────────
def test_slush_rush_doubles_under_snowscape():
    # gen-9 Snow surfaces as token "SNOWSCAPE" (poke-env Weather.SNOWSCAPE / Showdown |-weather|Snowscape),
    # NOT "SNOW" (the canonical one-hot alias). Slush Rush must ×2 under it.
    assert _close(_spe(_mon(ability="Slush Rush"), weather="SNOWSCAPE"), BASE_SPE * 2.0)
    assert _close(_spe(_mon(ability="Slush Rush"), weather="HAIL"), BASE_SPE * 2.0)      # legacy Hail token
    assert _close(_spe(_mon(ability="Slush Rush"), weather="RAINDANCE"), BASE_SPE)       # wrong weather
    assert _close(_spe(_mon(ability="Swift Swim"), weather="SNOWSCAPE"), BASE_SPE)       # wrong ability


def test_other_weather_speed_abilities_unbroken():
    assert _close(_spe(_mon(ability="Swift Swim"), weather="RAINDANCE"), BASE_SPE * 2.0)
    assert _close(_spe(_mon(ability="Chlorophyll"), weather="SUNNYDAY"), BASE_SPE * 2.0)
    assert _close(_spe(_mon(ability="Sand Rush"), weather="SANDSTORM"), BASE_SPE * 2.0)
