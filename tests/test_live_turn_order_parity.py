"""Live turn-order effective-speed must resolve item/ability through BELIEF, not raw poke-env attrs.

Regression for the high-severity serve-time parity bug found in the v10 pre-bake hunt: the OFFLINE
encoder's _effective_speed resolves the Choice Scarf x1.5 and weather-speed-ability x2 multipliers via
resolve_item_json / resolve_ability_json, which for an unrevealed OPPONENT fall back to the Pikalytics
top prior. The live _live_effective_speed used to read raw mon.item ('unknown_item') / mon.ability (None),
silently dropping both multipliers for every unrevealed opponent — a 1.5-2x who-moves-first divergence on
the turn-order global channels that the gap-#6 opponent splice does NOT correct.
"""
from __future__ import annotations

from types import SimpleNamespace

from v_dance.encoders.live_state_encoder import LiveStateEncoder


class _StubBelief:
    """Minimal belief whose top prior is a Choice Scarf + Swift Swim opponent."""
    def top_item(self, species):
        return "Choice Scarf"

    def top_ability(self, species):
        return "Swift Swim"

    def expected_stats_weighted(self, *a, **k):
        return ({"spe": 100}, 0.5)


def _enc():
    enc = LiveStateEncoder(belief=_StubBelief())
    # isolate the multiplier logic from the est-stats / mega-detection machinery
    enc._live_est_stats = lambda mon, is_own: ({"spe": 100}, 0.5)
    enc._is_mega_forme = lambda mon: False
    return enc


def _unrevealed_opp():
    # exactly what poke-env exposes for an opponent that has not revealed its item/ability
    return SimpleNamespace(species="someopp", base_species="someopp",
                           item="unknown_item", ability=None, boosts={}, status=None)


def test_belief_choice_scarf_multiplier_applied_live():
    spe, known = _enc()._live_effective_speed(_unrevealed_opp(), is_own=False,
                                              side_tailwind=False, weather=None)
    assert spe == 150.0, "belief-top Choice Scarf x1.5 not applied on the live turn-order path"
    assert known == 0.5


def test_belief_weather_speed_ability_multiplier_applied_live():
    # Swift Swim in rain stacks on the Scarf: 100 * 1.5 * 2 = 300
    spe, _ = _enc()._live_effective_speed(_unrevealed_opp(), is_own=False,
                                          side_tailwind=False, weather="RAINDANCE")
    assert spe == 300.0, "belief-top Swift Swim x2 (rain) not applied on the live turn-order path"


def test_no_weather_speed_ability_outside_matching_weather():
    # Swift Swim does NOTHING in sun — only the Scarf applies
    spe, _ = _enc()._live_effective_speed(_unrevealed_opp(), is_own=False,
                                          side_tailwind=False, weather="SUNNYDAY")
    assert spe == 150.0


def test_revealed_item_still_wins_over_belief():
    # a concrete (revealed) non-Scarf item must NOT get the belief Scarf multiplier
    enc = _enc()
    mon = SimpleNamespace(species="someopp", base_species="someopp",
                          item="leftovers", ability=None, boosts={}, status=None)
    spe, _ = enc._live_effective_speed(mon, is_own=False, side_tailwind=False, weather=None)
    assert spe == 100.0, "revealed Leftovers wrongly got the belief Choice Scarf multiplier"
