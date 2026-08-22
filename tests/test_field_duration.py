"""Field-duration tracking (weather/terrain/screens) — parser counts turns-ACTIVE (elapsed).

The parser counts how long each condition has been up and removes it only on the explicit end
event (|-weather|none / |-fieldend| / |-sideend|), NOT by auto-expiring at 5. So a Light Clay /
weather-rock 8-turn instance is tracked correctly, and `elapsed > 5` itself reveals the extended
(8-turn) instances identically offline (parser) and live (poke-env start-turn). See state_encoder
field-duration block + battle_models.FieldConditions/SideConditions.
"""
from __future__ import annotations

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser

_HEAD = """|gen|9
|player|p1|Me|1|
|player|p2|Foe|2|
|teamsize|p1|2
|teamsize|p2|2
|poke|p1|Torkoal, L50, F|
|poke|p2|Grimmsnarl, L50, M|
|start
|switch|p1a: Torkoal|Torkoal, L50, F|100/100
|switch|p2a: Grimmsnarl|Grimmsnarl, L50, M|100/100
"""


def _weather_log(active_turns: int) -> str:
    log = _HEAD + "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal\n|turn|1\n"
    log += "|move|p2a: Grimmsnarl|Reflect|p2a: Grimmsnarl\n|-sidestart|p2: Foe|move: Reflect\n"
    for t in range(active_turns):
        log += "|-weather|SunnyDay|[upkeep]\n|upkeep\n" + f"|turn|{t + 2}\n"
    return log


def test_weather_and_screen_count_up():
    p = ShowdownReplayParser(_weather_log(3), our_player="p1")
    p.parse()
    assert p.field_conditions.weather == "SunnyDay"
    # weather set pre-|turn|1 → aged at each of turns 1..4 (turn-anchored, matches poke-env)
    assert p.field_conditions.weather_turns == 4
    assert p.side_conditions["p2"].screens == {"reflect": 3}  # set during turn 1 → aged turns 2..4


def test_screen_survives_past_five_turns():
    # the old code auto-expired a screen at turn 5; an 8-turn (Light Clay) screen must NOT vanish
    p = ShowdownReplayParser(_weather_log(7), our_player="p1")
    p.parse()
    assert "reflect" in p.side_conditions["p2"].screens
    assert p.side_conditions["p2"].screens["reflect"] == 7   # elapsed > 5 ⇒ Light Clay instance


def test_end_event_removes_condition():
    log = _weather_log(2)
    log += "|-weather|none\n|-sideend|p2: Foe|Reflect\n|upkeep\n|turn|5\n"
    p = ShowdownReplayParser(log, our_player="p1")
    p.parse()
    assert p.field_conditions.weather is None
    assert p.field_conditions.weather_turns == 0
    assert "reflect" not in p.side_conditions["p2"].screens


def test_to_dict_exposes_elapsed():
    p = ShowdownReplayParser(_weather_log(2), our_player="p1")
    p.parse()
    fd = p.field_conditions.to_dict()
    assert fd["weather_turns_active"] == 3 and fd["terrain_turns_active"] == 0   # turns 1..3
    assert p.side_conditions["p2"].to_dict()["screens"] == {"reflect": 2}        # turns 2..3


# ── live ↔ offline parity for the 11-channel field-duration block (v10) ───────────
import numpy as np  # noqa: E402
import pytest  # noqa: E402

try:
    import tests._parity_harness as H  # noqa: E402
    from v_dance.encoders.state_encoder import STATE_DIM  # noqa: E402
    _HARNESS = True
except Exception:  # pragma: no cover
    _HARNESS = False

_FD = slice(STATE_DIM - 15, STATE_DIM - 4) if _HARNESS else None  # 11 fd channels before the 4 team-counts

_PARITY_LOG = """|player|p1|Me|1|
|player|p2|Foe|2|
|teamsize|p1|2
|teamsize|p2|2
|gen|9
|clearpoke
|poke|p1|Torkoal, L50, F|
|poke|p1|Grimmsnarl, L50, M|
|poke|p2|Pikachu, L50, M|
|poke|p2|Snorlax, L50, M|
|teampreview
|start
|switch|p1a: Torkoal|Torkoal, L50, F|100/100
|switch|p1b: Grimmsnarl|Grimmsnarl, L50, M|100/100
|switch|p2a: Pikachu|Pikachu, L50, M|100/100
|switch|p2b: Snorlax|Snorlax, L50, M|100/100
|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal
|turn|1
|move|p1b: Grimmsnarl|Reflect|p1b: Grimmsnarl
|-sidestart|p1: Me|move: Reflect
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|2
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|3
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|4
"""


@pytest.mark.skipif(not _HARNESS, reason="poke-env parity harness unavailable")
def test_field_duration_live_offline_parity():
    # splice=True → the live path uses the gap-#6 vod_parser reconstruction for the field block
    # (the production-honest path; poke-env's weather value is unusable as a start-turn).
    live = H.live_vectors_per_turn(_PARITY_LOG, "Me", splice=True)
    offline = H.offline_vectors_per_turn(_PARITY_LOG, "p1")
    assert offline, "offline produced no turns"
    checked = 0
    for turn in sorted(offline):
        assert turn in live, f"turn {turn} missing live"
        np.testing.assert_allclose(
            live[turn][_FD], offline[turn][_FD], atol=1e-6,
            err_msg=f"field-duration block diverges at turn {turn}: "
                    f"live={live[turn][_FD]} offline={offline[turn][_FD]}")
        checked += 1
    assert checked > 0
    # and the age actually grew (weather/reflect aged each turn, not stuck at 0)
    assert offline[max(offline)][_FD].max() > offline[min(offline)][_FD].max()


_TERRAIN_LOG = """|player|p1|Me|1|
|player|p2|Foe|2|
|teamsize|p1|2
|teamsize|p2|2
|gen|9
|clearpoke
|poke|p1|Pikachu, L50, M|
|poke|p1|Grimmsnarl, L50, M|
|poke|p2|Snorlax, L50, M|
|poke|p2|Garchomp, L50, M|
|teampreview
|start
|switch|p1a: Pikachu|Pikachu, L50, M|100/100
|switch|p1b: Grimmsnarl|Grimmsnarl, L50, M|100/100
|switch|p2a: Snorlax|Snorlax, L50, M|100/100
|switch|p2b: Garchomp|Garchomp, L50, M|100/100
|turn|1
|move|p1a: Pikachu|Electric Terrain|p1a: Pikachu
|-fieldstart|move: Electric Terrain
|upkeep
|turn|2
|upkeep
|turn|3
|upkeep
|turn|4
"""


@pytest.mark.skipif(not _HARNESS, reason="poke-env parity harness unavailable")
def test_field_duration_fallback_recovers_terrain_age():
    """The reconstruction-failure FALLBACK (splice=False) used to zero BOTH weather and terrain age.
    Terrain has a stable poke-env start-turn in battle.fields (like Trick Room), so the fix recovers
    the terrain channel; weather has no stable start-turn so it stays the documented residual 0."""
    live_fb = H.live_vectors_per_turn(_TERRAIN_LOG, "Me", splice=False)   # the production fallback path
    offline = H.offline_vectors_per_turn(_TERRAIN_LOG, "p1")
    assert offline, "offline produced no turns"
    weather_ch = STATE_DIM - 15 + 0      # channel 0 of the 11-channel field-duration block
    terrain_ch = STATE_DIM - 15 + 1      # channel 1 = terrain age
    saw_terrain = 0
    for turn in sorted(offline):
        if turn not in live_fb:
            continue
        # terrain age now MATCHES offline on the fallback (was forced 0 before the fix)
        np.testing.assert_allclose(live_fb[turn][terrain_ch], offline[turn][terrain_ch], atol=1e-6,
                                   err_msg=f"fallback terrain age != offline at turn {turn}")
        saw_terrain += int(offline[turn][terrain_ch] > 0)
        # weather stays the documented residual 0 on the fallback (no stable poke-env start-turn)
        assert live_fb[turn][weather_ch] == 0.0
    assert saw_terrain > 0, "fixture never had terrain up — test inert"
