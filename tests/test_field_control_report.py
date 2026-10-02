"""The weather / terrain control report (2026-10-02, USER: "terrain wars and weather wars"). Locked on a real ladder
game read by hand (…2691473369): Indeedee's Psychic Terrain at the lead → Rillaboom's Grassy Terrain on turn 1 (a
LOSS with Indeedee still on the field, never reclaimed) → Politoed's rain on turn 4 → Tyranitar's sand on turn 5 →
Grassy ends, Rillaboom re-sets it on turn 6 → we win on turn 8."""
from __future__ import annotations

from v_dance.eval.field_control_report import parse_replay, team_setters

REPLAY = """|player|p1|VictoriousDancing|cynthia|1203
|player|p2|Temeoxd|101|1236
|switch|p1a: Corviknight|Corviknight, L50, F|205/205
|switch|p1b: Indeedee|Indeedee, L50, M|135/135
|switch|p2a: Gengar|Gengar, L50, M|100/100
|switch|p2b: Incineroar|Incineroar, L50, M|100/100
|-fieldstart|move: Psychic Terrain|[from] ability: Psychic Surge|[of] p1b: Indeedee
|turn|1
|switch|p2a: Rillaboom|Rillaboom, L50, F|100/100
|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge|[of] p2a: Rillaboom
|turn|2
|turn|3
|turn|4
|switch|p2a: Politoed|Politoed, L50, F|100/100
|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Politoed
|turn|5
|switch|p1b: Excadrill|Excadrill, L50, M|187/187
|switch|p1a: Tyranitar|Tyranitar, L50, M|192/192
|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar
|-weather|Sandstorm|[upkeep]
|-fieldend|move: Grassy Terrain
|turn|6
|-weather|Sandstorm|[upkeep]
|switch|p2b: Rillaboom|Rillaboom, L50, F|83/100
|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge|[of] p2b: Rillaboom
|turn|7
|turn|8
|win|VictoriousDancing</script>"""

PASTE = """Tyranitar @ Tyranitarite
Ability: Sand Stream
Jolly Nature
- Rock Slide

Indeedee-F (F) @ Choice Scarf
Ability: Psychic Surge
Modest Nature
- Expanding Force

Excadrill @ Focus Sash
Ability: Sand Rush
- Iron Head
"""


def test_team_setters_reads_the_sheet():
    assert team_setters(PASTE) == {"weather": {"sand": "tyranitar"}, "terrain": {"psychic": "indeedee"}}


def test_the_hand_read_game():
    g = parse_replay(REPLAY, {"victoriousdancing", "encorefn"}, team_setters(PASTE))
    assert g["won"] is True and g["turns"] == 8
    assert g["held"] == {"weather": 3, "terrain": 1} and g["lost"] == {"weather": 1, "terrain": 6}
    assert g["lead"] == {"weather": None, "terrain": "ours"}
    assert g["opp_kinds"] == {"weather": ["rain"], "terrain": ["grassy"]}
    assert [(e["domain"], e["turn"], e["setter_alive"], e["setter_active"], e["reclaimed"]) for e in g["events"]] == \
        [("terrain", 1, True, True, False)]                 # rain arrived over NO weather: not a loss of ours


def test_the_second_account_is_recognised_and_a_reclaim_is_seen():
    text = (REPLAY.replace("VictoriousDancing", "EncoreFN")
            .replace("|turn|2\n", "|turn|2\n|switch|p1b: Corviknight|Corviknight, L50, F|205/205\n"
                     "|switch|p1b: Indeedee|Indeedee, L50, M|135/135\n"
                     "|-fieldstart|move: Psychic Terrain|[from] ability: Psychic Surge|[of] p1b: Indeedee\n"))
    g = parse_replay(text, {"victoriousdancing", "encorefn"}, team_setters(PASTE))
    assert g is not None and g["won"] is True
    assert g["events"][0]["reclaimed"] is True and g["events"][0]["reclaim_turn"] == 2
    assert parse_replay(REPLAY, {"someoneelse"}, team_setters(PASTE)) is None    # not our game
