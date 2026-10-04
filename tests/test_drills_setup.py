"""The DRILL league's launch-time setup and per-generation scoreboard (``v_dance.selfplay.drills``, 2026-10-02).

Under test, offline (pastes written to ``tmp_path``, passed as ABSOLUTE paths like ``resolve_team_path`` returns):
  · ``setup_drill``: opponent weights live only on exact pool entries and sum to 1, our own team is never an
    opponent, the target keys are the conflict (or focus-hit) team names, the field drill's ``ctx["ours"]`` is
    filled, bias 0 turns pressure off, a bad bias / an unknown drill raise ValueError;
  · ``parse_spec`` / ``get_drill``;
  · ``gen_scoreboard`` on the hand-read ladder game from ``test_field_control_report`` (our player
    VictoriousDancing): mirror + fallback rows left out and counted, an undecided log counted as skipped, the
    targets / by_kind splits, the pressure stats passed through, JSON-safe output; ``format_gen_line``;
  · the FOCUS drill scoring a Salamence-vs-Rillaboom set of games.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from test_field_control_report import REPLAY            # the hand-read ladder game (pytest prepend import)
from v_dance.selfplay.drills import (DrillSetup, format_gen_line, gen_scoreboard, get_drill, known_drills,
                                     parse_spec, setup_drill)

OUR = "VictoriousDancing"

# ── the team pastes ──────────────────────────────────────────────────────────
OURS = """Tyranitar @ Tyranitarite
Ability: Sand Stream
Jolly Nature
- Rock Slide

Indeedee-F (F) @ Psychic Seed
Ability: Psychic Surge
Modest Nature
- Expanding Force

Excadrill @ Focus Sash
Ability: Sand Rush
- Iron Head
"""
PASTES = {
    "Ours_Sand_Psy": OURS,
    "Rain_Grassy": "Pelipper @ Damp Rock\nAbility: Drizzle\n- Hurricane\n\n"
                   "Rillaboom @ Miracle Seed\nAbility: Grassy Surge\n- Grassy Glide\n",
    "Grassy_Only": "Rillaboom @ Assault Vest\nAbility: Grassy Surge\n- Fake Out\n\n"
                   "Incineroar @ Sitrus Berry\nAbility: Intimidate\n- Fake Out\n",
    "Chary_Y": "Charizard (M) @ Charizardite Y\nAbility: Blaze\n- Heat Wave\n\n"
               "Venusaur @ Life Orb\nAbility: Chlorophyll\n- Sleep Powder\n",
    "Sand_Mirror": "Hippowdon @ Leftovers\nAbility: Sand Stream\n- Earthquake\n",
    "Plain_A": "Garchomp @ Life Orb\nAbility: Rough Skin\n- Earthquake\n",
    "Plain_B": "Salamence @ Salamencite\nAbility: Intimidate\n- Draco Meteor\n\n"
               "Kingambit @ Black Glasses\nAbility: Defiant\n- Kowtow Cleave\n",
}
CONFLICT = ("Rain_Grassy", "Grassy_Only", "Chary_Y")
NEUTRAL = ("Sand_Mirror", "Plain_A", "Plain_B")
OURS_SETTERS = {"weather": {"sand": "tyranitar"}, "terrain": {"psychic": "indeedee"}}


def _write(folder: Path, name: str, text: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text(text, encoding="utf-8")
    return str(p.resolve())


@pytest.fixture
def teams(tmp_path):
    """{name: absolute path} of every paste, all in one reg folder."""
    return {n: _write(tmp_path / "M-C", n, t) for n, t in PASTES.items()}


@pytest.fixture
def field_setup(teams):
    pool = [teams[n] for n in PASTES]                       # our own team IS in the pool (as in a real run)
    return setup_drill("field", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)


# ── parse_spec / get_drill ───────────────────────────────────────────────────
def test_parse_spec_splits_name_and_args():
    assert parse_spec("field") == ("field", {})
    assert parse_spec("focus:opp=rillaboom,mon=salamence") == ("focus", {"opp": "rillaboom", "mon": "salamence"})
    assert parse_spec(" field : neutral = 0.3 , pressure=off ,") == ("field", {"neutral": "0.3", "pressure": "off"})
    assert parse_spec("focus:opp=rillaboom+amoonguss") == ("focus", {"opp": "rillaboom+amoonguss"})
    assert parse_spec("") == ("", {}) and parse_spec(None) == ("", {})


def test_known_drills_and_a_fresh_drill_has_an_empty_ctx():
    assert known_drills() == ["field", "focus", "megatime"]                 # 2026-10-03: + the mega-timing drill
    d = get_drill("field")
    assert d.name == "field" and d.pressure == "field" and d.ctx == {}     # only build_pool fills it (main process)
    assert get_drill("field:pressure=off").pressure is None
    assert get_drill("focus:opp=rillaboom").pressure is None
    m = get_drill("megatime")
    assert m.name == "megatime" and m.pressure == "field" and m.ctx == {}


def test_unknown_drill_and_an_empty_focus_raise_valueerror(teams):
    pool = list(teams.values())
    with pytest.raises(ValueError, match="unknown drill"):
        setup_drill("weather", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    with pytest.raises(ValueError, match="unknown drill"):
        get_drill("")
    with pytest.raises(ValueError, match="focus drill needs"):
        get_drill("focus")
    with pytest.raises(ValueError, match="no pool team carries"):
        setup_drill("focus:opp=kyogre", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)


# ── setup_drill: the FIELD drill ─────────────────────────────────────────────
def test_field_setup_weights_live_on_pool_entries_and_sum_to_one(teams, field_setup):
    s = field_setup
    pool = list(teams.values())
    assert isinstance(s, DrillSetup)
    assert set(s.opp_weights) <= set(pool)
    assert teams["Ours_Sand_Psy"] not in s.opp_weights          # never our own team
    assert set(s.opp_weights) == {teams[n] for n in CONFLICT + NEUTRAL}
    assert abs(sum(s.opp_weights.values()) - 1.0) < 1e-9
    assert abs(sum(s.pool.weights.values()) - 1.0) < 1e-9
    assert all(w > 0 and math.isfinite(w) for w in s.opp_weights.values())


def test_field_setup_follows_the_ladder_mix(teams, field_setup):
    """Default LADDER_MIX: grassy 678 · rain 356 · sun 195 share the 80 % conflict slice (6 teams: the 6 % cap is
    infeasible, so plain normalise). Rain_Grassy collects rain + half of grassy; Charizardite Y is a SUN setter."""
    w = field_setup.opp_weights
    tot = 678 + 356 + 195
    assert w[teams["Rain_Grassy"]] == pytest.approx(0.8 * (356 + 678 / 2) / tot)
    assert w[teams["Grassy_Only"]] == pytest.approx(0.8 * (678 / 2) / tot)
    assert w[teams["Chary_Y"]] == pytest.approx(0.8 * 195 / tot)
    for n in NEUTRAL:                                            # the sand mirror is no fight: neutral
        assert w[teams[n]] == pytest.approx(0.2 / 3)
    sh = field_setup.pool.shares
    assert sh["neutral"] == pytest.approx(0.2, abs=1e-6)
    assert sh["weather:sun"] == pytest.approx(0.8 * 195 / tot, abs=1e-6)
    assert field_setup.pool.missing == ["weather:snow", "terrain:electric", "terrain:misty"]


def test_field_setup_targets_ctx_and_pressure(teams, field_setup):
    s = field_setup
    assert s.target_keys == frozenset(n.lower() for n in CONFLICT)          # team_key = lower-cased file name
    assert s.drill.ctx["ours"] == OURS_SETTERS                              # filled by build_pool
    assert s.pool.ours == OURS_SETTERS
    assert s.pressure == "field" and s.bias == 2.5
    assert "(3 target teams)" in s.pool.summary()


def test_neutral_arg_moves_the_neutral_share(teams):
    pool = list(teams.values())
    s = setup_drill("field:neutral=0.5", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    assert sum(s.opp_weights[teams[n]] for n in NEUTRAL) == pytest.approx(0.5)
    assert sum(s.opp_weights[teams[n]] for n in CONFLICT) == pytest.approx(0.5)


def test_own_team_is_excluded_under_another_reg_folder_too(tmp_path, teams):
    other = _write(tmp_path / "M-B", "Ours_Sand_Psy", OURS)           # same name, other folder → still us
    pool = [other] + list(teams.values())
    s = setup_drill("field", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    assert other not in s.opp_weights and teams["Ours_Sand_Psy"] not in s.opp_weights


def test_same_named_team_maps_to_the_first_pool_entry(tmp_path, teams):
    """The league de-duplicates by file name and keeps the FIRST entry (gauntlet.own_team_matchups): a team listed
    only under M-C, with an M-B entry of the same name earlier in the pool, must be keyed by the M-B entry."""
    mb = _write(tmp_path / "M-B", "Grassy_Only", PASTES["Grassy_Only"])
    pool = [mb] + [teams[n] for n in PASTES if n != "Grassy_Only"]
    s = setup_drill("field", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    assert mb in s.opp_weights and teams["Grassy_Only"] not in s.opp_weights
    assert set(s.opp_weights) <= set(pool)
    assert "grassy_only" in s.target_keys


def test_a_team_in_two_reg_folders_weighs_as_one_team(tmp_path, teams):
    mb = _write(tmp_path / "M-B", "Chary_Y", PASTES["Chary_Y"])
    other_sun = _write(tmp_path / "M-C", "Chary_Z", PASTES["Chary_Y"])         # an identical sun team, listed once
    pool = [mb] + list(teams.values()) + [other_sun]
    s = setup_drill("field", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    assert s.opp_weights[mb] == pytest.approx(s.opp_weights[other_sun])


def test_the_per_team_cap_holds_on_the_league_weights(tmp_path, teams):
    mb = _write(tmp_path / "M-B", "Grassy_Only", PASTES["Grassy_Only"])
    plains = [_write(tmp_path / "M-C", f"Plain_{i:02d}", PASTES["Plain_A"]) for i in range(20)]
    pool = [mb] + list(teams.values()) + plains                    # 26 opponents: the 6 % cap is feasible
    s = setup_drill("field", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    assert max(s.pool.weights.values()) <= 0.06 + 1e-9              # holds per PATH ...
    assert max(s.opp_weights.values()) <= 0.06 + 1e-9               # ... and must hold per TEAM


@pytest.mark.parametrize("bias", [-0.5, float("nan"), float("inf"), float("-inf")])
def test_a_bad_bias_raises_valueerror(teams, bias):
    with pytest.raises(ValueError, match="drill-bias"):
        setup_drill("field", own_team_path=teams["Ours_Sand_Psy"], team_pool=list(teams.values()), bias=bias)


def test_bias_zero_turns_pressure_off_but_keeps_the_pool(teams):
    pool = list(teams.values())
    s = setup_drill("field", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool, bias=0)
    assert s.pressure is None and s.bias == 0.0
    assert s.opp_weights and s.target_keys
    off = setup_drill("field:pressure=off", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool, bias=2.5)
    assert off.pressure is None and off.bias == 2.5


def test_a_pool_without_conflict_teams_has_no_targets(teams):
    pool = [teams[n] for n in ("Ours_Sand_Psy",) + NEUTRAL]
    s = setup_drill("field", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    assert s.target_keys == frozenset()
    assert s.opp_weights == pytest.approx({teams[n]: 1 / 3 for n in NEUTRAL})
    sb = gen_scoreboard(s, [_row(REPLAY, kind="snapshot", team_b=teams["Plain_A"])])
    assert "targets" not in sb and sb["games"] == 1


def test_an_empty_pool_raises_valueerror(teams):
    with pytest.raises(ValueError, match="no pool team got a weight"):
        setup_drill("field", own_team_path=teams["Ours_Sand_Psy"], team_pool=[teams["Ours_Sand_Psy"]])


# ── setup_drill: the FOCUS drill ─────────────────────────────────────────────
def test_focus_setup_weights_and_targets(teams):
    pool = list(teams.values())
    s = setup_drill("focus:opp=rillaboom,mon=salamence", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    hits, rest = ("Rain_Grassy", "Grassy_Only"), ("Chary_Y", "Sand_Mirror", "Plain_A", "Plain_B")
    assert set(s.opp_weights) == {teams[n] for n in hits + rest}            # own team excluded
    for n in hits:
        assert s.opp_weights[teams[n]] == pytest.approx(0.7 / 2)             # share 0.7 over the hits
    for n in rest:
        assert s.opp_weights[teams[n]] == pytest.approx(0.3 / 4)
    assert abs(sum(s.opp_weights.values()) - 1.0) < 1e-9
    assert s.target_keys == frozenset(n.lower() for n in hits)
    assert s.pressure is None                                                # focus has no opponent bias
    assert s.pool.shares == {"opp:rillaboom": pytest.approx(0.7), "other": pytest.approx(0.3)}
    assert s.pool.ours == OURS_SETTERS


def test_focus_cap_and_mon_only(teams):
    pool = list(teams.values())
    s = setup_drill("focus:opp=rillaboom,cap=0.3", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    assert max(s.opp_weights.values()) <= 0.3 + 1e-9                         # 6 teams x 0.3 ≥ 1: feasible cap
    assert s.opp_weights[teams["Plain_A"]] == pytest.approx(0.1)             # the excess spread over the rest
    m = setup_drill("focus:mon=salamence", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    assert set(m.opp_weights) == {teams[n] for n in CONFLICT + NEUTRAL}      # no opp= → every team is a hit
    assert m.opp_weights == pytest.approx({teams[n]: 1 / 6 for n in CONFLICT + NEUTRAL})


# ── gen_scoreboard (FIELD) ───────────────────────────────────────────────────
LOSS = REPLAY.replace(f"|win|{OUR}", "|win|Temeoxd")
NO_WIN = REPLAY.replace(f"|win|{OUR}</script>", "")
TRICKY = REPLAY.replace("|turn|2\n", "|move|p1a: Corviknight|Trick|p2a: Gengar\n"
                                     "|move|p1b: Indeedee|Switcheroo|p2b: Incineroar\n"
                                     "|move|p2a: Gengar|Trick|p1a: Corviknight\n|turn|2\n")   # theirs: not counted


def _row(text, *, our=OUR, kind="snapshot", team_b="", mirror=False, fallback=False, pressure=False):
    return {"our": our, "text": text, "kind": kind, "opp_ref": None, "team_b": team_b,
            "mirror": mirror, "fallback": fallback, "pressure": pressure}


@pytest.fixture
def field_rows(teams):
    return [
        _row(TRICKY, kind="snapshot", team_b=teams["Rain_Grassy"], pressure=True),          # target, won
        _row(LOSS, kind="clone", team_b="teams/Champions/M-C/Grassy_Only", pressure=True),  # target by NAME, lost
        _row(REPLAY, kind="latest", team_b=teams["Plain_A"]),                                # neutral, won
        _row(REPLAY, kind="latest", team_b=teams["Ours_Sand_Psy"], mirror=True, pressure=True),
        _row(REPLAY, kind="snapshot", team_b=teams["Ours_Sand_Psy"], mirror=True, fallback=True),
        _row(LOSS, kind="clone", team_b=teams["Chary_Y"], fallback=True),
        _row(NO_WIN, kind="scripted", team_b=teams["Chary_Y"], pressure=True),               # undecided
        _row(REPLAY, our="SomeoneElse", kind="snapshot", team_b=teams["Chary_Y"]),           # not our game
        _row(None, kind="snapshot", team_b=teams["Chary_Y"]),                                # empty log
    ]


def test_scoreboard_excludes_and_counts_mirror_fallback_and_skipped(field_setup, field_rows):
    sb = gen_scoreboard(field_setup, field_rows, {"fired": 7, "taken": 3})
    assert sb["drill"] == "field"
    assert sb["games"] == 3
    assert sb["fallback"] == 2                         # a fallback that is ALSO a mirror counts as a fallback
    assert sb["mirror"] == 1
    assert sb["skipped"] == 3                          # no |win| · not our game · empty log
    assert sb["games"] + sb["fallback"] + sb["mirror"] + sb["skipped"] == len(field_rows)


def test_scoreboard_field_metrics_on_the_hand_read_game(field_setup, field_rows):
    """Each scored game is the hand-read one (8 turns: we hold weather 3, terrain 1; one terrain loss with Indeedee
    alive, never reclaimed; rain over NO weather is not a loss of ours); one of the three is a loss."""
    sb = gen_scoreboard(field_setup, field_rows)
    assert sb["win"] == 0.667
    assert sb["trick_per_game"] == 0.667               # Trick + Switcheroo by our side in one game; theirs ignored
    assert sb["weather_fights"] == 3 and sb["terrain_fights"] == 3
    assert sb["weather_fight_win"] == 0.667 and sb["terrain_fight_win"] == 0.667
    assert sb["weather_share"] == 0.375 and sb["terrain_share"] == 0.125
    assert sb["weather_reclaim"] is None               # no weather loss event at all
    assert sb["terrain_reclaim"] == 0.0
    assert sb["weather_setter_used"] == 1.0 and sb["terrain_setter_used"] == 1.0


def test_scoreboard_targets_and_by_kind_splits(field_setup, field_rows):
    sb = gen_scoreboard(field_setup, field_rows)
    t = sb["targets"]
    assert t["games"] == 2 and t["win"] == 0.5         # Rain_Grassy (abs path) + Grassy_Only (repo-relative form)
    assert set(sb["by_kind"]) == {"snapshot", "clone", "latest"}        # only SCORED rows' kinds
    assert sb["by_kind"]["snapshot"]["games"] == 1 and sb["by_kind"]["snapshot"]["win"] == 1.0
    assert sb["by_kind"]["clone"]["games"] == 1 and sb["by_kind"]["clone"]["win"] == 0.0
    assert sb["by_kind"]["latest"]["games"] == 1 and sb["by_kind"]["latest"]["win"] == 1.0


def test_scoreboard_pressure_stats_pass_through(field_setup, field_rows):
    sb = gen_scoreboard(field_setup, field_rows, {"fired": 7, "taken": 3})
    assert sb["pressure"] == {"games": 2, "fired": 7, "taken": 3}       # games = SCORED rows with pressure
    assert gen_scoreboard(field_setup, field_rows)["pressure"] == {"games": 2}
    assert gen_scoreboard(field_setup, field_rows, None)["pressure"] == {"games": 2}


def test_scoreboard_is_json_safe(field_setup, field_rows):
    sb = gen_scoreboard(field_setup, field_rows, {"fired": 7, "taken": 3})
    assert json.loads(json.dumps(sb, allow_nan=False)) == sb
    empty = gen_scoreboard(field_setup, [], None)
    json.dumps(empty, allow_nan=False)
    assert empty == {"drill": "field", "games": 0, "skipped": 0, "mirror": 0, "fallback": 0,
                     "targets": {"games": 0}, "by_kind": {}, "pressure": {"games": 0}}
    assert gen_scoreboard(field_setup, None)["games"] == 0
    only_skipped = gen_scoreboard(field_setup, [_row(NO_WIN)])
    assert only_skipped["games"] == 0 and only_skipped["skipped"] == 1
    json.dumps(only_skipped, allow_nan=False)


def test_scoreboard_does_not_mutate_rows(field_setup, field_rows):
    before = json.dumps(field_rows, sort_keys=True)
    gen_scoreboard(field_setup, field_rows, {"fired": 1})
    assert json.dumps(field_rows, sort_keys=True) == before


def test_a_drill_rebuilt_without_build_pool_scores_nothing(field_setup, field_rows):
    """Why scoring is MAIN-process only: a drill re-resolved by name has an empty ctx → every game is a swallowed
    KeyError, counted as skipped (never a crash)."""
    fresh = DrillSetup(drill=get_drill("field"), pool=field_setup.pool, opp_weights=field_setup.opp_weights,
                       target_keys=field_setup.target_keys, pressure="field", bias=2.5)
    sb = gen_scoreboard(fresh, field_rows)
    assert sb["games"] == 0 and sb["skipped"] == 6 and sb["mirror"] == 1 and sb["fallback"] == 2


def test_format_gen_line(field_setup, field_rows):
    line = format_gen_line(gen_scoreboard(field_setup, field_rows, {"fired": 7, "taken": 3}))
    assert line.startswith("DRILL field: ")
    assert "games 3" in line and "win 67%" in line and "skipped 3" in line and "mirror 1" in line
    assert "fallback 2" in line
    assert "trick_per_game 0.67" in line               # a per-game count, not a percentage
    assert "weather_reclaim —" in line                 # None prints as a dash
    assert "| targets: games 2 · win 50%" in line
    assert line.endswith("| pressure: 2 g, fired 7, taken 3")
    assert "by_kind" not in line                       # nested dicts are not flattened into the line


def test_format_gen_line_on_an_empty_generation(field_setup):
    line = format_gen_line(gen_scoreboard(field_setup, []))
    assert line == "DRILL field: games 0 · skipped 0 · mirror 0 · fallback 0"
    err = format_gen_line({"games": 0, "error": "KeyError('ours')"})       # generation.py's failure stub
    assert err.startswith("DRILL ?: ") and "error KeyError('ours')" in err


# ── the FOCUS drill: Salamence vs Rillaboom ──────────────────────────────────
def _focus_log(*, ours_side="p1", leads=("Salamence", "Incineroar"), later=(),
               theirs=("Rillaboom", "Amoonguss"), we_win=True):
    opp_side = "p2" if ours_side == "p1" else "p1"
    names = {ours_side: OUR, opp_side: "Rival"}
    out = [f"|player|p1|{names['p1']}|1|1200", f"|player|p2|{names['p2']}|2|1200"]
    for slot, mon in zip("ab", leads):
        out.append(f"|switch|{ours_side}{slot}: {mon}|{mon}, L50, M|100/100")
    for slot, mon in zip("ab", theirs):
        out.append(f"|switch|{opp_side}{slot}: {mon}|{mon}, L50, F|100/100")
    out.append("|turn|1")
    for mon in later:
        out.append(f"|switch|{ours_side}a: {mon}|{mon}, L50, M|100/100")
    out += ["|turn|2", f"|win|{OUR if we_win else 'Rival'}"]
    return "\n".join(out)


def test_focus_drill_scores_salamence_vs_rillaboom(teams):
    pool = list(teams.values())
    s = setup_drill("focus:opp=rillaboom,mon=salamence", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    rows = [
        # A: Salamence LED vs Rillaboom, won
        _row(_focus_log(), kind="snapshot", team_b=teams["Rain_Grassy"]),
        # B: Salamence brought later (after turn 1, Mega forme) vs Rillaboom, lost
        _row(_focus_log(leads=("Garchomp", "Incineroar"), later=("Salamence-Mega",), we_win=False),
             kind="clone", team_b=teams["Rain_Grassy"]),
        # C: no Salamence vs Rillaboom, lost
        _row(_focus_log(leads=("Garchomp", "Incineroar"), we_win=False), kind="clone", team_b=teams["Grassy_Only"]),
        # D: we are p2; Rillaboom on OUR side only, the opponent never shows one → not an opp hit; won
        _row(_focus_log(ours_side="p2", leads=("Salamence", "Rillaboom"), theirs=("Kyogre", "Tornadus")),
             kind="latest", team_b=teams["Plain_A"]),
        _row(_focus_log(), kind="latest", team_b=teams["Ours_Sand_Psy"], mirror=True),
    ]
    sb = gen_scoreboard(s, rows)
    assert sb["drill"] == "focus" and sb["mirror"] == 1 and sb["skipped"] == 0
    assert sb["games"] == 4 and sb["win"] == 0.5
    assert sb["opp_seen_games"] == 3 and sb["opp_seen_win"] == 0.333
    assert sb["salamence_brought"] == 0.667            # A + B of the 3 Rillaboom games
    assert sb["salamence_led"] == 0.333                # A only (B came in after turn 1)
    assert sb["win_with_salamence"] == 0.5             # A won, B lost
    assert sb["win_without_salamence"] == 0.0          # C lost
    t = sb["targets"]                                  # A, B (Rain_Grassy) + C (Grassy_Only)
    assert t["games"] == 3 and t["win"] == 0.333 and t["opp_seen_games"] == 3
    assert sb["by_kind"]["latest"]["games"] == 1 and sb["by_kind"]["latest"]["opp_seen_games"] == 0
    assert sb["by_kind"]["latest"]["salamence_brought"] is None      # no opp-hit game in that split
    json.dumps(sb, allow_nan=False)
    line = format_gen_line(sb)
    assert line.startswith("DRILL focus: ") and "salamence_brought 67%" in line and "| targets: games 3" in line


def test_focus_drill_without_mon_and_an_undecided_game(teams):
    pool = list(teams.values())
    s = setup_drill("focus:opp=rillaboom", own_team_path=teams["Ours_Sand_Psy"], team_pool=pool)
    no_win = _focus_log().rsplit("\n", 1)[0]
    sb = gen_scoreboard(s, [_row(_focus_log(), team_b=teams["Grassy_Only"]), _row(no_win)])
    assert sb["games"] == 1 and sb["skipped"] == 1 and sb["win"] == 1.0 and sb["opp_seen_games"] == 1
    assert not any(k.endswith("_brought") or k.startswith("win_with") for k in sb)   # no mon= → no mon metrics
