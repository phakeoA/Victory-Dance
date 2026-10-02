"""The DRILL team pool (2026-10-02, USER: "fight for terrain control and weather control"). Under test: conflict
teams = a weather / terrain KIND our team does not set; shares follow the ladder mix; ordinary teams keep
``neutral_frac``; one team never dominates; our own team is never an opponent; unfieldable kinds are reported."""
from __future__ import annotations

import collections

from v_dance.selfplay.drill_pool import build_drill_pool, classify

OURS = """Tyranitar @ Tyranitarite
Ability: Sand Stream
- Rock Slide

Indeedee-F @ Choice Scarf
Ability: Psychic Surge
- Expanding Force
"""
PASTES = {
    "ours": OURS,
    "rain_grassy": "Pelipper @ Damp Rock\nAbility: Drizzle\n- Hurricane\n\nRillaboom @ Miracle Seed\nAbility: Grassy Surge\n- Grassy Glide\n",
    "grassy_a": "Rillaboom @ Assault Vest\nAbility: Grassy Surge\n- Fake Out\n",
    "grassy_b": "Rillaboom @ Miracle Seed\nAbility: Grassy Surge\n- Wood Hammer\n",
    "sun": "Torkoal @ Charcoal\nAbility: Drought\n- Eruption\n",
    "psy_mirror": "Indeedee-F @ Psychic Seed\nAbility: Psychic Surge\n- Follow Me\n",
    "sand_mirror": "Hippowdon @ Leftovers\nAbility: Sand Stream\n- Earthquake\n",
    "plain": "Garchomp @ Life Orb\nAbility: Rough Skin\n- Earthquake\n",
}
POOL = list(PASTES)
MIX = {("terrain", "grassy"): 600, ("weather", "rain"): 300, ("weather", "sun"): 100, ("terrain", "electric"): 50}


def reader(p):
    return PASTES[p]


def test_conflict_is_a_kind_we_do_not_set():
    from v_dance.eval.field_control_report import team_setters
    kinds = {t.path: t.kinds for t in classify(POOL, team_setters(OURS), reader=reader)}
    assert kinds["rain_grassy"] == (("weather", "rain"), ("terrain", "grassy"))
    assert kinds["sun"] == (("weather", "sun"),)
    assert kinds["psy_mirror"] == () and kinds["sand_mirror"] == () and kinds["plain"] == ()   # no fight


def test_shares_follow_the_ladder_mix_and_keep_a_neutral_slice():
    dp = build_drill_pool(POOL, "ours", mix=MIX, neutral_frac=0.2, slots=400, max_team_share=1.0, reader=reader)
    assert "ours" not in dp.pool                                  # our own team is never the opponent
    c = collections.Counter(dp.pool)
    assert set(c) == {"rain_grassy", "grassy_a", "grassy_b", "sun", "psy_mirror", "sand_mirror", "plain"}
    assert abs(dp.shares["neutral"] - 0.2) < 0.03
    # grassy 600 : rain 300 : sun 100 of the 80 % conflict share; rain_grassy carries rain AND a third of grassy
    assert dp.shares["weather:sun"] < dp.shares["weather:rain"] < dp.shares["terrain:grassy"]
    assert abs(dp.shares["weather:sun"] - 0.8 * 100 / 1000) < 0.02
    assert dp.missing == ["terrain:electric"]                     # reported, not silently dropped
    assert "NO TEAM for: terrain:electric" in dp.summary()


def test_one_team_never_dominates():
    dp = build_drill_pool(POOL, "ours", mix=MIX, neutral_frac=0.2, slots=400, max_team_share=0.2, reader=reader)
    assert max(dp.weights.values()) <= 0.2 + 1e-9                 # water-filled: the excess went to the others
    assert abs(sum(dp.weights.values()) - 1.0) < 1e-9
    assert dp.weights["rain_grassy"] == max(dp.weights.values())  # still the heaviest — just capped


def test_an_infeasible_cap_falls_back_to_plain_weights():
    from v_dance.selfplay.drill_pool import cap_weights
    w = cap_weights({"a": 3.0, "b": 1.0}, cap=0.1)                # 2 teams can never all sit under 10 %
    assert w == {"a": 0.75, "b": 0.25}


def test_pool_weights_keys_are_the_first_pool_entry_per_team_name():
    from v_dance.selfplay.drill_pool import DrillPool, pool_weights
    pool = ["teams/Champions/M-B/The_Big_6_v2", "teams/Champions/M-C/The_Big_6_v2", "teams/Champions/M-C/X"]
    dp = DrillPool(pool=[], ours={}, weights={"teams/Champions/M-C/The_Big_6_v2": 0.6, "teams/Champions/M-C/X": 0.4})
    w = pool_weights(dp, pool)
    assert w == {"teams/Champions/M-B/The_Big_6_v2": 0.6, "teams/Champions/M-C/X": 0.4}   # what the league keeps
    assert set(w) <= set(pool)


def test_a_pool_without_conflict_teams_is_all_neutral():
    dp = build_drill_pool(["ours", "plain", "psy_mirror"], "ours", mix=MIX, reader=reader)
    assert set(dp.pool) == {"plain", "psy_mirror"} and dp.shares == {"neutral": 1.0}
