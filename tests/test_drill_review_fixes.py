"""The 2026-10-02 drill-league review fixes (14 findings, each upheld by 2 skeptics) not already pinned elsewhere:
mega-stone setters + the one-mega rule, the unbiased drill draw, the no-ladder-data weight, unreadable teams reported,
the focus drill's true team-preview pick, and the DRILL line's per-opponent-kind split."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from v_dance.eval.field_control_report import ITEM_TERRAIN, ITEM_WEATHER, team_setters
from v_dance.eval.gauntlet import own_team_matchups
from v_dance.play import field_fight as FF
from v_dance.selfplay import drills as DR
from v_dance.selfplay.drill_pool import build_drill_pool


def test_mega_stones_that_set_a_field_are_setters():
    assert ITEM_WEATHER["charizarditey"] == "sun" and ITEM_WEATHER["froslassite"] == "snow"
    assert ITEM_WEATHER["abomasite"] == "snow" and ITEM_TERRAIN["raichunitex"] == "electric"
    froslass = "Froslass @ Froslassite\nAbility: Cursed Body\n- Blizzard\n"
    raichu = "Raichu @ Raichunite X\nAbility: Lightning Rod\n- Thunderbolt\n"
    assert team_setters(froslass) == {"weather": {"snow": "froslass"}, "terrain": {}}
    assert team_setters(raichu) == {"weather": {}, "terrain": {"electric": "raichu"}}


def test_a_mega_stone_setter_counts_only_while_the_side_can_still_mega():
    zard = SimpleNamespace(ability="blaze", item="charizarditey")
    assert FF.setter_kinds(zard, SimpleNamespace(used_mega_evolve=False)) == {"weather": "sun"}
    assert FF.setter_kinds(zard, SimpleNamespace(used_mega_evolve=True)) == {}      # the one mega is spent
    megazard = SimpleNamespace(ability="drought", item="charizarditey")              # already mega'd: its ability
    assert FF.setter_kinds(megazard, SimpleNamespace(used_mega_evolve=True)) == {"weather": "sun"}
    raichu = SimpleNamespace(ability="lightningrod", item="raichunitex")
    assert FF.setter_kinds(raichu) == {"terrain": "electric"}                        # no battle = can still mega


def test_the_drill_draw_is_unbiased_reproducible_and_reaches_small_weights():
    pool = ["own"] + [f"t{i:02d}" for i in range(60)]
    w = {"t00": 0.5, **{f"t{i:02d}": 0.5 / 59 for i in range(1, 60)}}                # one heavy team + 59 light ones
    a = own_team_matchups("own", pool, 100, mirror_frac=0.2, weights=w, seed=3, draw="multinomial")
    b = own_team_matchups("own", pool, 100, mirror_frac=0.2, weights=w, seed=3, draw="multinomial")
    assert a == b and sum(n for _, _, n in a) == 100                                 # reproducible, exact total
    light = sum(n for _, t, n in a if t not in ("own", "t00"))
    assert 25 <= light <= 55                                                         # ~40 expected (0.5 × 80)
    old = own_team_matchups("own", pool, 100, mirror_frac=0.2, weights=w, seed=3)
    assert old == own_team_matchups("own", pool, 100, mirror_frac=0.2, weights=w, seed=3,
                                    draw="largest_remainder")                        # the default is unchanged


PASTES = {
    "ours": "Torkoal @ Charcoal\nAbility: Drought\n- Eruption\n",                      # we set SUN only
    "sand": "Tyranitar @ Leftovers\nAbility: Sand Stream\n- Rock Slide\n",            # sand: NOT in LADDER_MIX
    "grassy": "Rillaboom @ Miracle Seed\nAbility: Grassy Surge\n- Grassy Glide\n",
    "plain": "Garchomp @ Life Orb\nAbility: Rough Skin\n- Earthquake\n",
}


def _reader(p):
    if p not in PASTES:
        raise OSError(p)
    return PASTES[p]


def test_a_kind_without_ladder_data_gets_the_mean_weight_and_is_reported():
    dp = build_drill_pool(["ours", "sand", "grassy", "plain", "ghost"], "ours", neutral_frac=0.2,
                          max_team_share=1.0, reader=_reader)
    assert dp.no_ladder_data == ["weather:sand"]
    assert dp.weights["sand"] > 0.1                                                  # was ~0.001 at the floor weight
    assert dp.unreadable == ["ghost"] and "UNREADABLE: ghost" in dp.summary()


def test_the_focus_drill_scores_brought_from_the_team_preview():
    d = DR.get_drill("focus:opp=rillaboom,mon=salamence")
    g = {"won": True, "brought": False, "led": False, "opp_hit": True}               # never switched in (log)
    g = d.refine(g, {"tp_brought": ["salamence", "tyranitar", "indeedee", "excadrill"], "tp_led": ["tyranitar"]})
    assert g["brought"] is True and g["entered"] is False and g["led"] is False
    keep = d.refine({"won": True, "brought": True, "led": True, "opp_hit": True}, {"tp_brought": None})
    assert keep["brought"] is True                                                   # no pick on the row: the log


def test_tp_species_reads_the_trajectory_meta():
    from v_dance.selfplay.mp_collect import _tp_species
    meta = SimpleNamespace(own_team=["Tyranitar", "Indeedee-F", "Salamence", "Excadrill", "Sneasler", "Corviknight"],
                           tp_bring=[0, 1, 2, 3], tp_leads=[2, 0])
    assert _tp_species(meta, "tp_bring") == ["tyranitar", "indeedee", "salamence", "excadrill"]
    assert _tp_species(meta, "tp_leads") == ["salamence", "tyranitar"]
    assert _tp_species(SimpleNamespace(), "tp_bring") is None


def test_the_drill_line_splits_by_opponent_kind():
    sb = {"drill": "field", "games": 10, "win": 0.5,
          "by_kind": {"latest": {"games": 6, "win": 0.5, "terrain_share": 0.4},
                      "snapshot": {"games": 4, "win": 0.25, "weather_share": 0.3}},
          "pressure": {"games": 4, "fired": 9, "taken": 5}}
    line = DR.format_gen_line(sb)
    assert "by opponent: latest 6g win 50% T 40% · snapshot 4g win 25% W 30%" in line
    assert "pressure: 4 g, fired 9, taken 5" in line
