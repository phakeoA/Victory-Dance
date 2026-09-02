"""Matchup book (USER 2026-09-02) — per-regulation, per-OUR-team opponent-species tables for the
Online tab: which Pokémon we face most, our win % against them, the item they ran (revealed) else
the belief default marked "(belief)". USER ruling: M-B/The_Big_6, M-B/maw_zard and M-A/maw_zard are
three DIFFERENT tables. Torch-free; the dossier loader is exercised on the real file schema."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from v_dance.play import matchup_book as MB
from v_dance.play.matchup_book import ALL_TEAMS, UNKNOWN_TEAM, MatchupBook

MB_FMT = "gen9championsvgc2026regmb"
MA_FMT = "gen9championsvgc2026regma"


class FakeBelief:
    def top_item(self, sp):
        return {"incineroar": "Leftovers", "garchomp": "Life Orb"}.get(sp)

    def item_distribution(self, sp, top_k=5):
        return {"incineroar": [{"name": "Leftovers", "p": 0.6026}, {"name": "Life Orb", "p": 0.24}],
                "garchomp": [{"name": "Life Orb", "p": 0.41}]}.get(sp, [])

    def top_ability(self, sp):
        return {"incineroar": "Intimidate"}.get(sp)

    def _resolve(self, sp):
        return {"incineroar": "Incineroar", "garchomp": "Garchomp"}.get(sp)


@pytest.fixture(autouse=True)
def _no_data_files(monkeypatch):
    # keep the team-sheet display maps out of the test — the title-case fallback is asserted on
    monkeypatch.setattr(MB, "_display", lambda kind, raw: MB._title(raw))


def _tag(fmt, n):
    return f"battle-{fmt}-{n}"


def _mons(*specs):
    """'species:item/ability' shorthand -> the hook's dict shape."""
    out = []
    for s in specs:
        sp, _, rest = s.partition(":")
        item, _, ab = rest.partition("/")
        out.append({"species": sp, "item": item or None, "ability": ab or None})
    return out


def test_record_and_summary_math():
    b = MatchupBook()
    assert b.record_game(MB_FMT, "The_Big_6", "ai", _mons("incineroar", "garchomp"),
                         tag=_tag(MB_FMT, 1), opponent="A")
    assert b.record_game(MB_FMT, "The_Big_6", "human", _mons("incineroar"),
                         tag=_tag(MB_FMT, 2), opponent="B")
    assert b.record_game(MB_FMT, "The_Big_6", "draw", _mons("incineroar", "garchomp"),
                         tag=_tag(MB_FMT, 3), opponent="C")
    s = b.summary(MB_FMT, team="The_Big_6", belief=FakeBelief())
    rows = {r["species"]: r for r in s["rows"]}
    inc = rows["incineroar"]
    assert (inc["games"], inc["wins"], inc["losses"], inc["draws"]) == (3, 1, 1, 1)
    assert inc["win_pct"] == 50                       # draws are not decided games
    g = rows["garchomp"]
    assert (g["games"], g["wins"], g["losses"], g["win_pct"]) == (2, 1, 0, 100)
    assert s["rows"][0]["species"] == "incineroar"    # most games first
    assert s["footer"] == {"games": 3, "opponents": 3, "species": 2, "teams": ["The_Big_6"]}
    assert inc["display"] == "Incineroar" and inc["opponents"] == 3
    assert b.summary(MB_FMT, team="The_Big_6", top=1)["rows"][0]["species"] == "incineroar"
    assert len(b.summary(MB_FMT, team="The_Big_6", top=1)["rows"]) == 1


def test_item_precedence_seen_then_belief_then_none():
    b, bel = MatchupBook(), FakeBelief()
    b.record_game(MB_FMT, "T", "ai", _mons("incineroar:unknown_item", "garchomp", "sneasler"),
                  tag=_tag(MB_FMT, 1), opponent="A")
    rows = {r["species"]: r for r in b.summary(MB_FMT, team="T", belief=bel)["rows"]}
    inc = rows["incineroar"]
    # belief-only: the prior renormalised over the belief's listed items (0.6026 / 0.8426 = 72 %)
    assert (inc["item"], inc["item_src"], inc["item_pct"], inc["item_n"]) == ("Leftovers", "belief", 72, 0)
    assert [(e["name"], e["p"], e["seen"]) for e in inc["items"]] == [("Leftovers", 72, 0), ("Life Orb", 28, 0)]
    assert (inc["ability"], inc["ability_src"]) == ("Intimidate", "belief")
    assert rows["sneasler"]["item"] is None and rows["sneasler"]["item_src"] == "none"
    assert rows["sneasler"]["ability_src"] == "none"
    # no belief at all -> "none", never an exception
    assert b.summary(MB_FMT, team="T")["rows"][0]["item_src"] == "none"
    # a revealed item beats the belief and counts ONCE per opponent per species
    b.record_game(MB_FMT, "T", "ai", _mons("incineroar:lifeorb/intimidate"), tag=_tag(MB_FMT, 2), opponent="B")
    b.record_game(MB_FMT, "T", "human", _mons("incineroar:lifeorb"), tag=_tag(MB_FMT, 3), opponent="B")
    b.record_game(MB_FMT, "T", "human", _mons("incineroar:sitrusberry"), tag=_tag(MB_FMT, 4), opponent="C")
    inc = {r["species"]: r for r in b.summary(MB_FMT, team="T", belief=bel)["rows"]}["incineroar"]
    # a seen id that the belief also lists takes the belief's display name ("Life Orb", not "Lifeorb")
    assert (inc["item"], inc["item_src"], inc["item_n"]) == ("Life Orb", "seen", 1)
    assert (inc["ability"], inc["ability_src"], inc["ability_n"]) == ("Intimidate", "seen", 1)
    assert inc["games"] == 4 and inc["opponents"] == 3
    # "" (knocked off / confirmed itemless) is not a run item -> still the belief
    b.record_game(MB_FMT, "T", "ai", [{"species": "garchomp", "item": ""}], tag=_tag(MB_FMT, 5), opponent="D")
    g = {r["species"]: r for r in b.summary(MB_FMT, team="T", belief=bel)["rows"]}["garchomp"]
    assert (g["item"], g["item_src"]) == ("Life Orb", "belief")


def test_per_team_and_per_format_tables_are_separate():
    b = MatchupBook()
    b.record_game(MB_FMT, "The_Big_6", "ai", _mons("incineroar"), tag=_tag(MB_FMT, 1), opponent="A")
    b.record_game(MB_FMT, "maw_zard", "human", _mons("incineroar"), tag=_tag(MB_FMT, 2), opponent="B")
    b.record_game(MA_FMT, "maw_zard", "human", _mons("incineroar"), tag=_tag(MA_FMT, 3), opponent="C")
    b.record_game(MB_FMT, None, "ai", _mons("garchomp"), tag=_tag(MB_FMT, 4), opponent="D")
    big = b.summary(MB_FMT, team="The_Big_6")["rows"]
    assert [(r["species"], r["wins"], r["losses"]) for r in big] == [("incineroar", 1, 0)]
    maw_b = b.summary(MB_FMT, team="maw_zard")
    assert [(r["species"], r["wins"], r["losses"]) for r in maw_b["rows"]] == [("incineroar", 0, 1)]
    maw_a = b.summary(MA_FMT, team="maw_zard")               # same team, other regulation
    assert [(r["species"], r["wins"], r["losses"]) for r in maw_a["rows"]] == [("incineroar", 0, 1)]
    assert maw_a["footer"]["games"] == 1 and maw_b["footer"]["games"] == 1
    every = b.summary(MB_FMT, team=ALL_TEAMS)
    rows = {r["species"]: r for r in every["rows"]}
    assert (rows["incineroar"]["wins"], rows["incineroar"]["losses"]) == (1, 1)
    assert rows["garchomp"]["wins"] == 1
    assert every["footer"]["games"] == 3
    assert every["footer"]["teams"] == sorted([UNKNOWN_TEAM, "The_Big_6", "maw_zard"])
    assert b.teams(MB_FMT) == [{"team": UNKNOWN_TEAM, "games": 1}, {"team": "The_Big_6", "games": 1},
                               {"team": "maw_zard", "games": 1}]
    assert b.formats() == [MA_FMT, MB_FMT]
    assert b.summary("gen9championsvgc2026regmc", team=ALL_TEAMS)["rows"] == []
    assert b.summary(MB_FMT, team="never_played")["footer"]["games"] == 0


def test_session_scope_is_this_process_only():
    b = MatchupBook()
    b.record_game(MB_FMT, "T", "ai", _mons("incineroar"), tag=_tag(MB_FMT, 1), session_id="S1", opponent="A")
    b.record_game(MB_FMT, "T", "human", _mons("incineroar"), tag=_tag(MB_FMT, 2), opponent="B")  # a seed row
    s1 = b.summary(MB_FMT, session_id="S1")
    assert [(r["species"], r["wins"], r["losses"]) for r in s1["rows"]] == [("incineroar", 1, 0)]
    assert s1["footer"] == {"games": 1, "opponents": 1, "species": 1, "teams": []}
    assert b.summary(MB_FMT, session_id="S2") == {"rows": [], "footer": {"games": 0, "opponents": 0,
                                                                         "species": 0, "teams": []}}
    alltime = b.summary(MB_FMT, team="T")
    assert (alltime["rows"][0]["wins"], alltime["rows"][0]["losses"]) == (1, 1)
    assert alltime["footer"]["games"] == 2


def test_tag_dedupe_and_bad_input_never_raise():
    b = MatchupBook()
    assert b.record_game(None, "T", "ai", _mons("incineroar"),
                         tag="battle-gen9championsvgc2026regmb-77-abcpw", opponent="A")
    assert not b.record_game(MB_FMT, "T", "ai", _mons("incineroar"), tag=_tag(MB_FMT, 77), opponent="A")
    assert not b.record_game(None, "T", "ai", _mons("incineroar"), tag="not-a-tag")
    assert b.record_game(MB_FMT, "T", "bogus-result", [None, 42, "garchomp", {"species": ""}],
                         tag=_tag(MB_FMT, 78))
    s = b.summary(MB_FMT, team="T")
    assert s["footer"]["games"] == 2
    rows = {r["species"]: r for r in s["rows"]}
    assert rows["garchomp"]["games"] == 1 and rows["garchomp"]["win_pct"] is None
    assert b.record_battle(object(), {}) is False


def test_load_dossiers_real_schema(tmp_path: Path):
    d = tmp_path / "dossiers"
    d.mkdir()
    (d / "alice.json").write_text(json.dumps({
        "opponent": "Alice", "wins_vs_us": 1, "losses_vs_us": 1, "draws_vs_us": 0,
        "games": [
            {"ts": "2026-07-21T13:41:01Z", "battle_tag": _tag(MB_FMT, 1), "result": "human", "turns": 8,
             "our_team": "maw_zard", "note": "online", "revealed": ["blaziken", "incineroar"]},
            {"ts": "2026-09-01T10:00:00Z", "battle_tag": _tag(MB_FMT, 2), "result": "ai", "turns": 5,
             "our_team": "The_Big_6", "note": "online", "revealed": ["incineroar", "garchomp"]},
            {"ts": "2026-09-01T11:00:00Z", "battle_tag": None, "result": "ai",
             "our_team": "The_Big_6", "revealed": ["garchomp"]},                 # tag-less -> skipped
        ],
        "mons": {"incineroar": {"species": "incineroar", "moves": [], "item": "lifeorb",
                                "ability": "intimidate", "times_seen": 2},
                 "blaziken": {"species": "blaziken", "moves": [], "item": None,
                              "ability": "speedboost", "times_seen": 1},
                 "garchomp": {"species": "garchomp", "moves": [], "item": None, "ability": None,
                              "times_seen": 1}}}), encoding="utf-8")
    (d / "bob.json").write_text(json.dumps({
        "opponent": "Bob",
        "games": [{"ts": "2026-08-01T00:00:00Z", "battle_tag": _tag(MA_FMT, 9), "result": "ai",
                   "our_team": "maw_zard", "revealed": ["incineroar"]}],
        "mons": {"incineroar": {"item": "leftovers"}}}), encoding="utf-8")
    (d / "corrupt.json").write_text("{not json", encoding="utf-8")
    b = MatchupBook()
    assert b.load_dossiers(d) == (3, 2)
    assert b.skipped == 2                             # the corrupt file + the tag-less game
    big = {r["species"]: r for r in b.summary(MB_FMT, team="The_Big_6")["rows"]}
    assert (big["incineroar"]["item"], big["incineroar"]["item_src"]) == ("Lifeorb", "seen")
    assert (big["incineroar"]["ability"], big["incineroar"]["ability_src"]) == ("Intimidate", "seen")
    assert (big["incineroar"]["wins"], big["incineroar"]["losses"]) == (1, 0)
    maw_b = {r["species"]: r for r in b.summary(MB_FMT, team="maw_zard")["rows"]}
    assert set(maw_b) == {"blaziken", "incineroar"} and maw_b["incineroar"]["losses"] == 1
    assert maw_b["blaziken"]["ability"] == "Speedboost" and maw_b["blaziken"]["item_src"] == "none"
    maw_a = {r["species"]: r for r in b.summary(MA_FMT, team="maw_zard")["rows"]}
    assert (maw_a["incineroar"]["item"], maw_a["incineroar"]["wins"]) == ("Leftovers", 1)
    # the item proxy is per OPPONENT: Alice's Incineroar counts once though it played 2 games
    every = {r["species"]: r for r in b.summary(MB_FMT, team=ALL_TEAMS)["rows"]}
    assert every["incineroar"]["item_n"] == 1 and every["incineroar"]["games"] == 2
    assert every["incineroar"]["last_seen"] == "2026-09-01T10:00:00Z"
    assert b.teams(MB_FMT) == [{"team": "The_Big_6", "games": 1}, {"team": "maw_zard", "games": 1}]
    assert "matchup book: 3 game(s) from 2 dossier(s)" in b.banner() and "2 skipped" in b.banner()
    # a live game already in the seed (same base tag) is not double-counted
    assert not b.record_game(MB_FMT, "The_Big_6", "ai", _mons("incineroar"), tag=_tag(MB_FMT, 2),
                             opponent="Alice", live=True)
    assert b.live_games == 0
    # a missing directory is a no-op
    assert MatchupBook().load_dossiers(tmp_path / "nope") == (0, 0)


def test_record_battle_from_poke_env_like_objects():
    class Mon:
        def __init__(self, species, item, ability):
            self.species, self.item, self.ability = species, item, ability

    class Battle:
        battle_tag = _tag(MB_FMT, 5) + "-privsuffix"
        opponent_username = "Carol"
        opponent_team = {"p2: Incineroar": Mon("incineroar", "unknown_item", "intimidate"),
                         "p2: Garchomp": Mon("garchomp", "lifeorb", None)}

    b = MatchupBook()
    row = {"battle_tag": _tag(MB_FMT, 5), "ai_team": "The_Big_6", "result": "ai",
           "opponent": "Carol", "session_id": "S9", "ts": "2026-09-02T10:00:00Z"}
    assert b.record_battle(Battle(), row)
    rows = {r["species"]: r for r in b.summary(MB_FMT, session_id="S9", belief=FakeBelief())["rows"]}
    assert (rows["incineroar"]["item_src"], rows["incineroar"]["ability_src"]) == ("belief", "seen")
    assert (rows["garchomp"]["item"], rows["garchomp"]["item_src"]) == ("Lifeorb", "seen")
    assert b.live_games == 1
    assert b.summary(MB_FMT, team="The_Big_6")["footer"]["games"] == 1
    # the session id can come from the controller when the row lacks it
    row2 = {"battle_tag": _tag(MB_FMT, 6), "ai_team": "The_Big_6", "result": "human"}
    assert b.record_battle(Battle(), row2, session_id="S9")
    assert b.summary(MB_FMT, session_id="S9")["footer"]["games"] == 2


def test_one_off_sightings_blend_with_the_belief_instead_of_becoming_fact():
    """USER 09-02: 'we can't expect all Sneaslers from 63 games to be running Pressure, Life Orb'.
    Revealed sightings are evidence for the opponents who showed them; the belief covers the rest."""
    class SneaslerBelief(FakeBelief):
        def item_distribution(self, sp, top_k=5):
            return [{"name": "Focus Sash", "p": 0.45}, {"name": "Life Orb", "p": 0.20},
                    {"name": "Choice Band", "p": 0.15}, {"name": "Sitrus Berry", "p": 0.05}]

        def ability_distribution(self, sp, top_k=4):
            return [{"name": "Poison Touch", "p": 0.55}, {"name": "Unburden", "p": 0.40},
                    {"name": "Pressure", "p": 0.05}]

    b = MatchupBook()
    for i in range(60):                                        # 60 opponents, one Life Orb / Pressure
        mons = _mons("sneasler:lifeorb/pressure") if i == 0 else _mons("sneasler")
        b.record_game(MB_FMT, "T", "ai" if i % 2 else "human", mons, tag=_tag(MB_FMT, i),
                      opponent=f"opp{i}")
    r = b.summary(MB_FMT, team="T", belief=SneaslerBelief())["rows"][0]
    assert r["pop"] == 60 and r["item_known"] == 1 and r["ability_known"] == 1
    assert [(e["name"], e["p"], e["seen"]) for e in r["items"]] == \
        [("Focus Sash", 52, 0), ("Life Orb", 25, 1), ("Choice Band", 17, 0)]      # prior renormalised
    assert [(e["name"], e["p"], e["seen"]) for e in r["abilities"]] == \
        [("Poison Touch", 54, 0), ("Unburden", 39, 0), ("Pressure", 7, 1)]
    assert (r["item"], r["item_src"], r["item_pct"]) == ("Focus Sash", "belief", 52)   # not "Life Orb"
    assert (r["ability"], r["ability_src"]) == ("Poison Touch", "belief")
    # 15 Life Orbs among 53 opponents → the sighting share leads and says so
    b2 = MatchupBook()
    for i in range(53):
        b2.record_game(MB_FMT, "T", "ai", _mons("garchomp:lifeorb" if i < 15 else "garchomp"),
                       tag=_tag(MB_FMT, 100 + i), opponent=f"g{i}")
    r2 = b2.summary(MB_FMT, team="T", belief=SneaslerBelief())["rows"][0]
    assert r2["items"][0]["name"] == "Life Orb" and r2["items"][0]["seen"] == 15
    assert r2["items"][0]["p"] == int(round(100 * (15 + 38 * 0.20 / 0.85) / 53))
    # no belief: the rest is honestly "unknown", and a lone sighting never becomes the headline
    r3 = b.summary(MB_FMT, team="T")["rows"][0]
    assert [(e["name"], e["p"], e["seen"]) for e in r3["items"]] == [("?", 98, 0), ("Lifeorb", 2, 1)]
    assert r3["item"] is None and r3["item_src"] == "none"
    assert r3["abilities"][1] == {"name": "Pressure", "p": 2, "seen": 1, "mega": False}


def test_mega_evolution_proves_the_stone_and_labels_the_ability():
    """USER bug 09-02: a Gardevoir row read 'Choice Scarf 5% (belief) · Pixilate' — Pixilate IS
    Mega Gardevoir's ability, so the item is Gardevoirite (revealed by the mega), not a belief."""
    if MB._dex() is None:
        pytest.skip("data/pokedex.json not available")
    assert MB.mega_of("gardevoir", "pixilate") == {"forme": "Gardevoir-Mega", "ability": "Pixilate",
                                                   "stone": "Gardevoirite"}
    assert MB.mega_of("gardevoir", "trace") is None                     # a base ability
    assert MB.mega_of("charizard", "drought")["stone"] == "Charizardite Y"
    assert MB.mega_of("charizard", "toughclaws")["forme"] == "Charizard-Mega-X"
    assert MB.mega_of("charizard", None, is_mega=True) is None          # X or Y? unknowable
    assert MB.mega_of("latios", "levitate") is None                     # shared base/mega ability
    assert MB.mega_of("latios", "levitate", is_mega=True)["stone"] == "Latiosite"
    assert MB.mega_of("scizor", "technician") is None and MB.mega_of("scizor", "technician", is_mega=True)
    assert MB.mega_of("gardevoirmega", None)["stone"] == "Gardevoirite"  # a mega forme id itself
    assert MB.mega_of("incineroar", "intimidate") is None and MB.mega_of("", "x") is None
    b, bel = MatchupBook(), FakeBelief()
    b.record_game(MB_FMT, "T", "human", _mons("gardevoir:/pixilate"), tag=_tag(MB_FMT, 1), opponent="A")
    b.record_game(MB_FMT, "T", "ai", _mons("gardevoir:/trace"), tag=_tag(MB_FMT, 2), opponent="B")
    b.record_game(MB_FMT, "T", "ai", [{"species": "latios", "ability": "levitate", "mega": True}],
                  tag=_tag(MB_FMT, 3), opponent="C")
    rows = {r["species"]: r for r in b.summary(MB_FMT, team="T", belief=bel)["rows"]}
    g = rows["gardevoir"]
    assert (g["item"], g["item_src"], g["item_n"]) == ("Gardevoirite", "seen", 1)
    assert (g["ability"], g["ability_src"], g["ability_mega"], g["mega"]) == ("Pixilate", "seen", True, 1)
    lat = rows["latios"]
    assert (lat["item"], lat["item_src"], lat["ability"], lat["ability_mega"]) == ("Latiosite", "seen", "Levitate", False)
    assert lat["mega"] == 1
    # the seed path infers the same from a dossier's last-seen mega ability
    b2 = MatchupBook()
    b2.record_game(MB_FMT, "T", "ai", [{"species": "gardevoir", "item": None, "ability": "pixilate"}],
                   tag=_tag(MB_FMT, 9), opponent="Z")
    r = b2.summary(MB_FMT, team="T")["rows"][0]
    assert r["item"] == "Gardevoirite" and r["ability_mega"] and r["mega"] == 1


def test_display_species_fallbacks():
    assert MB.display_species("incineroar", FakeBelief()) == "Incineroar"
    # no belief hit -> the pokedex name (title-cased) when data is present, else the id itself
    out = MB.display_species("ninetalesalola", FakeBelief())
    assert out in ("Ninetales-Alola", "ninetalesalola")
    assert MB.display_species("", None) == ""
    assert MB.fmt_of_tag(">battle-gen9championsvgc2026regmb-12-xyz") == MB_FMT
    assert MB.base_tag("battle-gen9championsvgc2026regmb-12-xyz") == _tag(MB_FMT, 12)
    assert MB.fmt_of_tag("garbage") is None and MB.base_tag(None) == ""
