"""2026-09-03 (USER): the matchup tables — 'mausholdfour' displayed as its raw id (an unresolved belief key
echoed the id), 'unknown 100%' on a cosmetic forme (the belief keys the base forme), the Mega Aggron the USER
had just seen hidden by the 12-row cap; plus sort controls, scrolling, and mini sprites with a '?' fallback."""
from __future__ import annotations

from v_dance.play import matchup_book as MB
from v_dance.play.matchup_book import MatchupBook, base_species_id, display_species, sprite_id

FMT = "gen9championsvgc2026regmb"


class EchoBelief:
    """A belief whose resolver ECHOES an unknown id (what the real one does) and only knows the BASE forme."""
    def _resolve(self, sp):
        return {"maushold": "Maushold", "aggron": "Aggron"}.get(sp, sp)

    def item_distribution(self, sp, top_k=5):
        return {"maushold": [{"name": "Chople Berry", "p": 0.23}, {"name": "Wide Lens", "p": 0.17}],
                "aggron": [{"name": "Aggronite", "p": 0.99}]}.get(sp, [])

    def ability_distribution(self, sp, top_k=5):
        return {"maushold": [{"name": "Friend Guard", "p": 0.7}]}.get(sp, [])

    def top_ability(self, sp):
        return None


def test_forme_ids_get_their_dex_display_name_not_the_echoed_id():
    assert display_species("mausholdfour", EchoBelief()) == "Maushold-Four"
    assert display_species("mausholdfour", None) == "Maushold-Four"
    assert display_species("maushold", EchoBelief()) == "Maushold"          # the belief's own key still wins
    assert display_species("charizardmegay", None) == "Charizard-Mega-Y"
    assert display_species("zzznotamon", EchoBelief()) == "zzznotamon"      # unknown everywhere: the id


def test_cosmetic_forme_borrows_the_base_formes_belief_prior():
    assert base_species_id("mausholdfour") == "maushold" and base_species_id("aggron") is None
    assert MB._prior(EchoBelief(), "mausholdfour", "items")[0][0] == "Chople Berry"
    assert MB._prior(EchoBelief(), "mausholdfour", "abilities")[0][0] == "Friend Guard"
    assert MB._prior(EchoBelief(), "zzznotamon", "items") == []


def test_sprite_ids_follow_showdowns_file_names():
    assert sprite_id("mausholdfour") == "maushold-four" and sprite_id("Maushold-Four") == "maushold-four"
    assert sprite_id("aggronmega") == "aggron-mega" and sprite_id("Aggron-Mega") == "aggron-mega"
    assert sprite_id("charizardmegay") == "charizard-megay" and sprite_id("raichualola") == "raichu-alola"
    assert sprite_id("aggron") == "aggron" and sprite_id("zzznotamon") == "zzznotamon"


def test_summary_returns_every_row_by_default_with_sprites_and_the_mega_sprite():
    b = MatchupBook()
    for i in range(15):
        b.record_game(FMT, "T", "ai", [{"species": f"mon{i}", "item": None, "ability": None}],
                      tag=f"battle-{FMT}-{i}", opponent=f"opp{i}")
    b.record_game(FMT, "T", "human", [{"species": "aggron", "item": "aggronite", "ability": "filter"},
                                      {"species": "mausholdfour", "item": None, "ability": None}],
                  tag=f"battle-{FMT}-99", opponent="djd54")
    s = b.summary(FMT, team="T", belief=EchoBelief())
    assert len(s["rows"]) == 17 and s["footer"]["species"] == 17           # no 12-row cap
    assert len(b.summary(FMT, team="T", top=2)["rows"]) == 2                 # an explicit cap still caps
    rows = {r["species"]: r for r in s["rows"]}
    ag = rows["aggron"]
    assert ag["sprite"] == "aggron" and ag["mega"] >= 1                     # mega = the sighting count
    assert ag["mega_name"] == "Aggron-Mega" and ag["mega_sprite"] == "aggron-mega"
    mf = rows["mausholdfour"]
    assert mf["display"] == "Maushold-Four" and mf["sprite"] == "maushold-four" and mf["mega_sprite"] is None
    assert mf["items"][0]["name"] == "Chople Berry"                          # the base forme's prior, not "unknown"
    assert rows["mon3"]["sprite"] == "mon3"                                  # unknown species: plain id, the UI shows '?'


def test_both_pages_ship_the_sort_controls_and_the_sprite_fallback():
    from v_dance.play import bot_control_ui as bcu
    from v_dance.datatools import mission_control as mc
    panel = bcu._PANEL_HTML
    assert 'id="muSSort"' in panel and 'id="muASort"' in panel and "function pkMissing" in panel
    assert "play.pokemonshowdown.com/sprites/gen5/" in panel and "function muSorted" in panel
    html = mc._HTML_PATH.read_text(encoding="utf-8")
    assert 'id="ob-mu-s-sort"' in html and 'id="ob-mu-a-sort"' in html and "function pkMissing" in html
    assert "max-height:460px" in html and "function muSorted" in html
