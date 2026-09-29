"""B5 (2026-09-29): the MunchStats page parser and the belief_sources per-field merge."""
from v_dance.datatools.belief_sources import merge
from v_dance.datatools.scrapers.scrape_munchstats import parse_page

# Mirrors a live page's text order: nav + sidebar index (#rank rows) BEFORE "Base Stats", then the
# stat sections, then the footer that ends the last section.
_PAGE = """<html><head><title>Salamence | MunchStats</title><script>var x = "Moves";</script></head><body>
<div>Salamence</div><div>Rank</div><div>#4</div>
<a>Rillaboom</a><span>#1</span><a>Sneasler</a><span>#2</span>
<h3>Base Stats</h3><div>HP</div><div>95</div>
<div>Usage Rank Trend</div>
<h3>Moves</h3><div>Protect</div><div>93.9%</div><div>Hyper Voice</div><div>90.7%</div>
<h3>Teammates</h3><div>Rillaboom</div><div>#1</div><div>Sneasler</div><div>#2</div>
<h3>Items</h3><div>Salamencite</div><div>97.8%</div><div>Life Orb</div><div>0.5%</div>
<h3>Abilities</h3><div>Intimidate</div><div>99.0%</div><div>Moxie</div><div>1.0%</div>
<h3>Stat Point Spreads</h3><div>2/0/0/32/0/32</div><div>44.2%</div><div>32/0/2/0/0/32</div><div>10.0%</div>
<h3>Natures</h3><div>Timid</div><div>51.6%</div><div>Modest</div><div>23.1%</div>
<div>Stats Graph (Beta)</div><div>Usage</div><div>Top Teams</div>
<p style="display: block;">Game data last scraped: September 28, 2026 at 12:04 UTC</p>
</body></html>"""


def test_parse_page_reads_every_section():
    d = parse_page(_PAGE)
    assert d["usage_rank"] == 4 and d["usage_pct"] is None
    assert d["moves"] == [{"name": "Protect", "pct": 93.9}, {"name": "Hyper Voice", "pct": 90.7}]
    assert d["items"][0] == {"name": "Salamencite", "pct": 97.8}        # the stone lives on the BASE species
    assert d["abilities"][0] == {"name": "Intimidate", "pct": 99.0}
    assert d["teammates"] == [{"name": "Rillaboom", "rank": 1}, {"name": "Sneasler", "rank": 2}]
    assert d["natures"][0] == {"nature": "Timid", "pct": 51.6}
    # spreads carry the MODAL nature (same rule as the Pikalytics parser)
    assert d["spreads"][0] == {"nature": "Timid", "evs": [2, 0, 0, 32, 0, 32], "pct": 44.2}
    assert len(d["spreads"]) == 2


def test_parse_page_without_stats_is_none():
    assert parse_page("<html><body><div>Base Stats</div><div>Usage</div></body></html>") is None
    assert parse_page("<html><body>Not found</body></html>") is None


def _mon(**kw):
    return {"usage_pct": None, "moves": [], "items": [], "abilities": [], "spreads": [], "natures": [],
            "teammates": [], **kw}


def test_merge_takes_the_first_non_empty_source_per_field():
    munch = {"pokemon": {"Salamence": _mon(moves=[{"name": "Protect", "pct": 93.9}],
                                           spreads=[{"nature": "Timid", "evs": [2, 0, 0, 32, 0, 32], "pct": 44.2}],
                                           usage_rank=4)}}
    pika = {"pokemon": {"Salamence": _mon(usage_pct=23.2, moves=[{"name": "Protect", "pct": 30.0}]),
                        "Rillaboom": _mon(usage_pct=37.0, moves=[{"name": "Fake Out", "pct": 57.0},
                                                                 {"name": "Grassy Glide", "pct": 43.0}])}}
    prev = {"pokemon": {"Rillaboom": _mon(usage_pct=1.0, moves=[{"name": "Wood Hammer", "pct": 90.0}],
                                          spreads=[{"nature": "Adamant", "evs": [32, 32, 0, 0, 0, 2], "pct": 50.0}]),
                        "Incineroar": _mon(moves=[{"name": "Fake Out", "pct": 99.0}])}}
    out = merge(munch, pika, prev, "regmb")

    sala = out["Salamence"]
    assert sala["moves"] == [{"name": "Protect", "pct": 93.9}] and sala["sources"]["moves"] == "munchstats"
    assert sala["usage_pct"] == 23.2 and sala["sources"]["usage_pct"] == "pikalytics"   # MunchStats has rank only
    assert sala["usage_rank"] == 4

    rilla = out["Rillaboom"]
    # Pikalytics' M-C per-slot share (sums to ~100) is rescaled to per-set and tagged
    assert rilla["moves"] == [{"name": "Fake Out", "pct": 100.0}, {"name": "Grassy Glide", "pct": 100.0}]
    assert rilla["moves_scale"] == "per-slot x4"
    # the previous reg only fills spreads / natures / teammates, never usage or moves
    assert rilla["spreads"][0]["evs"] == [32, 32, 0, 0, 0, 2] and rilla["sources"]["spreads"] == "pikalytics_regmb"
    assert rilla["usage_pct"] == 37.0
    # ...and never ADDS a species the current reg's sources do not have
    assert "Incineroar" not in out
