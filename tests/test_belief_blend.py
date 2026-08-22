"""Era-retrain step 0 — belief blend (Pikalytics prior × observed ladder meta).

The blend must: weight each species by its OBSERVED sample (w = n/(n+k)), convex-blend the
moves/items/abilities lists over the union of normalised names (observed ids ↔ Pikalytics display
names), pass spreads/natures/teammates through untouched (not observable / rank-vs-pct mismatch),
treat a null prior usage as 0, and keep the output loadable by BeliefState."""
from __future__ import annotations

import math

from v_dance.datatools.belief_blend import blend, _blend_block


def _pika():
    return {
        "format": "gen9championsvgc2026regmb", "scraped_at": "2026-07-10T07:56:48Z",
        "pokemon": {
            "Garchomp": {
                "usage_pct": None,                       # null prior usage (teammate-only entry)
                "moves": [{"name": "Dragon Claw", "pct": 80.0}, {"name": "Protect", "pct": 60.0}],
                "items": [{"name": "Life Orb", "pct": 50.0}],
                "abilities": [{"name": "Rough Skin", "pct": 100.0}],
                "spreads": [{"nature": "Jolly", "evs": [0, 252, 0, 0, 4, 252], "pct": 40.0}],
                "natures": [{"nature": "Jolly", "pct": 70.0}],
                "teammates": [{"name": "Whimsicott", "rank": 1}],
            },
            "Pelipper": {"usage_pct": 10.0, "moves": [{"name": "Hurricane", "pct": 90.0}],
                         "items": [], "abilities": [], "spreads": [], "natures": [], "teammates": []},
        },
    }


def _observed(n_games=100):
    return {
        "n_games": n_games, "scraped_at": "2026-07-10T20:00:00Z",
        "pokemon": {
            # 40% of 100 games = 40 sightings → w = 40/60 = 2/3 with k=20
            "Garchomp": {"usage_pct": 40.0,
                         "moves": [{"name": "earthquake", "pct": 60.0},
                                   {"name": "dragonclaw", "pct": 30.0}],
                         "items": [{"name": "lifeorb", "pct": 90.0}],
                         "abilities": [], "spreads": [], "natures": [],
                         "teammates": [{"name": "Kingambit", "pct": 30.0}]},
            # a species the prior has never heard of
            "Sinistcha": {"usage_pct": 12.0, "moves": [{"name": "matchagotcha", "pct": 100.0}],
                          "items": [], "abilities": [], "spreads": [], "natures": [],
                          "teammates": []},
        },
    }


def test_blend_block_union_and_display_names():
    w = 0.5
    out = _blend_block([{"name": "Dragon Claw", "pct": 80.0}],
                       [{"name": "dragonclaw", "pct": 30.0}, {"name": "earthquake", "pct": 60.0}], w)
    by = {e["name"]: e["pct"] for e in out}
    assert math.isclose(by["Dragon Claw"], 0.5 * 80 + 0.5 * 30)     # display name kept, blended
    assert math.isclose(by["earthquake"], 0.5 * 60)                 # observed-only entry enters
    assert out[0]["pct"] >= out[-1]["pct"]                          # sorted desc


def test_blend_weights_and_blocks():
    doc = blend(_pika(), _observed(), k=20.0)
    g = doc["pokemon"]["Garchomp"]
    w = 40.0 / 60.0
    assert math.isclose(g["blend_w"], round(w, 3), abs_tol=1e-3)
    by = {e["name"]: e["pct"] for e in g["moves"]}
    assert math.isclose(by["Dragon Claw"], round((1 - w) * 80 + w * 30, 2), abs_tol=0.01)
    assert math.isclose(by["earthquake"], round(w * 60, 2), abs_tol=0.01)
    assert math.isclose(g["usage_pct"], round(w * 40.0, 2), abs_tol=0.01)   # null prior = 0
    # pass-through blocks untouched
    assert g["spreads"] == _pika()["pokemon"]["Garchomp"]["spreads"]
    assert g["natures"] == _pika()["pokemon"]["Garchomp"]["natures"]
    assert g["teammates"] == _pika()["pokemon"]["Garchomp"]["teammates"]


def test_unseen_prior_species_pass_through_and_observed_only_added():
    doc = blend(_pika(), _observed(), k=20.0)
    assert doc["pokemon"]["Pelipper"] == _pika()["pokemon"]["Pelipper"]     # not seen → untouched
    assert doc["pokemon"]["Sinistcha"]["usage_pct"] == 12.0                 # observed-only added
    st = doc["blend_provenance"]["counts"]
    assert st == {"blended": 1, "passthrough": 1, "added": 1}


def test_low_sample_species_stays_near_prior():
    obs = _observed(n_games=100)
    obs["pokemon"]["Garchomp"]["usage_pct"] = 1.0        # 1 sighting → w = 1/21 ≈ 0.048
    doc = blend(_pika(), obs, k=20.0)
    g = doc["pokemon"]["Garchomp"]
    assert g["blend_w"] < 0.05
    by = {e["name"]: e["pct"] for e in g["moves"]}
    assert by["Dragon Claw"] > 76.0                      # barely moved off the prior


def test_output_loads_as_beliefstate(tmp_path):
    import json
    from v_dance.parser.belief_state import BeliefState
    p = tmp_path / "blend.json"
    p.write_text(json.dumps(blend(_pika(), _observed(), k=20.0)), encoding="utf-8")
    b = BeliefState(p)
    assert b is not None
