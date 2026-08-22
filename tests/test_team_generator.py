"""M4 team-builder AI (execution_plan_4a_m4 PART 2): generator, paste round-trip,
graceful scoring, and the two Flask endpoints (stubbed — no node, no nets, no server)."""
from types import SimpleNamespace

import pytest

from v_dance.datatools import team_generator as tg


class _StubBelief:
    """Six-species toy meta with a fixed co-occurrence structure."""
    _SPECIES = ["Mawile", "Charizard", "Torkoal", "Rillaboom", "Incineroar",
                "Amoonguss", "Pelipper", "Archaludon"]
    _PAIRS = {("Mawile", "Torkoal"): 40.0, ("Charizard", "Torkoal"): 50.0,
              ("Pelipper", "Archaludon"): 60.0}

    def known(self, s):
        return s in self._SPECIES

    def usage(self, s):
        return 50.0 - self._SPECIES.index(s) * 4 if s in self._SPECIES else 0.0

    def usage_ranking(self):
        return [(s, self.usage(s)) for s in self._SPECIES]

    def teammates(self, s, top_k=16):
        out = []
        for (a, b), p in self._PAIRS.items():
            if a == s:
                out.append({"name": b, "p": p})
            elif b == s:
                out.append({"name": a, "p": p})
        return out

    def ability_distribution(self, s, top_k=4):
        return [{"name": "Intimidate", "p": 0.9}]

    def item_distribution(self, s, top_k=5):
        return [{"name": "Focus Sash", "p": 0.5}]

    def move_distribution(self, s, top_k=8):
        return [{"name": m, "p": 0.5} for m in
                ("Protect", "Fake Out", "Sucker Punch", "Play Rough", "Icy Wind")]

    def spread_distribution(self, s, top_k=5, revealed_nature=None):
        return [{"nature": "Adamant", "evs": [32, 32, 0, 0, 2, 0],
                 "evs_actual": [252, 252, 0, 0, 4, 0], "p": 0.6}]


def test_generate_rosters_deterministic_and_valid():
    b = _StubBelief()
    r1 = tg.generate_rosters(["Mawile"], 3, b)
    r2 = tg.generate_rosters(["Mawile"], 3, b)
    assert r1 == r2                                   # pure argsort, no RNG
    assert len(r1) == 3
    for team in r1:
        assert team[0] == "Mawile" and len(team) == 6 and len(set(team)) == 6
    with pytest.raises(ValueError):
        tg.generate_rosters(["NotAMon"], 2, b)


def test_fill_sets_and_paste_roundtrip():
    from v_dance.parser.vod_parser.team_sheet import parse_showdown_team
    b = _StubBelief()
    mons = tg.fill_sets(["Mawile", "Torkoal"], b)
    paste = tg.to_paste(mons)
    back = parse_showdown_team(paste)
    assert [m["species"] for m in back] == ["Mawile", "Torkoal"]
    assert back[0]["item"] == "Focus Sash" and back[0]["nature"] == "Adamant"
    assert len(back[0]["moves"]) == 4 and back[0]["moves"][0] == "Protect"
    # Champions stat-point budget: pastes carry the 0-32 BUCKET scale, never 0-252.
    assert back[0]["evs"].get("hp") == 32 and back[0]["evs"].get("spd") == 2


def test_score_team_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(tg, "_artifact", lambda: {"k": 10})
    import v_dance.datatools.team_archetypes as ta
    monkeypatch.setattr(ta, "assign_team_sheet", lambda mons, art, encoder=None: (8, 120.5))
    monkeypatch.setattr(tg, "_matrix", lambda: None)
    fake_tp = (SimpleNamespace(use_set_head=False), {}, {})
    s = tg.score_team([{"species": "Mawile"}], _StubBelief(),
                      opp_rosters=[[{"species": "Pelipper"}]], tp_bundle=fake_tp)
    assert s["archetype"] == {"z": 8, "dist": 120.5}
    assert s["tp_confidence"] is None                 # no set head on the fake TP
    assert s["matchup_prior"] is None and "router_matrix" in s["matchup_note"]


def test_generate_teams_pipeline_no_validate_no_score():
    out = tg.generate_teams(["Mawile"], 2, _StubBelief(), validate=False, score=False)
    assert len(out["teams"]) == 2 and out["dropped_illegal"] == 0
    assert all("paste" in t and len(t["roster"]) == 6 for t in out["teams"])


# ── B5 endpoints (Flask test client; generator + belief stubbed) ───────────────
@pytest.fixture()
def client(monkeypatch):
    from v_dance.datatools import server as srv
    monkeypatch.setattr(srv, "_get_belief", lambda: _StubBelief())
    return srv.app.test_client()


def test_api_generate(client, monkeypatch):
    import v_dance.datatools.team_generator as tgen
    monkeypatch.setattr(tgen, "generate_teams",
                        lambda core, n, belief, **kw: {"teams": [{"roster": core * 6,
                                                                  "paste": "x"}],
                                                       "dropped_illegal": 1})
    r = client.post("/api/teams/generate", json={"core": ["Mawile"], "n": 3})
    assert r.status_code == 200 and r.get_json()["dropped_illegal"] == 1
    assert client.post("/api/teams/generate", json={"core": []}).status_code == 400


def test_api_score(client, monkeypatch):
    import v_dance.datatools.team_generator as tgen
    monkeypatch.setattr(tgen, "score_team",
                        lambda mons, belief, **kw: {"archetype": {"z": 1, "dist": 1.0}})
    r = client.post("/api/teams/score", json={"paste": "Mawile @ Focus Sash\n- Protect\n"})
    assert r.status_code == 200 and r.get_json()["scores"]["archetype"]["z"] == 1
    assert client.post("/api/teams/score", json={"paste": ""}).status_code == 400
