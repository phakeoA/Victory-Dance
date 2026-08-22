"""Phase-4b router v1 (DS-4b): policy unit tests. The archetype assigner is
monkeypatched (thin wrapper over already-tested assign_team_sheet); what's under
test is the routing POLICY: lost-last-game trigger, prior-table argmax, avoid
repeating the losing team, and never-raise / no-opinion fallbacks."""
import json

import pytest

from v_dance.play import opponent_dossier, team_router


POOL = ["maw_zard", "rain_counter", "tr_counter"]


def _dossier(tmp_path, monkeypatch, games, mons=None):
    monkeypatch.setattr(opponent_dossier, "DOSSIER_DIR", tmp_path)
    # mons={} must SURVIVE (the no-archetype case) — `or` would swallow it and the
    # route would fall through to the real (now data-seeded) priors file.
    d = {"opponent": "Rival",
         "mons": mons if mons is not None else {"pelipper": {"species": "Pelipper"}},
         "games": games}
    (tmp_path / "rival.json").write_text(json.dumps(d), encoding="utf-8")


def _priors(tmp_path, table):
    p = tmp_path / "router_priors.json"
    p.write_text(json.dumps(table), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _fixed_archetype(monkeypatch):
    monkeypatch.setattr(team_router, "_assign_archetype", lambda mons: 8 if mons else None)


def test_no_dossier_no_opinion(tmp_path, monkeypatch):
    monkeypatch.setattr(opponent_dossier, "DOSSIER_DIR", tmp_path)
    team, why = team_router.route("Nobody", POOL)
    assert team is None and "no dossier" in why


def test_won_last_game_keeps_course(tmp_path, monkeypatch):
    _dossier(tmp_path, monkeypatch, [{"result": "ai", "our_team": "maw_zard"}])
    team, why = team_router.route("Rival", POOL)
    assert team is None and "keep course" in why


def test_lost_last_game_routes_to_best_prior(tmp_path, monkeypatch):
    _dossier(tmp_path, monkeypatch, [{"result": "human", "our_team": "maw_zard"}])
    p = _priors(tmp_path, {"8": {"rain_counter": 2.0, "tr_counter": 1.0, "maw_zard": 3.0}})
    team, why = team_router.route("Rival", POOL, priors_path=p)
    # maw_zard scores highest but is the team we just LOST with → next best
    assert team == "rain_counter" and "z8" in why


def test_losing_team_allowed_when_only_candidate(tmp_path, monkeypatch):
    _dossier(tmp_path, monkeypatch, [{"result": "human", "our_team": "maw_zard"}])
    p = _priors(tmp_path, {"8": {"maw_zard": 1.0}})
    team, _ = team_router.route("Rival", POOL, priors_path=p)
    assert team == "maw_zard"


def test_unseeded_or_corrupt_priors_no_opinion(tmp_path, monkeypatch):
    _dossier(tmp_path, monkeypatch, [{"result": "human", "our_team": "maw_zard"}])
    team, why = team_router.route("Rival", POOL, priors_path=tmp_path / "missing.json")
    assert team is None and "z8" in why
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    team, _ = team_router.route("Rival", POOL, priors_path=bad)
    assert team is None


def test_no_mons_no_archetype_no_opinion(tmp_path, monkeypatch):
    _dossier(tmp_path, monkeypatch, [{"result": "human", "our_team": "maw_zard"}], mons={})
    team, why = team_router.route("Rival", POOL)
    assert team is None and "archetype" in why


def test_router_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(opponent_dossier, "load", lambda o: 1 / 0)
    team, why = team_router.route("Rival", POOL)
    assert team is None and "router error" in why
