"""Tests for server.py — /health, /data, /parse, /export — using Flask's
test client (no live server needed). Run against the real example VOD."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
import v_dance.datatools.server as server  # noqa: E402


@pytest.fixture(scope="module")
def client():
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


def _entry():
    return {
        "_meta": {"yourSide": "p1", "winner": None,
                  "p1name": "stevenhe vgc", "p2name": "speedyturtle87"},
        "p1": {"Floette-Eternal": {"nature": None, "item": None,
                                   "ability": "Flower Veil",
                                   "ev_spread": None, "moves": None}},
        "p2": {},
    }


# ── /health ───────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["pokedex_loaded"] is True   # data/pokedex.json exists


# ── /data ─────────────────────────────────────────────────────────────────

def test_data_serves_pokedex(client):
    r = client.get("/data/pokedex.json")
    assert r.status_code == 200
    dex = r.get_json()
    assert dex["floettemega"]["abilities"] == {"0": "Fairy Aura"}


def test_data_path_traversal_blocked(client):
    r = client.get("/data/../scripts/server.py")
    assert r.status_code in (403, 404)


# ── /parse ────────────────────────────────────────────────────────────────

def test_parse_replay(client, vod_path):
    with vod_path.open("rb") as f:
        r = client.post("/parse", data={
            "replay_html": (f, vod_path.name),
            "known_teams": json.dumps(_entry()),
        }, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["turns"]
    # Bug 8 fields present in the preview's revealed_info
    ri = body["revealed_info"]["p1:Floette-Eternal"]
    assert ri["is_mega"] is True
    assert ri["mega_ability"] == "Fairy Aura"
    assert ri["pre_mega_ability"] is None
    assert ri["possible_abilities"] == ["Flower Veil", "Symbiosis"]
    # User-supplied entry surfaced as overrides
    assert body["known_team_overrides"]["p1:Floette-Eternal"]["ability"] == "Flower Veil"


def test_parse_missing_file(client):
    r = client.post("/parse", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_parse_bad_known_teams_json(client, vod_path):
    with vod_path.open("rb") as f:
        r = client.post("/parse", data={
            "replay_html": (f, vod_path.name),
            "known_teams": "{not json",
        }, content_type="multipart/form-data")
    assert r.status_code == 400


# ── /export ───────────────────────────────────────────────────────────────

def test_export_jsonl(client, vod_html):
    r = client.post("/export", json={
        "battle_id": "test-battle",
        "known_teams_entry": _entry(),
        "replay_html": vod_html,
    })
    assert r.status_code == 200
    count = int(r.headers["X-Transition-Count"])
    assert count > 0

    lines = r.data.decode("utf-8").strip().split("\n")
    assert len(lines) == count
    transitions = [json.loads(l) for l in lines]

    # No source_type → Type B → BOTH perspectives exported (each ranked
    # player is a behavioural-cloning target; doubles per-replay yield)
    assert {t["perspective"] for t in transitions} == {"p1", "p2"}

    # Bug 8 end-to-end: mega'd Floette in the export carries the fixed mega
    # ability as active and the injected base ability as pre_mega_ability.
    megas = [m for t in transitions
             for m in t["state_after_actions"]["our_active"].values()
             if m["species"] == "Floette-Mega"]
    assert megas
    for m in megas:
        assert m["known_ability"] == "Fairy Aura"
        assert m["pre_mega_ability"] == "Flower Veil"
        assert m["mega_ability"] == "Fairy Aura"


def test_export_source_type_passthrough(client, vod_html):
    """The UI's Type A/B/C/D selector must reach the exported transitions —
    without it everything silently trains as Type B."""
    r = client.post("/export", json={
        "battle_id": "test-battle",
        "known_teams_entry": _entry(),
        "replay_html": vod_html,
        "source_type": "own_vod",
    })
    assert r.status_code == 200
    transitions = [json.loads(l) for l in
                   r.data.decode("utf-8").strip().split("\n")]
    assert {t["source_type"] for t in transitions} == {"own_vod"}
    # Non-B types keep the yourSide restriction (the opponent's perspective
    # would invert the exact/distribution stats-quality semantics)
    assert {t["perspective"] for t in transitions} == {"p1"}


def test_export_defaults_to_type_b(client, vod_html):
    r = client.post("/export", json={
        "battle_id": "test-battle",
        "known_teams_entry": _entry(),
        "replay_html": vod_html,
    })
    assert r.status_code == 200
    transitions = [json.loads(l) for l in
                   r.data.decode("utf-8").strip().split("\n")]
    assert {t["source_type"] for t in transitions} == {"ranked_player_vod"}


def test_export_type_b_doubles_perspectives(client, vod_html):
    """One Type B replay = two ranked players' decisions: explicit
    source_type B must export p1 AND p2 perspectives, twice the turns."""
    r = client.post("/export", json={
        "battle_id": "test-battle",
        "known_teams_entry": _entry(),        # yourSide=p1 must NOT restrict
        "replay_html": vod_html,
        "source_type": "ranked_player_vod",
    })
    assert r.status_code == 200
    transitions = [json.loads(l) for l in
                   r.data.decode("utf-8").strip().split("\n")]
    assert {t["perspective"] for t in transitions} == {"p1", "p2"}
    by_persp = {}
    for t in transitions:
        by_persp[t["perspective"]] = by_persp.get(t["perspective"], 0) + 1
    assert by_persp["p1"] == by_persp["p2"]
    # Each perspective's transitions are self-consistent: its own actions
    # under its own mask
    for t in transitions:
        assert t["players"]["our_side"] == t["perspective"]
        for a in t["our_actions"]:
            if a["action_index"] is not None:
                assert t["action_mask"][a["slot"]][a["action_index"]] == 1


def test_export_requires_html(client):
    r = client.post("/export", json={
        "battle_id": "x", "known_teams_entry": {}, "replay_html": "",
    })
    assert r.status_code == 400


def test_export_requires_json_body(client):
    r = client.post("/export", data="nope", content_type="text/plain")
    assert r.status_code == 400
