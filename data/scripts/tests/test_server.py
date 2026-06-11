"""Tests for server.py — /health, /data, /parse, /export — using Flask's
test client (no live server needed). Run against the real example VOD."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import server  # noqa: E402


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

    # yourSide=p1 → only p1-perspective transitions
    assert {t["perspective"] for t in transitions} == {"p1"}

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


def test_export_requires_html(client):
    r = client.post("/export", json={
        "battle_id": "x", "known_teams_entry": {}, "replay_html": "",
    })
    assert r.status_code == 400


def test_export_requires_json_body(client):
    r = client.post("/export", data="nope", content_type="text/plain")
    assert r.status_code == 400
