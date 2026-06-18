"""3c.6e-2: Flask dashboard server — serves dashboard files + live JSON feeds."""
from __future__ import annotations

import json

import pytest

from v_dance.datatools.dashboard_server import create_app


@pytest.fixture
def client(tmp_path):
    # real dashboard dir (files exist), temp archive dir for the live feeds
    app = create_app(archive_dir=tmp_path)
    app.testing = True
    return app, app.test_client(), tmp_path


def test_index_serves_dashboard_html(client):
    _, c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert b"Victory-Dance" in r.data and b"dashboard.js" in r.data


def test_static_assets_served(client):
    _, c, _ = client
    assert c.get("/dashboard.css").status_code == 200
    assert c.get("/dashboard.js").status_code == 200


def test_unlisted_asset_is_404(client):
    _, c, _ = client
    assert c.get("/secrets.txt").status_code == 404            # not in the allow-list


def test_manifest_default_when_missing(client):
    _, c, _ = client
    r = c.get("/manifest.json")
    assert r.status_code == 200
    j = r.get_json()
    assert j["n_generations"] == 0 and j["generations"] == []
    assert "no-store" in r.headers.get("Cache-Control", "")


def test_manifest_served_from_archive(client):
    _, c, arch = client
    (arch / "manifest.json").write_text(json.dumps({"n_generations": 4, "generations": [1, 2]}), encoding="utf-8")
    j = c.get("/manifest.json").get_json()
    assert j["n_generations"] == 4 and len(j["generations"]) == 2


def test_status_default_idle_when_missing(client):
    _, c, _ = client
    r = c.get("/status.json")
    assert r.status_code == 200
    j = r.get_json()
    assert j["live"] is False and j["run"]["phase"] == "idle"
    assert "no-store" in r.headers.get("Cache-Control", "")


def test_status_served_from_archive_live(client):
    _, c, arch = client
    (arch / "status.json").write_text(json.dumps({"live": True, "run": {"phase": "collecting"},
                                                   "active_battles": [{"tag": "battle-x-1"}]}), encoding="utf-8")
    j = c.get("/status.json").get_json()
    assert j["live"] is True and j["run"]["phase"] == "collecting"
    assert j["active_battles"][0]["tag"] == "battle-x-1"


def test_live_log_default_and_served(client):
    _, c, arch = client
    j = c.get("/live_log.json").get_json()
    assert j["log"] == [] and j["n_lines"] == 0                # default when no run
    (arch / "live_log.json").write_text(json.dumps(
        {"tag": "battle-z-1", "turn": 3, "n_lines": 2, "log": ["|turn|1", "|move|x"]}), encoding="utf-8")
    j = c.get("/live_log.json").get_json()
    assert j["tag"] == "battle-z-1" and len(j["log"]) == 2


def test_live_battles_default_empty_and_aggregates(client):
    """#18 multi-battle spectate: /live_battles.json aggregates the file-per-battle feed."""
    _, c, arch = client
    j = c.get("/live_battles.json").get_json()
    assert j["n"] == 0 and j["battles"] == []                  # default when no run
    from v_dance.selfplay.status import LiveBattles
    live = arch / "live"
    live.mkdir()
    lb = LiveBattles(live)
    lb.update("battle-a-1", p1="SP1", p2="SP2", turn=2, log=["|turn|2"])
    lb.update("battle-b-2", p1="SP3", p2="SP4", turn=5, log=["|turn|5"])
    j = c.get("/live_battles.json").get_json()
    assert j["n"] == 2
    tags = sorted(b["tag"] for b in j["battles"])
    assert tags == ["battle-a-1", "battle-b-2"]                # SEVERAL concurrent battles
    assert "no-store" in c.get("/live_battles.json").headers.get("Cache-Control", "")
