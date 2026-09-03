"""2026-09-03 (USER): the Dashboard tab (the self-play dashboard embedded in Mission Control: auto-start,
full height, a run strip) and the bandit ARMS PANEL (win rate / retired / LEARNING label) in both UIs.
The dashboard is exercised through Flask's test client; the MC proxy on a closed port; the pages as text."""
from __future__ import annotations

import json

import pytest


def test_dashboard_run_info_names_the_run_it_follows_and_carries_its_status(tmp_path):
    flask = pytest.importorskip("flask")   # noqa: F841 — the dashboard is a Flask app
    from v_dance.datatools.dashboard_server import _IDLE_STATUS, create_app
    app = create_app(archive_dir=tmp_path)
    app.config["TESTING"] = True
    c = app.test_client()
    r = c.get("/run_info.json")
    js = r.get_json()
    assert r.status_code == 200 and js["status"] == _IDLE_STATUS       # no run yet -> the idle default
    assert js["archive_dir"] == str(tmp_path.resolve()) and isinstance(js["run_name"], str)
    assert "no-store" in r.headers.get("Cache-Control", "")
    # a run subfolder holding a status.json is what the feeds follow (resolve_run_dir) — and it is NAMED
    run = tmp_path / "era5b_v2_from_era2"
    run.mkdir()
    st = {**_IDLE_STATUS, "live": True,
          "run": {**_IDLE_STATUS["run"], "phase": "collecting", "generation": 3, "n_generations": 40}}
    (run / "status.json").write_text(json.dumps(st), encoding="utf-8")
    js = c.get("/run_info.json").get_json()
    assert js["run_name"] == "era5b_v2_from_era2" and js["run_dir"] == str(run.resolve())
    assert js["status"]["live"] is True and js["status"]["run"]["generation"] == 3
    # a corrupt status.json degrades to the idle default, never a 500
    (run / "status.json").write_text("{not json", encoding="utf-8")
    js = c.get("/run_info.json").get_json()
    assert js["status"] == _IDLE_STATUS and js["run_name"] == "era5b_v2_from_era2"


def test_mission_control_proxies_the_dashboard_and_ships_the_tab_and_the_arms_panel(monkeypatch):
    from v_dance.datatools import mission_control as mc
    monkeypatch.setattr(mc, "_port_open", lambda port: False)
    assert mc._dashboard_status() == {"up": False}                    # :5175 down -> a clean 'down'
    html = mc._HTML_PATH.read_text(encoding="utf-8")
    # the Dashboard tab: renamed, wide, auto-start + the run strip fed by the proxy
    assert '["monitor","Dashboard"]' in html and '<div class="page wide" id="page-monitor">' in html
    assert 'iframe class="embed tall"' in html and "/api/dashboard/status" in html
    assert 'id: "svc_dashboard"' in html and "dashAutoStarted" in html
    # the arms panel replaces the text line: table + badges + the rule note
    assert 'id="ob-arms"' in html and "function renderArms" in html and "function armBadges" in html
    assert "🧠 learning" in html and "⛔ retired" in html and "P(worse)" in html
    assert 'set("ob-bandit", arms.length ? "Bandit (' not in html          # the old text line is gone
