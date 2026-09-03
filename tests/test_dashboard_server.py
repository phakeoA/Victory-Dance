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


# ── #H: saved gauntlet-replay browser ────────────────────────────────────────────────────────
def _make_eval_tree(arch, *, stamp="2026-06-18_03-00-00", gen=3):
    base = arch / "live" / stamp / f"gen_{gen}" / "eval"
    (base / "league").mkdir(parents=True)
    (base / "heuristic").mkdir(parents=True)
    (base / "league" / "gen3_vs_gen0_battle-x-1.html").write_text("<!doctype html>LEAGUE", encoding="utf-8")
    (base / "heuristic" / "gen3_vs_heuristic_battle-y-2.html").write_text("<!doctype html>H", encoding="utf-8")
    return base


def test_eval_replays_default_empty(client):
    _, c, _ = client
    assert c.get("/eval_replays.json").get_json()["sections"] == []          # no gen -> empty
    j = c.get("/eval_replays.json?gen=3").get_json()
    assert j["gen"] == 3 and j["sections"] == []                             # gen has no files yet


def test_eval_replays_lists_grouped_league_first(client):
    _, c, arch = client
    _make_eval_tree(arch)
    j = c.get("/eval_replays.json?gen=3").get_json()
    sects = j["sections"]
    assert sects[0]["section"] == "league"                                   # league/championship FIRST
    assert {s["section"] for s in sects} == {"league", "heuristic"}
    assert sects[0]["files"][0]["name"] == "gen3_vs_gen0_battle-x-1.html"
    assert "no-store" in c.get("/eval_replays.json?gen=3").headers.get("Cache-Control", "")


def test_eval_replay_serves_html_and_guards(client):
    _, c, arch = client
    _make_eval_tree(arch)
    r = c.get("/eval_replay/3/league/gen3_vs_gen0_battle-x-1.html")
    assert r.status_code == 200 and b"LEAGUE" in r.data                       # the real saved replay
    assert c.get("/eval_replay/3/bogus/x.html").status_code == 404            # kind not whitelisted
    assert c.get("/eval_replay/3/league/x.json").status_code == 404           # non-.html rejected
    assert c.get("/eval_replay/3/league/missing.html").status_code == 404     # absent file


# ── self-play launcher UI (Step 3): /launcher, /config-schema, /launch, /stop ─────
import v_dance.datatools.dashboard_server as DS  # noqa: E402


class _FakeProc:
    def __init__(self, pid=4242):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return None                                  # always "running" for the test


def test_launcher_is_a_dashboard_tab_and_assets_served(client):
    _, c, _ = client
    # the launcher is now a TAB inside the dashboard page, not a separate /launcher page
    home = c.get("/").data
    assert b'data-tab="launcher"' in home and b'id="tab-launcher"' in home
    assert b"launcher.css" in home and b"launcher.js" in home
    assert c.get("/launcher.css").status_code == 200
    assert c.get("/launcher.js").status_code == 200
    assert c.get("/launcher").status_code == 404                 # standalone page removed


def test_spawn_run_streams_unbuffered_for_live_console(tmp_path, monkeypatch):
    import subprocess
    captured = {}

    class _FakePopen:
        def __init__(self, *a, **k):
            captured["argv"] = a[0]
            captured["env"] = k.get("env")
            self.pid = 1

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    log = tmp_path / "run.log"
    DS._spawn_run(cfg, log)
    # unbuffered so the Console tail isn't stuck behind a block buffer; encoding preserved
    assert captured["env"].get("PYTHONUNBUFFERED") == "1"
    assert captured["env"].get("PYTHONIOENCODING") == "utf-8"
    assert log.exists()                                          # the log file was opened for stdout


def test_console_is_a_dashboard_tab_and_assets_served(client):
    _, c, _ = client
    home = c.get("/").data
    assert b'data-tab="console"' in home and b'id="tab-console"' in home
    assert b"console.css" in home and b"console.js" in home
    assert c.get("/console.css").status_code == 200
    assert c.get("/console.js").status_code == 200


def test_config_schema_route(client, monkeypatch):
    _, c, _ = client
    monkeypatch.setattr(DS, "_config_schema", lambda: {"sections": {
        "generation": {"games": {"default": 100, "type": "int", "nullable": False}},
        "train": {"actor_lr": {"default": 3e-5, "type": "float", "nullable": False}}}})
    j = c.get("/config-schema").get_json()
    assert "generation" in j["sections"] and j["sections"]["train"]["actor_lr"]["type"] == "float"


def test_launch_rejects_non_dict_body(client):
    _, c, _ = client
    r = c.post("/launch", json=["not", "a", "dict"])
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_launch_rejects_when_a_run_is_live(client):
    import time
    _, c, arch = client
    (arch / "status.json").write_text(json.dumps({"live": True, "updated_at": time.time()}),
                                      encoding="utf-8")
    r = c.post("/launch", json={"generation": {"games": 5}})
    assert r.status_code == 409 and "already in progress" in r.get_json()["error"]


def test_run_is_live_treats_stale_flag_as_not_live(client):
    import time
    _, _, arch = client
    (arch / "status.json").write_text(json.dumps({"live": True, "updated_at": time.time() - 99999}),
                                      encoding="utf-8")
    assert DS._run_is_live(arch) is False                        # crashed run's stale flag -> not live
    (arch / "status.json").write_text(json.dumps({"live": True, "updated_at": time.time()}),
                                      encoding="utf-8")
    assert DS._run_is_live(arch) is True                         # fresh -> live


def test_launch_allowed_over_a_stale_live_flag(client, monkeypatch):
    import time
    _, c, arch = client
    (arch / "status.json").write_text(json.dumps({"live": True, "updated_at": time.time() - 99999}),
                                      encoding="utf-8")
    monkeypatch.setattr(DS, "_spawn_run", lambda cfg, log: _FakeProc(pid=909))
    r = c.post("/launch", json={"generation": {"games": 5}})
    assert r.status_code == 200 and r.get_json()["pid"] == 909   # stale flag doesn't deadlock launch


def test_launch_spawns_when_idle(client, monkeypatch):
    _, c, arch = client
    spawned = {}

    def _fake_spawn(cfg_path, log_path):
        spawned["cfg"] = cfg_path
        return _FakeProc(pid=777)

    monkeypatch.setattr(DS, "_spawn_run", _fake_spawn)
    r = c.post("/launch", json={"generation": {"games": 5}, "train": {"actor_lr": 1e-4}})
    j = r.get_json()
    assert r.status_code == 200 and j["ok"] is True and j["pid"] == 777
    # the config the UI sent was written to disk and handed to the spawn
    assert spawned["cfg"].exists()
    written = json.loads(spawned["cfg"].read_text(encoding="utf-8"))
    assert written["train"]["actor_lr"] == 1e-4
    assert (arch / "launched.pid").read_text(encoding="utf-8").strip() == "777"   # PID persisted
    # a second launch is refused while the first is "running"
    assert c.post("/launch", json={"generation": {}}).status_code == 409


def test_stop_without_active_run_is_409(client):
    _, c, _ = client
    r = c.post("/stop")                                          # no UI proc, no status.json
    assert r.status_code == 409 and r.get_json()["ok"] is False


def test_stop_clears_a_stale_live_flag(client):
    import time
    _, c, arch = client
    (arch / "status.json").write_text(json.dumps({"live": True, "updated_at": time.time() - 99999,
                                                  "run": {"phase": "collecting"}}), encoding="utf-8")
    r = c.post("/stop")                                          # no UI proc -> clears the stale flag
    assert r.status_code == 200 and r.get_json().get("cleared_live_flag") is True
    assert json.loads((arch / "status.json").read_text(encoding="utf-8"))["live"] is False
    assert DS._run_is_live(arch) is False                        # launcher can start now


def test_pid_alive_self_true_bogus_false():
    import os
    assert DS._pid_alive(os.getpid()) is True
    assert DS._pid_alive(2_000_000_000) is False


def test_stop_via_pidfile_after_dashboard_restart(client, monkeypatch):
    import time
    _, c, arch = client                       # fresh app -> _LAUNCHED is None (simulates a restart)
    (arch / "launched.pid").write_text("4242", encoding="utf-8")
    (arch / "status.json").write_text(json.dumps({"live": True, "updated_at": time.time()}),
                                      encoding="utf-8")
    killed = {}
    monkeypatch.setattr(DS, "_pid_alive", lambda p: int(p) == 4242)
    monkeypatch.setattr(DS, "_kill_pid_tree", lambda p: killed.setdefault("pid", int(p)))
    j = c.post("/stop").get_json()
    assert j["ok"] is True and j["stopped_pid"] == 4242 and j.get("via") == "pidfile"
    assert killed["pid"] == 4242
    assert json.loads((arch / "status.json").read_text(encoding="utf-8"))["live"] is False


def test_launch_state_reads_pidfile_when_no_handle(client, monkeypatch):
    _, c, arch = client
    (arch / "launched.pid").write_text("4242", encoding="utf-8")
    monkeypatch.setattr(DS, "_pid_alive", lambda p: int(p) == 4242)
    j = c.get("/launch_state").get_json()
    assert j["pid"] == 4242 and j["running"] is True


def test_launch_state_idle(client):
    _, c, _ = client
    j = c.get("/launch_state").get_json()
    assert j["running"] is False and j["pid"] is None


# ── Console tab: /launch_log incremental tail of the launched run's stdout/stderr ─
def test_launch_log_unavailable_when_no_log(client):
    _, c, _ = client
    r = c.get("/launch_log")
    j = r.get_json()
    assert j["available"] is False and j["data"] == "" and j["running"] is False
    assert "no-store" in r.headers.get("Cache-Control", "")


def test_launch_log_initial_tail_then_incremental(client):
    _, c, arch = client
    logs = arch / "logs"
    logs.mkdir()
    lf = logs / "launch_20260620_120000.log"
    lf.write_text("line1\nline2\n", encoding="utf-8")
    # first poll (no offset) -> reset + full content + offset parked at EOF
    j = c.get("/launch_log").get_json()
    assert j["available"] is True and j["reset"] is True and j["path"] == lf.name
    assert "line1" in j["data"] and "line2" in j["data"]
    off = j["offset"]
    assert off == lf.stat().st_size
    # no new bytes -> empty incremental chunk, offset unchanged, no reset
    j2 = c.get(f"/launch_log?offset={off}&path={lf.name}").get_json()
    assert j2["data"] == "" and j2["reset"] is False and j2["offset"] == off
    # append -> ONLY the new bytes come back
    with open(lf, "a", encoding="utf-8") as fh:
        fh.write("line3\n")
    j3 = c.get(f"/launch_log?offset={off}&path={lf.name}").get_json()
    # ONLY the appended bytes come back (CRLF-tolerant: the log is written in text mode on Windows)
    assert j3["data"].replace("\r\n", "\n") == "line3\n" and j3["reset"] is False


def test_launch_log_resync_on_rotation_and_path_change(client):
    import os
    import time
    _, c, arch = client
    logs = arch / "logs"
    logs.mkdir()
    lf = logs / "launch_20260620_120000.log"
    lf.write_text("abcdef\n", encoding="utf-8")
    size = lf.stat().st_size
    # offset past EOF (file truncated/rotated under us) -> reset from tail
    j = c.get(f"/launch_log?offset={size + 999}&path={lf.name}").get_json()
    assert j["reset"] is True and "abcdef" in j["data"]
    # a NEWER run's log supersedes -> resolves to it; the stale client path forces a reset
    newer = logs / "launch_20260620_130000.log"
    newer.write_text("NEWRUN\n", encoding="utf-8")
    os.utime(newer, (time.time() + 10, time.time() + 10))     # guarantee newest by mtime
    j2 = c.get(f"/launch_log?offset=3&path={lf.name}").get_json()
    assert j2["path"] == newer.name and j2["reset"] is True and "NEWRUN" in j2["data"]


def test_launch_log_prefers_in_memory_path_over_glob(client):
    _, c, arch = client
    logs = arch / "logs"
    logs.mkdir()
    (logs / "launch_old.log").write_text("OLD\n", encoding="utf-8")
    chosen = logs / "launch_chosen.log"
    chosen.write_text("CHOSEN\n", encoding="utf-8")
    c.application.config["_LAUNCH_LOG"] = str(chosen)             # explicit handle wins over newest-glob
    j = c.get("/launch_log").get_json()
    assert j["path"] == chosen.name and "CHOSEN" in j["data"]


# ── auto-find the newest run subfolder (2026-09-03) ───────────────────────────
# A run launched with --archive artifacts/self_play_archive/<name> writes status.json into that
# SUBFOLDER; the server's default archive dir is the ROOT, so the dashboard used to show nothing
# unless the operator passed --archive too. resolve_run_dir() closes that gap.
def _write_status(d, mtime=None, live=True):
    import json
    import os
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({"live": live, "run": {"generation": 1}}),
                                   encoding="utf-8")
    if mtime is not None:
        os.utime(d / "status.json", (mtime, mtime))
    return d


def test_resolve_run_dir_prefers_the_base_when_it_has_a_status(tmp_path):
    """An EXPLICIT --archive pointing straight at a run must be returned unchanged."""
    from v_dance.datatools.dashboard_server import resolve_run_dir
    _write_status(tmp_path)
    _write_status(tmp_path / "some_run", mtime=9_000_000_000)      # newer, but base wins
    assert resolve_run_dir(tmp_path) == tmp_path


def test_resolve_run_dir_finds_the_newest_subfolder(tmp_path):
    from v_dance.datatools.dashboard_server import resolve_run_dir
    _write_status(tmp_path / "old_run", mtime=1_000_000_000)
    newest = _write_status(tmp_path / "new_run", mtime=2_000_000_000)
    (tmp_path / "not_a_run").mkdir()                               # no status.json -> ignored
    assert resolve_run_dir(tmp_path) == newest


def test_resolve_run_dir_falls_back_to_base_when_nothing_ran(tmp_path):
    from v_dance.datatools.dashboard_server import resolve_run_dir
    (tmp_path / "empty").mkdir()
    assert resolve_run_dir(tmp_path) == tmp_path
    assert resolve_run_dir(tmp_path / "does_not_exist") == tmp_path / "does_not_exist"


def test_status_and_manifest_served_from_the_newest_subfolder(tmp_path):
    """The whole point: point the server at the ROOT and still get the run's live feeds."""
    import json
    from v_dance.datatools.dashboard_server import create_app
    run = _write_status(tmp_path / "era5b_v2_from_era2", mtime=2_000_000_000)
    (run / "manifest.json").write_text(json.dumps({"champion_generation": 28}), encoding="utf-8")
    _write_status(tmp_path / "older_run", mtime=1_000_000_000)
    app = create_app(archive_dir=tmp_path)
    c = app.test_client()
    assert c.get("/status.json").get_json()["run"]["generation"] == 1
    assert c.get("/manifest.json").get_json()["champion_generation"] == 28


def test_a_newer_run_is_followed_without_restarting_the_server(tmp_path):
    """Resolution is per REQUEST, so starting a second run switches the dashboard over."""
    import json
    from v_dance.datatools.dashboard_server import create_app
    first = _write_status(tmp_path / "run_a", mtime=1_000_000_000)
    (first / "manifest.json").write_text(json.dumps({"champion_generation": 1}), encoding="utf-8")
    app = create_app(archive_dir=tmp_path)
    c = app.test_client()
    assert c.get("/manifest.json").get_json()["champion_generation"] == 1
    second = _write_status(tmp_path / "run_b", mtime=2_000_000_000)
    (second / "manifest.json").write_text(json.dumps({"champion_generation": 2}), encoding="utf-8")
    assert c.get("/manifest.json").get_json()["champion_generation"] == 2   # same app object


def test_launcher_state_stays_on_the_base_archive_dir(tmp_path):
    """A NEW run's config / log / pid must never land inside an OLD run's folder."""
    from v_dance.datatools.dashboard_server import create_app
    _write_status(tmp_path / "old_run", mtime=1_000_000_000, live=False)
    app = create_app(archive_dir=tmp_path)
    assert app.config["ARCHIVE_DIR"] == tmp_path.resolve()


def test_empty_archive_root_serves_the_idle_placeholders(tmp_path):
    from v_dance.datatools.dashboard_server import create_app
    c = create_app(archive_dir=tmp_path).test_client()
    assert c.get("/status.json").status_code == 200
    assert c.get("/manifest.json").status_code == 200
