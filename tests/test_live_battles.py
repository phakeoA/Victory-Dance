"""Offline tests for the file-per-active-battle spectate feed (task #18.1).

Each concurrent battle (any process — main or an MP collection worker) writes its own
``<live_dir>/<tag>.json``; the dashboard server globs them. No server / poke-env."""
from __future__ import annotations

import json

from v_dance.selfplay.status import (LiveBattles, read_live_battles, _safe_tag,
                                     run_stamp, gen_kind_dir, current_live_dir)


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_safe_tag_sanitizes():
    assert _safe_tag("battle-gen9vgc2026-123") == "battle-gen9vgc2026-123"
    assert _safe_tag("weird/tag:1") == "weird_tag_1"
    assert _safe_tag("") == "battle"


def test_update_writes_one_file_per_battle(tmp_path):
    lb = LiveBattles(tmp_path)
    lb.update("battle-a-1", p1="SP1", p2="SP2", turn=3, log=["|turn|3", "|move|x"])
    lb.update("battle-b-2", p1="SP3", p2="SP4", turn=1, log=["|turn|1"])
    files = sorted(p.name for p in tmp_path.glob("*.json"))
    assert files == ["battle-a-1.json", "battle-b-2.json"]
    a = json.loads((tmp_path / "battle-a-1.json").read_text())
    assert a["tag"] == "battle-a-1" and a["p1"] == "SP1" and a["turn"] == 3 and a["n_lines"] == 2


def test_remove_drops_the_file(tmp_path):
    lb = LiveBattles(tmp_path)
    lb.update("battle-a-1", turn=1)
    assert (tmp_path / "battle-a-1.json").exists()
    lb.remove("battle-a-1")
    assert not (tmp_path / "battle-a-1.json").exists()
    lb.remove("battle-a-1")          # removing a missing one is a no-op (no raise)


def test_update_throttles_per_battle(tmp_path):
    clk = _Clock(100.0)
    lb = LiveBattles(tmp_path, clock=clk, min_interval=0.5)
    lb.update("b", turn=1)
    clk.t = 100.2                     # within the throttle window
    lb.update("b", turn=2)           # skipped
    assert json.loads((tmp_path / "b.json").read_text())["turn"] == 1
    clk.t = 100.8                     # past the window
    lb.update("b", turn=3)
    assert json.loads((tmp_path / "b.json").read_text())["turn"] == 3
    # throttle is PER battle — a different tag isn't blocked by b's recent write
    clk.t = 100.85
    lb.update("c", turn=9)
    assert (tmp_path / "c.json").exists()


def test_read_live_battles_aggregates_and_sorts(tmp_path):
    clk = _Clock(500.0)
    lb = LiveBattles(tmp_path, clock=clk)
    lb.update("battle-z", turn=1)
    lb.update("battle-a", turn=2)
    got = read_live_battles(tmp_path, clock=clk)
    assert [b["tag"] for b in got] == ["battle-a", "battle-z"]   # sorted by tag


def test_read_live_battles_drops_stale(tmp_path):
    import os
    import time
    lb = LiveBattles(tmp_path)
    lb.update("fresh", turn=1)
    lb.update("stale", turn=1)
    old = time.time() - 60           # the reader pre-filters on the cheap file mtime
    os.utime(tmp_path / "stale.json", (old, old))
    got = read_live_battles(tmp_path, stale_after=15.0)
    assert [b["tag"] for b in got] == ["fresh"]


def test_read_live_battles_empty_dir(tmp_path):
    assert read_live_battles(tmp_path) == []
    assert read_live_battles(tmp_path / "does-not-exist") == []


# ── structured saved-replay layout (#18b) ─────────────────────────────────────
def test_run_stamp_format():
    import datetime
    assert run_stamp(datetime.datetime(2026, 6, 18, 14, 30, 5)) == "2026-06-18_14-30-05"


def test_gen_kind_dir():
    assert gen_kind_dir("/run", 3, "replays").as_posix().endswith("/run/gen_3/replays")
    assert gen_kind_dir("/run", 1, "eval").as_posix().endswith("/run/gen_1/eval")
    import pytest
    with pytest.raises(AssertionError):
        gen_kind_dir("/run", 0, "bogus")


def test_finalize_keeps_file_but_marks_done(tmp_path):
    import json
    lb = LiveBattles(tmp_path)
    lb.update("b", turn=2, log=["|turn|2"])
    lb.finalize("b", turn=5, log=["|turn|2", "|win|x"])     # save: keep file, mark done
    d = json.loads((tmp_path / "b.json").read_text())
    assert d["done"] is True and d["turn"] == 5 and d["n_lines"] == 2
    assert read_live_battles(tmp_path) == []                # done → not LIVE (but file remains)
    assert (tmp_path / "b.json").exists()


def test_current_live_dir_scopes_to_latest_run_and_gen(tmp_path):
    # review #18b: the dashboard scans only the current run's latest gen, not the whole tree
    root = tmp_path / "live"
    LiveBattles(gen_kind_dir(root / "2026-06-18_01-00-00", 0, "replays")).update("old", turn=1)
    cur = root / "2026-06-18_02-00-00"
    LiveBattles(gen_kind_dir(cur, 0, "replays")).update("g0", turn=1)
    LiveBattles(gen_kind_dir(cur, 1, "eval")).update("g1", turn=1)
    scope = current_live_dir(root)
    assert scope == cur / "gen_1"                    # newest run, newest gen
    # reading the scope sees only that gen's battles, not the old run's
    tags = [b["tag"] for b in read_live_battles(scope)]
    assert tags == ["g1"]


def test_current_live_dir_fallbacks(tmp_path):
    assert current_live_dir(tmp_path / "nope") == (tmp_path / "nope")   # missing -> itself
    flat = tmp_path / "live"
    LiveBattles(flat).update("b", turn=1)            # flat layout (no run-stamp dirs)
    assert current_live_dir(flat) == flat            # -> the root itself


def test_read_live_battles_recurses_and_annotates_gen_kind(tmp_path):
    # structured tree: <run>/gen_0/replays/<tag>.json  +  <run>/gen_1/eval/<tag>.json
    LiveBattles(gen_kind_dir(tmp_path, 0, "replays")).update("battle-c", p1="SP1", p2="SP2", turn=3)
    LiveBattles(gen_kind_dir(tmp_path, 1, "eval")).update("battle-e", p1="BC", p2="heuristic", turn=2)
    got = read_live_battles(tmp_path)
    by_tag = {b["tag"]: b for b in got}
    assert by_tag["battle-c"]["gen"] == 0 and by_tag["battle-c"]["kind"] == "replays"
    assert by_tag["battle-e"]["gen"] == 1 and by_tag["battle-e"]["kind"] == "eval"
