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


# ── Fix A: --save-replays now writes a real playable .html replay, not litter JSON ────────────
class _StubBattle:
    """Duck-typed stand-in for poke-env's battle: exposes ``_replay_data`` (List[List[str]] — each a
    ``|``-split protocol message), which the save path rejoins into the raw ``|``-log and renders via
    our Showdown emitter. ``raises`` makes reading it blow up (exercises the crash-proof path)."""
    def __init__(self, tag, lines=None, raises=False):
        self.battle_tag = tag
        self._raises = raises
        self._lines = lines if lines is not None else ["|gametype|doubles", "|turn|1", "|win|TrainerRed"]

    @property
    def _replay_data(self):
        if self._raises:
            raise RuntimeError("boom")
        return [ln.split("|") for ln in self._lines]           # "|move|x" -> ['', 'move', 'x']


def test_save_html_replay_writes_html_and_drops_json(tmp_path):
    lb = LiveBattles(tmp_path)
    lb.update("battle-x-9", turn=4, log=["|turn|4"])          # the LIVE-feed json exists
    assert (tmp_path / "battle-x-9.json").exists()
    out = lb.save_html_replay("battle-x-9", _StubBattle("battle-x-9",
                              lines=["|tier|[Gen 9] VGC", "|win|steven"]))
    assert out == tmp_path / "battle-x-9.html"
    txt = out.read_text(encoding="utf-8")
    assert out.exists() and 'class="battle-log-data"' in txt and "replay-embed.js" in txt  # Showdown format
    assert "|win|steven" in txt                              # the raw server log embedded faithfully
    assert not (tmp_path / "battle-x-9.json").exists()        # the live json is dropped on finish
    assert read_live_battles(tmp_path) == []                  # nothing LIVE remains


def test_save_html_replay_sanitizes_filename(tmp_path):
    lb = LiveBattles(tmp_path)
    out = lb.save_html_replay("battle/odd:tag", _StubBattle("battle/odd:tag"))
    assert out == tmp_path / f"{_safe_tag('battle/odd:tag')}.html" and out.exists()


def test_save_html_replay_failure_is_crash_proof_and_still_drops_json(tmp_path):
    lb = LiveBattles(tmp_path)
    lb.update("b", turn=1, log=["|turn|1"])
    out = lb.save_html_replay("b", _StubBattle("b", raises=True))   # save blows up
    assert out is None                                       # reported, not raised
    assert not (tmp_path / "b.json").exists()                # live json still dropped


def test_save_replays_does_not_enable_pokeenv_native_dump(tmp_path):
    """Regression: ``SplicingVGCPlayerBase`` must NOT reuse poke-env's ``_save_replays`` attribute.
    poke-env's ``Player`` stores its OWN native-save flag there and ``_create_battle`` reads it to
    stamp every battle; clobbering it back to True re-enabled poke-env's uncategorized
    ``./replays/<user> - <tag>.html`` dump (the stray root + selfplay folders). Our flag lives under
    a DISTINCT name so poke-env's stays False and only the structured ``<live_dir>/<tag>.html`` lands."""
    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    from v_dance.play.player import VGCPlayer
    team = R.load_team(R.resolve_team_path("WolfeGlick"))
    p = VGCPlayer(model_path=None, account_configuration=AccountConfiguration("BCnodump", None),
                  battle_format=R.BATTLE_FORMAT, team=team, save_replays=True,
                  live_dir=str(tmp_path), start_listening=False)
    assert p._save_replays is False           # poke-env's NATIVE dump stays DISABLED
    assert p._save_html_replays is True        # our structured-HTML flag is the one that's set


def test_close_evicts_per_player_logger(tmp_path):
    """Leak fix: poke-env interns a unique-named logger+handler per player in the process-global
    loggerDict forever; with 22d gen-salted usernames that's one leak per player per gen. close()
    must evict it so a long run doesn't accumulate tens of thousands of Logger/Handler objects."""
    import gc
    import logging
    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    from v_dance.play.player import VGCPlayer
    team = R.load_team(R.resolve_team_path("WolfeGlick"))
    name = "LG0x999x777"                       # a representative gen-salted single-use account name
    p = VGCPlayer(model_path=None, account_configuration=AccountConfiguration(name, None),
                  battle_format=R.BATTLE_FORMAT, team=team, start_listening=False)
    assert name in logging.Logger.manager.loggerDict          # poke-env interned it on construction
    p.close()
    gc.collect()
    assert name not in logging.Logger.manager.loggerDict      # close() reclaimed it (no leak)


def test_save_html_replay_real_poke_env_double_battle(tmp_path):
    # the actual path the finished-callback relies on: read the real battle's accumulated raw
    # _replay_data and render OUR Showdown replay HTML (no poke-env save_replay).
    import logging
    from poke_env.battle.double_battle import DoubleBattle
    b = DoubleBattle("battle-gen9vgc2026regma-77", "TrainerRed",
                     logging.getLogger("test"), gen=9)
    b._replay_data.append(["", "tier", "[Gen 9 Champions] VGC 2026 Reg M-A"])
    b._replay_data.append(["", "turn", "1"])
    b._replay_data.append(["", "win", "TrainerRed"])          # the server's ground-truth stream
    lb = LiveBattles(tmp_path)
    out = lb.save_html_replay("battle-gen9vgc2026regma-77", b)
    assert out is not None and out.exists() and out.suffix == ".html"
    txt = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in txt and 'class="battle-log-data"' in txt and "replay-embed.js" in txt
    assert "|turn|1" in txt and "|win|TrainerRed" in txt      # raw server log faithfully embedded
    assert "battle-gen9vgc2026regma-77" in txt                # replayid


def test_render_replay_html_is_showdown_format():
    """The emitter that replaces poke-env's save_replay — Showdown's canonical replay format."""
    from v_dance.selfplay.replay_html import render_replay_html
    log = ["|tier|[Gen 9 Champions] VGC 2026 Reg M-A", "|player|p1|Alice|101|1500",
           "|player|p2|Bob|rosa|1500", "|move|p1a: Charizard|Flamethrower|p2a: Venusaur", "|win|Alice"]
    h = render_replay_html(log, replayid="b-1")
    assert h.startswith("<!DOCTYPE html>")
    assert 'class="battle-log-data"' in h and "replay-embed.js" in h      # CDN-animated player
    assert "|move|p1a: Charizard|Flamethrower|p2a: Venusaur" in h         # raw log embedded
    assert "[Gen 9 Champions] VGC 2026 Reg M-A" in h                      # format in title/header
    assert "Alice vs. Bob" in h                                           # players parsed from |player|


def test_render_replay_html_neutralises_script_close():
    from v_dance.selfplay.replay_html import render_replay_html
    h = render_replay_html(["|c|x|oops </script> hi"])
    assert "<\\/script>" in h and "oops <\\/script> hi" in h              # </ escaped, tag can't close early


# ── task E: eval replays routed by opponent into eval/<kind>/ + eval/league/ ───────────────────
def test_eval_replay_routing_scripted():
    from v_dance.eval.gauntlet import eval_replay_routing
    assert eval_replay_routing("heuristic", 3) == ("heuristic", "gen3_vs_heuristic")
    assert eval_replay_routing("random", 0) == ("random", "gen0_vs_random")
    assert eval_replay_routing("max_damage", 12) == ("max_damage", "gen12_vs_max_damage")


def test_eval_replay_routing_league_gen_vs_gen():
    from v_dance.eval.gauntlet import eval_replay_routing
    # prev_best mirror / HoF: opp checkpoint gen -> eval/league/gen<N>_vs_gen<M>
    assert eval_replay_routing("prev_best", 3, opp_ref="x/checkpoints/gen1.pt") == ("league", "gen3_vs_gen1")
    # opponent path isn't a gen<M>.pt (e.g. the base BC) -> 'champion'
    assert eval_replay_routing("prev_best", 3, opp_ref="x/bc_best.pt") == ("league", "gen3_vs_champion")
    # unknown candidate gen (standalone gauntlet on a non-gen ckpt) -> gen?
    assert eval_replay_routing("prev_best", None, opp_ref="a/gen2.pt") == ("league", "gen?_vs_gen2")


def test_list_saved_eval_replays_orders_league_first_latest_run(tmp_path):
    # #H: the saved-replay browser lists the LATEST run's gen_<N>/eval/, league section first
    from v_dance.selfplay.status import list_saved_eval_replays, latest_run_dir
    live = tmp_path / "live"
    old = live / "2026-06-18_01-00-00" / "gen_2" / "eval" / "random"
    old.mkdir(parents=True)
    (old / "old.html").write_text("x")                       # an EARLIER run — must be ignored
    new = live / "2026-06-18_05-00-00" / "gen_4" / "eval"
    (new / "league").mkdir(parents=True)
    (new / "heuristic").mkdir(parents=True)
    (new / "league" / "gen4_vs_gen1_battle-a-1.html").write_text("x")
    (new / "heuristic" / "gen4_vs_heuristic_battle-b-2.html").write_text("x")
    assert latest_run_dir(live).name == "2026-06-18_05-00-00"
    out = list_saved_eval_replays(live, 4)
    assert [s["section"] for s in out] == ["league", "heuristic"]    # league FIRST, then scripted alpha
    assert out[0]["files"][0]["name"] == "gen4_vs_gen1_battle-a-1.html"
    assert list_saved_eval_replays(live, 99) == []                   # a gen with none -> empty


def test_save_html_replay_out_dir_and_label(tmp_path):
    # task E: live JSON lives in self.dir; the HTML lands in out_dir with the descriptive prefix
    lb = LiveBattles(tmp_path)
    lb.update("battle-z-7", turn=2, log=["|turn|2"])        # the live-feed json (self.dir)
    sub = tmp_path / "heuristic"
    out = lb.save_html_replay("battle-z-7", _StubBattle("battle-z-7", lines=["|win|R"]),
                              out_dir=sub, label="gen3_vs_heuristic")
    assert out == sub / "gen3_vs_heuristic_battle-z-7.html"
    assert out.exists() and "|win|R" in out.read_text(encoding="utf-8")
    assert not (tmp_path / "battle-z-7.json").exists()      # the live json (self.dir) was dropped


def test_save_html_replay_copy_dir_writes_corpus_copy(tmp_path):
    """2026-07-10 (USER): ``copy_dir`` drops a SECOND identical HTML into the training-corpus
    folder (data/vods/Type_C at serve), name prefixed with out_dir's basename (the session id) —
    local battle tags restart with the server, so the bare tag is not collision-safe.
    2026-07-19 (USER): a RATED game keeps that contract; an UNRATED (private/challenge/local)
    game routes to the sibling ``<copy_dir>_private`` instead, its name led by the OPPONENT's
    username (cherry-pick folder reads by opponent; session+tag stem keeps repeats unique)."""
    lb = LiveBattles(tmp_path)
    sess = tmp_path / "replays" / "20260710-1234-99"
    corpus = tmp_path / "Type_C"
    rated = _StubBattle("battle-c-1", lines=["|rated|", "|win|R"])
    out = lb.save_html_replay("battle-c-1", rated,
                              out_dir=sess, label="online", copy_dir=corpus)
    assert out == sess / "online_battle-c-1.html" and out.exists()
    copy = corpus / "20260710-1234-99_online_battle-c-1.html"
    assert copy.exists()
    assert copy.read_text(encoding="utf-8") == out.read_text(encoding="utf-8")   # identical html

    unrated = _StubBattle("battle-c-2", lines=["|win|R"])
    unrated.opponent_username = "djmed90"
    out2 = lb.save_html_replay("battle-c-2", unrated,
                               out_dir=sess, label="online", copy_dir=corpus)
    assert out2 == sess / "online_battle-c-2.html" and out2.exists()   # main save unchanged
    priv = tmp_path / "Type_C_private" / "djmed90_20260710-1234-99_online_battle-c-2.html"
    assert priv.exists()
    assert not (corpus / "20260710-1234-99_online_battle-c-2.html").exists()   # never in Type_C


def test_save_html_replay_copy_dir_failure_never_breaks_the_save(tmp_path, monkeypatch):
    lb = LiveBattles(tmp_path)
    bad = tmp_path / "corpus_file_not_dir"
    bad.write_text("occupied", encoding="utf-8")             # mkdir(copy_dir) will fail
    out = lb.save_html_replay("battle-c-2", _StubBattle("battle-c-2", lines=["|win|R"]),
                              out_dir=tmp_path / "s", label="online", copy_dir=bad)
    assert out is not None and out.exists()                  # the primary save still landed


def test_save_html_replay_no_copy_dir_is_byte_identical(tmp_path):
    lb = LiveBattles(tmp_path)
    out = lb.save_html_replay("battle-c-3", _StubBattle("battle-c-3", lines=["|win|R"]))
    assert out == tmp_path / "battle-c-3.html" and out.exists()
