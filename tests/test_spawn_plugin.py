"""W2 throughput step 1 (2026-09-03) — the server-side spawn plugin (rlspawn.ts) and its installer.

Offline: the installer copies this repo's plugin into a (fake) pokemon-showdown clone, is idempotent,
and the source carries every command the spawner client will rely on. LIVE (opt-in,
``VD_SPAWN_LIVE_TEST=1``): install into the real clone, start a local server on a spare port (the
launcher rebuilds → the plugin compiles), log two poke-env bot accounts in, set their VGC teams with
``/utm``, ``/rlautospawn`` two battles between them, and see both accounts FINISH games the server
created on its own — no challenge protocol involved.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import pytest

from v_dance.selfplay import spawn_plugin as SP


def _fake_clone(tmp_path: Path) -> Path:
    d = tmp_path / "pokemon-showdown"
    (d / "server" / "chat-plugins").mkdir(parents=True)
    return d


def test_plugin_source_carries_the_contract():
    src = SP.PLUGIN_SRC.read_text(encoding="utf-8")
    for cmd in SP.COMMANDS:
        assert f"\t{cmd}(target, room, user)" in src, cmd
    assert "export const commands: Chat.ChatCommands" in src
    assert "battleSettings" in src and "createBattle(" in src           # VGC team per account (/utm)
    assert "_rlBorn" in src and "RL_MAX_LIFESPAN_MS" in src            # GC of plugin-born rooms
    assert "queryresponse|rlactive|" in src and "queryresponse|rlstatus|" in src
    assert "MIT" in src and "Nebraskinator/ps-ppo" in src              # attribution


def test_installer_copies_once_and_reports(tmp_path: Path, capsys):
    clone = _fake_clone(tmp_path)
    assert not SP.is_installed(clone)
    r1 = SP.install_rlspawn(clone)
    assert r1["changed"] and r1["installed"] and not r1["built"]
    assert Path(r1["dest"]).read_bytes() == SP.PLUGIN_SRC.read_bytes() and SP.is_installed(clone)
    r2 = SP.install_rlspawn(clone)
    assert not r2["changed"]                                          # idempotent
    Path(r1["dest"]).write_text("stale", encoding="utf-8")
    assert not SP.is_installed(clone) and SP.install_rlspawn(clone)["changed"]
    assert SP.compiled_path(clone).name == "rlspawn.js" and not SP.compiled_path(clone).exists()
    assert SP.main(["--check", "--showdown-dir", str(clone)]) == 0
    assert "CURRENT" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        SP.install_rlspawn(tmp_path / "nope")
    assert SP.main(["--check", "--showdown-dir", str(tmp_path / "nope")]) == 1


@pytest.mark.skipif(os.environ.get("VD_SPAWN_LIVE_TEST") != "1",
                    reason="opt-in: VD_SPAWN_LIVE_TEST=1 starts a local Showdown server (~1-2 min)")
def test_live_server_spawns_battles_between_two_bot_accounts():
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R
    from v_dance.play import parallel_battles as PB
    from poke_env import AccountConfiguration
    from poke_env.concurrency import POKE_LOOP
    from poke_env.player.baselines import RandomPlayer

    assert SP.SHOWDOWN_DIR.is_dir(), "pokemon-showdown clone missing (setup.sh)"
    SP.install_rlspawn()                                              # the server start rebuilds it
    port = int(os.environ.get("VD_SPAWN_LIVE_PORT", "8011"))
    proc = R.start_showdown_on(port)
    assert R._port_open(R.SHOWDOWN_HOST, port), "local Showdown did not come up"
    fmt = R.BATTLE_FORMAT
    paths = R.discover_teams(reg=fmt)[:2]
    assert len(paths) == 2
    teams = [R.load_team(R.resolve_team_path(p)) for p in paths]
    srv = R.localhost_server_config(port)
    salt = time.strftime("%H%M%S")
    players = [RandomPlayer(account_configuration=AccountConfiguration(f"RLsp{s}{salt}", None),
                            server_configuration=srv, battle_format=fmt, team=t,
                            max_concurrent_battles=4, start_listening=True, log_level=logging.WARNING)
               for s, t in zip("AB", teams)]
    pA, pB = players
    n_target = 2

    async def drive():
        for p in players:
            await asyncio.wait_for(p.ps_client.logged_in.wait(), 45)
        for p in players:                                             # the plugin reads battleSettings.team
            await p.ps_client.send_message("/utm " + p._team.yield_team())
        await pA.ps_client.send_message(f"/rlautospawn {pA.username}, {pB.username}, {fmt}, {n_target}")
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and min(p.n_finished_battles for p in players) < n_target:
            await asyncio.sleep(1.0)
        await pA.ps_client.send_message(f"/rlautooff {pA.username}, {pB.username}, {fmt}")
        await pA.ps_client.send_message("/rlstatus")
        await asyncio.sleep(1.0)
        return [p.n_finished_battles for p in players], [len(p.battles) for p in players]

    try:
        finished, seen = asyncio.run_coroutine_threadsafe(drive(), POKE_LOOP).result(timeout=320)
        print(f"[live] finished per account {finished}, battle rooms seen {seen}")
        assert min(finished) >= n_target, f"battles never finished: {finished}"
        assert min(seen) >= n_target                                  # rooms the SERVER created for them
    finally:
        try:
            asyncio.run(PB.close_players(*players))
        except Exception:
            pass
        R.stop_showdown(proc)
