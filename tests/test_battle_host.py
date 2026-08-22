"""Tests for v_dance.play.browser.battle_host.BattleHost — the connection-less frame→decision core
that the "AI plays through the browser" transport reuses (local now, online later).

These validate the TRANSPORT PLUMBING with NO Showdown server and NO model (model_path=None → random
fallback): the host builds a real VGCPlayer that never opens a socket, routes raw protocol frames
through poke-env's own parser, creates the battle, accumulates the gap-#6 _proto_log, and CAPTURES the
commands poke-env would have websocket-sent (so the transport can ship them into the tab). The full
request→/choose decision parity is exercised by the live spike scratch/browser_battlehost_spike.py
(needs a real battle), not here."""
from __future__ import annotations

import pytest

pytest.importorskip("poke_env")

from v_dance.formats import DEFAULT_FORMAT
from v_dance.play.browser.battle_host import BattleHost

_TAG = f"battle-{DEFAULT_FORMAT}-1"
# A minimal, well-formed VGC-doubles battle-init frame (the shape poke-env's PSClient delivers).
_INIT_FRAME = (
    f">{_TAG}\n"
    "|init|battle\n"
    "|title|Alice vs. Bob\n"
    "|gametype|doubles\n"
    "|player|p1|Alice|1|\n"
    "|player|p2|Bob|2|\n"
    "|turn|1"
)


def _host():
    # model_path/team_chooser_path=None → random fallback (no torch load); we only exercise plumbing.
    return BattleHost(team=None, model_path=None, team_chooser_path=None)


def test_constructs_offline_without_connecting():
    h = _host()
    # A real poke-env player exists but is NOT connected (start_listening=False → no websocket set).
    assert h.player is not None
    assert not hasattr(h.player.ps_client, "websocket") or \
        getattr(h.player.ps_client, "websocket", None) is None
    assert h.battles == {}


def test_init_frame_creates_battle_and_routes_through_pokeenv_parser():
    h = _host()
    produced = h.feed(_INIT_FRAME)
    # poke-env created the battle from the frame (its parser is being driven, no socket).
    assert _TAG in h.battles
    # VGC init makes poke-env emit an open-team-sheets decision; the host defaults to REJECT (matching
    # production make_player) — we CAPTURE it (not websocket-send it).
    assert (_TAG, "/rejectopenteamsheets") in produced
    # the gap-#6 opponent splice captured the public protocol for this battle.
    assert len(h.player._proto_log.get(_TAG, [])) > 0


def test_drain_returns_and_clears_buffer():
    h = _host()
    h.feed(_INIT_FRAME)
    drained = h.drain()
    assert any(msg == "/rejectopenteamsheets" for _room, msg in drained)
    assert h.drain() == []        # cleared


def test_end_battle_reclaims_state():
    h = _host()
    h.feed(_INIT_FRAME)
    assert _TAG in h.battles and len(h.player._proto_log.get(_TAG, [])) > 0
    h.end_battle(_TAG)
    assert _TAG not in h.battles                  # battle object evicted
    assert _TAG not in h.player._proto_log        # splice log evicted
    assert h.drain() == []                        # outgoing buffer cleared


def test_captured_message_carries_room_and_is_not_sent_over_a_socket():
    h = _host()
    produced = h.feed(_INIT_FRAME)
    # every captured item is (room, message); the room is this battle's tag.
    for room, msg in produced:
        assert isinstance(msg, str) and msg.startswith("/")
        assert room == _TAG
