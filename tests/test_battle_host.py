"""Tests for v_dance.online.browser.battle_host.BattleHost — the connection-less frame→decision core
that the "AI plays through the browser" transport reuses (local now, online later).

These validate the TRANSPORT PLUMBING with NO Showdown server and NO model (model_path=None → random
fallback): the host builds a real VGCPlayer that never opens a socket, routes raw protocol frames
through poke-env's own parser, creates the battle, accumulates the gap-#6 _proto_log, and CAPTURES the
commands poke-env would have websocket-sent (so the transport can ship them into the tab). The full
request→/choose decision parity is exercised by the live spike scratch/archive/browser/browser_battlehost_spike.py
(needs a real battle), not here."""
from __future__ import annotations

import pytest

pytest.importorskip("poke_env")

from v_dance.formats import DEFAULT_FORMAT
from v_dance.online.browser.battle_host import BattleHost

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


# ── 2026-10-01: frames for a battle with NO battle object park instead of blocking forever ──────────
def _turn(n):
    return f">{_TAG}\n|\n|t:|1790000000\n|turn|{n}"


def test_a_frame_before_the_init_parks_instantly_and_replays_after_it():
    h = _host()
    assert h.feed(_turn(2), timeout=5) == []                 # returns at once (used to wait forever)
    assert _TAG not in h.battles and [t for t, _a in h.parked_older_than(0)] == [_TAG]
    h.feed(_INIT_FRAME)                                      # the battle appears → the parked frame is replayed
    assert h.battles[_TAG].turn == 2 and h.parked_older_than(0) == []


def test_after_a_reconnect_forget_the_replayed_init_supersedes_the_parked_frames():
    h = _host()
    h.feed(_INIT_FRAME)
    h.forget_battle(_TAG)                                    # reconnect: the server will re-send the full log
    h.feed(_turn(5), timeout=5)                              # a stale pre-disconnect frame: parked, not fed
    h.feed(_INIT_FRAME)                                      # the REPLAY (turn 1 here)
    assert h.battles[_TAG].turn == 1 and h.parked_older_than(0) == []   # stale frame discarded, not re-applied


def test_parked_age_and_end_battle_cleanup():
    h = _host()
    t = [100.0]
    h._clock = lambda: t[0]
    h.feed(_turn(2), timeout=5)
    t[0] = 109.0
    assert h.parked_older_than(8) == [(_TAG, 9.0)]
    h.touch_parked(_TAG)
    assert h.parked_older_than(8) == []
    h.end_battle(_TAG)
    t[0] = 200.0
    assert h.parked_older_than(0) == []
