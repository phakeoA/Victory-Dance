"""2026-09-04 — the pacing gate INSIDE the real consumer (USER: the team picker said "typed too quickly" and no
Pokémon was picked). Five rooms start at once under the immediate timer: the five ``/timer on`` use the burst
inline, the five team choices that follow are DEFERRED and paced (here with a fast refill), all ten reach the
page in order (timers, then decisions), a server throttle notice in a frame is counted, and the panel's status
carries the gate's counters. Driven with the timer tests' fakes — no Playwright, no server."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("poke_env")

import test_battle_timer_lanes as T                        # the consumer driver + fakes (pytest prepend import)
import v_dance.play.play_vs_human_browser as _pvhb
from v_dance.play.send_gate import SendGate

FMT = "gen9championsvgc2026regmb"
ROOMS = [f"battle-{FMT}-77{i}" for i in range(5)]


def test_five_lanes_at_team_preview_are_paced_not_dropped(monkeypatch):
    monkeypatch.setattr(_pvhb, "TIMER_IMMEDIATE", True)
    monkeypatch.setattr(SendGate, "REFILL_S", 0.05)          # 0.65 s in production; fast for the test clock
    frames = [(0.0, f">{r}\n|init|battle") for r in ROOMS]                                   # 5 timers (inline)
    frames += [(0.0, f">{r}\n|request|{{\"teamPreview\":true}}") for r in ROOMS]             # 5 /team (deferred)
    frames += [(0.05, f">{ROOMS[0]}\n|raw|<strong class=\"message-throttle-notice\">Your message was not sent "
                      "because you've been typing too quickly.</strong>")]
    page, _host = asyncio.run(T._drive(frames, total_s=0.9))
    msgs = [(a["r"], a["m"]) for _, a in page.evals if isinstance(a, dict) and "m" in a]
    timers = [r for r, m in msgs if m == "/timer on"]
    teams = [r for r, m in msgs if m.startswith("/team")]
    assert sorted(timers) == sorted(ROOMS) and sorted(teams) == sorted(ROOMS)   # nothing dropped, once per room
    assert max(i for i, (_, m) in enumerate(msgs) if m == "/timer on") < min(
        i for i, (_, m) in enumerate(msgs) if m.startswith("/team"))              # the burst went to the timers
    g = _pvhb.SEND_GATE[0]
    st = g.stats()
    assert st["sent"] == 10 and st["deferred"] == 5 and st["queued"] == 0 and st["errors"] == 0
    assert st["notices"] == 1                                    # the server notice in the frame was counted
    assert st["max_wait_s"] > 0.0


def test_single_room_flow_is_byte_identical_to_the_pre_gate_wire_format(monkeypatch):
    monkeypatch.setattr(_pvhb, "TIMER_IMMEDIATE", True)
    A = ROOMS[0]
    frames = [(0.0, f">{A}\n|init|battle"), (0.05, f">{A}\n|request|{{\"teamPreview\":true}}")]
    page, _ = asyncio.run(T._drive(frames, total_s=0.4))
    sends = [(js, a) for js, a in page.evals if isinstance(a, dict) and "m" in a]
    assert all(js == "(d) => app.socket.send(d.r + '|' + d.m)" for js, _ in sends)   # the same JS + arg shape
    assert [(a["r"], a["m"]) for _, a in sends] == [(A, "/timer on"), (A, "/team 1234")]
    assert _pvhb.SEND_GATE[0].stats()["deferred"] == 0        # under the burst: inline, no pump


def test_panel_status_carries_the_gate_counters():
    from v_dance.play import bot_control_ui as bcu
    src = (bcu.__file__)
    text = open(src, encoding="utf-8").read()
    assert '"send_gate"' in text and "SEND_GATE" in text


def test_a_stale_team_choice_after_a_rejoin_is_not_shipped_once_the_battle_is_under_way(monkeypatch):
    """2026-09-06 (the juanrava game): on a reconnect rejoin poke-env re-emits the team-preview pick from the replayed
    |teampreview|; when the replay already shows |start| / |turn| the preview is over and the server only answers
    "[Invalid choice] Can't choose for Team Preview" (which then made the retry guard perturb the real turn-1 move).
    The consumer drops that /team; a fresh room's /team still ships; the live |request| completes the rejoin."""
    monkeypatch.setattr(_pvhb, "TIMER_IMMEDIATE", False)
    monkeypatch.setattr(SendGate, "REFILL_S", 0.01)
    rejoined, fresh = ROOMS[0], ROOMS[1]
    _pvhb.REJOINING.add(rejoined)
    try:
        frames = [(0.0, f">{rejoined}\n|init|battle"),
                  (0.0, f">{rejoined}\n|start|\n|turn|1\n|request|{{}}"),          # the replay: preview long over
                  (0.0, f">{fresh}\n|init|battle"),
                  (0.0, f">{fresh}\n|request|{{\"teamPreview\":true}}")]
        page, _host = asyncio.run(T._drive(frames, total_s=0.5))
        msgs = [(a["r"], a["m"]) for _, a in page.evals if isinstance(a, dict) and "m" in a]
        assert (fresh, "/team 1234") in msgs and (rejoined, "/team 1234") not in msgs
        assert rejoined not in _pvhb.REJOINING                              # the live |request| completed the rejoin
    finally:
        _pvhb.REJOINING.clear()
