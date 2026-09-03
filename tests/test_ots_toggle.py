"""2026-09-03 (USER): the OPEN TEAM SHEETS toggle. The consumer swaps poke-env's init-time reject for an
accept (poke-env's own accept mode stays OFF: it would defer team preview until both sheets arrive — a hang
when the opponent declines); when both accept, the opponent's ``|showteam|`` sheet is stamped into the battle
net's snapshot sheet-authoritatively (the training corpus's OTS routine); the panel / Mission Control /
launch env carry the toggle; the recorder stamps ``ots`` per game."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("poke_env")

import v_dance.play.play_vs_human_browser as _pvhb
from v_dance.parser.vod_parser.team_sheet import parse_packed_team
from v_dance.play.ots_sheets import apply_ots_sheets, opp_sheet_mons, ots_known, room_base_tag, stamp_ots_sheets

PACKED = ("Charizard||charizarditey|solarpower|heatwave,solarbeam,protect,weatherball|Timid||||||50|,,,,,Fire]"
          "Whimsicott||focussash|prankster|tailwind,moonblast,encore,protect|Timid||||||50|]"
          "Garchomp|||roughskin|earthquake,dragonclaw,rockslide,protect|Jolly||||||50|]"
          "Kingambit||blackglasses|defiant|kowtowcleave,suckerpunch,swordsdance,protect|Adamant||||||50|")


def _id(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sheet_is_stamped_sheet_authoritative_and_skips_exact_or_transformed_mons():
    mons = parse_packed_team(PACKED)
    assert [m["species"] for m in mons] == ["Charizard", "Whimsicott", "Garchomp", "Kingambit"]
    snap = {"opp_active": {"opp_a": {"species": "Charizard", "revealed_moves": ["Struggle"]},
                           "opp_b": {"species": "Whimsicott", "is_transformed": True}},
            "opp_bench": [{"species": "Garchomp", "item_consumed": True},
                          {"species": "Kingambit", "exact": True, "known_moves": ["Protect"]},
                          {"species": "Incineroar"}]}                     # not on the sheet: untouched
    assert stamp_ots_sheets(snap, mons) == 2                             # Charizard + Garchomp
    z = snap["opp_active"]["opp_a"]
    assert _id(z["known_item"]) == "charizarditey" and _id(z["known_ability"]) == "solarpower"
    assert [_id(m) for m in z["known_moves"]] == ["heatwave", "solarbeam", "protect", "weatherball"]
    assert z["sheet_moves"] is True and z["nature"] == "Timid"           # the sheet's four, revealed Struggle out
    assert "known_moves" not in snap["opp_active"]["opp_b"]              # transformed: skipped
    g = snap["opp_bench"][0]
    assert len(g["known_moves"]) == 4 and not g.get("sheet_itemless")    # no item on the sheet, but consumed
    assert snap["opp_bench"][1] == {"species": "Kingambit", "exact": True, "known_moves": ["Protect"]}
    assert snap["opp_bench"][2] == {"species": "Incineroar"}
    assert apply_ots_sheets(snap, mons) is snap and apply_ots_sheets(None, mons) is None
    assert stamp_ots_sheets(snap, None) == 0 and stamp_ots_sheets({}, mons) == 0


def test_opp_sheet_lookup_uses_the_room_base_tag_and_the_opponent_side():
    mons = parse_packed_team(PACKED)
    player = SimpleNamespace(_ots_sheets={"battle-gen9vgc-77": {"p1": mons[:2], "p2": mons[2:]}})
    b = SimpleNamespace(battle_tag="battle-gen9vgc-77-abcdef", player_role="p1")
    assert room_base_tag(">battle-gen9vgc-77-abcdef") == "battle-gen9vgc-77"
    assert [m["species"] for m in opp_sheet_mons(player, b)] == ["Garchomp", "Kingambit"]
    b.player_role = "p2"
    assert [m["species"] for m in opp_sheet_mons(player, b)] == ["Charizard", "Whimsicott"]
    assert opp_sheet_mons(SimpleNamespace(), b) is None                   # closed game
    assert ots_known(player, "battle-gen9vgc-77-abcdef") is True
    player._ots_sheets["battle-gen9vgc-78"] = {"p1": mons[:2]}           # our sheet only (opponent declined)
    assert ots_known(player, "battle-gen9vgc-78") is False


def test_consumer_swaps_the_reject_for_an_accept_only_when_the_toggle_is_on(monkeypatch):
    assert _pvhb.OTS_ACCEPT is False                                     # module default = closed sheets
    assert _pvhb.ots_answer("/rejectopenteamsheets") == "/rejectopenteamsheets"
    monkeypatch.setattr(_pvhb, "OTS_ACCEPT", True)
    assert _pvhb.ots_answer("/rejectopenteamsheets") == "/acceptopenteamsheets"
    assert _pvhb.ots_answer("/choose move 1, move 2") == "/choose move 1, move 2"


def test_panel_toggle_status_and_page(monkeypatch):
    import asyncio
    from v_dance.play import bot_control_ui as bcu
    from v_dance.play.bot_control_ui import BotController
    tb = _mod("test_bot_control_ui")
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])
    monkeypatch.setenv("VD_SITE_POLL", "0")
    monkeypatch.setattr(_pvhb, "OTS_ACCEPT", False)
    loop = asyncio.new_event_loop()
    try:
        host = tb.FakeHost()
        host.player = SimpleNamespace(_ots_sheets={"battle-x-1": {"p1": [], "p2": []}})
        c = BotController(page=tb.FakePage(), host=host, tally={"ai": 0, "you": 0, "draw": 0},
                          ai_pool=tb.POOL, fmt=tb.FMT, username="VictoriousDancing", loop=loop,
                          env_path=Path("unused.env"))
        s = c.status()
        assert s["ots_accept"] is False and s["ots_games"] == 1
        c.set_ots_accept(True)
        assert _pvhb.OTS_ACCEPT is True and c.status()["ots_accept"] is True
        assert any("open team sheets: ACCEPT" in e for e in c.events)
        c.set_ots_accept(False)
        assert _pvhb.OTS_ACCEPT is False
    finally:
        loop.close()
    html = bcu._PANEL_HTML
    assert 'id="otsAccept"' in html and "ots_accept: $('otsAccept').checked" in html


def test_recorder_seals_the_ots_flag(tmp_path):
    tr = _mod("test_ladder_recorder")
    p = tr._player()
    p._ots_sheets = {}
    rec, _ = tr._recorder(tmp_path, p)
    tag = f"battle-{tr.FMT}-41"
    rec.record(tr._battle(tag), tr._state(1), 1, 2, 0, 0, "model", "turn")
    t = rec.finish(tag, tr._battle(tag, won=True), won=True)
    assert t.meta.sampling["ots"] is False
    tag2 = f"battle-{tr.FMT}-42"
    p._ots_sheets[tag2] = {"p1": [{"species": "A"}], "p2": [{"species": "B"}]}
    rec.record(tr._battle(tag2 + "-priv"), tr._state(1), 1, 2, 0, 0, "model", "turn")
    t2 = rec.finish(tag2 + "-priv", tr._battle(tag2, won=True), won=True)
    assert t2.meta.sampling["ots"] is True


def test_mission_control_carries_the_launch_key_and_both_checkboxes():
    from v_dance.datatools import mission_control as mc
    assert "VD_OTS_ACCEPT" in mc._ENV_READ_KEYS and "VD_OTS_ACCEPT" in mc._ENV_WRITE_KEYS
    html = mc._HTML_PATH.read_text(encoding="utf-8")
    assert 'id="ob-launch-ots"' in html and 'id="ob-ots"' in html and "ots_accept: $(\"ob-ots\").checked" in html
    assert 'envRow("VD_OTS_ACCEPT", "bool")' in html
