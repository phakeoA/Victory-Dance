"""M5 (DS-M5, 2026-07-11): OTS |showteam| → TP opp_known serve wiring.

Contract: flag-gated (VD_TP_OTS_OVERLAY, default OFF = byte-identical serve);
sheets keyed by the room's BASE tag (private-suffix rooms must hit the same
entry); the opponent side is picked by our player_role; mons with nothing
revealed contribute no overlay; the builder never raises.
"""
from types import SimpleNamespace

import pytest

from v_dance.parser.vod_parser.team_sheet import parse_showteam_sides
from v_dance.play import player as player_mod
from v_dance.play.player import ots_opp_known, room_base_tag


def test_room_base_tag():
    assert room_base_tag("battle-gen9x-123") == "battle-gen9x-123"
    assert room_base_tag(">battle-gen9x-123-abc9pw") == "battle-gen9x-123"
    assert room_base_tag("lobby") == "lobby"


def _fake(role="p1", tag="battle-gen9x-123-suffixpw", sheets=None, cfg=None):
    # cfg defaults to a ckpt that CERTIFIES OTS-overlay training (ots_overlay_trained) so the
    # overlay path is exercised; pass cfg={} to model a closed-sheet net (guard suppresses).
    p = SimpleNamespace(_ots_sheets=sheets or {},
                        _tc_cfg={"ots_overlay_trained": True} if cfg is None else cfg)
    b = SimpleNamespace(player_role=role, battle_tag=tag)
    return p, b


_SIDES = {"p1": [{"species": "Torkoal", "ability": "Drought", "moves": ["Eruption"]}],
          "p2": [{"species": "Pelipper", "ability": "Drizzle", "moves": ["Hurricane", ""]},
                 {"species": "Ditto", "ability": None, "moves": []}]}   # nothing revealed


@pytest.fixture()
def _flag_on(monkeypatch):
    monkeypatch.setattr(player_mod, "TP_OTS_OVERLAY", True)


def test_flag_off_returns_none():
    p, b = _fake(sheets={"battle-gen9x-123": _SIDES})
    assert ots_opp_known(p, b) is None                     # default OFF = byte-identical


def test_opp_side_by_role_and_base_tag(_flag_on):
    p, b = _fake(role="p1", sheets={"battle-gen9x-123": _SIDES})
    known = ots_opp_known(p, b)                            # suffixed tag → base-tag hit
    assert set(known) == {"pelipper"}                      # Ditto revealed nothing → skipped
    ok = known["pelipper"]
    assert ok.ability == "Drizzle" and list(ok.moves) == ["Hurricane"]
    p2, b2 = _fake(role="p2", sheets={"battle-gen9x-123": _SIDES})
    assert set(ots_opp_known(p2, b2)) == {"torkoal"}       # role p2 → p1 is the opponent


def test_closed_or_uncaptured_returns_none(_flag_on):
    p, b = _fake(sheets={})
    assert ots_opp_known(p, b) is None
    p2 = SimpleNamespace(_tc_cfg={"ots_overlay_trained": True})  # store attr missing
    _, b2 = _fake()
    assert ots_opp_known(p2, b2) is None


def test_guard_suppresses_overlay_without_ckpt_marker(_flag_on):
    # flag ON + sheets present, but the ckpt config does NOT certify OTS-overlay training
    # → the overlay is OOD for a closed-sheet net (e.g. checkpoints_set), so serve without it.
    p, b = _fake(role="p1", sheets={"battle-gen9x-123": _SIDES}, cfg={})
    assert ots_opp_known(p, b) is None


def test_never_raises(_flag_on):
    p = SimpleNamespace(_ots_sheets={"battle-gen9x-123": {"p2": [{"species": 42}]}},
                        _tc_cfg={"ots_overlay_trained": True})
    _, b = _fake()
    assert ots_opp_known(p, b) is None                     # junk mon → warn + None, no raise


def test_parse_showteam_payload_roundtrip():
    payload = (">battle-gen9x-123\n"
               "|showteam|p1|Torkoal||heatrock|Drought|Eruption,Protect|||||50|]\n"
               "|j|someone")
    sides = parse_showteam_sides(payload)
    assert "p1" in sides and sides["p1"][0]["species"].lower().startswith("torkoal")
