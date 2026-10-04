"""``tp_ckpt: "none"`` — a bandit arm that plays WITHOUT a team picker (2026-10-03).

The self-play learner has never had a team picker (both collection paths build it without one), so every self-play
checkpoint trained on the first-4 roster heuristic. Offline vs era2, g50 on Baltimore won 77.5 % with first-4 vs
62.7 % with the served picker — this lets the ladder A/B it. ``"none"`` must load no picker (the player then falls to
``vgc_base._heuristic_team_order``), must not be mistaken for a missing file, and must not touch other arms.
"""
from __future__ import annotations

import json
from pathlib import Path

from v_dance.play import serve_bandit as SB
from v_dance.play.vgc_base import _heuristic_team_order


class _P:
    pass


class _Host:
    def __init__(self):
        self.player = _P()


def test_none_arm_is_kept_without_a_file_check(tmp_path: Path):
    cfg = {"arms": [
        {"name": "g50", "battle_ckpt": "ai_train_scripts/g50/battle_base.pt", "tp_ckpt": "default"},
        {"name": "g50_first4", "battle_ckpt": "ai_train_scripts/g50/battle_base.pt", "tp_ckpt": "none"},
        {"name": "typo", "battle_ckpt": "ai_train_scripts/g50/battle_base.pt", "tp_ckpt": "missing_tp.pt"},
    ]}
    p = tmp_path / "bandit.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    checked = []

    def exists(q):
        checked.append(str(q))
        return "missing" not in str(q)

    arms = SB.load_arms(p, exists=exists)
    assert [a.name for a in arms] == ["g50", "g50_first4"]           # a real missing TP path is still dropped
    assert arms[1].uses_no_tp() and not arms[1].uses_default("tp") and not arms[0].uses_no_tp()
    assert not any(c.endswith("none") for c in checked)               # "none" is never looked up as a file


def test_none_arm_loads_no_picker_and_the_player_falls_to_first4():
    loads = []

    def loader(kind, path):
        loads.append(kind)
        return (("M", path), ("H",)) if kind == "battle" else (("TC", path), {"v": 1}, {"cfg": 1})

    host, cache = _Host(), {}
    a = SB.Arm(name="g50_first4", battle_ckpt="ai_train_scripts/g50/battle_base.pt", tp_ckpt="none")
    SB.apply_arm(host, a, cache, default_battle="D_B.pt", default_tp="D_TP.pt", loader=loader)
    p = host.player
    assert p._team_chooser is None and p._tc_vocab is None and p._tc_cfg is None
    assert loads == ["battle"]                                        # the TP loader is never called
    b = SB.Arm(name="g50", battle_ckpt="ai_train_scripts/g50/battle_base.pt", tp_ckpt="default")
    SB.apply_arm(host, b, cache, default_battle="D_B.pt", default_tp="D_TP.pt", loader=loader)
    assert p._team_chooser == ("TC", Path("D_TP.pt"))                  # a normal arm still gets its picker
    mons = [type("Mon", (), {"species": s})() for s in ("salamence", "sneasler", "indeedee", "tyranitar",
                                                         "corviknight", "excadrill")]
    battle = type("B", (), {"teampreview_team": mons})()
    assert _heuristic_team_order(battle)[:4] == [0, 1, 2, 3]          # what a picker-less player brings


def test_an_arm_still_listing_matchup_rules_loads_and_says_they_are_ignored(tmp_path: Path, capsys):
    # the hand-written matchup rules were removed 2026-10-04: an old config arm keeps loading, the key is ignored
    p = tmp_path / "bandit.json"
    p.write_text(json.dumps({"arms": [{"name": "x", "battle_ckpt": "default", "tp_ckpt": "default",
                                       "matchup_rules": ["rillaboom_salamence"]}]}), encoding="utf-8")
    arms = SB.load_arms(p, exists=lambda q: True)
    assert [a.name for a in arms] == ["x"] and not hasattr(arms[0], "matchup_rules")
    assert "IGNORED" in capsys.readouterr().out
