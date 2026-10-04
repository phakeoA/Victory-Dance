"""panel_eval diagnostic knobs (2026-10-03, USER: teach the Rillaboom lesson to both the team picker and the battle net).

Found while wiring it: BOTH self-play collection paths build the LEARNER without a team chooser, so it brings the
first four of its sheet every game (Baltimore: Salamence / Sneasler / Indeedee / Tyranitar, leads Salamence +
Sneasler) while eval and the ladder use the team picker. These knobs let a re-test compare the candidate's team
preview policies — the picker, the first-4 heuristic (``--candidate-tp none``), the picker + a matchup rule — on an
opponent subset (``--opp-has rillaboom --opp-lacks archaludon``). The CANDIDATE side only; unset = byte-identical.
"""
from __future__ import annotations

import pytest

from v_dance.eval.panel_eval import filter_pool
from v_dance.selfplay import mp_eval as ME

_SHEETS = {
    "teams/M-C/Baltimore_Sand_Psy": ["Salamence", "Sneasler", "Indeedee-F", "Tyranitar", "Corviknight", "Excadrill"],
    "teams/M-C/Rilla_A": ["Rillaboom", "Incineroar", "Sinistcha"],
    "teams/M-C/Rilla_Archa": ["Rillaboom", "Archaludon", "Pelipper"],
    "teams/M-C/Rain": ["Pelipper", "Basculegion", "Archaludon"],
}


def test_filter_pool_keeps_matching_opponents_and_the_own_team():
    out = filter_pool(list(_SHEETS), "Baltimore_Sand_Psy", has=["rillaboom"], lacks=["archaludon"],
                      species_of=lambda t: _SHEETS[t])
    assert out == ["teams/M-C/Baltimore_Sand_Psy", "teams/M-C/Rilla_A"]
    assert filter_pool(list(_SHEETS), "Baltimore_Sand_Psy", species_of=lambda t: _SHEETS[t]) == list(_SHEETS)


def test_candidate_overrides_default_none_and_path(monkeypatch):
    monkeypatch.delenv("VD_EVAL_CANDIDATE_TP", raising=False)
    monkeypatch.delenv("VD_EVAL_CANDIDATE_RULES", raising=False)
    assert ME.candidate_overrides("tp.pt") == ("tp.pt", ())             # unset = the eval default, byte-identical
    monkeypatch.setenv("VD_EVAL_CANDIDATE_TP", "none")
    assert ME.candidate_overrides("tp.pt") == (None, ())                # the self-play learner's first-4 heuristic
    monkeypatch.setenv("VD_EVAL_CANDIDATE_TP", "other.pt")
    monkeypatch.setenv("VD_EVAL_CANDIDATE_RULES", "rillaboom_salamence, ")
    assert ME.candidate_overrides("tp.pt") == ("other.pt", ("rillaboom_salamence",))


def test_build_eval_players_overrides_only_the_candidate(monkeypatch):
    import v_dance.play.run_local_battle as R
    import v_dance.eval.gauntlet as G
    made = []

    class _P:
        pass

    def fake_make_player(name, team, *, model_path=None, team_chooser_path=None, **kw):
        p = _P()
        p.name, p.model_path, p.tc = name, model_path, team_chooser_path
        made.append(p)
        return p

    monkeypatch.setattr(R, "make_player", fake_make_player)
    monkeypatch.setattr(R, "load_team", lambda path: "TEAM")
    monkeypatch.setattr(R, "resolve_team_path", lambda t: t)
    monkeypatch.setattr(G, "_make_opponent", lambda *a, **k: pytest.fail("panel specs never build a scripted opp"))
    monkeypatch.setenv("VD_EVAL_CANDIDATE_TP", "none")
    monkeypatch.setenv("VD_EVAL_CANDIDATE_RULES", "rillaboom_salamence")
    spec = ME.EvalSpec(ME.PANEL_PREFIX + "era2", "Baltimore_Sand_Psy", "Rilla_A", 4, 1, 900, opp_ckpt="era2.pt")
    cand, opp = ME._build_eval_players_real("g50.pt", None, "tp.pt", spec)
    assert cand.tc is None and cand._matchup_rules == ("rillaboom_salamence",)
    assert opp.tc == "tp.pt" and not hasattr(opp, "_matchup_rules")   # the opponent is untouched
