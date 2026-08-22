"""Cross-folder team picker (this session): a native multi-select dialog only spans
ONE folder, so pick_team_files now LOOPS ("add more from another folder?") and dedups
across batches — letting you pick some teams from M-A and some from M-B in one session.
The GUI loop is thin tkinter glue; the testable seam is _normalize_team_selection."""
from __future__ import annotations

from pathlib import Path

import v_dance.play.run_local_battle as R


def test_normalize_accumulates_across_folders_and_dedups():
    repo = Path("/repo")
    # As if the user picked from M-A in one dialog batch, then M-B in the next, with one
    # team accidentally picked twice across the two folders.
    picked = [
        repo / "teams" / "Champions" / "M-A" / "team1",
        repo / "teams" / "Champions" / "M-A" / "WolfeGlick",
        repo / "teams" / "Champions" / "M-B" / "blaze_flo",
        repo / "teams" / "Champions" / "M-A" / "team1",     # duplicate -> dropped
    ]
    out = R._normalize_team_selection([str(p) for p in picked], repo)
    assert out == [
        "teams/Champions/M-A/team1",
        "teams/Champions/M-A/WolfeGlick",
        "teams/Champions/M-B/blaze_flo",
    ]  # deduped, order-preserved, spans BOTH folders, repo-relative POSIX


def test_normalize_keeps_outside_repo_absolute():
    repo = Path("/repo")
    outside = Path("/elsewhere/teams/weird")
    out = R._normalize_team_selection([str(repo / "teams" / "x"), str(outside)], repo)
    assert out[0] == "teams/x"
    assert out[1] == str(outside)        # outside the repo -> left absolute


def test_normalize_empty_is_empty():
    assert R._normalize_team_selection([], Path("/repo")) == []


# ── team-paste EV/IV separator normalization (Champions bucket-scale export quirk) ──
import pytest  # noqa: E402


def test_normalize_team_paste_adds_spaces_around_bare_slash():
    raw = "EVs: 32 HP/15 Def/19 Spe"
    assert R._normalize_team_paste(raw) == "EVs: 32 HP / 15 Def / 19 Spe"


def test_normalize_team_paste_handles_spa_and_is_idempotent():
    fixed = "EVs: 1 HP / 3 Atk / 30 SpA / 32 Spe"
    assert R._normalize_team_paste("EVs: 1 HP/3 Atk/30 SpA/32 Spe") == fixed
    assert R._normalize_team_paste(fixed) == fixed                # already-spaced unchanged


def test_normalize_team_paste_leaves_non_ev_lines_untouched():
    raw = ("Floette-Mega (F) @ Floettite\nAbility: Flower Veil\n"
           "EVs: 252 HP / 4 Def\n- Moonblast")
    assert R._normalize_team_paste(raw) == raw                    # correct EVs + other lines unchanged


def test_wyrdeer_flo_team_parses_via_load_team():
    # regression: this real M-B team used bare '/' EV separators -> poke-env KeyError 'hp/15'.
    from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
    p = Path(__file__).resolve().parents[1] / "teams" / "Champions" / "M-B" / "wyrdeer_flo"
    if not p.exists():
        pytest.skip("wyrdeer_flo team not present")
    ConstantTeambuilder(R.load_team(p))                           # must not raise
