"""Format registry/resolver (v_dance/formats.py) — backwards-compat across
Champions-doubles regs. The bot must target M-B by default yet still select M-A
(and any future reg), with a spawn-safe env override and a Pikalytics fallback so
a not-yet-scraped reg never zeroes the belief prior.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from v_dance import formats


def test_default_is_active_regmb_and_registry_lists_both():
    assert formats.default_format() == "gen9championsvgc2026regmb"
    known = formats.known_formats()
    assert "gen9championsvgc2026regmb" in known
    assert "gen9championsvgc2026regma" in known  # M-A retained for backwards-compat


def test_env_override_is_spawn_safe(monkeypatch):
    """A worker inherits VDANCE_BATTLE_FORMAT and must resolve to it (env > registry)."""
    monkeypatch.setenv(formats.ENV_FORMAT_KEY, "gen9championsvgc2026regma")
    assert formats.default_format() == "gen9championsvgc2026regma"
    monkeypatch.delenv(formats.ENV_FORMAT_KEY, raising=False)
    assert formats.default_format() == "gen9championsvgc2026regmb"


def test_set_active_format_sets_env_and_snapshot(monkeypatch):
    monkeypatch.delenv(formats.ENV_FORMAT_KEY, raising=False)
    prev = formats.DEFAULT_FORMAT
    try:
        formats.set_active_format("gen9championsvgc2026regma")
        import os
        assert os.environ[formats.ENV_FORMAT_KEY] == "gen9championsvgc2026regma"
        assert formats.DEFAULT_FORMAT == "gen9championsvgc2026regma"
    finally:
        monkeypatch.delenv(formats.ENV_FORMAT_KEY, raising=False)
        formats.DEFAULT_FORMAT = prev


def test_reg_token():
    assert formats.reg_token("gen9championsvgc2026regmb") == "regmb"
    assert formats.reg_token("gen9championsvgc2026regma") == "regma"
    assert formats.reg_token("gen9championsvgc2026regmc") == "regmc"  # future reg
    assert formats.reg_token("gen9ou") is None
    # Bo3 / Blitz variant ids share the reg (audit fix — anchored regex mis-parsed these)
    assert formats.reg_token("gen9championsvgc2026regmb-bo3") == "regmb"
    assert formats.reg_token("gen9championsvgc2026regmbbo3") == "regmb"


def test_pikalytics_filename():
    assert formats.pikalytics_filename("gen9championsvgc2026regmb") == "pikalytics_regmb.json"
    assert formats.pikalytics_filename("gen9championsvgc2026regma") == "pikalytics_regma.json"
    assert formats.pikalytics_filename("gen9championsvgc2026regmb-bo3") == "pikalytics_regmb.json"


def test_is_champions_doubles():
    assert formats.is_champions_doubles("gen9championsvgc2026regmb")
    assert formats.is_champions_doubles("gen9championsvgc2026regma")
    assert formats.is_champions_doubles("gen9championsvgc2027regxa")  # future
    assert not formats.is_champions_doubles("gen9vgc2026regg")        # plain VGC, not Champions
    assert not formats.is_champions_doubles("gen9ou")
    assert not formats.is_champions_doubles(None)


def test_pikalytics_regmb_resolves_to_its_own_scraped_file():
    """M-B's pikalytics_regmb.json has now been scraped (17.3) -> the resolver serves
    it DIRECTLY (no longer the M-A fallback). M-A still resolves to its own file. The
    absent-reg -> M-A fallback path is covered (mocked) by the test below."""
    p = formats.pikalytics_path_for("gen9championsvgc2026regmb")
    assert p is not None and p.name == "pikalytics_regmb.json"
    pa = formats.pikalytics_path_for("gen9championsvgc2026regma")
    assert pa is not None and pa.name == "pikalytics_regma.json"


def test_pikalytics_prefers_format_specific_when_present(monkeypatch, tmp_path):
    """Once pikalytics_<reg>.json exists it is preferred over the fallback."""
    (tmp_path / "pikalytics_regma.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pikalytics_regmb.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(formats, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(formats, "_FALLBACK_PIKALYTICS", tmp_path / "pikalytics_regma.json")
    assert formats.pikalytics_path_for("gen9championsvgc2026regmb").name == "pikalytics_regmb.json"
    assert formats.pikalytics_path_for("gen9championsvgc2026regma").name == "pikalytics_regma.json"
    # an unknown reg with no file falls back to M-A
    assert formats.pikalytics_path_for("gen9championsvgc2099regzz").name == "pikalytics_regma.json"


def test_belief_default_path_resolves_via_formats():
    """belief_state's default path must come THROUGH the resolver (not hardcoded), so it
    tracks the ACTIVE format's scraped file (now regmb) rather than a baked-in constant.
    The equality is the durable assertion; the filename documents the current active reg."""
    from v_dance.parser import belief_state
    assert belief_state._DEFAULT_PIKALYTICS_PATH == formats.pikalytics_path_for(formats.DEFAULT_FORMAT)
    assert belief_state._DEFAULT_PIKALYTICS_PATH.name == "pikalytics_regmb.json"
    assert belief_state._DEFAULT_PIKALYTICS_PATH.exists()


def test_run_local_battle_format_follows_registry():
    import v_dance.play.run_local_battle as R
    assert R.BATTLE_FORMAT == formats.DEFAULT_FORMAT == "gen9championsvgc2026regmb"


def test_belief_resolves_active_format_fresh(monkeypatch, tmp_path):
    """Import-order fix: the belief path follows the ACTIVE format resolved fresh,
    not a value frozen at belief_state import (the consumers now call
    pikalytics_path_for(default_format()) at use-time)."""
    (tmp_path / "pikalytics_regma.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pikalytics_regmb.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(formats, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(formats, "_FALLBACK_PIKALYTICS", tmp_path / "pikalytics_regma.json")
    monkeypatch.setenv(formats.ENV_FORMAT_KEY, "gen9championsvgc2026regmb")
    assert formats.pikalytics_path_for(formats.default_format()).name == "pikalytics_regmb.json"
    monkeypatch.setenv(formats.ENV_FORMAT_KEY, "gen9championsvgc2026regma")
    assert formats.pikalytics_path_for(formats.default_format()).name == "pikalytics_regma.json"


def test_discover_teams_reg_subdir_filter(tmp_path):
    """A backwards-compat run can restrict the default pool to its reg's subfolder."""
    import v_dance.play.run_local_battle as R
    (tmp_path / "M-A").mkdir()
    (tmp_path / "M-B").mkdir()
    assert R._reg_team_subdir(tmp_path, "gen9championsvgc2026regma").name == "M-A"
    assert R._reg_team_subdir(tmp_path, "gen9championsvgc2026regmb").name == "M-B"
    assert R._reg_team_subdir(tmp_path, "gen9ou") is None
