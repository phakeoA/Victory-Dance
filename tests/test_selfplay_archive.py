"""Task 3c.5: generation archive (manifest data) + Type_D Showdown replay HTML.

The Type_D HTML is the standard Showdown replay format (raw |-log in a
<script class="battle-log-data"> + replay-embed.js) — the same format the VOD parser
ingests (Type_B IS this) — NOT a custom viewer, and NO static graphs (the dashboard,
3c.6, renders metrics from the manifest). Pure — no poke-env / torch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
from v_dance.selfplay.generation import GenerationHistory, GenerationRecord  # noqa: E402
from v_dance.selfplay.league import OpponentLeague  # noqa: E402
from v_dance.selfplay import archive as AR  # noqa: E402


def _history():
    h = GenerationHistory()
    h.add(GenerationRecord(0, 200, 48, 100, 1040.0, "promote", True))
    h.add(GenerationRecord(1, 200, 56, 100, 1080.0, "promote", True))
    h.add(GenerationRecord(2, 200, 55, 100, None, "hold", False))      # eval Elo missing
    h.best_path, h.best_scripted = "gen1.pt", (56, 100)
    return h


_LOG = [
    "|gametype|doubles", "|player|p1|Alice|101|1500", "|player|p2|Bob|rosa|1500",
    "|gen|9", "|tier|[Gen 9 Champions] VGC 2026 Reg M-A", "|teamsize|p1|4", "|start",
    "|switch|p1a: Charizard|Charizard, L50|100/100", "|turn|1",
    "|move|p1a: Charizard|Flamethrower|p2a: Venusaur", "|faint|p2a: Venusaur", "|win|Alice",
]


# ── manifest (data store for the dashboard) ───────────────────────────────────
def test_manifest(tmp_path):
    h = _history()
    lg = OpponentLeague(latest_path="x"); lg.admit("gen0", "gen0.pt", 0); lg.admit("gen1", "gen1.pt", 1)
    m = AR.build_manifest(h, lg)
    assert m["n_generations"] == 3 and m["best_path"] == "gen1.pt"
    assert m["league"] == ["gen0", "gen1"]
    assert m["generations"][0]["scripted_win_rate"] == pytest.approx(0.48)
    assert m["generations"][2]["model_elo"] is None and not m["generations"][2]["promoted"]
    p = AR.write_manifest(tmp_path, h, lg)
    assert json.loads(p.read_text(encoding="utf-8"))["n_generations"] == 3


def test_manifest_enriched_health_deltas_and_summary():
    h = GenerationHistory()
    h.add(GenerationRecord(0, 200, 48, 100, 1000.0, "promote", True,
                           update_stats={"loss": 0.5, "kl_to_bc": 0.002,
                                         "explained_variance": 0.8, "clip_fraction": 0.02,
                                         "halted": False}))
    h.add(GenerationRecord(1, 200, 56, 100, 1080.0, "promote", True,
                           update_stats={"loss": 0.4, "kl_to_bc": 0.003,
                                         "explained_variance": 0.85, "clip_fraction": 0.03,
                                         "halted": False}))
    h.add(GenerationRecord(2, 200, 52, 100, 1050.0, "hold", False,
                           update_stats={"loss": 0.45, "kl_to_bc": 0.004,
                                         "explained_variance": 0.7, "clip_fraction": 0.05,
                                         "halted": True}))
    h.best_path, h.best_scripted = "gen1.pt", (56, 100)
    m = AR.build_manifest(h)
    g = m["generations"]
    # PPO health passthrough (halted bool collapses to 0/1 for charting)
    assert g[0]["update_stats"]["loss"] == pytest.approx(0.5)
    assert g[2]["update_stats"]["halted"] == 1.0 and g[0]["update_stats"]["halted"] == 0.0
    # improvement deltas
    assert g[0]["elo_delta"] is None and g[0]["win_rate_delta"] is None        # first gen, no prior
    assert g[1]["elo_delta"] == pytest.approx(80.0)
    assert g[1]["win_rate_delta"] == pytest.approx(0.08)
    assert g[2]["elo_delta"] == pytest.approx(-30.0)                           # a regression gen
    # is_best = highest win-rate so far
    assert g[0]["is_best"] and g[1]["is_best"] and not g[2]["is_best"]
    assert g[0]["n_trajectories"] == 200
    # top-level summary
    assert m["best_generation"] == 1 and m["best_win_rate"] == pytest.approx(0.56)
    assert m["best_elo"] == pytest.approx(1080.0) and m["n_promotions"] == 2


def test_manifest_stars_the_champion_not_argmax_scripted():
    """The dashboard star must follow the gate's CHAMPION (latest promoted), NOT the argmax
    scripted-win-rate gen — those diverge once scripted saturates (red-team observability fix)."""
    h = GenerationHistory()
    h.add(GenerationRecord(0, 0, 50, 100, 1000.0, "promote", True, champion_elo=1000.0))
    h.add(GenerationRecord(1, 0, 65, 100, 1100.0, "hold", False, champion_elo=1000.0))   # max scripted wr
    h.add(GenerationRecord(2, 0, 60, 100, 1080.0, "promote", True, champion_elo=1164.0))  # the champion
    h.best_path, h.best_scripted, h.champion_elo = "gen2.pt", (60, 100), 1164.0
    m = AR.build_manifest(h)
    assert m["champion_generation"] == 2 and m["champion_path"] == "gen2.pt"
    assert m["champion_elo"] == pytest.approx(1164.0)
    assert m["best_generation"] == 2                       # star = champion (gen2), not gen1
    assert m["best_scripted_generation"] == 1              # argmax-scripted tracked separately
    assert m["generations"][2]["is_champion"] and not m["generations"][1]["is_champion"]
    assert m["generations"][2]["champion_elo"] == pytest.approx(1164.0)


# ── Type_D = standard Showdown replay format ──────────────────────────────────
def test_render_replay_is_showdown_format():
    h = AR.render_replay_html(_LOG)
    assert h.startswith("<!DOCTYPE html>")
    assert '<script type="text/plain" class="battle-log-data">' in h    # the canonical tag
    assert "replay-embed.js" in h                                       # the renderer
    assert "|move|p1a: Charizard|Flamethrower|p2a: Venusaur" in h       # raw log embedded
    assert "[Gen 9 Champions] VGC 2026 Reg M-A" in h                    # format from |tier|
    assert "Alice vs. Bob" in h                                         # players from |player|


def test_replay_html_neutralises_script_close():
    h = AR.render_replay_html(["|c|x|oops </script> hi"])
    assert "<\\/script>" in h and "oops <\\/script> hi" in h            # log's </ escaped


def test_parse_replay_meta():
    meta = AR._parse_replay_meta(_LOG)
    assert meta["p1"] == "Alice" and meta["p2"] == "Bob"
    assert meta["format"] == "[Gen 9 Champions] VGC 2026 Reg M-A"


def test_write_type_d(tmp_path):
    p = AR.write_type_d(tmp_path, 3, "show/case:odd", _LOG)
    assert p.name == "gen_3_show_case_odd.html"                         # tag sanitised
    txt = p.read_text(encoding="utf-8")
    assert "battle-log-data" in txt and "|win|Alice" in txt            # parser-ingestible


def test_write_generation_artifacts(tmp_path):
    out = AR.write_generation_artifacts(tmp_path, _history(),
                                        type_d_dir=tmp_path / "Type_D",
                                        showcase_log=_LOG, tag="g2")
    assert Path(out["manifest"]).exists() and Path(out["type_d_html"]).exists()
    assert "gen_2_g2.html" in out["type_d_html"]                        # tagged with latest gen
    assert "elo_curve" not in out                                       # NO static graph


def test_write_generation_artifacts_no_showcase(tmp_path):
    out = AR.write_generation_artifacts(tmp_path, _history())           # no log -> no Type_D
    assert "manifest" in out and "type_d_html" not in out


def test_type_d_default_dir_is_data_vods():
    assert AR.TYPE_D_DIR.parent.name == "vods" and AR.TYPE_D_DIR.name == "Type_D"
