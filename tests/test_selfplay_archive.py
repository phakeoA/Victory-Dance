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


# ── write_generation_artifacts (manifest only; Type_D replays removed → --save-replays) ──
def test_write_generation_artifacts_writes_manifest_only(tmp_path):
    out = AR.write_generation_artifacts(tmp_path, _history())
    assert Path(out["manifest"]).exists()                              # the dashboard data source
    assert set(out) == {"manifest"}                                    # no Type_D / replay html
    assert "elo_curve" not in out                                      # NO static graph
    assert not list(tmp_path.glob("*.html"))                           # nothing renders replay HTML here
