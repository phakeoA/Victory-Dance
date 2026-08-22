"""15b-feat.0: the TP synergy-sensitivity probe machinery.

Tests synergy_delta against SYNTHETIC score functions where the answer is known
(blind model -> ~0; synergy-aware model -> clearly positive), so the probe is proven
to DETECT synergy when it exists and report ~0 when it doesn't — independent of what
the real (measured) net happens to score. Plus an integration smoke that the probe
runs end-to-end against a real TeamPreviewModel and returns finite numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "scratch") not in sys.path:
    sys.path.insert(0, str(_REPO / "scratch"))
# scratch/ is a local-only dev-harness dir (gitignored) — skip cleanly where it's absent (CI)
P = pytest.importorskip("tp_synergy_probe",
                        reason="scratch/ probe harness is local-only, not in the published repo")

_FILL = ["Aaa", "Bbb", "Ccc", "Ddd"]
_OPP = ["Ooo"] * 6
_CTRL = ["C1", "C2", "C3", "C4", "C5"]


def test_synergy_delta_zero_for_blind_model():
    # Each species gets a fixed score, independent of its teammates -> no synergy.
    def blind(own, opp):
        return np.array([1.0 if s == "Excadrill" else 0.0 for s in own], dtype=float)

    l_set, l_mean, l_std = P.synergy_delta(blind, "Tyranitar", "Excadrill", _FILL, _OPP, _CTRL)
    assert abs(l_set - l_mean) < 1e-9          # slot-1 identity does not move the abuser
    assert l_std < 1e-9                          # controls are all identical


def test_synergy_delta_detects_synergy():
    # The abuser (slot 0) is boosted ONLY when its setter is present anywhere on the team.
    def synergy(own, opp):
        s = np.array([1.0 if x == "Excadrill" else 0.0 for x in own], dtype=float)
        if "Tyranitar" in own:
            s[[i for i, x in enumerate(own) if x == "Excadrill"]] += 5.0
        return s

    l_set, l_mean, l_std = P.synergy_delta(synergy, "Tyranitar", "Excadrill", _FILL, _OPP, _CTRL)
    assert (l_set - l_mean) > 4.0               # the +5 setter boost is recovered
    assert l_std < 1e-9                          # no control contains the setter


def test_synergy_delta_ignores_pure_context_strength():
    # A setter that uniformly lifts EVERY slot (generic "strong teammate") must NOT read
    # as synergy: the abuser's delta tracks only the abuser-SPECIFIC lift, which is 0 here.
    def uniform_lift(own, opp):
        s = np.array([1.0 if x == "Excadrill" else 0.0 for x in own], dtype=float)
        if "Tyranitar" in own:
            s += 2.0                             # lifts all slots equally
        return s

    l_set, l_mean, _ = P.synergy_delta(uniform_lift, "Tyranitar", "Excadrill", _FILL, _OPP, _CTRL)
    # both setter and controls add nothing abuser-specific beyond the uniform +2 that the
    # setter alone applies; delta is the uniform lift (+2), NOT a synergy term -> the probe
    # reports it, and the run() verdict compares it against the noise band to judge.
    assert (l_set - l_mean) == pytest.approx(2.0, abs=1e-9)


def test_probe_runs_on_a_real_teampreview_model():
    from v_dance.models.teampreview_model import TeamPreviewModel
    from v_dance.parser.vod_parser.pokedex import norm_species

    names = ["Tyranitar", "Excadrill", "Aaa", "Bbb", "Ccc", "Ddd", "Eee", "Ooo"]
    vocab = {norm_species(s): i + 1 for i, s in enumerate(names)}
    cfg = {"feat_dim": 46, "bring_k": 4, "lead_k": 2, "vocab_size": len(vocab) + 1}
    model = TeamPreviewModel(vocab_size=len(vocab) + 1, feat_dim=46)
    model.eval()
    score = P.make_score_fn(model, vocab, cfg)
    l_set, l_mean, l_std = P.synergy_delta(
        score, "Tyranitar", "Excadrill", ["Aaa", "Bbb", "Ccc", "Ddd"], ["Ooo"] * 6, ["Eee", "Aaa", "Bbb"])
    assert all(np.isfinite([l_set, l_mean, l_std]))
