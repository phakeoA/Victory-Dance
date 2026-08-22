"""TP near-tie sampling (era-4 Phase 1a, 2026-07-20): VD_TP_TIE_EPS samples among
set-head subset/lead candidates within eps logits of the top; 0/unset = exact argmax
(original code path). The preview is one joint decision — no pair-coherence risk (N1)."""
import numpy as np
import pytest

from v_dance.play.model_io import _near_tie_sample


def test_eps_zero_is_argmax():
    scores = np.array([0.1, 3.0, 2.99, -1.0])
    idx, dev = _near_tie_sample(scores, 0.0)
    assert idx == 1 and dev is False


def test_no_candidates_within_eps_is_argmax():
    scores = np.array([0.0, 5.0, 1.0])
    idx, dev = _near_tie_sample(scores, 0.2)
    assert idx == 1 and dev is False


def test_near_ties_sampled_only_from_candidates():
    scores = np.array([3.0, 2.95, 2.7, -4.0])   # eps=0.1 → candidates {0, 1} only
    np.random.seed(0)
    picks = {_near_tie_sample(scores, 0.1)[0] for _ in range(200)}
    assert picks == {0, 1}


def test_deviation_flag_only_on_non_argmax_pick():
    scores = np.array([1.0, 0.99])
    np.random.seed(1)
    for _ in range(50):
        idx, dev = _near_tie_sample(scores, 0.5)
        assert dev == (idx != 0)


def test_env_off_keeps_set_head_argmax(monkeypatch):
    """eps unset/0 must leave _set_head_order on the literal original argmax path."""
    monkeypatch.delenv("VD_TP_TIE_EPS", raising=False)
    from v_dance.play.model_io import _tp_tie_eps
    assert _tp_tie_eps() == 0.0
    monkeypatch.setenv("VD_TP_TIE_EPS", "")
    assert _tp_tie_eps() == 0.0
    monkeypatch.setenv("VD_TP_TIE_EPS", "not-a-float")
    assert _tp_tie_eps() == 0.0
    monkeypatch.setenv("VD_TP_TIE_EPS", "0.2")
    assert _tp_tie_eps() == pytest.approx(0.2)
