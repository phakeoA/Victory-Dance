"""
Level C / A3b — opp_action_prior unit tests (2026-06-30).

``model_io.opp_action_prior`` = softmax(opp-head logits) masked by ``build_opp_action_mask`` — the
opponent model the Level-C search plans against. Also covers the ``_masked_softmax`` helper.
"""
from __future__ import annotations

import numpy as np
import pytest

from v_dance.play import model_io
from v_dance.play.model_io import _masked_softmax

torch = pytest.importorskip("torch")
A = 16


def test_masked_softmax_basic():
    logits = [float(i) for i in range(8)]
    mask = [1, 0, 1, 0, 1, 0, 0, 0]
    p = _masked_softmax(logits, mask)
    assert len(p) == 8
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[1] == 0 and p[3] == 0 and p[5] == 0           # illegal → zero
    assert p[0] > 0 and p[2] > 0 and p[4] > 0
    assert p[4] > p[2] > p[0]                              # higher logit → more mass (within legal)


def test_masked_softmax_no_legal():
    p = _masked_softmax([1.0, 2.0, 3.0], [0, 0, 0])
    assert p.sum() == 0.0 and len(p) == 3                  # no legal action → all zero


def test_masked_softmax_short_mask_padded():
    # a mask shorter than the logits treats the tail as illegal
    p = _masked_softmax([1.0, 2.0, 3.0, 4.0], [1, 1])
    assert p[2] == 0 and p[3] == 0 and abs(p.sum() - 1.0) < 1e-9


def _fake_model(actions: dict):
    def m(t):
        return actions, None, None     # (actions, gimmicks, value)
    return m


def test_opp_action_prior_masks_and_normalises(monkeypatch):
    actions = {
        "our_a": torch.zeros(A), "our_b": torch.zeros(A),
        "opp_a": torch.tensor([float(i) for i in range(A)]),       # increasing
        "opp_b": torch.tensor([float(A - i) for i in range(A)]),   # decreasing
    }
    monkeypatch.setattr(
        "v_dance.encoders.action_codec.build_opp_action_mask",
        lambda snap: {"opp_a": [1, 0, 1, 0] + [0] * 12, "opp_b": [0, 1, 0, 1] + [0] * 12})
    pri = model_io.opp_action_prior(_fake_model(actions), np.zeros(8, dtype=np.float32), {})
    assert pri is not None
    a, b = pri["opp_a"], pri["opp_b"]
    assert len(a) == A and abs(a.sum() - 1.0) < 1e-6 and abs(b.sum() - 1.0) < 1e-6
    assert a[1] == 0 and a[3] == 0 and a[0] > 0 and a[2] > 0     # masked to legal
    assert a[2] > a[0]                                           # higher opp_a logit → more mass
    assert b[0] == 0 and b[2] == 0 and b[1] > 0 and b[3] > 0
    assert b[1] > b[3]                                           # opp_b logits decrease → idx1 > idx3


def test_opp_action_prior_none_without_opp_heads():
    actions = {"our_a": torch.zeros(A), "our_b": torch.zeros(A)}   # our-only net
    assert model_io.opp_action_prior(_fake_model(actions), np.zeros(8, dtype=np.float32), {}) is None
