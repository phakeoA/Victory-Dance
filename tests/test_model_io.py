"""Unit tests for local_battle/model_io.py (#13): loading dict checkpoints and
mask-aware decoding of the two-head BC policy + the team-preview scorer.

The checkpoint-dependent tests skip cleanly when the trained .pt files (or torch)
are absent, so a fresh clone without trained models still passes the suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
import v_dance.play.model_io as M  # noqa: E402  (local_battle/model_io.py)

_BC_CKPT = _REPO / "ai_train_scripts" / "BC_model" / "checkpoints_attn" / "battle_selfplay_gen141.pt"
_TP_CKPT = _REPO / "ai_train_scripts" / "teamPreview_model" / "checkpoints" / "teampreview_sbda.pt"


# ── masked_argmax (pure, no torch / no checkpoint needed) ─────────────────────
def test_masked_argmax_picks_highest_legal_not_highest_overall():
    # logit 10 at index 0 is ILLEGAL; the best LEGAL is index 2 (8.0).
    assert M.masked_argmax([10.0, 5.0, 8.0], [False, True, True]) == 2


def test_masked_argmax_all_illegal_returns_none():
    assert M.masked_argmax([3.0, 1.0, 2.0], [False, False, False]) is None


def test_masked_argmax_single_legal():
    assert M.masked_argmax([0.0, 0.0, 0.0, 9.0], [False, False, False, True]) == 3


# ── masked_sample (serve-side temperature / top-p, pure, no checkpoint) ───────
def test_masked_sample_temp0_equals_argmax():
    logits = [10.0, 5.0, 8.0, 1.0]
    mask = [False, True, True, True]
    # T=0 must reproduce masked_argmax exactly (highest LEGAL, ties→first)
    assert M.masked_sample(logits, mask, temperature=0.0) == M.masked_argmax(logits, mask)
    assert M.masked_sample(logits, mask, temperature=0.0) == 2


def test_masked_sample_never_picks_illegal():
    import numpy as np
    rng = np.random.default_rng(0)
    logits = [100.0, 0.0, 0.0]          # the huge logit is ILLEGAL
    mask = [False, True, True]
    for _ in range(200):
        assert M.masked_sample(logits, mask, temperature=1.0, rng=rng) in (1, 2)


def test_masked_sample_all_illegal_returns_none():
    assert M.masked_sample([1.0, 2.0], [False, False], temperature=1.0) is None


def test_masked_sample_is_deterministic_with_seed():
    import numpy as np
    logits = [2.0, 1.5, 1.0, 0.5]
    mask = [True, True, True, True]
    a = [M.masked_sample(logits, mask, temperature=1.0, rng=np.random.default_rng(7))
         for _ in range(5)]
    b = [M.masked_sample(logits, mask, temperature=1.0, rng=np.random.default_rng(7))
         for _ in range(5)]
    assert a == b                        # same seed → same draws


def test_masked_sample_topp_restricts_to_nucleus():
    import numpy as np
    rng = np.random.default_rng(1)
    # index 0 dominates; a tight top_p must collapse to argmax-only
    logits = [10.0, 0.0, 0.0, 0.0]
    mask = [True, True, True, True]
    picks = {M.masked_sample(logits, mask, temperature=1.0, top_p=0.5, rng=rng)
             for _ in range(100)}
    assert picks == {0}                  # nucleus is just the top action


def test_masked_sample_low_temp_concentrates_on_top():
    import numpy as np
    rng = np.random.default_rng(2)
    logits = [1.0, 0.9, 0.8, 0.7]        # close logits
    mask = [True, True, True, True]
    # T=0.02 → top-action prob ~0.985 (exp(-5) tail); expected ~197/200
    cold = [M.masked_sample(logits, mask, temperature=0.02, rng=rng) for _ in range(200)]
    assert cold.count(0) > 185           # near-deterministic when cold


# ── W3b-1a (2026-09-02): the behaviour log-prob at the sampling site ─────────
def test_masked_sample_logp_is_the_same_draw_plus_its_log_probability():
    import math
    import numpy as np
    logits, mask = [0.1, 2.0, 1.0, -3.0], [True, True, True, False]
    picks = [M.masked_sample(logits, mask, temperature=1.0, rng=np.random.default_rng(3))
             for _ in range(20)]
    both = [M.masked_sample_logp(logits, mask, temperature=1.0, rng=np.random.default_rng(3))
            for _ in range(20)]
    assert [i for i, _ in both] == picks                        # same draw, same RNG consumption
    p = M._masked_softmax(logits, mask)                         # tau 1 → the stash's own softmax
    for i, lp in both:
        assert lp == pytest.approx(math.log(p[i]))
    # tempered: p_tau ∝ exp(l / tau) over the legal set
    i, lp = M.masked_sample_logp(logits, mask, temperature=0.5, rng=np.random.default_rng(1))
    z = np.array([l / 0.5 for l, ok in zip(logits, mask) if ok]); pt = np.exp(z - z.max()); pt /= pt.sum()
    assert lp == pytest.approx(math.log(pt[[k for k, ok in enumerate(mask) if ok].index(i)]))
    # argmax convention: probability 1 → log-prob 0; no legal action → (None, None)
    assert M.masked_sample_logp(logits, mask, temperature=0.0) == (1, 0.0)
    assert M.masked_sample_logp(logits, [False] * 4, temperature=1.0) == (None, None)


def test_masked_sample_logp_top_p_reports_the_truncated_distribution():
    import math
    import numpy as np
    logits, mask = [3.0, 2.5, -5.0, -5.0], [True] * 4
    i, lp = M.masked_sample_logp(logits, mask, temperature=1.0, top_p=0.9,
                                 rng=np.random.default_rng(0))
    assert i in (0, 1)                                          # the nucleus = the two big ones
    z = np.array(logits) - 3.0; p = np.exp(z); p /= p.sum()
    keep = p[:2] / p[:2].sum()                                  # renormalised over the nucleus
    assert lp == pytest.approx(math.log(keep[i]))               # NOT the untruncated softmax


class _FakeNet:
    """A two-head 'model' for the independent decode path (no pair_cond flag)."""
    memory_dim = 0

    def __init__(self, l0, l1):
        import torch
        self.l0, self.l1 = torch.tensor(l0), torch.tensor(l1)

    def __call__(self, t, **kw):
        return {"our_a": self.l0, "our_b": self.l1}


def test_bc_action_indices_stashes_the_per_slot_behaviour_logp():
    pytest.importorskip("torch")
    import math
    import numpy as np
    net = _FakeNet([0.0, 1.0, 2.0, 0.5], [2.0, 0.0, -1.0, 1.0])
    m0, m1 = [True, True, True, True], [True, False, True, True]
    x = np.zeros(4, np.float32)
    a0, a1 = M.bc_action_indices(net, ("our_a", "our_b"), x, m0, m1, temperature=1.0,
                                 rng=np.random.default_rng(5))
    rec = M.decode_record()
    assert rec["picks"] == (a0, a1) and rec["pair"] is False and rec["tau"] == 1.0
    lp0, lp1 = rec["logp"]
    assert lp0 == pytest.approx(math.log(M._masked_softmax(net.l0.numpy(), m0)[a0]))
    assert lp1 == pytest.approx(math.log(M._masked_softmax(net.l1.numpy(), m1)[a1]))
    assert rec["masks"] == (m0, m1)
    # argmax serve → 0.0 by convention; an action-less slot → None
    M.bc_action_indices(net, ("our_a", "our_b"), x, m0, [False] * 4)
    assert M.decode_record()["logp"] == (0.0, None) and M.decode_record()["picks"] == (2, None)


def test_merge_dedup_records_keeps_slot0_from_the_first_and_slot1_from_the_second():
    first = {"masks": ([1, 1], [1, 1]), "logp": (-0.5, -0.6), "picks": (7, 7), "tau": 0.3, "pair": True}
    second = {"masks": ([1, 1], [1, 0]), "logp": (-0.9, -0.8), "picks": (3, 0), "tau": 0.3, "pair": True}
    out = M.merge_dedup_records(first, second)
    assert out["masks"] == ([1, 1], [1, 0]) and out["logp"] == (-0.5, -0.8) and out["picks"] == (7, 0)
    assert out["dedup_resample"] is True and out["tau"] == 0.3
    assert M.merge_dedup_records({}, {})["logp"] == (None, None)   # robust to empty records


# ── BC policy load + decode (needs torch + trained checkpoint) ────────────────
@pytest.fixture(scope="module")
def bc_loaded():
    pytest.importorskip("torch")
    if not _BC_CKPT.exists():
        pytest.skip(f"BC checkpoint not found: {_BC_CKPT}")
    try:
        model, heads = M.load_bc_policy(_BC_CKPT)
    except ValueError as e:
        # The committed bc_best.pt may predate the current state layout (#5):
        # the stale-layout guard rejects it until the re-export + retrain lands.
        if "layout" in str(e) or "state_dim" in str(e):
            pytest.skip(f"bc_best.pt is a stale-layout checkpoint (retrain pending): {e}")
        raise
    return model, heads


def test_load_bc_policy_reconstructs_two_head_module(bc_loaded):
    import torch
    from v_dance.encoders.state_encoder import STATE_DIM, ACTION_DIM
    model, heads = bc_loaded
    assert heads == ("our_a", "our_b")
    assert model.state_dim == STATE_DIM
    actions, gimmicks, value = model(torch.zeros(STATE_DIM, dtype=torch.float32))
    assert set(actions) == {"our_a", "our_b"}
    assert set(gimmicks) == {"our_a", "our_b"}
    assert actions["our_a"].numel() == ACTION_DIM
    assert value.numel() == 1                        # scalar position value present


def test_value_logit_returns_probability_and_flag(bc_loaded):
    import numpy as np
    from v_dance.encoders.state_encoder import STATE_DIM
    model, _heads = bc_loaded
    v = M.value_logit(model, np.zeros(STATE_DIM, dtype=np.float32))
    assert v is not None and 0.0 <= v <= 1.0         # sigmoid → win prob in [0,1]
    # production bc_best.pt is now the value-trained checkpoint (#2 promoted) →
    # the serve player is allowed to read its win-prob.  (A pre-value checkpoint
    # would load non-strict with value_trained False; that path is unit-tested via
    # the back-compat dataset/model tests.)
    assert M.value_trained(model) is True


def test_bc_action_indices_returns_masked_argmax(bc_loaded):
    """The decoded action per head equals a manual masked-argmax over that head's
    raw logits — i.e. always the highest-logit LEGAL action."""
    import numpy as np, torch
    from v_dance.encoders.state_encoder import STATE_DIM
    model, heads = bc_loaded
    sv = np.random.RandomState(1).randn(STATE_DIM).astype("float32")
    # legal subsets that exclude the (possibly) top logit, to make masking bite.
    mask0 = [i in (1, 4, 13) for i in range(16)]
    mask1 = [i in (0, 2, 3, 12, 14) for i in range(16)]
    with torch.no_grad():
        actions, _gimmicks, _value = model(torch.as_tensor(sv))
    exp0 = M.masked_argmax(actions["our_a"].numpy().ravel(), mask0)
    exp1 = M.masked_argmax(actions["our_b"].numpy().ravel(), mask1)
    a0, a1 = M.bc_action_indices(model, heads, sv, mask0, mask1)
    assert (a0, a1) == (exp0, exp1)
    # and the chosen indices are legal
    assert mask0[a0] and mask1[a1]


def test_bc_action_indices_none_when_no_legal(bc_loaded):
    import numpy as np
    from v_dance.encoders.state_encoder import STATE_DIM
    model, heads = bc_loaded
    sv = np.zeros(STATE_DIM, dtype="float32")
    a0, a1 = M.bc_action_indices(model, heads, sv, [False] * 16, [True] + [False] * 15)
    assert a0 is None and a1 == 0


# ── Gimmick head load + decode (Task #5d) ─────────────────────────────────────
def _save_bc_ckpt(tmp_path, with_gimmick: bool, gimmick_trained=None, name=None):
    import torch
    from conftest import tiny_attn_policy
    from v_dance.encoders.state_encoder import STATE_DIM, get_state_layout_version, get_gimmick_dim
    m = tiny_attn_policy()
    sd = m.state_dict()
    if not with_gimmick:
        # Pre-gimmick checkpoint: strip the gimmick-head weights so they load non-strict (at init);
        # the config still BUILDS the heads (gimmick_heads set), exactly the legacy scenario.
        sd = {k: v for k, v in sd.items() if not k.startswith("gimmick_heads.")}
    cfg = {"model_type": "attn", "state_dim": STATE_DIM, "action_dim": 16,
           "d_model": 32, "n_heads": 4, "n_layers": 1, "ff_mult": 2, "dropout": 0.0,
           "value_readout": "mean", "heads": ["our_a", "our_b"],
           "gimmick_heads": ["our_a", "our_b"],
           "state_layout_version": get_state_layout_version()}
    if with_gimmick:
        cfg["gimmick_dim"] = get_gimmick_dim()      # v11 Phase D: 3 (matches the saved head width)
    if gimmick_trained is not None:
        cfg["gimmick_trained"] = gimmick_trained
    p = tmp_path / (name or ("g.pt" if with_gimmick else "old.pt"))
    torch.save({"model_state": sd, "config": cfg}, p)
    return p


def test_gimmick_checkpoint_sets_trained_flag_and_logits(tmp_path):
    pytest.importorskip("torch")
    import numpy as np
    from v_dance.encoders.state_encoder import STATE_DIM
    model, heads = M.load_bc_policy(_save_bc_ckpt(tmp_path, with_gimmick=True))
    assert M.gimmick_trained(model) is True
    g0, g1 = M.gimmick_logits(model, heads, np.zeros(STATE_DIM, "float32"))
    assert g0.shape == (3,) and g1.shape == (3,)            # v11 Phase D: 3-wide {none, mega, tera}
    assert M.masked_argmax(g0, [True, False, False]) == 0   # only 'none' legal
    assert M.masked_argmax(g0, [True, True, True]) in (0, 1, 2)


def test_gimmick_head_present_but_config_untrained_is_flagged_untrained(tmp_path):
    """A checkpoint with gimmick weights but trained on pre-gimmick data (config
    gimmick_trained=False) must be flagged untrained — the head is at init."""
    pytest.importorskip("torch")
    model, _ = M.load_bc_policy(
        _save_bc_ckpt(tmp_path, with_gimmick=True, gimmick_trained=False, name="untrained.pt"))
    assert M.gimmick_trained(model) is False


def test_pre_gimmick_checkpoint_loads_nonstrict_and_flags_untrained(tmp_path):
    """An OLD checkpoint (no gimmick_heads) loads with the gimmick head at init,
    flagged untrained, and its action head still decodes — so the live bot never
    acts on an untrained gimmick head."""
    pytest.importorskip("torch")
    import numpy as np
    from v_dance.encoders.state_encoder import STATE_DIM
    model, heads = M.load_bc_policy(_save_bc_ckpt(tmp_path, with_gimmick=False))
    assert M.gimmick_trained(model) is False
    a0, a1 = M.bc_action_indices(
        model, heads, np.zeros(STATE_DIM, "float32"),
        [True] + [False] * 15, [True] + [False] * 15,
    )
    assert a0 == 0 and a1 == 0


# ── Team-chooser load + order (needs torch + trained checkpoint) ──────────────
@pytest.fixture(scope="module")
def tp_loaded():
    pytest.importorskip("torch")
    if not _TP_CKPT.exists():
        pytest.skip(f"team-chooser checkpoint not found: {_TP_CKPT}")
    return M.load_team_chooser(_TP_CKPT)


def test_load_team_chooser_has_vocab_and_config(tp_loaded):
    model, vocab, cfg = tp_loaded
    assert isinstance(vocab, dict) and len(vocab) > 50
    assert cfg["feat_dim"] > 0 and cfg["vocab_size"] >= len(vocab)


def test_team_order_returns_n_distinct_in_range(tp_loaded):
    model, vocab, cfg = tp_loaded
    our = ["charizard", "incineroar", "rillaboom", "amoonguss", "urshifu", "fluttermane"]
    opp = ["miraidon", "ironhands", "chiyu", "landorustherian", "ogerpon", "ironbundle"]
    order = M.team_order(model, vocab, cfg, our, opp, n=4)
    assert len(order) == 4
    assert len(set(order)) == 4
    assert all(0 <= i < 6 for i in order)


def test_team_order_leads_are_within_brought(tp_loaded):
    """The first lead_k entries (the leads) must be a subset of the full bring,
    and the whole order must be distinct roster positions."""
    model, vocab, cfg = tp_loaded
    our = ["gholdengo", "ragingbolt", "ironhands", "fluttermane", "ogerpon", "rillaboom"]
    opp = ["miraidon", "ironbundle", "chiyu", "landorustherian", "amoonguss", "urshifu"]
    order = M.team_order(model, vocab, cfg, our, opp, n=4)
    lead_k = int(cfg.get("lead_k", 2))
    leads = set(order[:lead_k])
    assert leads <= set(order)        # leads are part of the bring
    assert len(set(order)) == len(order)
