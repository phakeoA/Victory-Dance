"""Era-4 2b — autoregressive pair decode on AttnBCPolicy (design §2b).

Stage-1 exit criteria (offline, no retrain):
  1. pair_cond=False (default) is UNCHANGED: identical key-set + shapes.
  2. Warm-start surgery (init_pair_model_from_ckpt): the widened model fed an
     unconditioned checkpoint reproduces its logits to float tolerance (NOT
     bitwise: widening the head matmul with zero columns is an exact-arithmetic
     no-op that still reorders the reduction, and float32 is not associative,
     so some BLAS backends -- CI Linux -- differ by ~1 ULP) when the pair
     cond is zeros / absent — the kill switch.
  3. A non-zero partner one-hot CHANGES the OUR-head logits once the pair
     columns are non-zero, and our_a/our_b condition INDEPENDENTLY.
  4. Aux opp heads / gimmicks / value are untouched by the pair input.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch")

from v_dance.encoders.state_encoder import get_action_dim, get_state_dim
from v_dance.models.bc_model_attn import (
    AttnBCPolicy,
    init_pair_model_from_ckpt,
)

_TINY = dict(d_model=32, n_heads=4, n_layers=1, dropout=0.0,
             heads=("our_a", "our_b", "opp_a", "opp_b"),
             gimmick_heads=("our_a", "our_b"))
A = get_action_dim()


def _tiny(**over):
    kw = dict(_TINY)
    kw.update(over)
    return AttnBCPolicy(**kw)


def _rand_x(*shape):
    g = torch.Generator().manual_seed(7)
    return torch.rand(*shape, get_state_dim(), generator=g)


def _onehot(i, n=A):
    v = torch.zeros(n)
    v[i] = 1.0
    return v


def test_default_arch_unchanged():
    base, pair_off = _tiny(), _tiny(pair_cond=False)
    sd_a, sd_b = base.state_dict(), pair_off.state_dict()
    assert set(sd_a) == set(sd_b)
    assert all(sd_a[k].shape == sd_b[k].shape for k in sd_a)


def test_pair_heads_grow_by_action_dim_only():
    base, pair = _tiny(), _tiny(pair_cond=True)
    sb, sp = base.state_dict(), pair.state_dict()
    assert set(sb) == set(sp)
    for k in sb:
        if k in ("heads.our_a.weight", "heads.our_b.weight"):
            assert sp[k].shape[1] == sb[k].shape[1] + A
        else:
            assert sp[k].shape == sb[k].shape, k


def test_zero_cond_bit_exact_after_surgery():
    torch.manual_seed(0)
    donor = _tiny().eval()
    pair = init_pair_model_from_ckpt(_tiny(pair_cond=True), donor.state_dict()).eval()
    x = _rand_x(3)
    with torch.no_grad():
        a0, g0, v0 = donor(x)
        # both "no cond" spellings must reproduce the donor
        a1, g1, v1 = pair(x)
        a2, g2, v2 = pair(x, partner_actions={"our_a": torch.zeros(3, A),
                                              "our_b": torch.zeros(3, A)})

    # Float tolerance, NOT torch.equal: the zero-column widening is an exact-
    # arithmetic no-op that still reorders the matmul reduction, so backends
    # disagree by ~1e-7 (bitwise on this box, not on every CI runner). atol=1e-6
    # matches what test_phase2_adv_archetypes already uses for the value readout,
    # and it stays sharp -- mutation-checked 2026-09-01: a misaligned transplant
    # reads 1.8e+00, one donor weight off by 1e-3 reads 9.5e-4, and even 1e-5 on
    # a single weight reads 9.5e-6, all far above the bar.
    def _same(p, q):
        return torch.allclose(p, q, rtol=0, atol=1e-6)

    for k in a0:
        assert _same(a0[k], a1[k]) and _same(a0[k], a2[k]), k
    for k in g0:
        assert _same(g0[k], g1[k]) and _same(g0[k], g2[k]), k
    assert _same(v0, v1) and _same(v0, v2)


def test_partner_input_conditions_each_our_head_independently():
    torch.manual_seed(1)
    pair = _tiny(pair_cond=True).eval()          # fresh init: pair columns non-zero
    x = _rand_x(2)
    with torch.no_grad():
        base_a, base_g, base_v = pair(x)
        cond_a, cond_g, cond_v = pair(x, partner_actions={
            "our_a": _onehot(3).expand(2, A)})   # only our_a sees a partner action
    assert not torch.equal(base_a["our_a"], cond_a["our_a"])   # conditioned head moved
    assert torch.equal(base_a["our_b"], cond_a["our_b"])       # the other did not
    # opp heads / gimmicks / value never read the pair input
    for k in ("opp_a", "opp_b"):
        assert torch.equal(base_a[k], cond_a[k])
    for k in base_g:
        assert torch.equal(base_g[k], cond_g[k])
    assert torch.equal(base_v, cond_v)
    # different partner actions -> different logits (the cond is action-specific)
    with torch.no_grad():
        cond2_a, _, _ = pair(x, partner_actions={"our_a": _onehot(9).expand(2, A)})
    assert not torch.equal(cond_a["our_a"], cond2_a["our_a"])


def test_surgery_refuses_non_pair_model():
    donor = _tiny()
    with pytest.raises(ValueError, match="pair_cond=False"):
        init_pair_model_from_ckpt(_tiny(), donor.state_dict())


def test_unbatched_forward_with_partner_vec():
    torch.manual_seed(2)
    pair = _tiny(pair_cond=True).eval()
    x = _rand_x()                                 # (state_dim,) unbatched serve shape
    with torch.no_grad():
        a, _, _ = pair(x, partner_actions={"our_b": _onehot(5)})
    assert a["our_a"].shape[-1] == A and a["our_a"].dim() == 1


# ── stage 2: trainer teacher-forcing ──────────────────────────────────────────

def _fake_batch(m, bsz=8):
    g = m.gimmick_dim
    return {
        "x": _rand_x(bsz),
        "target": torch.randint(0, A, (bsz, 2)),
        "mask": torch.ones(bsz, 2, A),
        "valid": torch.ones(bsz, 2),
        "gimmick_target": torch.zeros(bsz, 2, dtype=torch.long),
        "gimmick_mask": torch.ones(bsz, 2, g),
        "gimmick_valid": torch.ones(bsz, 2),
        "value_target": torch.ones(bsz),
        "value_valid": torch.zeros(bsz),      # value head inert in this test
    }


def test_run_epoch_teacher_forcing_trains_pair_columns():
    from v_dance.training.train_bc import run_epoch
    torch.manual_seed(3)
    m = _tiny(pair_cond=True)
    batch = _fake_batch(m)
    before_a = m.heads["our_a"].weight[:, -A:].clone()
    before_b = m.heads["our_b"].weight[:, -A:].clone()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    torch.manual_seed(4)                       # deterministic decode-order/dropout draws
    run_epoch(m, [batch], "cpu", opt, pair_dropout=0.0)
    # teacher forcing routed gradient into the pair columns of both OUR heads
    assert not torch.equal(before_a, m.heads["our_a"].weight[:, -A:])
    assert not torch.equal(before_b, m.heads["our_b"].weight[:, -A:])


def test_run_epoch_eval_is_zero_cond_and_pair_off_unchanged():
    from v_dance.training.train_bc import run_epoch
    torch.manual_seed(5)
    m = _tiny(pair_cond=True)
    batch = _fake_batch(m)
    stats = run_epoch(m, [batch], "cpu", optimizer=None)   # eval: no cond, no grads
    assert stats["n"] > 0
    # pair_cond=False model passes through run_epoch untouched by the 2b branch
    m0 = _tiny()
    opt = torch.optim.SGD(m0.parameters(), lr=0.1)
    stats0 = run_epoch(m0, [batch], "cpu", opt, pair_dropout=0.0)
    assert stats0["n"] > 0


# ── stage 3: serve sequential decode ──────────────────────────────────────────

def _rigged_pair_model():
    """A pair model whose decode is fully determined by head biases/pair columns:
    our_a always picks 5 (high bias, high confidence); our_b picks 2 zero-cond
    but 7 when conditioned on partner action 5 (a huge pair-column weight)."""
    m = _tiny(pair_cond=True).eval()
    with torch.no_grad():
        for h in ("our_a", "our_b", "opp_a", "opp_b"):
            m.heads[h].weight.zero_()
            m.heads[h].bias.zero_()
        m.heads["our_a"].bias[5] = 10.0            # slot 0: confident action 5
        m.heads["our_b"].bias[2] = 1.0             # slot 1 zero-cond: action 2 (low conf)
        m.heads["our_b"].weight[7, -A + 5] = 100.0  # partner=5 -> action 7 wins
    return m


def test_pair_decode_sequential_conditions_second_slot():
    from v_dance.play.model_io import bc_action_indices
    m = _rigged_pair_model()
    x = _rand_x().numpy()
    mask = [True] * A
    m._pair_decode = True
    a0, a1 = bc_action_indices(m, ("our_a", "our_b"), x, mask, mask)
    assert (a0, a1) == (5, 7)                      # conditioned: our_b flips 2 -> 7
    m._pair_decode = False                         # kill switch: independent decode
    a0, a1 = bc_action_indices(m, ("our_a", "our_b"), x, mask, mask)
    assert (a0, a1) == (5, 2)


def test_pair_decode_no_legal_first_slot_degrades_to_zero_cond():
    from v_dance.play.model_io import bc_action_indices
    m = _rigged_pair_model()
    m._pair_decode = True
    x = _rand_x().numpy()
    none_mask = [False] * A
    a0, a1 = bc_action_indices(m, ("our_a", "our_b"), x, none_mask, [True] * A)
    assert a0 is None and a1 == 2                  # zero-cond pick for the survivor


# ── stage 4: bc_val_report teacher-forced go/no-go metric ─────────────────────

def _pair_val_examples(n=8):
    import numpy as np
    from v_dance.encoders.state_encoder import ACTIONS_PER_SLOT
    rng = np.random.RandomState(9)
    mask = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
    mask[:4] = 1.0
    return [{
        "x": rng.rand(get_state_dim()).astype(np.float32),
        "targets": {"our_a": 1, "our_b": 3},
        "masks": {"our_a": mask, "our_b": mask},
        "gimmick_targets": {}, "gimmick_masks": {},
        "replay_id": f"r{i}", "perspective": "p1",
        "rating": None, "rating_delta": 0.0, "won": True,
        "turn": (i % 6) + 1, "decision_type": "turn",
    } for i in range(n)]


def _save_ckpt(m, path, pair):
    from v_dance.encoders.state_encoder import (
        get_gimmick_dim, get_state_layout_version)
    cfg = {"model_type": "attn", "state_dim": get_state_dim(),
           "state_layout_version": get_state_layout_version(),
           "action_dim": A, "gimmick_dim": get_gimmick_dim(),
           "d_model": 32, "n_heads": 4, "n_layers": 1, "ff_mult": 2, "dropout": 0.0,
           "heads": list(m.head_names), "gimmick_heads": list(m.gimmick_head_names),
           "value_trained": True, "gimmick_trained": True, "pair_cond": pair}
    torch.save({"model_state": m.state_dict(), "config": cfg}, path)
    return path


def test_val_report_teacher_forced_gap(tmp_path, capsys):
    from v_dance.eval.bc_val_report import evaluate_checkpoint, print_report
    m = _rigged_pair_model()
    with torch.no_grad():                          # zero-cond our_b picks 0 (wrong,
        m.heads["our_b"].bias.zero_()              # target 3); TF partner=1 -> 3
        m.heads["our_b"].bias[0] = 5.0
        m.heads["our_b"].weight.zero_()
        m.heads["our_b"].weight[3, -A + 1] = 100.0
        m.heads["our_a"].bias.zero_()
        m.heads["our_a"].bias[1] = 10.0            # our_a always right (target 1)
    p = _save_ckpt(m, tmp_path / "pair.pt", pair=True)
    r = evaluate_checkpoint(str(p), _pair_val_examples(), device="cpu", batch_size=4)
    assert r["pair_tf_head"] is not None
    assert r["by_head"]["our_b"].rate() == 0.0             # zero-cond misses
    assert r["pair_tf_head"]["our_b"].rate() == 1.0        # teacher-forced nails it
    assert r["pair_tf_head"]["our_a"].rate() == 1.0        # unharmed
    print_report([r])
    out = capsys.readouterr().out
    assert "2b pair-cond" in out and "gap +1.0000" in out


def test_val_report_non_pair_ckpt_skips_metric(tmp_path, capsys):
    from v_dance.eval.bc_val_report import evaluate_checkpoint, print_report
    m = _tiny()
    p = _save_ckpt(m, tmp_path / "plain.pt", pair=False)
    r = evaluate_checkpoint(str(p), _pair_val_examples(), device="cpu", batch_size=4)
    assert r["pair_tf_head"] is None
    print_report([r])
    assert "2b pair-cond" not in capsys.readouterr().out


def test_load_bc_policy_stamps_pair_decode(tmp_path, monkeypatch, capsys):
    from v_dance.play import model_io
    from v_dance.play.model_io import load_bc_policy
    m = _tiny(pair_cond=True)
    p = _save_ckpt(m, tmp_path / "battle_pair.pt", pair=True)
    monkeypatch.setattr(model_io, "_PAIR_ECHO_DONE", False)        # echo is once-per-process
    loaded, heads = load_bc_policy(str(p))
    assert getattr(loaded, "_pair_decode", False) is True          # default ON
    assert "pair decode ACTIVE" in capsys.readouterr().out
    loaded1b, _ = load_bc_policy(str(p))                           # second load: no re-echo
    assert "pair decode" not in capsys.readouterr().out
    monkeypatch.setenv("VD_PAIR_DECODE", "0")                      # kill switch
    monkeypatch.setattr(model_io, "_PAIR_ECHO_DONE", False)
    loaded2, _ = load_bc_policy(str(p))
    assert getattr(loaded2, "_pair_decode", True) is False
    assert "DISABLED" in capsys.readouterr().out


def test_actor_critic_roundtrips_pair_cond_policy(tmp_path):
    # Regression (2026-07-24, exploit-meter crash): an ActorCritic warm-started from a
    # pair_cond target must SAVE a config that rebuilds the widened heads — the verify
    # reload inside ac.save() is the gate that caught the missing pair_cond stamp.
    from v_dance.selfplay.actor_critic import ActorCritic
    m = _tiny(pair_cond=True)
    src = _save_ckpt(m, tmp_path / "target_pair.pt", pair=True)
    ac = ActorCritic.from_bc_checkpoint(str(src), require_value_trained=False)
    out = tmp_path / "exploiter.pt"
    ac.save(out, generation=1, verify=True)       # raises on a bad config stamp
    from v_dance.play.model_io import load_bc_policy
    reloaded, _ = load_bc_policy(str(out))
    assert bool(getattr(reloaded, "pair_cond", False)) is True
    assert reloaded.heads["our_a"].weight.shape[1] == m.heads["our_a"].weight.shape[1]
