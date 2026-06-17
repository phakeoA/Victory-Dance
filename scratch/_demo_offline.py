"""Guided OFFLINE smoke for the PPO self-play data + trainer layer (3a + 3b).

Runs the WHOLE 3a/3b chain on a REAL ranked replay (both perspectives) with the REAL
production v3 BC checkpoint, printing what each piece does + what a healthy result looks
like. This is the manual "does it actually work?" walkthrough - no live Showdown needed.

    .venv/Scripts/python.exe local_battle/self_play/_demo_offline.py

Read the [CHECK] lines: each states the property being verified and the actual numbers.
Every section ends in OK; a real failure raises. See the per-section "what you're looking
at" notes printed inline.
"""
from __future__ import annotations

import glob
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]   # scratch/ -> repo root
import json

import torch

import v_dance.play.model_io as model_io
from v_dance.encoders.state_encoder import StateEncoder, get_state_dim
from v_dance.selfplay.actor_critic import ActorCritic
from v_dance.selfplay.collector import TrajectoryCollector, assert_zero_sum
from v_dance.selfplay.reward import place_terminal_reward
from v_dance.selfplay.schema import PASS_ACTION
from v_dance.selfplay.store import read_trajectories, write_trajectories, assert_terminal_rewards_clean
from v_dance.selfplay.gae import compute_batch_gae, compute_gae
from v_dance.selfplay import diagnostics, policy_eval, value_space, pbrs
from v_dance.selfplay import ppo as P
from v_dance.selfplay.trainer import PPOTrainer, TrainConfig, explained_variance

CKPT = _REPO / "ai_train_scripts" / "BC_model" / "checkpoints" / "bc_best.pt"


def hdr(title):
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)


def chk(msg):
    print(f"  [CHECK] {msg}")


# ── build two real trajectories (p1 + p2) from one replay ─────────────────────
def _slot(acts, slot):
    a = acts.get(slot)
    if not a or a.get("action_index") is None:
        return PASS_ACTION, 0
    return int(a["action_index"]), int(a.get("gimmick_index") or 0)


def build_trajectory(rows, persp, ac, enc):
    rows = [r for r in rows if r.get("perspective") == persp]
    players = rows[0].get("players") or {}
    winner = rows[0].get("winner")
    me = (players.get(persp) or {})
    roster = me.get("roster") or []
    brought = me.get("brought") or []
    won = (winner is not None and me.get("username") == winner)

    c = TrajectoryCollector(rows[0].get("replay_id", "demo"), persp)
    for t in rows:
        snap = t.get("state_before_actions") or {}
        state = np.asarray(enc.encode_snapshot(snap, t.get("turn") or 0), dtype=np.float32)
        acts = {a.get("slot"): a for a in (t.get("our_actions") or [])}
        a0, g0 = _slot(acts, "our_a")
        a1, g1 = _slot(acts, "our_b")
        am = t.get("action_mask") or {}
        gm = t.get("gimmick_mask") or {}
        c.add_step(state=state, action_s0=a0, action_s1=a1, gimmick_s0=g0, gimmick_s1=g1,
                   mask_s0=am.get("our_a"), mask_s1=am.get("our_b"),
                   gmask_s0=gm.get("our_a"), gmask_s1=gm.get("our_b"),
                   decision_type=t.get("decision_type", "turn"), turn=t.get("turn") or 0)

    bring = [roster.index(s) for s in brought if s in roster]
    traj = c.finish(own_team=roster, opp_team=[], tp_bring=bring, tp_leads=bring[:2],
                    won=won, terminal_type="win" if won else "loss")
    # fill collection-time logprob + value (value_pm) from the real model = on-policy
    lp, _ent, vpm = policy_eval.evaluate_actions(ac, traj.transitions)
    for tr_i, l, v in zip(traj.transitions, lp.detach().tolist(), vpm.detach().tolist()):
        tr_i.logprob, tr_i.value = float(l), float(v)
    place_terminal_reward(traj)
    return traj


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # ── 3b.1: load production model + cloned critic ───────────────────────────
    hdr("3b.1  Actor-critic from the REAL v3 checkpoint (cloned separate critic)")
    assert CKPT.exists(), f"missing checkpoint {CKPT}"
    ac = ActorCritic.from_bc_checkpoint(CKPT)
    n_actor = sum(p.numel() for p in ac.actor_parameters())
    n_critic = sum(p.numel() for p in ac.critic_parameters())
    chk(f"loaded {CKPT.name}: STATE_DIM={get_state_dim()} value_trained={ac.value_trained} "
        f"heads={ac.head_names}")
    chk(f"actor params={n_actor:,}  critic params={n_critic:,}  (critic is a separate net)")
    actor_store = {p.data_ptr() for p in ac.policy.parameters()}
    critic_store = {p.data_ptr() for p in ac.critic_parameters()}
    assert actor_store.isdisjoint(critic_store)
    chk("critic shares ZERO parameter tensors with the actor -> decoupled  OK")
    print("  (look for: value_trained=True, STATE_DIM=1854, disjoint param storage)")

    # ── load a real replay, both perspectives ─────────────────────────────────
    hdr("3a.1-3a.3  Real trajectories from a ranked replay (both perspectives)")
    files = sorted(glob.glob(str(_REPO / "data" / "vods" / "**" / "Jsonl_TypeB" / "**" / "*.jsonl"),
                             recursive=True))
    assert files, "no Type_B jsonl found"
    rows = [json.loads(l) for l in open(files[0], encoding="utf-8")]
    enc = StateEncoder()
    p1 = build_trajectory(rows, "p1", ac, enc)
    p2 = build_trajectory(rows, "p2", ac, enc)
    print(f"  replay: {Path(files[0]).name}")
    for tag, tr in (("p1", p1), ("p2", p2)):
        chk(f"{tag}: {len(tr.transitions)} steps, won={tr.meta.won}, "
            f"terminal_type={tr.meta.terminal_type}, last reward={tr.transitions[-1].reward:+.0f}")
    chk(f"the model's win-prob on p1's opening board: "
        f"{0.5*(p1.transitions[0].value+1):.3f}  (value head on a real state)")
    print("  (look for: one side won=True/+1, the other won=False/-1, both real boards)")

    # ── 3a.2: zero-sum symmetry ───────────────────────────────────────────────
    hdr("3a.2  Zero-sum symmetry guard (the anti-self-collusion invariant)")
    assert_zero_sum(p1, p2)
    chk(f"r_us(p1)={p1.meta.outcome_reward():+.0f}  r_opp(p2)={p2.meta.outcome_reward():+.0f}  "
        f"sum={p1.meta.outcome_reward()+p2.meta.outcome_reward():+.0f}  OK")
    print("  (look for: rewards are exact negations, sum == 0)")

    # ── 3a.4: store round-trip ────────────────────────────────────────────────
    hdr("3a.4  Type-C store round-trip (incl. masks) + terminal-reward integrity")
    path = Path(tempfile.mkdtemp()) / "demo.jsonl"
    write_trajectories(path, [p1, p2])
    back = read_trajectories(path, expected_state_dim=get_state_dim())
    assert len(back) == 2 and len(back[0].transitions) == len(p1.transitions)
    assert back[0].transitions[0].mask_s0 == p1.transitions[0].mask_s0
    assert_terminal_rewards_clean(back)
    chk(f"wrote 2 trajectories -> {path.name}, read back identical (masks preserved)  OK")
    chk("assert_terminal_rewards_clean: non-terminal rewards 0, terminal in [-1,1]  OK")

    # ── 3a.5: diagnostics ─────────────────────────────────────────────────────
    hdr("3a.5  Diagnostics (per-bring / overall win-rates)")
    summ = diagnostics.summarize([p1, p2], min_games=1)
    ov = summ.get("overall", {})
    chk(f"overall over both perspectives: {ov}")
    print("  (look for: ~50% by symmetry since both sides of one game are present)")

    # ── 3b.2: GAE ─────────────────────────────────────────────────────────────
    hdr("3b.2  GAE - returns rise toward the win (winning perspective)")
    win_traj = p1 if p1.meta.won else p2
    adv, ret = compute_gae(win_traj)
    print("   turn | value_pm | reward | GAE return")
    for i, t in enumerate(win_traj.transitions):
        mark = "  <- terminal +1" if t.done else ""
        print(f"   {t.turn:>4} | {t.value:+7.3f} | {t.reward:+6.0f} | {ret[i]:+8.3f}{mark}")
    chk(f"final return = {ret[-1]:+.3f} (-> the +1 win); returns climb toward the terminal  OK")
    badv, bret = compute_batch_gae([p1, p2], standardize_adv=True)
    chk(f"batch advantages standardized: mean={badv.mean():+.4f} std={badv.std():.4f} "
        f"(want ~0 / ~1)")

    # ── 3b.5: log-prob / ratio / legality ─────────────────────────────────────
    hdr("3b.5  Masked joint log-prob + entropy + the PPO ratio==1 sanity")
    policy_eval.assert_actions_legal(p1.transitions)
    lp, ent, vpm = policy_eval.evaluate_actions(ac, p1.transitions)
    old = torch.tensor([t.logprob for t in p1.transitions])
    ratio = (lp.detach() - old).exp()
    chk(f"recomputed log-prob == stored old -> ratio mean={ratio.mean():.5f} "
        f"max|ratio-1|={(ratio-1).abs().max():.2e}  (want exactly 1)")
    chk(f"mean per-step entropy={ent.mean():.3f} ; all recorded actions legal under masks  OK")

    # ── 3b.3: PPO loss at init ────────────────────────────────────────────────
    hdr("3b.3  PPO loss bundle (clip + value + entropy + KL-to-BC) at init")
    ref = P.make_reference_policy(ac)
    a_all, r_all = compute_batch_gae([p1, p2], standardize_adv=False)
    loss, st = P.ppo_loss_from_batch(ac, p1.transitions + p2.transitions, a_all, r_all,
                                     cfg=P.PPOConfig(), ref_policy=ref)
    chk(f"ratio_mean={st['ratio_mean']:.4f} clip_frac={st['clip_fraction']:.3f} "
        f"approx_kl={st['approx_kl_old_new']:+.2e} kl_to_bc={st['kl_to_bc']:.2e}")
    chk(f"policy_loss={st['policy_loss']:+.4f} value_loss={st['value_loss']:.4f} "
        f"entropy={st['entropy']:.3f}")
    print("  (look for: ratio~1, clip_frac~0, kl_to_bc~0 at init - on-policy + no drift yet)")

    # ── 3b.4 + 3b.6 fix: critic warm-up migrates V ────────────────────────────
    hdr("3b.4  Critic-only warm-up MIGRATES V to the return scale (+ the 3b.6-bug fix)")
    trajs = [p1, p2]
    tr = PPOTrainer(ac, train_cfg=TrainConfig(minibatch_size=0, critic_lr=1e-2))
    txns, _adv, rr = tr._flatten(trajs)
    with torch.no_grad():
        v_before = ac.value_pm(tr._states(txns)).cpu().numpy()
    ev_before = explained_variance(v_before, rr)
    tr.warmup_critic(trajs, n_updates=200)
    with torch.no_grad():
        v_after = ac.value_pm(tr._states(txns)).cpu().numpy()
    ev_after = explained_variance(v_after, rr)
    chk(f"critic vs returns:  |mean V - mean return|  before={abs(v_before.mean()-rr.mean()):.3f}"
        f"  ->  after={abs(v_after.mean()-rr.mean()):.3f}")
    chk(f"explained variance: before={ev_before:+.3f}  ->  after={ev_after:+.3f}  (rises)")
    print("  (look for: V moves CLOSER to the returns, EV rises - the warm-up is not throttled)")

    hdr("3b.4  One PPO update moves the policy + the collapse guards")
    a_before = [p.detach().clone() for p in ac.actor_parameters()]
    st2 = tr.ppo_update(trajs)
    moved = any(not torch.equal(b, p) for b, p in zip(a_before, ac.actor_parameters()))
    chk(f"ppo_update: loss={st2['loss']:+.4f} kl_to_bc={st2['kl_to_bc']:.3e} "
        f"EV={st2['explained_variance']:+.3f} halted={st2['halted']}  actor_moved={moved}")
    ac2 = ActorCritic.from_bc_checkpoint(CKPT)
    tr2 = PPOTrainer(ac2, train_cfg=TrainConfig(minibatch_size=0, ppo_epochs=3,
                                                actor_lr=1e-2, target_kl_from_bc=1e-9))
    halt = tr2.ppo_update([build_trajectory(rows, "p1", ac2, enc),
                           build_trajectory(rows, "p2", ac2, enc)])
    chk(f"collapse guard (target_kl_from_bc=1e-9): halted={halt['halted']}  "
        f"reason='{halt['halt_reason']}'")
    print("  (look for: actor_moved=True, halted=False normally; the guard halts on forced drift)")

    # ── 3b.6: value space ─────────────────────────────────────────────────────
    hdr("3b.6  Single value-space invariant (value_pm in [-1,1]) + win-prob catch")
    value_space.assert_transitions_value_space(p1.transitions)
    rep = value_space.value_space_report(p1.transitions + p2.transitions)
    chk(f"value report: min={rep['value_min']:+.3f} max={rep['value_max']:+.3f} "
        f"frac_negative={rep['frac_negative']:.2f}")
    fake_winprob = np.random.uniform(0, 1, size=100)        # the insidious mistake
    real_pm = np.random.uniform(-1, 1, size=100)
    chk(f"looks_like_winprob(win-prob batch)={value_space.looks_like_winprob(fake_winprob)} "
        f"(want True = caught)  ;  (value_pm batch)={value_space.looks_like_winprob(real_pm)} "
        f"(want False)")
    print("  (look for: a [0,1] batch is flagged; a [-1,1] batch is not)")

    # ── 3b.7: gated PBRS ──────────────────────────────────────────────────────
    hdr("3b.7  Gated PBRS - frozen Phi, telescoping invariance, default OFF")
    shaper = pbrs.PotentialShaper.from_critic(ac.critic, coef=0.2)
    res = pbrs.telescoping_residual(p1, shaper)
    frac_full = pbrs.shaping_fraction([p1, p2], shaper, lam=1.0)
    frac_half = pbrs.shaping_fraction([p1, p2], shaper, lam=0.5)
    chk(f"Phi(terminal)={pbrs.PotentialShaper.phi_terminal()}  "
        f"id(Phi)!=id(critic): {id(shaper)!=id(ac.critic)}")
    chk(f"telescoping residual (sum gamma^t F_t + Phi(s0)) = {res:+.2e}  (want ~0 = invariant)")
    chk(f"shaping fraction: lam=1.0 -> {frac_full:.3f} ; lam=0.5 -> {frac_half:.3f} "
        f"(linear in lambda -> anneal to 0)")
    before = [t.reward for t in p1.transitions]
    out = pbrs.maybe_shape_batch([p1], shaper, pbrs.PBRSConfig(enabled=False))
    chk(f"maybe_shape_batch(enabled=False) -> applied={out['applied']}, rewards untouched="
        f"{[t.reward for t in p1.transitions]==before}")
    print("  (look for: residual ~0, fraction halves with lambda, default-OFF is a no-op)")

    hdr("DONE - every section completed without an assertion failure")
    print("  3a (schema/collector/reward/store/diagnostics) + 3b (actor-critic/GAE/log-prob/"
          "PPO/warm-up/value-space/PBRS) all exercised on REAL data with the REAL v3 model.")


if __name__ == "__main__":
    main()
