"""PPO training loop — optimisers, epochs/minibatches, critic-only warm-up, and the
warm-start-collapse guards — task 3b.4.

3b.3 produced the per-batch loss; this is the LOOP that drives it, plus the sec-2
warm-start protocol:

  * **Separate optimisers** over ``ac.actor_parameters()`` and ``ac.critic_parameters()``
    (the cloned critic optimises independently of the policy; sec 2).
  * **Critic-only warm-up** (``warmup_critic``): freeze the actor, regress the critic
    onto the GAE returns for K passes so V migrates from the BC flat-{0,1} scale to the
    gamma=0.997 discounted-return scale BEFORE the actor unfreezes at a SMALL LR (sec 2).
  * **PPO update** (``ppo_update``): ``ppo_epochs`` passes of shuffled minibatches, each
    a clip+value+entropy+KL-to-BC step on both optimisers, grad-norm clipped.
  * **Collapse guards** (sec 2/10): early-halt the update if KL-from-BC exceeds a target
    (the policy is drifting off the BC prior) or the critic's explained variance falls
    below a floor (the value surface is being wrecked). Reported via ``stats['halted']``
    so the generation loop (3c.3) can react (revert / lower LR).

Everything runs in eval mode (dropout off) so updates are deterministic — autograd is
unaffected by eval mode. The optimiser state + RNG live on the trainer so 3c.4 can
snapshot/restore them for resumable multi-session training.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
import torch
from torch.nn.utils import clip_grad_norm_

import logging

from v_dance.selfplay import policy_eval, ppo as P
from v_dance.selfplay.gae import DEFAULT_GAMMA, DEFAULT_LAM, compute_batch_gae
from v_dance.selfplay.ppo import PPOConfig, make_reference_policy
from v_dance.selfplay.value_space import assert_gae_identity, assert_transitions_value_space

log = logging.getLogger(__name__)


def explained_variance(values, returns) -> float:
    """``1 - Var(returns - values) / Var(returns)`` — how much of the return variance
    the critic explains. 1 = perfect, 0 = no better than predicting the mean, <0 =
    worse. A drop is the warm-start-collapse signal (sec 10). 0 for a constant-return
    batch (no variance to explain), NaN for empty."""
    v = np.asarray(values, np.float64)
    r = np.asarray(returns, np.float64)
    if r.size == 0:
        return float("nan")
    var_r = r.var()
    if var_r <= 1e-12:
        return 0.0
    return float(1.0 - (r - v).var() / var_r)


@dataclass
class TrainConfig:
    actor_lr: float = 3e-5          # SMALL (warm-start; sec 2)
    critic_lr: float = 1e-3
    ppo_epochs: int = 4
    minibatch_size: int = 256       # 0 / None -> one full-batch minibatch
    max_grad_norm: float = 0.5
    gamma: float = DEFAULT_GAMMA    # sec 13 FLOOR
    lam: float = DEFAULT_LAM
    target_kl_from_bc: Optional[float] = None       # early-halt guard (None = off)
    target_kl_relax_per_gen: float = 0.0            # relax that guard by this per gen (0 = off)
    target_kl_max: Optional[float] = None           # cap for the relaxed guard (None = uncapped)
    min_explained_variance: Optional[float] = None  # post-update halt guard (None = off)
    assert_value_space: bool = True                 # 3b.6 value_pm backstop each update
    # W3b-2 (2026-09-02, the ladder PPO recipe — docs/w3b_ladder_ppo_design.md §5 / ps-ppo review):
    actor_weight_decay: float = 0.0     # > 0 → AdamW on the actor (decoupled decay); 0 = Adam, byte-identical
    backbone_lr_scale: float = 1.0      # < 1 → the trunk (everything but the action / gimmick heads) trains
                                        # at actor_lr × scale; 1.0 = one param group, byte-identical
    approx_kl_stop: Optional[float] = None  # per-EPOCH early stop: the epoch's mean approx KL(old||new)
                                            # above this halts the remaining epochs (recipe: 0.02); None = off


class PPOTrainer:
    def __init__(self, ac, ppo_cfg: Optional[PPOConfig] = None,
                 train_cfg: Optional[TrainConfig] = None, ref_policy=None,
                 device: str = "cpu", seed: int = 0):
        self.ac = ac
        self.cfg = ppo_cfg or PPOConfig()
        self.tcfg = train_cfg or TrainConfig()
        self.device = device
        self.ref = ref_policy if ref_policy is not None else make_reference_policy(ac, device)
        self.actor_opt = self._make_actor_opt()
        self.critic_opt = torch.optim.Adam(ac.critic_parameters(), lr=self.tcfg.critic_lr)
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.warmups = 0

    def _make_actor_opt(self):
        """The actor optimiser. W3b-2 options: AdamW (``actor_weight_decay`` > 0) and a backbone
        LR scale (the action / gimmick heads at ``actor_lr``, the trunk at ``actor_lr ×
        backbone_lr_scale``). Defaults = one Adam group at ``actor_lr`` — byte-identical."""
        lr = float(self.tcfg.actor_lr)
        scale = float(self.tcfg.backbone_lr_scale)
        wd = float(self.tcfg.actor_weight_decay)
        named = [(n, p) for n, p in self.ac.policy.named_parameters()
                 if not n.startswith("value_head.")]           # == ac.actor_parameters()
        if scale == 1.0:
            groups = [{"params": [p for _, p in named]}]
        else:
            is_head = lambda n: n.startswith(("heads.", "gimmick_heads."))   # noqa: E731
            groups = [{"params": [p for n, p in named if is_head(n)], "lr": lr},
                      {"params": [p for n, p in named if not is_head(n)], "lr": lr * scale}]
            groups = [g for g in groups if g["params"]]
        if wd > 0.0:
            return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
        return torch.optim.Adam(groups, lr=lr)

    def reset_optimizers(self) -> None:
        """Re-initialise the Adam optimisers, clearing the first/second-moment estimates.
        Called on a collapse REVERT: restoring the champion WEIGHTS while keeping the stale
        Adam moments that DROVE the collapse would re-apply large momentum steps and shove
        the just-restored policy back off the same cliff. Fresh moments let recovery start
        clean at the configured LRs (the champion checkpoint also carries critic_state, so
        ``restore_from`` reloads a matching critic — value baseline + policy stay aligned)."""
        self.actor_opt = self._make_actor_opt()
        self.critic_opt = torch.optim.Adam(self.ac.critic_parameters(), lr=self.tcfg.critic_lr)

    # ── batch helpers ─────────────────────────────────────────────────────────
    def _flatten(self, trajectories):
        """GAE (UN-standardised — per-minibatch standardisation happens in the loss) +
        the flattened transition list, aligned 1:1 with the concatenated adv/returns."""
        adv, ret = compute_batch_gae(trajectories, self.tcfg.gamma, self.tcfg.lam,
                                     standardize_adv=False)
        txns = [t for tr in trajectories for t in tr.transitions]
        assert len(txns) == len(adv) == len(ret), \
            f"flatten misalignment: {len(txns)} txns vs {len(adv)} adv (GAE order mismatch)"
        if self.tcfg.assert_value_space:             # 3b.6: ONE space (value_pm ∈ [-1,1])
            assert_transitions_value_space(txns)
            assert_gae_identity(adv, ret, [t.value for t in txns])
        return txns, adv, ret

    def _minibatches(self, n: int):
        idx = np.arange(n)
        self.rng.shuffle(idx)
        mb = self.tcfg.minibatch_size or n
        for s in range(0, n, mb):
            yield idx[s:s + mb]

    def _states(self, txns):
        return torch.as_tensor(np.stack([np.asarray(t.state, np.float32) for t in txns]),
                               device=self.device)

    def _critic_ev(self, txns, ret) -> float:
        with torch.no_grad():
            v = self.ac.value_pm(self._states(txns)).cpu().numpy()
        return explained_variance(v, ret)

    def _kl_from_bc(self, txns) -> float:
        cfg = self.cfg
        pm = None if cfg.replacement_policy else policy_eval.replacement_policy_mask(txns, self.device)
        with torch.no_grad():
            ev = policy_eval.ppo_forward(self.ac, txns, cfg.tau, self.device,
                                         ref_policy=self.ref, gimmick_kl_weight=cfg.gimmick_kl_weight,
                                         pair=cfg.pair_decode, order=cfg.pair_order,   # W3b-1b parity
                                         gimmick_terms=cfg.gimmick_terms, policy_mask=pm)
        return float(ev.kl_to_ref.mean())

    # ── critic-only warm-up (actor frozen) ────────────────────────────────────
    def warmup_critic(self, trajectories, n_updates: int = 1) -> Dict[str, float]:
        """K critic-only passes: regress the critic onto the GAE returns, actor frozen
        (no actor grad is ever taken). Returns the final value loss + explained variance."""
        txns, adv, ret = self._flatten(trajectories)
        if not txns:
            return {"value_loss": float("nan"), "explained_variance": float("nan")}
        last_vloss = float("nan")
        for _ in range(n_updates):
            for mb in self._minibatches(len(txns)):
                mb_txns = [txns[i] for i in mb]
                states = self._states(mb_txns)
                old_v = torch.as_tensor(np.array([txns[i].value for i in mb], np.float32),
                                        device=self.device)
                tgt = torch.as_tensor(ret[mb].astype(np.float32), device=self.device)
                # C51: regress the distributional CE on the per-atom head; else the scalar value loss.
                atoms_logits = None
                if self.cfg.value_loss_mode == "c51":
                    atoms_logits = self.ac.critic.value_atoms_logits(states)
                    value_pm = (atoms_logits.softmax(dim=-1) * self.ac.critic.support).sum(dim=-1)
                else:
                    value_pm = self.ac.value_pm(states)                # critic only
                # clip=False: the warm-up regresses on a FIXED cold-start anchor, so the
                # value-clip would freeze V within ±value_clip of the BC value (3b.4 bug fix).
                vloss = P.value_loss(value_pm, old_v, tgt, self.cfg, clip=False, atoms_logits=atoms_logits,
                                     support=(self.ac.critic.support if atoms_logits is not None else None))
                self.critic_opt.zero_grad()
                vloss.backward()
                clip_grad_norm_(self.ac.critic_parameters(), self.tcfg.max_grad_norm)
                self.critic_opt.step()
                last_vloss = float(vloss.detach())
            self.warmups += 1
        return {"value_loss": last_vloss, "explained_variance": self._critic_ev(txns, ret)}

    # ── #23: rebase collection-time values onto the post-warm-up critic ────────
    def rebase_values(self, trajectories) -> int:
        """Overwrite each transition's collection-time ``value`` with the critic's CURRENT V(s)
        (``value_pm`` space). Call at gen 0 AFTER ``warmup_critic`` and BEFORE ``ppo_update``.

        The warm-up migrates the critic OFF the BC value scale, so each step's stored
        collection-time value (the pre-warm-up BC estimate) is one critic-migration STALE — and
        that value is reused below as BOTH the value-clip trust-region anchor (``old_value_pm`` in
        ``_value_loss``) AND the GAE baseline (``compute_batch_gae`` reads ``t.value``). A stale
        clip anchor pins V within ±``value_clip`` of the dead BC estimate (throttling the very
        gradient the warm-up exists to enable), and a stale baseline biases the gen-0 advantages.
        Rebasing makes gen-0 behave like gen-N, where ``old_value`` IS the critic at the start of
        the update (no intervening shift). Mutates the transitions in place (gen-0 trajectories are
        discarded after the update). Returns the count rebased."""
        txns = [t for tr in trajectories for t in tr.transitions]
        if not txns:
            return 0
        with torch.no_grad():
            v = self.ac.value_pm(self._states(txns)).cpu().numpy()
        for t, vi in zip(txns, v):
            t.value = float(vi)
        return len(txns)

    # ── full PPO update ───────────────────────────────────────────────────────
    def ppo_update(self, trajectories, traj_weights=None) -> Dict[str, float]:
        """``ppo_epochs`` passes of shuffled minibatches (clip + value + entropy +
        KL-to-BC), both optimisers stepped, grad-norm clipped. Early-halts on the
        collapse guards; always reports explained variance + ``halted``/``halt_reason``.

        ``traj_weights`` (2026-09-04 W3b B2, one float per trajectory, same order): the advantages of every
        transition of trajectory i are multiplied by ``traj_weights[i]`` BEFORE the loss (returns untouched —
        the critic still regresses the true outcome). The ladder update weights games by the opponent's
        rating so a tanked band does not train the policy on beating weak players. The per-minibatch
        advantage standardisation keeps the RELATIVE weights and removes any overall scale."""
        txns, adv, ret = self._flatten(trajectories)
        if not txns:
            return {"halted": False, "halt_reason": "empty batch", "n_steps": 0}
        w_stats: Dict[str, float] = {}
        if traj_weights is not None:
            if len(traj_weights) != len(trajectories):
                raise ValueError(f"traj_weights: {len(traj_weights)} weights for {len(trajectories)} trajectories")
            w = np.concatenate([np.full(len(tr.transitions), float(wi), dtype=np.float32)
                                for tr, wi in zip(trajectories, traj_weights)])
            assert len(w) == len(adv), "traj_weights: transition count mismatch"
            adv = adv * w
            w_stats = {"traj_weight_mean": float(np.mean(traj_weights)),
                       "traj_weight_min": float(np.min(traj_weights)),
                       "traj_weight_max": float(np.max(traj_weights))}
        agg: List[Dict[str, float]] = []
        halted, reason = False, None
        nonfinite_skips = 0     # defensive: minibatches dropped for a NaN/inf loss or grad (never step them)
        epochs_run = 0
        for _epoch in range(self.tcfg.ppo_epochs):
            epoch_start = len(agg)
            epochs_run += 1
            for mb in self._minibatches(len(txns)):
                mb_txns = [txns[i] for i in mb]
                loss, st = P.ppo_loss_from_batch(self.ac, mb_txns, adv[mb], ret[mb],
                                                 cfg=self.cfg, ref_policy=self.ref,
                                                 device=self.device)
                # NaN/inf GUARD (no isfinite check existed anywhere in the train path): a single degenerate
                # minibatch would otherwise poison the weights AND the Adam moments AND every per-gen snapshot,
                # which the gate's policy-only revert cannot clean (esp. at gen 0). Skip such a minibatch
                # entirely — never backward/step on non-finite values — and surface the count.
                if not bool(torch.isfinite(loss)):
                    log.error("PPO update: non-finite LOSS on a minibatch — skipping it (no backward/step) "
                              "to protect the weights + optimizer state.")
                    nonfinite_skips += 1
                    continue
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                gn_a = clip_grad_norm_(self.ac.actor_parameters(), self.tcfg.max_grad_norm)
                gn_c = clip_grad_norm_(self.ac.critic_parameters(), self.tcfg.max_grad_norm)
                if not (bool(torch.isfinite(gn_a)) and bool(torch.isfinite(gn_c))):
                    log.error("PPO update: non-finite GRAD norm (actor=%s critic=%s) — skipping the optimizer "
                              "step to protect the weights + optimizer state.", float(gn_a), float(gn_c))
                    nonfinite_skips += 1
                    continue
                self.actor_opt.step()
                self.critic_opt.step()
                agg.append(st)
            if self.tcfg.target_kl_from_bc is not None:
                kl_bc = self._kl_from_bc(txns)
                if kl_bc > self.tcfg.target_kl_from_bc:
                    halted, reason = True, f"kl_from_bc {kl_bc:.4f} > {self.tcfg.target_kl_from_bc}"
                    break
            # W3b-2: the recipe's per-epoch approx-KL early stop (a BENIGN stop — the epochs that ran
            # are kept; it is not a collapse guard). Reported through halt_reason as "approx_kl …".
            if self.tcfg.approx_kl_stop is not None and len(agg) > epoch_start:
                akl = float(np.mean([s.get("approx_kl_old_new", 0.0) for s in agg[epoch_start:]]))
                if akl > self.tcfg.approx_kl_stop:
                    halted, reason = True, (f"approx_kl {akl:.4f} > {self.tcfg.approx_kl_stop} "
                                            f"after epoch {epochs_run}")
                    break
        self.updates += 1

        stats = _mean_stats(agg)
        stats.update(w_stats)
        ev = self._critic_ev(txns, ret)
        stats["explained_variance"] = ev
        stats["n_steps"] = len(txns)
        stats["epochs_run"] = epochs_run
        stats["nonfinite_skips"] = nonfinite_skips
        if nonfinite_skips:
            log.warning("PPO update: %d minibatch(es) skipped this gen for non-finite loss/grad.", nonfinite_skips)
        if nonfinite_skips and not agg:        # every minibatch was non-finite → NO weights changed
            halted, reason = True, f"all minibatches non-finite ({nonfinite_skips}) — no update applied"
        if not halted and self.tcfg.min_explained_variance is not None \
                and ev < self.tcfg.min_explained_variance:
            halted, reason = True, f"explained_variance {ev:.3f} < {self.tcfg.min_explained_variance}"
        stats["halted"] = halted
        stats["halt_reason"] = reason
        return stats


def _mean_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average each numeric stat across the minibatch steps. audit: drop NON-FINITE entries per key before
    averaging (seed NaN only if EVERY entry was NaN) — e.g. `opp_ce` is NaN on a minibatch with zero valid
    opp-action targets (all-PASS), and a single NaN would otherwise poison the whole generation's reported
    metric. Training is unaffected (the loss term is excluded for those minibatches); this keeps the telemetry
    a true mean over the minibatches that actually contributed."""
    if not stats_list:
        return {}
    out: Dict[str, float] = {}
    for k in stats_list[0].keys():
        finite = [s[k] for s in stats_list if isinstance(s.get(k), (int, float)) and np.isfinite(s[k])]
        out[k] = float(np.mean(finite)) if finite else float("nan")
    return out
