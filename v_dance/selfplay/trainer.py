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

from v_dance.selfplay import policy_eval, ppo as P
from v_dance.selfplay.gae import DEFAULT_GAMMA, DEFAULT_LAM, compute_batch_gae
from v_dance.selfplay.ppo import PPOConfig, make_reference_policy
from v_dance.selfplay.value_space import assert_gae_identity, assert_transitions_value_space


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


class PPOTrainer:
    def __init__(self, ac, ppo_cfg: Optional[PPOConfig] = None,
                 train_cfg: Optional[TrainConfig] = None, ref_policy=None,
                 device: str = "cpu", seed: int = 0):
        self.ac = ac
        self.cfg = ppo_cfg or PPOConfig()
        self.tcfg = train_cfg or TrainConfig()
        self.device = device
        self.ref = ref_policy if ref_policy is not None else make_reference_policy(ac, device)
        self.actor_opt = torch.optim.Adam(ac.actor_parameters(), lr=self.tcfg.actor_lr)
        self.critic_opt = torch.optim.Adam(ac.critic_parameters(), lr=self.tcfg.critic_lr)
        self.rng = np.random.default_rng(seed)
        self.updates = 0
        self.warmups = 0

    def reset_optimizers(self) -> None:
        """Re-initialise the Adam optimisers, clearing the first/second-moment estimates.
        Called on a collapse REVERT: restoring the champion WEIGHTS while keeping the stale
        Adam moments that DROVE the collapse would re-apply large momentum steps and shove
        the just-restored policy back off the same cliff. Fresh moments let recovery start
        clean at the configured LRs (the champion checkpoint also carries critic_state, so
        ``restore_from`` reloads a matching critic — value baseline + policy stay aligned)."""
        self.actor_opt = torch.optim.Adam(self.ac.actor_parameters(), lr=self.tcfg.actor_lr)
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
        with torch.no_grad():
            ev = policy_eval.ppo_forward(self.ac, txns, self.cfg.tau, self.device,
                                         ref_policy=self.ref)
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
                value_pm = self.ac.value_pm(self._states(mb_txns))     # critic only
                old_v = torch.as_tensor(np.array([txns[i].value for i in mb], np.float32),
                                        device=self.device)
                tgt = torch.as_tensor(ret[mb].astype(np.float32), device=self.device)
                # clip=False: the warm-up regresses on a FIXED cold-start anchor, so the
                # value-clip would freeze V within ±value_clip of the BC value (3b.4 bug fix).
                vloss = P.value_loss(value_pm, old_v, tgt, self.cfg, clip=False)
                self.critic_opt.zero_grad()
                vloss.backward()
                clip_grad_norm_(self.ac.critic_parameters(), self.tcfg.max_grad_norm)
                self.critic_opt.step()
                last_vloss = float(vloss.detach())
            self.warmups += 1
        return {"value_loss": last_vloss, "explained_variance": self._critic_ev(txns, ret)}

    # ── full PPO update ───────────────────────────────────────────────────────
    def ppo_update(self, trajectories) -> Dict[str, float]:
        """``ppo_epochs`` passes of shuffled minibatches (clip + value + entropy +
        KL-to-BC), both optimisers stepped, grad-norm clipped. Early-halts on the
        collapse guards; always reports explained variance + ``halted``/``halt_reason``."""
        txns, adv, ret = self._flatten(trajectories)
        if not txns:
            return {"halted": False, "halt_reason": "empty batch", "n_steps": 0}
        agg: List[Dict[str, float]] = []
        halted, reason = False, None
        for _epoch in range(self.tcfg.ppo_epochs):
            for mb in self._minibatches(len(txns)):
                mb_txns = [txns[i] for i in mb]
                loss, st = P.ppo_loss_from_batch(self.ac, mb_txns, adv[mb], ret[mb],
                                                 cfg=self.cfg, ref_policy=self.ref,
                                                 device=self.device)
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                clip_grad_norm_(self.ac.actor_parameters(), self.tcfg.max_grad_norm)
                clip_grad_norm_(self.ac.critic_parameters(), self.tcfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()
                agg.append(st)
            if self.tcfg.target_kl_from_bc is not None:
                kl_bc = self._kl_from_bc(txns)
                if kl_bc > self.tcfg.target_kl_from_bc:
                    halted, reason = True, f"kl_from_bc {kl_bc:.4f} > {self.tcfg.target_kl_from_bc}"
                    break
        self.updates += 1

        stats = _mean_stats(agg)
        ev = self._critic_ev(txns, ret)
        stats["explained_variance"] = ev
        stats["n_steps"] = len(txns)
        if not halted and self.tcfg.min_explained_variance is not None \
                and ev < self.tcfg.min_explained_variance:
            halted, reason = True, f"explained_variance {ev:.3f} < {self.tcfg.min_explained_variance}"
        stats["halted"] = halted
        stats["halt_reason"] = reason
        return stats


def _mean_stats(stats_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average each numeric stat across the minibatch steps."""
    if not stats_list:
        return {}
    keys = stats_list[0].keys()
    return {k: float(np.mean([s[k] for s in stats_list])) for k in keys}
