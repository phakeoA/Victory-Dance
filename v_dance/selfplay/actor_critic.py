"""Actor-critic initialisation FROM the BC checkpoint — task 3b.1.

The first torch-bearing module in ``self_play``. Builds the PPO actor-critic by
reusing the trained BC policy + value head (docs/ppo_reward_design.md sec 2 — the
single biggest sample-efficiency lever):

  * **Actor** = the loaded ``BCPolicy`` (trunk -> per-slot action heads + per-slot
    gimmick heads). Its own ``value_head`` is VESTIGIAL here (the separate critic
    owns the value function) and is excluded from the actor optimiser group.
  * **Critic** = a *separate* deep-copy of ``trunk + value_head`` from the BC
    weights. Decoupling protects the BCE-calibrated value surface from
    policy-gradient drift / warm-start collapse (sec 2 default). ``id(critic) !=
    id(actor)`` and the two share NO parameter tensors, so a critic update leaves
    the actor bit-identical and vice-versa.

Value space (the #1 silent-bug surface, sec 3 / locked down in 3b.6):
  * the critic head is a raw win-LOGIT;
  * ``winprob(x) = sigmoid(logit)``      in [0,1]  -> BCE critic target + live readout;
  * ``value_pm(x) = 2*sigmoid(logit)-1`` in [-1,1] -> the GAE baseline, in the SAME
    +-1 space as the terminal reward (so ``adv = r + gamma*V' - V`` is consistent).
The collection-time ``Transition.value`` the GAE module consumes is this ``value_pm``.

Resumability (3c.4) leans on ``ActorCritic`` being a single ``nn.Module``: its
``state_dict`` carries both ``policy.*`` and ``critic.*`` so one save/load round-trips
the whole actor-critic.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import List, Tuple

# ── Path bootstrap: local_battle/ (for model_io) + data/scripts + BC_model ────
_REPO_ROOT = Path(__file__).resolve().parents[2]
import torch
import torch.nn as nn

import v_dance.play.model_io as model_io


class Critic(nn.Module):
    """Separate value network: a clone of the BC ``trunk + value_head``.

    forward(x) -> (B,) raw win-LOGIT. ``winprob`` / ``value_pm`` map it to the two
    use-time spaces (see module docstring). Holds its OWN parameter tensors (the
    factory deep-copies the BC modules), so it optimises independently of the actor.
    """

    def __init__(self, trunk: nn.Module, value_head: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.value_head = value_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(x)).squeeze(-1)  # (B,) raw win-logit

    def winprob(self, x: torch.Tensor) -> torch.Tensor:
        """Win probability in [0,1] — the BCE critic target space + live readout."""
        return torch.sigmoid(self.forward(x))

    def value_pm(self, x: torch.Tensor) -> torch.Tensor:
        """Value in [-1,1] (``2*sigmoid(logit)-1``) — the GAE baseline, matching the
        +-1 terminal reward space (sec 1/3)."""
        return 2.0 * torch.sigmoid(self.forward(x)) - 1.0


def _clone_critic(policy) -> Critic:
    """Deep-copy ``trunk + value_head`` off the loaded BCPolicy into a fresh module.

    ``copy.deepcopy`` of an ``nn.Module`` allocates NEW parameter tensors with the
    weights copied — so the critic starts identical to the BC value surface but is
    fully decoupled from the actor's parameters."""
    return Critic(copy.deepcopy(policy.trunk), copy.deepcopy(policy.value_head))


class ActorCritic(nn.Module):
    """PPO actor-critic = BC actor + a separate cloned critic (sec 2)."""

    def __init__(self, policy, critic: Critic, head_names, gimmick_head_names,
                 value_trained: bool):
        super().__init__()
        self.policy = policy            # BCPolicy actor (action + gimmick heads)
        self.critic = critic            # separate value net (sec 2 default)
        self.head_names: Tuple[str, ...] = tuple(head_names)
        self.gimmick_head_names: Tuple[str, ...] = tuple(gimmick_head_names)
        self.value_trained = bool(value_trained)
        # Structural separateness guard (sec 2): policy and critic are distinct
        # objects and must not alias the same parameters.
        assert id(self.policy) is not id(self.critic)

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_bc_checkpoint(cls, path, device: str = "cpu",
                           require_value_trained: bool = True) -> "ActorCritic":
        """Load a BC checkpoint via ``model_io`` and build the actor-critic.

        ``require_value_trained`` (default True): refuse a checkpoint whose value
        head was never trained on outcome labels — initialising the critic from an
        untrained head is just xavier noise and forfeits the whole sample-efficiency
        rationale of sec 2. The production v3 ``bc_best.pt`` has value_trained=True.
        """
        policy, head_names = model_io.load_bc_policy(path, device)
        vt = model_io.value_trained(policy)
        if require_value_trained and not vt:
            raise ValueError(
                f"BC checkpoint at {path} has an UNTRAINED value head "
                f"(value_trained=False). The cloned critic would init from noise, "
                f"defeating sec 2's calibrated-critic warm-start. Pass "
                f"require_value_trained=False only for tests / a deliberate cold critic."
            )
        critic = _clone_critic(policy).to(device)
        critic.eval()
        gimmick_head_names = tuple(getattr(policy, "gimmick_head_names", ()))
        return cls(policy, critic, head_names, gimmick_head_names, vt)

    # ── forward / value-space helpers ─────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """(B,state_dim) -> (actions, gimmicks, value_logit).

        ``actions`` / ``gimmicks`` come from the ACTOR; ``value_logit`` from the
        separate CRITIC (the actor's own value head is ignored — the critic is the
        value source of truth, since the actor trunk drifts under policy gradient)."""
        actions, gimmicks, _actor_value = self.policy(x)
        return actions, gimmicks, self.critic(x)

    def value_logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def winprob(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic.winprob(x)

    def value_pm(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic.value_pm(x)

    # ── optimiser param groups (separate actor / critic optimisers, sec 2) ────
    def actor_parameters(self) -> List[nn.Parameter]:
        """Actor policy parameters = trunk + action heads + gimmick heads, EXCLUDING
        the vestigial value head (the separate critic owns value)."""
        return [p for n, p in self.policy.named_parameters()
                if not n.startswith("value_head.")]

    def critic_parameters(self) -> List[nn.Parameter]:
        return list(self.critic.parameters())

    # ── checkpointing (gauntlet-loadable + full-AC restorable) ────────────────
    def state_checkpoint(self, generation: Optional[int] = None) -> dict:
        """A checkpoint dict that is BOTH a valid BC checkpoint (``model_state`` +
        ``config``, so ``model_io.load_bc_policy`` / the gauntlet can load the policy)
        AND carries the separate critic (``critic_state``) for a full actor-critic
        restore (3c.3b revert / 3c.4 resume). ``config`` is reconstructed from the
        live policy + the current encoder layout."""
        from v_dance.encoders.state_encoder import get_state_layout_version
        hidden = [m.out_features for m in self.policy.trunk if isinstance(m, nn.Linear)]
        # Reconstruct dropout from the trunk: BCPolicy only INSERTS Dropout modules when
        # dropout>0, so omitting it (default 0.0) rebuilds a dropout-free trunk whose
        # module indices DON'T match a dropout-trained checkpoint's state_dict (the 2nd
        # Linear lands at index 2 vs 3) -> load_state_dict fails. bc_best.pt has dropout=0.1.
        dropout = next((float(m.p) for m in self.policy.trunk if isinstance(m, nn.Dropout)), 0.0)
        cfg = {
            "state_dim": self.policy.state_dim, "action_dim": self.policy.action_dim,
            "hidden_dims": hidden, "dropout": dropout, "heads": list(self.head_names),
            "gimmick_dim": self.policy.gimmick_dim,
            "gimmick_heads": list(self.gimmick_head_names),
            "gimmick_trained": bool(getattr(self.policy, "_gimmick_trained", True)),
            "value_trained": bool(self.value_trained),
            "state_layout_version": get_state_layout_version(),
        }
        ck = {"model_state": self.policy.state_dict(), "config": cfg,
              "critic_state": self.critic.state_dict()}
        if generation is not None:
            ck["generation"] = int(generation)
        return ck

    def save(self, path, generation: Optional[int] = None, verify: bool = True) -> None:
        """Write a checkpoint. ``verify`` (default True) reloads it via the BC loader to
        confirm the policy round-trips — a broken save (e.g. a config/architecture
        mismatch) would otherwise only surface as a SILENT no-model fallback when the
        gauntlet later tries to load it, and the promotion gate would run on garbage."""
        torch.save(self.state_checkpoint(generation), path)
        if verify:
            model_io.load_bc_policy(path)   # raises loudly if the saved checkpoint won't load

    def restore_from(self, path, device: str = "cpu") -> None:
        """Reload BOTH policy and critic from a ``state_checkpoint`` (collapse-revert /
        resume). A plain BC checkpoint with no ``critic_state`` reverts the policy only."""
        ck = torch.load(path, map_location=device, weights_only=False)
        self.policy.load_state_dict(ck["model_state"])
        if "critic_state" in ck:
            self.critic.load_state_dict(ck["critic_state"])

    # ── hybrid CPU-collection / GPU-update (3c.8b, sec 20) ─────────────────────
    def inference_copy(self, device: str = "cpu") -> "ActorCritic":
        """A DETACHED, eval-mode deep-copy on ``device`` for COLLECTION. Per sec 20 the
        update lives on the GPU but collection runs the (tiny) model on the CPU — games are
        async so per-turn forwards can't be batched and the CPU<->GPU transfer per single
        sample would make collection slower. This copy has its OWN parameter tensors, so
        collection never touches the persistent actor-critic / optimiser graph; remake it
        each generation so it reflects the latest trained weights."""
        copyac = copy.deepcopy(self).to(device)
        copyac.eval()
        return copyac
