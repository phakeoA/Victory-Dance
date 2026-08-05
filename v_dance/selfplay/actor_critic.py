"""Actor-critic initialisation FROM the BC checkpoint — task 3b.1.

The first torch-bearing module in ``self_play``. Builds the PPO actor-critic by
reusing the trained BC policy + value head (docs/ppo_reward_design.md sec 2 — the
single biggest sample-efficiency lever):

  * **Actor** = the loaded ``AttnBCPolicy`` (per-mon encoder + self-attention -> per-slot
    action heads + per-slot gimmick heads). Its own ``value_head`` is VESTIGIAL here (the
    separate critic owns the value function) and is excluded from the actor optimiser group.
  * **Critic** = an ``AttnCritic``: a *separate* deep-copy of the whole AttnBCPolicy (its
    value path is computed inside forward). Decoupling protects the BCE-calibrated value
    surface from policy-gradient drift / warm-start collapse (sec 2 default). ``id(critic) !=
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
from pathlib import Path
from typing import List, Optional, Tuple

# ── Path bootstrap: local_battle/ (for model_io) + data/scripts + BC_model ────
_REPO_ROOT = Path(__file__).resolve().parents[2]
import torch
import torch.nn as nn

import v_dance.play.model_io as model_io


class AttnCritic(nn.Module):
    """Separate value network for the AttnBCPolicy (#23 23-critic = ``shared_trunk``).

    Holds a DEEP-COPY of the whole ``AttnBCPolicy`` and reads ``value = net(x)[2]``.
    The attn value path needs the full mon-encoder + self-attention + global stack, so
    a 'separable value head' would clone nearly the entire net anyway — and the deepcopy
    gives DISJOINT parameter tensors (a critic step leaves the actor bit-identical and
    vice-versa). forward(x) -> (B,) raw win-LOGIT; ``winprob`` / ``value_pm`` map it to the
    two use-time spaces (see module docstring)."""

    def __init__(self, net: nn.Module, support: Optional[torch.Tensor] = None):
        super().__init__()
        self.net = net
        # C51 (distributional value): when ``support`` (1-D atom locations in value_pm space
        # [-1,1]) is given, the critic reads the net's per-atom head and ``value_pm`` is the
        # distribution MEAN Σ p_i z_i — STILL ∈ [-1,1], so every downstream consumer (GAE,
        # advantages, value-clip, value_space asserts, rebase) is unchanged. None = scalar critic.
        if support is not None:
            self.register_buffer("support", support.detach().clone().float())
        else:
            self.support = None

    @property
    def is_c51(self) -> bool:
        return self.support is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[2]                               # (B,) raw win-logit (scalar head)

    def winprob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def value_atoms_logits(self, x: torch.Tensor) -> torch.Tensor:
        """C51: raw per-atom value logits — (B, n_atoms). Raises on a scalar critic."""
        if self.support is None:
            raise RuntimeError("value_atoms_logits requires a C51 critic (support is None)")
        return self.net.value_atoms_logits(x)

    def value_dist(self, x: torch.Tensor) -> torch.Tensor:
        """C51: softmax categorical over the value support — (B, n_atoms). Raises on a scalar critic."""
        return torch.softmax(self.value_atoms_logits(x), dim=-1)

    def value_pm(self, x: torch.Tensor) -> torch.Tensor:
        if self.support is not None:
            return (self.value_dist(x) * self.support).sum(dim=-1)   # mean ∈ [-1,1]
        return 2.0 * torch.sigmoid(self.forward(x)) - 1.0


def _clone_critic(policy):
    """Deep-copy the BC value surface into a fresh, decoupled critic module.

    ``copy.deepcopy`` allocates NEW parameter tensors, so the critic starts identical to the
    BC value surface but shares NO tensors with the actor (a critic step leaves the actor
    bit-identical and vice-versa). The attn value path is computed INSIDE ``forward``, so the
    critic deep-copies the WHOLE policy and reads ``net(x)[2]`` (the design's shared_trunk decision)."""
    return AttnCritic(copy.deepcopy(policy))


def init_value_atoms_from_scalar(net, support, alpha: float = 2.0) -> None:
    """Warm-start a freshly-added C51 atoms head FROM the trained scalar value head. Set the atom
    logits ``L_i(x) = alpha * z_i * s(x)`` where ``s(x)`` is the scalar win-logit (``value_head``):
    then the softmax MEAN ``sum(p_i z_i)`` is a monotone-increasing function of s(x) — mass shifts
    toward +1 when the BC value is positive and toward -1 when negative (and is exactly 0 at s=0,
    since the support is symmetric). So the distributional critic STARTS aligned with the calibrated
    scalar value; the critic warm-up then sharpens the shape (docs/c51_value_head_design.md, dec. 3)."""
    z = support.detach()                                    # (N,)
    sw = net.value_head.weight.detach()[0]                  # (val_in,)
    sb = float(net.value_head.bias.detach()[0])
    with torch.no_grad():
        net.value_atoms_head.weight.copy_(alpha * z.unsqueeze(1) * sw.unsqueeze(0))   # (N, val_in)
        net.value_atoms_head.bias.copy_(alpha * z * sb)                                # (N,)


class ActorCritic(nn.Module):
    """PPO actor-critic = BC actor + a separate cloned critic (sec 2)."""

    def __init__(self, policy, critic: "AttnCritic", head_names, gimmick_head_names,
                 value_trained: bool):
        super().__init__()
        self.policy = policy            # AttnBCPolicy actor (action + gimmick heads)
        self.critic = critic            # separate value net (sec 2 default)
        self.head_names: Tuple[str, ...] = tuple(head_names)
        self.gimmick_head_names: Tuple[str, ...] = tuple(gimmick_head_names)
        self.value_trained = bool(value_trained)
        # Structural separateness guard (sec 2): policy and critic are distinct
        # objects and must not alias the same parameters. (audit: was `id(x) is not id(y)`, which
        # compares two freshly-boxed large ints and is ALWAYS True — it could never catch aliasing.)
        assert self.policy is not self.critic, "policy and critic must be distinct modules (sec 2)"

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_bc_checkpoint(cls, path, device: str = "cpu",
                           require_value_trained: bool = True,
                           n_value_atoms: int = 0, v_min: float = -1.0,
                           v_max: float = 1.0) -> "ActorCritic":
        """Load a BC checkpoint via ``model_io`` and build the actor-critic.

        ``require_value_trained`` (default True): refuse a checkpoint whose value
        head was never trained on outcome labels — initialising the critic from an
        untrained head is just xavier noise and forfeits the whole sample-efficiency
        rationale of sec 2. The production ``battle_selfplay_gen141.pt`` (and the BC anchor
        ``battle_base.pt``) have value_trained=True.

        ``n_value_atoms > 0`` builds a **C51 distributional critic** (value_loss_mode='c51'): the
        cloned critic gets a per-atom value head over a uniform support [``v_min``, ``v_max``],
        warm-started from the scalar value head (``init_value_atoms_from_scalar``). The ACTOR stays
        scalar (its value path is vestigial). ``n_value_atoms=0`` (default) is the byte-identical
        scalar critic.
        """
        _ck = torch.load(path, map_location=device, weights_only=False)   # load ONCE; reuse for the policy
        policy, head_names = model_io.load_bc_policy(path, device, _ckpt=_ck)   # build + the c51 auto-detect
        vt = model_io.value_trained(policy)
        if require_value_trained and not vt:
            raise ValueError(
                f"BC checkpoint at {path} has an UNTRAINED value head "
                f"(value_trained=False). The cloned critic would init from noise, "
                f"defeating sec 2's calibrated-critic warm-start. Pass "
                f"require_value_trained=False only for tests / a deliberate cold critic."
            )
        n_atoms, _vmin, _vmax = int(n_value_atoms), float(v_min), float(v_max)
        if n_atoms <= 0 and isinstance(_ck, dict):         # AUTO-DETECT a saved C51 critic (mp/resume/revert)
            _cfg, _cs = (_ck.get("config") or {}), (_ck.get("critic_state") or {})
            if int(_cfg.get("n_value_atoms") or 0) > 0:    # preferred: the stamped distributional config
                n_atoms = int(_cfg["n_value_atoms"])
                _vmin, _vmax = float(_cfg.get("v_min", -1.0)), float(_cfg.get("v_max", 1.0))
            elif "net.value_atoms_head.weight" in _cs:     # stamp-less c51 ckpt: infer from the saved shapes
                n_atoms = int(_cs["net.value_atoms_head.weight"].shape[0])
                if "support" in _cs:
                    _s = _cs["support"]
                    _vmin, _vmax = float(_s[0]), float(_s[-1])
        critic_net = copy.deepcopy(policy)
        support = None
        if n_atoms > 0:
            critic_net.add_value_atoms_head(n_atoms)
            support = torch.linspace(_vmin, _vmax, n_atoms)
        critic = AttnCritic(critic_net, support=support).to(device)
        if support is not None:                            # warm-start AFTER .to(device): one device
            init_value_atoms_from_scalar(critic.net, critic.support)   # overwritten by restore_from on resume/mp
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
        p = self.policy
        cfg = {
            "state_dim": p.state_dim, "action_dim": p.action_dim,
            "gimmick_dim": p.gimmick_dim,
            "heads": list(self.head_names),
            "gimmick_heads": list(self.gimmick_head_names),
            "gimmick_trained": bool(getattr(p, "_gimmick_trained", True)),
            "value_trained": bool(self.value_trained),
            "state_layout_version": get_state_layout_version(),
        }
        # #27 attn-only: stamp the AttnBCPolicy arch so model_io rebuilds the exact net.
        cfg["model_type"] = "attn"
        cfg["d_model"] = p.d_model
        cfg["n_heads"] = p.n_heads
        cfg["n_layers"] = p.n_layers
        cfg["ff_mult"] = p.ff_mult
        cfg["dropout"] = p.dropout
        cfg["value_readout"] = p.value_readout
        cfg["opp_cond"] = bool(getattr(p, "opp_cond", False))   # Level-B opp-conditioned our heads
        # Widening cores (era-4 2b caught this class: an exploiter warm-started from a
        # pair_cond target saved 528-wide our-heads with a config that rebuilt 512 —
        # the verify reload failed). Stamp EVERY head-widening flag from the live policy.
        cfg["pair_cond"] = bool(getattr(p, "pair_cond", False))
        cfg["memory_dim"] = int(getattr(p, "memory_dim", 0) or 0)
        if cfg["memory_dim"]:
            cfg["mem_layers"] = int(getattr(p, "mem_layers", 2))
            cfg["mem_heads"] = int(getattr(p, "mem_heads", 4))
            cfg["max_mem_len"] = int(getattr(p, "max_mem_len", 64))
        cfg["n_archetypes"] = int(getattr(p, "n_archetypes", 0) or 0)
        cfg["z_dim"] = int(getattr(p, "z_dim", 0) or 0)
        # C51: stamp the distributional value config so a COLD load (mp worker / resume / gauntlet)
        # rebuilds the matching critic via from_bc_checkpoint. Absent => scalar critic (back-compat).
        if getattr(self.critic, "support", None) is not None:
            _sup = self.critic.support
            cfg["n_value_atoms"] = int(_sup.numel())
            cfg["v_min"], cfg["v_max"] = float(_sup[0]), float(_sup[-1])
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
