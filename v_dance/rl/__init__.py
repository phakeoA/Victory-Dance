"""PPO core — the format-agnostic RL machinery (refactor Phase 2, 2026-09-10; formerly in ``v_dance.selfplay``).

- ``schema``        Transition / Trajectory / EpisodeMeta (+ ``PASS_ACTION``) — the trajectory record
- ``collector``     TrajectoryCollector (per-battle step capture, pair alignment)
- ``store``         JSON-lines trajectory store (``write_trajectories`` / ``iter_trajectories``)
- ``reward``        the reward bible: terminal ±1 only, model-driven-source accounting
- ``gae``           GAE + the default gamma / lambda
- ``pbrs``          gated potential-based shaping off the win-prob value head
- ``value_space``   the value-space identities the trainer asserts
- ``actor_critic``  ActorCritic (BC-net backbone + critic head; ``from_bc_checkpoint``)
- ``policy_eval``   the pair-mode evaluator (the served 2b decode) + legality asserts
- ``ppo``           PPOConfig, the loss, the KL-to-reference leash
- ``trainer``       PPOTrainer / TrainConfig (the update loop)

Consumers: ``v_dance.selfplay`` (the era loop + collection) and ``v_dance.ladder`` (W3b). The old
``v_dance.selfplay.<module>`` paths are ``sys.modules`` shims until Phase 6.
"""
