"""PPO self-play package (the RL strength path).

Built per docs/ppo_reward_design.md. Implemented modules (torch-free data + advantage layer):
  schema.py       — Trajectory / Transition / EpisodeMeta / TerminalType (3a.1)
  collector.py    — TrajectoryCollector + assert_zero_sum symmetry guard (3a.2)
  reward.py       — terminal reward placement + MODEL-DRIVEN% hard-fail (3a.3)
  store.py        — Type-C jsonl store / converter + integrity guard (3a.4)
  diagnostics.py  — per-bring / per-matchup win-rates (3a.5)
  gae.py          — GAE advantages + returns, gamma=0.997 floor (3b.2)
Still to land (torch / live): rest of 3b (BC-init cloned critic, per-head log-prob, PPO losses,
warm-up, gated PBRS) and 3c (game runner, league, resumability, Type_D archive). See NEXT_SESSION_PROMPT.md.
"""
