"""Learning from the ladder (W3b) — the recorder → nightly leashed-PPO chain → bandit arm loop.

- ``recorder``    the online bot's trajectory recorder (W3b-0; was ``v_dance.play.ladder_recorder`` until refactor Phase 2)
- ``update``      selection · leashed PPO step · gates · registration (W3b-2; was ``v_dance.selfplay.ladder_update``)
- ``ppo_update``  the nightly chain CLI (``python -m v_dance.ladder.ppo_update --dry-run`` / ``--run-gates --register``)
- ``bc_finetune`` B4, the ladder-wins BC fine-tune CLI (``python -m v_dance.ladder.bc_finetune --dry-run``)

Phase 1 (2026-09-10) moved the two CLIs here from the gitignored ``scratch/``; Phase 2 moved the two library modules
(the old paths are ``sys.modules`` shims until Phase 6). Design: ``docs/w3b_ladder_ppo_design.md``.
"""
