"""Learning from the ladder (W3b) — the recorder → nightly leashed-PPO chain → bandit arm loop.

Phase 1 of the 2026-09-10 refactor moved the two CLIs here from the gitignored ``scratch/``:

- ``ppo_update``  — the nightly chain step (``python -m v_dance.ladder.ppo_update --dry-run`` / ``--run-gates --register``)
- ``bc_finetune`` — B4, the ladder-wins BC fine-tune (``python -m v_dance.ladder.bc_finetune --dry-run``)

Phase 2 brings ``v_dance.play.ladder_recorder`` and ``v_dance.selfplay.ladder_update`` here as ``recorder`` / ``update``.
Design: ``docs/w3b_ladder_ppo_design.md``.
"""
