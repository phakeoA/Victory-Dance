"""Learning from the ladder (W3b) — the recorder → nightly leashed-PPO chain → bandit arm loop.

- ``recorder``    the online bot's trajectory recorder (W3b-0)
- ``update``      selection · leashed PPO step · gates · registration (W3b-2)
- ``ppo_update``  the nightly chain CLI (``python -m v_dance.ladder.ppo_update --dry-run`` / ``--run-gates --register``)

Design: ``docs/w3b_ladder_ppo_design.md``. Assembled here by the 2026-09-10 refactor (Phases 1-2).
"""
