"""The online bot and its operations (refactor Phase 3, 2026-09-10; formerly in ``v_dance.play``).

- ``play_online_browser``  the ladder bot (Playwright transport, LinkWatch, lanes, recorder hooks) — ``python -m v_dance.online.play_online_browser``
- ``play_vs_human_browser`` the vs-human browser player (headed / ``--self-test``)
- ``play_ladder``           the poke-env websocket ladder runner
- ``bot_control_ui``        the :8777 HTTP control panel
- ``send_gate``             SendGate (message pacing / Showdown limits)
- ``browser/``              ``battle_host`` — the Playwright battle host

``v_dance.play`` keeps the serve core + local harnesses. The old ``v_dance.play.<module>`` paths are ``sys.modules`` shims
until Phase 6 (they also still run under ``python -m``). ``online`` may import ``eval`` and ``ladder``.
"""
