"""The online bot and its operations (carved out of ``v_dance.play`` by the 2026-09-10 refactor).

- ``bot``                   the ladder bot (Playwright transport, LinkWatch, lanes, recorder hooks) — ``python -m v_dance.online.bot``
- ``panel``                 the :8777 HTTP control panel (started by the bot; ``VD_*`` toggles, arms, SendGate notices)
- ``play_vs_human_browser`` the vs-human browser player — ``python -m v_dance.online.play_vs_human_browser`` (``--self-test``)
- ``send_gate``             SendGate (message pacing / Showdown limits)
- ``browser/``              ``battle_host`` — the Playwright battle host

``v_dance.play`` keeps the serve core + local harnesses. ``online`` may import ``eval`` and ``ladder``.
"""
