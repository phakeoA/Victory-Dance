"""v_dance.play.browser — "AI plays THROUGH the browser" transport.

The bot's DECISION core (encoder + model_io + gen141 + SBDA TP) is transport-agnostic: it consumes a
poke-env battle and emits a ``/choose`` string. This package adds a SECOND transport (a Playwright-driven
browser tab) alongside the existing websocket player, reusing that decision core unchanged.

Reusable building blocks (used by BOTH the LOCAL browser mode and the future ONLINE mode):
  * ``battle_host.BattleHost`` — drives the production ``VGCPlayer`` decision pipeline from RAW protocol
    frames with NO live websocket (transport supplies frames, ships the captured ``/choose``).

See ``scratch/play_vs_human_browser_plan.md`` (local) and ``scratch/online_multitransport_plan.md`` (online).
"""
