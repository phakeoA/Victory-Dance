"""The three UI servers + their static web assets.

- ``mission_control``      Mission Control, http://127.0.0.1:8990 — ``python -m v_dance.ui.mission_control [--no-browser --port 8990]``
- ``dashboard_server``     the self-play dashboard, :5175 (MC's Dashboard tab starts it) — ``python -m v_dance.ui.dashboard_server``
- ``team_builder_server``  the Flask team builder, :5174 — ``python -m v_dance.ui.team_builder_server``
- ``static/{mission_control, dashboard, team_builder}/``  the html / js / css each server serves

Path constants: ``mission_control._HTML_PATH``, ``dashboard_server._DASH_DIR``, ``team_builder_server._UI_DIR`` all resolve
under ``static/`` next to this file. The scrapers live in ``v_dance/datatools/scrapers/`` (scripts, run by path).
"""
