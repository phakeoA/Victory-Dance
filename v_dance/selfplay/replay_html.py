"""Showdown replay HTML emitter — the SAME format the Showdown client's "Download replay" button
produces (and the ``data/vods/Type_B`` corpus): the raw ``|``-protocol log embedded in a
``<script class="battle-log-data">`` tag, animated by ``play.pokemonshowdown.com/js/replay-embed.js``.

This is what ``--save-replays`` writes, replacing poke-env's ``battle.save_replay`` — so the saved
gauntlet/collection replays are Showdown-native (Showdown's own log + replay template), not poke-env's
rendering. The log is the SERVER's ground-truth protocol stream (immune to poke-env's parsing
desyncs), and the page animates via the Showdown CDN client (needs internet, same as the live
spectate / a downloaded replay — by design; the user always has internet).

Pure (no poke-env / torch) — just string formatting over a ``|``-log.
"""
from __future__ import annotations

import html as _html
from typing import List, Optional

# The Showdown replay-log CSS (verbatim from a downloaded Type_B replay) so the page reads
# identically to a real Showdown replay even before replay-embed.js loads.
REPLAY_CSS = (
    "html,body {font-family:Verdana, sans-serif;font-size:10pt;margin:0;padding:0;}body{padding:12px 0;} "
    ".battle-log {font-family:Verdana, sans-serif;font-size:10pt;} .battle-log-inline "
    "{border:1px solid #AAAAAA;background:#EEF2F5;color:black;max-width:640px;margin:0 auto 80px;"
    "padding-bottom:5px;} .battle-log .inner {padding:4px 8px 0px 8px;} .battle-log .inner-preempt "
    "{padding:0 8px 4px 8px;} .battle-log .inner-after {margin-top:0.5em;} .battle-log h2 "
    "{margin:0.5em -8px;padding:4px 8px;border:1px solid #AAAAAA;background:#E0E7EA;border-left:0;"
    "border-right:0;font-family:Verdana, sans-serif;font-size:13pt;} .battle-log .chat "
    "{vertical-align:middle;padding:3px 0 3px 0;font-size:8pt;} .battle-log .chat strong "
    "{color:#40576A;} .battle-log .chat em {padding:1px 4px 1px 3px;color:#000000;font-style:normal;} "
    ".chat.mine {background:rgba(0,0,0,0.05);margin-left:-8px;margin-right:-8px;padding-left:8px;"
    "padding-right:8px;} .spoiler {color:#BBBBBB;background:#BBBBBB;padding:0px 3px;} .spoiler:hover, "
    ".spoiler:active, .spoiler-shown {color:#000000;background:#E2E2E2;padding:0px 3px;} .spoiler a "
    "{color:#BBBBBB;} .spoiler:hover a, .spoiler:active a, .spoiler-shown a {color:#2288CC;} .chat code, "
    ".chat .spoiler:hover code, .chat .spoiler:active code, .chat .spoiler-shown code "
    "{border:1px solid #C0C0C0;background:#EEEEEE;color:black;padding:0 2px;} .chat .spoiler code "
    "{border:1px solid #CCCCCC;background:#CCCCCC;color:#CCCCCC;} .battle-log .rated {padding:3px 4px;} "
    ".battle-log .rated strong {color:white;background:#89A;padding:1px 4px;border-radius:4px;} .spacer "
    "{margin-top:0.5em;} .message-announce {background:#6688AA;color:white;padding:1px 4px 2px;} "
    ".message-announce a, .broadcast-green a, .broadcast-blue a, .broadcast-red a {color:#DDEEFF;} "
    ".broadcast-green {background-color:#559955;color:white;padding:2px 4px;} .broadcast-blue "
    "{background-color:#6688AA;color:white;padding:2px 4px;} .infobox {border:1px solid #6688AA;"
    "padding:2px 4px;} .infobox-limited {max-height:200px;overflow:auto;overflow-x:hidden;} "
    ".broadcast-red {background-color:#AA5544;color:white;padding:2px 4px;} .message-learn-canlearn "
    "{font-weight:bold;color:#228822;text-decoration:underline;} .message-learn-cannotlearn "
    "{font-weight:bold;color:#CC2222;text-decoration:underline;} .message-effect-weak "
    "{font-weight:bold;color:#CC2222;} .message-effect-resist {font-weight:bold;color:#6688AA;} "
    ".message-effect-immune {font-weight:bold;color:#666666;} .message-learn-list {margin-top:0;"
    "margin-bottom:0;} .message-throttle-notice, .message-error {color:#992222;} .message-overflow, "
    ".chat small.message-overflow {font-size:0pt;} .message-overflow::before {font-size:9pt;"
    "content:'...';} .subtle {color:#3A4A66;}"
)


def parse_replay_meta(log_lines: List[str]) -> dict:
    """Pull format (``|tier|``) + player names (``|player|``) from the protocol log, for the page
    title/header — exactly what the Showdown replay header shows."""
    fmt, names = "Self-play battle", {}
    for ln in log_lines:
        p = ln.split("|")
        if len(p) >= 3 and p[1] == "tier":
            fmt = p[2]
        elif len(p) >= 4 and p[1] == "player" and p[3]:
            names[p[2]] = p[3]
    return {"format": fmt, "p1": names.get("p1", "p1"), "p2": names.get("p2", "p2")}


def render_replay_html(log_lines: List[str], *, replayid: str = "selfplay",
                       title: Optional[str] = None) -> str:
    """Render a raw ``|``-protocol log as a self-contained Showdown replay HTML page — the canonical
    ``battle-log-data`` + CDN ``replay-embed.js`` format (identical to the client's Download-replay
    output / the Type_B corpus). Animates in any browser (online)."""
    meta = parse_replay_meta(log_lines)
    fmt, p1, p2 = meta["format"], meta["p1"], meta["p2"]
    page_title = title or f"{fmt} replay: {p1} vs. {p2}"
    # `</` would close the inline <script> tag early — neutralise it (replay-embed un-escapes `\/`).
    battle_log = "\n".join(log_lines).replace("</", "<\\/")
    return (
        "<!DOCTYPE html>\n<meta charset=\"utf-8\" />\n<!-- version 1 -->\n"
        f"<title>{_html.escape(page_title)}</title>\n<style>\n{REPLAY_CSS}\n</style>\n"
        "<div class=\"wrapper replay-wrapper\" style=\"max-width:1180px;margin:0 auto\">\n"
        f"<input type=\"hidden\" name=\"replayid\" value=\"{_html.escape(replayid)}\" />\n"
        "<div class=\"battle\"></div><div class=\"battle-log\"></div>"
        "<div class=\"replay-controls\"></div><div class=\"replay-controls-2\"></div>\n"
        "<h1 style=\"font-weight:normal;text-align:center\"><strong>"
        f"{_html.escape(fmt)}</strong><br />{_html.escape(p1)} vs. {_html.escape(p2)}</h1>\n"
        f"<script type=\"text/plain\" class=\"battle-log-data\">{battle_log}</script>\n"
        "</div>\n<script>\nlet daily = Math.floor(Date.now()/1000/60/60/24);"
        "document.write('<script src=\"https://play.pokemonshowdown.com/js/"
        "replay-embed.js?version'+daily+'\"></'+'script>');\n</script>\n"
    )


def battle_replay_lines(battle) -> List[str]:
    """The raw ``|``-protocol log lines from a (poke-env) battle's accumulated ``_replay_data`` —
    each stored split-message rejoined with ``|`` (``_split_message_to_replay_event`` does the same).
    This is the SERVER's ground-truth stream, NOT poke-env's interpretation, so it's desync-proof."""
    data = getattr(battle, "_replay_data", None) or []
    return ["|".join(sm) for sm in data]
