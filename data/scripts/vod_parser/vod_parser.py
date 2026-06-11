"""
vod_parser.py
=============
Parses a Pokémon Showdown replay .html file into the Victory-Dance training JSON schema.

Supports Type B VODs (Ranked Player replays):
  - Both sides are public information extracted from the replay log.
  - Exact EVs/IVs are unknown; damage events are recorded so a back-calculator
    can later narrow down spread distributions.
  - `predicted_action_by_bot` is always null (bot wasn't present).
  - `source_type` is always "ranked_player_vod".

Usage:
    python vod_parser.py <replay.html> [--out output.json] [--player p1|p2]

The `--player` flag marks which side is "our" side vs. opponent.
If omitted, defaults to p1.

Module layout
-------------
  battle_models.py  — PokemonSlot, SideConditions, FieldConditions dataclasses
  replay_parser.py  — ShowdownReplayParser, extract_log_from_html,
                       extract_replay_id_from_html
  transitions.py    — parse_replay_for_preview, replay_to_transitions
  vod_parser.py     — (this file) re-exports the public API + CLI entry point
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Bootstrap: allow running this file directly (python vod_parser/vod_parser.py)
# by putting data/scripts on sys.path so `import vod_parser` resolves to the
# package.  When imported normally this is a no-op.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# ── Public re-exports (server.py and other callers import from here) ─────────
from vod_parser.battle_models import PokemonSlot, SideConditions, FieldConditions
from vod_parser.replay_parser import (
    ShowdownReplayParser,
    extract_log_from_html,
    extract_replay_id_from_html,
)
from vod_parser.transitions import parse_replay_for_preview, replay_to_transitions

__all__ = [
    # models
    "PokemonSlot",
    "SideConditions",
    "FieldConditions",
    # parser
    "ShowdownReplayParser",
    "extract_log_from_html",
    "extract_replay_id_from_html",
    # transitions / server API
    "parse_replay_for_preview",
    "replay_to_transitions",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Parse a Showdown replay HTML into Victory-Dance JSON.")
    ap.add_argument("replay_html", help="Path to the .html replay file")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <replay>.json)")
    ap.add_argument(
        "--player",
        choices=["p1", "p2"],
        default="p1",
        help="Which side is 'our' side (default: p1)",
    )
    args = ap.parse_args()

    src = Path(args.replay_html)
    if not src.exists():
        print(f"ERROR: File not found: {src}", file=sys.stderr)
        sys.exit(1)

    html = src.read_text(encoding="utf-8")

    log = extract_log_from_html(html)
    replay_id = extract_replay_id_from_html(html)

    parser = ShowdownReplayParser(log, our_player=args.player)
    result = parser.parse()
    result["replay_id"] = replay_id
    result["source_file"] = src.name

    out_path = Path(args.out) if args.out else src.with_suffix(".json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(result['turns'])} turns → {out_path}")


if __name__ == "__main__":
    main()
