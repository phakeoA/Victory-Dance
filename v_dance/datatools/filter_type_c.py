"""
filter_type_c.py
================
Pre-ingest hygiene for the Type_C corpus (our REAL online games), USER request 2026-07-10:
**turn-1 forfeits / turn-1 timeouts / disconnect-aborted games teach the net nothing** — the
opponent left before any Pokémon was played — yet they sit in ``data/vods/Type_C`` and would ride
into the next era-retrain. This script classifies every recorded replay and (with ``--apply``)
**DELETES the junk** (USER: delete, not quarantine; ``--quarantine`` moves to ``_excluded/``
instead). The training pipeline is DOUBLY protected: ``bulk_parse_replays`` runs the same
``classify_log`` gate itself and skips junk (+ removes its stale exports) even if this script was
never run, and ``play_online_browser`` auto-runs this filter (delete mode) at every session exit —
so the folder stays clean with zero manual steps.

Classification (per replay's embedded battle log):
  * ``no_result``      — no ``|win|``/``|tie|`` line at all (aborted recording / dead room)
  * ``turnN_forfeit``  — a ``forfeited`` message decided the game at turn ≤ ``--max-turn``
  * ``turnN_timeout``  — a ``lost due to inactivity`` message (timer/disconnect) at turn ≤ ``--max-turn``
  * ``no_moves``       — a result with ZERO ``|move|`` lines (nothing was ever clicked)
  * ``unreadable``     — the log could not be extracted (corrupt file)
  * ``ok``             — a real game (kept)

Every run (dry or applied) rewrites ``<input>/type_c_manifest.json`` — filename → status/turns/
winner/**pre-game ratings** — the ratings audit for training-time elo valuation. The ratings come
from the log's ``|player|p1|name|avatar|RATING`` lines, which the recorded replays DO carry and
which ``replay_parser`` already lifts into every transition's ``players.pN.rating_before`` — so
``train_bc --rating-weight`` / ``--rating-min`` value Type_C by the elo the game was played at
with NO further plumbing (verified end-to-end 2026-07-10).

Usage
-----
    python -m v_dance.datatools.filter_type_c                 # DRY-RUN report (default input)
    python -m v_dance.datatools.filter_type_c --apply         # DELETE the junk
    python -m v_dance.datatools.filter_type_c --apply --quarantine   # move to _excluded/ instead

Rerunnable + auto-run at every online-session exit; still worth a manual dry-run look before an
era-retrain ingest (the manifest is the audit trail).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from v_dance.parser.vod_parser.replay_parser import (
    extract_log_from_html, extract_replay_id_from_html,
)

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _REPO / "data" / "vods" / "Type_C"
_EXCLUDED_DIRNAME = "_excluded"
_MANIFEST_NAME = "type_c_manifest.json"


def classify_log(log: str, max_turn: int = 1) -> dict:
    """Classify one battle log. Returns {status, turns, moves, winner, tie, ratings, players}.
    ``ratings`` maps username → pre-game ladder rating (int) from the |player| lines — absent
    on unrated games (challenges)."""
    turns = 0
    moves = 0
    winner = None
    tie = False
    forfeit = False
    inactivity = False
    players: list = []
    ratings: dict = {}
    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if cmd == "turn" and len(parts) > 2:
            try:
                turns = max(turns, int(parts[2]))
            except ValueError:
                pass
        elif cmd == "move":
            moves += 1
        elif cmd == "win" and len(parts) > 2:
            winner = parts[2].strip()
        elif cmd == "tie":
            tie = True
        elif cmd == "player" and len(parts) > 3 and parts[3]:
            players.append(parts[3])
            if len(parts) > 5 and parts[5]:
                try:
                    ratings[parts[3]] = int(parts[5])
                except ValueError:
                    pass
        elif cmd in ("-message", "message") and len(parts) > 2:
            txt = parts[2].lower()
            if "forfeited" in txt:
                forfeit = True
            elif "lost due to inactivity" in txt or "lost due to disconnection" in txt:
                inactivity = True

    if winner is None and not tie:
        status = "no_result"
    elif forfeit and turns <= max_turn:
        status = f"turn{turns}_forfeit"
    elif inactivity and turns <= max_turn:
        status = f"turn{turns}_timeout"
    elif moves == 0:
        status = "no_moves"
    else:
        status = "ok"
    return {"status": status, "turns": turns, "moves": moves, "winner": winner,
            "tie": tie, "ratings": ratings, "players": players}


def scan(input_dir: Path, max_turn: int = 1) -> dict:
    """Classify every replay .html directly in ``input_dir`` (``_excluded/`` is never rescanned).
    Returns filename → classification dict (+ ``battle_tag``)."""
    out: dict = {}
    for f in sorted(input_dir.glob("*.html")):
        try:
            html = f.read_text(encoding="utf-8", errors="replace")
            log = extract_log_from_html(html)
            if not log:
                raise ValueError("no battle-log-data block")
            info = classify_log(log, max_turn=max_turn)
            info["battle_tag"] = extract_replay_id_from_html(html) or ""
        except Exception as exc:
            info = {"status": "unreadable", "error": str(exc)[:200], "turns": 0, "moves": 0,
                    "winner": None, "tie": False, "ratings": {}, "players": [], "battle_tag": ""}
        out[f.name] = info
    return out


def run(input_dir: Path, max_turn: int = 1, apply: bool = False, quarantine: bool = False) -> dict:
    """Scan + report; with ``apply`` DELETE the junk (USER 2026-07-10 — deletion, not quarantine;
    pass ``quarantine=True`` to move into ``_excluded/`` instead). Returns the manifest dict."""
    results = scan(input_dir, max_turn=max_turn)
    junk = {n: i for n, i in results.items() if i["status"] != "ok"}
    kept = {n: i for n, i in results.items() if i["status"] == "ok"}

    action = None
    if apply and junk:
        action = "quarantined" if quarantine else "deleted"
        if quarantine:
            excl = input_dir / _EXCLUDED_DIRNAME
            excl.mkdir(exist_ok=True)
            for name in junk:
                shutil.move(str(input_dir / name), str(excl / name))
        else:
            for name in junk:
                (input_dir / name).unlink()

    manifest = {name: {**info,
                       "removed": bool(apply and name in junk),
                       "action": (action if (apply and name in junk) else None)}
                for name, info in results.items()}
    (input_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    # ── report ────────────────────────────────────────────────────────────────
    by_status: dict = {}
    for i in results.values():
        by_status[i["status"]] = by_status.get(i["status"], 0) + 1
    print(f"[filter_type_c] {input_dir} — {len(results)} replays scanned "
          f"(max-turn threshold {max_turn}):")
    for st in sorted(by_status):
        print(f"    {st:>16}: {by_status[st]}")
    for name, i in sorted(junk.items()):
        print(f"    JUNK {i['status']:<16} turns={i['turns']:<3} moves={i['moves']:<3} "
              f"win={str(i['winner'])[:18]:<18} {name}")
    rated = [r for i in kept.values() for r in i["ratings"].values()]
    if rated:
        rated.sort()
        print(f"    kept {len(kept)} real games — pre-game ratings on file: "
              f"n={len(rated)} min={rated[0]} med={rated[len(rated) // 2]} max={rated[-1]} "
              f"(rides into training as players.pN.rating_before)")
    unrated_kept = sum(1 for i in kept.values() if not i["ratings"])
    if unrated_kept:
        print(f"    NOTE: {unrated_kept} kept game(s) carry NO rating (unrated challenges) — "
              f"train-time --rating-weight treats them as median, --rating-min keeps them.")
    verb = ("DRY-RUN (pass --apply to delete)" if not apply
            else "QUARANTINED" if quarantine else "DELETED")
    print(f"    {verb} — {len(junk)} junk file(s)"
          f"{' → ' + str(input_dir / _EXCLUDED_DIRNAME) if apply and quarantine and junk else ''}")
    print(f"    manifest → {input_dir / _MANIFEST_NAME}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter unlearnable games out of the Type_C corpus.")
    ap.add_argument("--input", default=str(_DEFAULT_INPUT),
                    help=f"replay folder (default {_DEFAULT_INPUT})")
    ap.add_argument("--max-turn", type=int, default=1,
                    help="forfeit/timeout games ending at turn <= this are junk (default 1).")
    ap.add_argument("--apply", action="store_true",
                    help="actually DELETE the junk (default: dry-run report).")
    ap.add_argument("--quarantine", action="store_true",
                    help="with --apply: move junk to _excluded/ instead of deleting.")
    args = ap.parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"[filter_type_c] not a directory: {input_dir}")
    run(input_dir, max_turn=args.max_turn, apply=args.apply, quarantine=args.quarantine)


if __name__ == "__main__":
    main()
