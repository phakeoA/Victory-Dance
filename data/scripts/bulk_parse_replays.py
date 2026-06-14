"""
bulk_parse_replays.py
=====================
Batch-parse a folder of Pokémon Showdown replay .html files into Victory-Dance
training JSONL — the headless equivalent of the team-builder UI's Export button,
run over a whole directory at once.

Each replay becomes one ``<replay-stem>.jsonl`` file in the output directory,
with one JSON transition per line.  Type B (ranked-player VODs) export BOTH
player perspectives — each side is a valid behaviour-cloning target — exactly
like server.py's /export endpoint, so the output is identical to what the UI
produces.

The parser already handles the Zoroark (Illusion) and Ditto (Imposter/Transform)
edge cases, and belief enrichment (Pikalytics) fills opponent distributions, so
no per-replay hand-annotation is needed for Type B bulk collection.

Two ways to run it
------------------
1. **Click "Run" in VS Code** (no arguments).  Two folder pickers open in turn:
   first choose the folder of replay .html files, then choose where the .jsonl
   files should go.  That's it — it runs and pops up a summary when done.
2. **Command line** (for the big batch / automation), passing the paths:

Usage
-----
    python bulk_parse_replays.py --input  <replay_dir> \
                                 --output <jsonl_dir>  [options]

Example (the Reg-MA Type B set)::

    python bulk_parse_replays.py \
        --input  data/vods/Type_B/gen9championsvgc2026regma \
        --output data/vods/Prepared_training_data/Regulation_MA/Jsonl_TypeB

Both paths are chosen by you and are not hard-coded — point --input at whatever
regulation/folder you are processing, and --output wherever you want the JSONL.

Type A (own VODs)
-----------------
Pass ``--team <paste file>`` to inject YOUR exact stats.  Each replay's side is
auto-detected by matching the team against the teampreview rosters, so one
command does a whole per-team folder::

    python bulk_parse_replays.py \
        --team   teams/M-A/Kronomono1 \
        --input  data/vods/Type_A/gen9champions2026regma/Kronomono1 \
        --output data/vods/Prepared_training_data/Regulation_MA/Jsonl_TypeA_Kronomono1

In GUI mode (no args) a file picker offers the team after the replays folder;
Cancel it to fall back to Type B.

Options
-------
    --type / -t      VOD type (default "B", or "own_vod" when --team is given).
    --team           Showdown team-paste file for your side (Type A).
    --pikalytics / -p  Pikalytics belief JSON for opponent enrichment
                       (default: data/pikalytics_regma.json).  If absent the
                       run still succeeds; distribution blocks are just empty.
    --recursive / -r   Recurse into sub-folders of --input.
    --limit / -n N     Only process the first N replays (handy for a test run).
    --overwrite        Re-export replays whose .jsonl already exists
                       (default: skip them, so re-runs are resumable).
    --players          Comma list to force perspectives (e.g. "p1").  Defaults
                       to both for Type B, else p1.

Run it with the project venv:  .venv\\Scripts\\python.exe data\\scripts\\bulk_parse_replays.py ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ── Bootstrap: make the sibling modules importable no matter the cwd ──────────
# This file lives in data/scripts/; its siblings are belief_state.py,
# state_encoder.py and the vod_parser package.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parents[1]            # data/scripts → data → root
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from vod_parser.transitions import replay_to_transitions   # noqa: E402
from vod_parser.replay_parser import extract_log_from_html  # noqa: E402
from vod_parser.team_sheet import (                          # noqa: E402
    parse_showdown_team, team_to_known_side, detect_our_side,
)

try:
    from belief_state import BeliefState, VodType            # noqa: E402
except ImportError:                                          # pragma: no cover
    BeliefState = None        # type: ignore
    VodType = None            # type: ignore

_DEFAULT_PIKALYTICS = _PROJECT_ROOT / "data" / "pikalytics_regma.json"


def _rosters_from_html(html_text: str) -> dict:
    """Pull the teampreview rosters ({"p1": [...], "p2": [...]}) from a replay.

    Used by the Type A --team flow to auto-detect which side the player is on
    by matching the team paste against each side's revealed six.
    """
    log = extract_log_from_html(html_text)
    rosters = {"p1": [], "p2": []}
    for m in re.finditer(r"^\|poke\|(p[12])\|([^,|\n]+)", log, re.MULTILINE):
        rosters[m.group(1)].append(m.group(2).strip())
    return rosters


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_belief(pika_path: Path | None):
    """Load the Pikalytics BeliefState, or return None (graceful) if missing."""
    if BeliefState is None:
        print("[warn] belief_state.py unavailable — exporting without Pikalytics "
              "enrichment (opponent distribution blocks will be empty).")
        return None
    if not pika_path or not pika_path.exists():
        print(f"[warn] Pikalytics file not found at {pika_path} — exporting "
              "without belief enrichment.")
        return None
    print(f"[belief] loading {pika_path} ...")
    return BeliefState(pika_path)


def _players_for(source_type: str, override: str | None) -> list[str]:
    """Perspectives to export.  Type B → both ranked players; else just p1.

    Mirrors server.py /export: a Type B replay is two ranked players, so each
    side's decisions are a valid BC target and the replay yields both halves.
    """
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    is_b = True
    if VodType is not None:
        try:
            is_b = VodType.coerce(source_type or "B") is VodType.B
        except Exception:
            is_b = (source_type or "B").upper() in ("B", "RANKED_PLAYER_VOD")
    return ["p1", "p2"] if is_b else ["p1"]


def _discover(input_dir: Path, recursive: bool) -> list[Path]:
    globber = input_dir.rglob if recursive else input_dir.glob
    return sorted(globber("*.html"))


# ── GUI folder pickers (used when --input/--output are omitted) ───────────────
# A single hidden Tk root is shared across the input picker, output picker, and
# the completion dialog so we never create more than one Tk instance.
_tk_root = None


def _gui_available() -> bool:
    try:
        import tkinter  # noqa: F401
        return True
    except Exception:
        return False


def _get_root():
    global _tk_root
    if _tk_root is None:
        import tkinter as tk
        _tk_root = tk.Tk()
        _tk_root.withdraw()                      # no blank window, just dialogs
        try:
            _tk_root.attributes("-topmost", True)  # dialogs above the editor
        except Exception:
            pass
    return _tk_root


def _pick_directory(title: str, initial: Path | None = None) -> Path | None:
    """Open a folder-browser dialog; return the chosen Path, or None if cancelled."""
    from tkinter import filedialog
    root = _get_root()
    root.update()
    path = filedialog.askdirectory(
        title=title,
        mustexist=False,
        initialdir=str(initial) if initial else None,
    )
    return Path(path).resolve() if path else None


def _pick_file(title: str, initial: Path | None = None) -> Path | None:
    """Open a file-browser dialog; return the chosen Path, or None if cancelled."""
    from tkinter import filedialog
    root = _get_root()
    root.update()
    path = filedialog.askopenfilename(
        title=title,
        initialdir=str(initial) if initial else None,
    )
    return Path(path).resolve() if path else None


def _gui_done(message: str) -> None:
    """Pop a completion summary so GUI users get visual confirmation."""
    try:
        from tkinter import messagebox
        messagebox.showinfo("Bulk replay export — done", message)
    except Exception:
        pass


def _gui_close() -> None:
    global _tk_root
    if _tk_root is not None:
        try:
            _tk_root.destroy()
        except Exception:
            pass
        _tk_root = None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    # Windows consoles default to cp1252, which can't encode → or accented
    # species names (Flabébé, Nidoran♀, …) that show up in log output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(
        description="Batch-parse Showdown replay .html files into training JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", "-i", default=None,
                    help="Directory containing replay .html files. "
                         "Omit to pick it from a folder dialog (VS Code Run).")
    ap.add_argument("--output", "-o", default=None,
                    help="Directory to write <replay-stem>.jsonl files into. "
                         "Omit to pick it from a folder dialog (VS Code Run).")
    ap.add_argument("--type", "-t", default="B",
                    help='VOD type / source_type (e.g. "B", "ranked_player_vod"). '
                         'With --team this defaults to Type A ("own_vod").')
    ap.add_argument("--team", default=None,
                    help="Showdown team-paste file for YOUR side (Type A). Each "
                         "replay's side is auto-detected by matching this team "
                         "against the teampreview rosters. Omit for Type B. In "
                         "GUI mode a file picker offers this (Cancel = Type B).")
    ap.add_argument("--pikalytics", "-p", default=str(_DEFAULT_PIKALYTICS),
                    help="Pikalytics belief JSON for opponent enrichment.")
    ap.add_argument("--recursive", "-r", action="store_true",
                    help="Recurse into sub-folders of --input.")
    ap.add_argument("--limit", "-n", type=int, default=None,
                    help="Process only the first N replays (test runs).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-export replays whose .jsonl already exists.")
    ap.add_argument("--players", default=None,
                    help='Force perspectives, e.g. "p1" or "p1,p2".')
    args = ap.parse_args()

    # GUI mode: any path the user didn't pass on the command line is chosen from
    # a folder dialog.  Clicking "Run" in VS Code passes no args → both dialogs.
    gui_mode = args.input is None or args.output is None
    if gui_mode and not _gui_available():
        print("ERROR: --input/--output were omitted and no folder-picker (tkinter) "
              "is available. Pass --input and --output on the command line.",
              file=sys.stderr)
        return 2

    if args.input:
        input_dir = Path(args.input).resolve()
    else:
        print("Opening folder picker: choose the REPLAYS folder ...")
        input_dir = _pick_directory("Select the folder containing replay .html files")
        if input_dir is None:
            print("Cancelled — no replays folder selected.")
            _gui_close()
            return 1

    # Team paste (Type A): from --team, or an optional GUI picker after the
    # replays folder.  Cancelling the picker means "Type B, no team".
    team_path = None
    if args.team:
        team_path = Path(args.team).resolve()
    elif gui_mode:
        print("Opening file picker: choose your TEAM file (Cancel = Type B / no team) ...")
        team_path = _pick_file(
            "Select your team paste file for Type A  (Cancel = Type B / no team)",
            initial=input_dir,
        )

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        print("Opening folder picker: choose where to SAVE the .jsonl files ...")
        output_dir = _pick_directory(
            "Select the folder to save the per-replay .jsonl files into",
            initial=input_dir,
        )
        if output_dir is None:
            print("Cancelled — no output folder selected.")
            _gui_close()
            return 1

    if not input_dir.is_dir():
        print(f"ERROR: input is not a directory: {input_dir}", file=sys.stderr)
        _gui_close()
        return 2

    # When folders are picked via the dialog, search recursively so the user can
    # choose any ancestor folder and still find every replay beneath it.
    recursive = args.recursive or gui_mode
    replays = _discover(input_dir, recursive)
    if args.limit is not None:
        replays = replays[: args.limit]
    if not replays:
        msg = (f"No .html replays found in:\n{input_dir}"
               f"{' (searched sub-folders too)' if recursive else ''}.")
        print(msg, file=sys.stderr)
        if gui_mode:
            _gui_done(msg)
        _gui_close()
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    belief = _load_belief(Path(args.pikalytics) if args.pikalytics else None)

    # ── Type A team injection setup ──────────────────────────────────────────
    team_mode = team_path is not None
    known_side = None
    team_base: set = set()
    source_type = args.type
    if team_mode:
        if not team_path.is_file():
            print(f"ERROR: team file not found: {team_path}", file=sys.stderr)
            _gui_close()
            return 2
        mons = parse_showdown_team(team_path.read_text(encoding="utf-8"))
        if not mons:
            print(f"ERROR: parsed 0 Pokémon from team file: {team_path}", file=sys.stderr)
            _gui_close()
            return 2
        known_side = team_to_known_side(mons)
        team_base = set(known_side)
        # Importing your own team makes this a Type A own-VOD; only fill in for
        # the default — an explicit --type the user passed still wins.
        if args.type == "B":
            source_type = "own_vod"

    players = None if team_mode else _players_for(source_type, args.players)

    print(f"[plan] {len(replays)} replay(s) | type={source_type}"
          f"{' (Type A — team injected, side auto-detected)' if team_mode else ''}")
    if team_mode:
        print(f"[plan] team: {team_path.name}  "
              f"({len(team_base)} mons: {', '.join(sorted(team_base))})")
    else:
        print(f"[plan] perspectives={players}")
    print(f"[plan] in : {input_dir}")
    print(f"[plan] out: {output_dir}\n")

    n_written = n_skipped = n_empty = n_no_side = 0
    total_transitions = 0
    errors: list[tuple[str, str]] = []
    t0 = time.time()

    for idx, html in enumerate(replays, 1):
        out_path = output_dir / f"{html.stem}.jsonl"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            print(f"[{idx}/{len(replays)}] skip (exists)  {html.name}")
            continue

        side = None
        try:
            if team_mode:
                # Auto-detect which side is the player by matching the team
                # paste against each side's teampreview roster.
                rosters = _rosters_from_html(html.read_text(encoding="utf-8"))
                side = detect_our_side(rosters, team_base)
                if side is None:
                    n_no_side += 1
                    print(f"[{idx}/{len(replays)}] no team match — skipped  {html.name}")
                    continue
                known_teams = {"team": {side: known_side,
                                        "_meta": {"yourSide": side}}}
                transitions = replay_to_transitions(
                    html, belief=belief, players=[side],
                    known_teams=known_teams, source_type=source_type,
                )
            else:
                transitions = replay_to_transitions(
                    html, belief=belief, players=players, source_type=source_type,
                )
        except Exception as exc:  # one bad replay must not kill the batch
            errors.append((html.name, f"{type(exc).__name__}: {exc}"))
            print(f"[{idx}/{len(replays)}] ERROR        {html.name} — {exc}")
            continue

        if not transitions:
            n_empty += 1
            print(f"[{idx}/{len(replays)}] empty (0 turns) {html.name}")
            continue

        # Same serialization as server.py /export: one JSON object per line.
        out_path.write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in transitions),
            encoding="utf-8",
        )
        n_written += 1
        total_transitions += len(transitions)
        tag = f" [you={side}]" if team_mode else ""
        print(f"[{idx}/{len(replays)}] ok  {len(transitions):>4} tr{tag}  → {out_path.name}")

    dt = time.time() - t0
    summary = (
        f"Done in {dt:.1f}s\n"
        f"  written   : {n_written} file(s), {total_transitions} transitions\n"
        f"  skipped   : {n_skipped} (already existed; use --overwrite to redo)\n"
        + (f"  no team   : {n_no_side} (team matched neither roster)\n" if team_mode else "")
        + f"  empty     : {n_empty} (0 turns)\n"
        f"  errored   : {len(errors)}"
    )
    print("\n" + "─" * 60)
    print(summary)
    for name, msg in errors:
        print(f"      - {name}: {msg}")
    print(f"\nOutput folder: {output_dir}")
    print("─" * 60)

    if gui_mode:
        _gui_done(f"{summary}\n\nOutput folder:\n{output_dir}")
    _gui_close()
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
