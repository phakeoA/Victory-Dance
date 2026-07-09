"""
ingest_hf_logs.py
=================
Batch-ingest the VGC-Bench open battle-log dataset (HuggingFace
``cameronangliss/vgc-battle-logs``) into Victory-Dance training JSONL.

The dataset stores one JSON object per format file::

    {"<battle-id>": ["<epoch-seconds>", "<raw showdown protocol log>"], ...}

where the battle-id (``gen9championsvgc2026regmb-2635876837``) doubles as the
canonical replay id.  Every log is raw Showdown protocol — the same thing our
replay HTMLs wrap — so each record goes straight through
``log_to_transitions`` (the exact pipeline the .html bulk exporter uses) and
comes out as one ``{replay_id}.jsonl`` file per battle, both perspectives.

Only the Champions Reg M-A / M-B files are OUR format (mega-on / tera-off);
anything else in a supplied file is skipped and counted.  These logs are
OTS-filtered upstream (open team sheets, ``|showteam|`` for both sides):

  * ``--ots-mode open``   (default) — the revealed sheets are stamped into
    every snapshot as decision-time knowledge (known_item / known_ability /
    known_moves / nature; EVs stay hidden → mons stay belief-estimated) and
    belief narrows around them.  Matches what a tournament player sees.
  * ``--ots-mode closed`` — the sheets are IGNORED; opponent knowledge stays
    progressive/inferred exactly like our ladder corpus.  Output files and
    replay_ids get a ``__closed`` suffix so both regimes can coexist in one
    corpus without tripping corpus_qa's duplicate-replay_id hard fail.

The Pikalytics belief prior is auto-selected PER BATTLE from the battle-id's
era token (regma → data/pikalytics_regma.json, regmb → …_regmb.json) and
fails loud when the era's file is missing (wrong-era bake risk — mirrors
bulk_parse_replays._load_belief).

Usage
-----
    python -m v_dance.datatools.ingest_hf_logs \
        --input  data/hf_vgc_bench/logs_gen9championsvgc2026regmb.json \
        --output data/vods/Prepared_training_data/Regulation_MB/Jsonl_HF_OTS \
        [--ots-mode open] [--rated-only] [--limit N] [--overwrite] [--workers N]

Notes
-----
    * RAM: each input file is json.load'ed whole (the 550 MB regmabo3 file
      peaks at a few GB) — process one file per run if memory is tight.
    * ``--workers N`` parses battles in parallel (each worker loads its own
      BeliefState once).  Output files are per-battle → no write races.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Champions battle ids: gen9championsvgc2026regma-123 / regmabo3-123 / regmb… / regmbbo3…
_KEY_RE = re.compile(r"gen9championsvgc2026regm([ab])(bo3)?-\d+$")
_RATED_RE = re.compile(r"^\|rated", re.MULTILINE)
_WIN_RE = re.compile(r"^\|win\|", re.MULTILINE)

# era token ("a"/"b") → BeliefState, loaded once per process (workers included).
_belief_cache: dict = {}


def _pika_path_for_era(era: str) -> Path:
    return _PROJECT_ROOT / "data" / f"pikalytics_regm{era}.json"


def _get_belief(era: str):
    """Era-matched Pikalytics prior, fail-loud on a missing file.

    A silent fallback here would bake the WRONG era's usage distribution into
    the export (the regma/regmb metas differ) — same rationale as
    bulk_parse_replays._load_belief's fatal path.
    """
    b = _belief_cache.get(era)
    if b is None:
        from v_dance.parser.belief_state import BeliefState
        p = _pika_path_for_era(era)
        if not p.exists():
            raise RuntimeError(
                f"pikalytics prior missing for era regm{era}: {p} — refusing to "
                f"fall back to another era's distribution (wrong-era bake risk)"
            )
        b = BeliefState(p)
        _belief_cache[era] = b
    return b


def _process_one(args_tuple) -> tuple:
    """Parse ONE battle log → write {replay_id}.jsonl.

    Module-level + single-tuple signature so it works under Windows-spawn
    ProcessPoolExecutor.map.  Returns (rid, n_transitions, error_or_None).
    """
    rid, log, era, out_dir_s, ots_open, tp_only = args_tuple
    try:
        from v_dance.parser.vod_parser.transitions import log_to_transitions
        replay_id = rid if ots_open else f"{rid}__closed"
        transitions = log_to_transitions(
            log, replay_id,
            belief=_get_belief(era),
            players=["p1", "p2"],
            source_type="B",          # ranked-player VODs — Type B enrichment
            ots=(True if ots_open else False),
        )
        if not transitions:
            return rid, 0, None
        if tp_only:
            # Team-preview slice: keep only the FIRST transition per perspective
            # (TP training reads exactly that) and drop the bulky post-action
            # state — the MA-BO3 77.8k file becomes tractable this way.  ⚠ a
            # tp-only folder is a TEAM-PREVIEW corpus, NEVER battle-BC --data.
            firsts: dict = {}
            for t in transitions:
                p = t.get("perspective") or "p1"
                if p not in firsts:
                    firsts[p] = t
            transitions = []
            for t in firsts.values():
                t.pop("state_after_actions", None)
                t.pop("damage_events", None)
                t.pop("state_vector", None)
                transitions.append(t)
        out_path = Path(out_dir_s) / f"{replay_id}.jsonl"
        # ATOMIC write (audit 2026-07-02): a tree-killed / disk-full run must never
        # leave a half-written file under the FINAL name — the resume pass skips
        # existing files, so a truncated battle would silently stay in the corpus
        # forever (corpus_qa cannot detect a clean line-boundary truncation).
        # os.replace is atomic on NTFS; a file under the final name is complete.
        tmp_path = out_path.with_suffix(".jsonl.tmp")
        tmp_path.write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in transitions),
            encoding="utf-8",
        )
        os.replace(tmp_path, out_path)
        return rid, len(transitions), None
    except Exception as exc:      # one bad log must not kill the batch
        return rid, 0, f"{type(exc).__name__}: {exc}"


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(
        description="Ingest VGC-Bench HF battle-log JSON into training JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", "-i", nargs="+", required=True,
                    help="One or more HF log JSON files ({battle-id: [time, log]}).")
    ap.add_argument("--output", "-o", required=True,
                    help="Directory for the per-battle .jsonl files.")
    ap.add_argument("--ots-mode", choices=("open", "closed"), default="open",
                    help="open = stamp the revealed team sheets (tournament info "
                         "regime); closed = ignore them (ladder info regime, "
                         "'__closed'-suffixed output).")
    ap.add_argument("--rated-only", action="store_true",
                    help="Skip logs without a |rated line.")
    ap.add_argument("--limit", "-n", type=int, default=None,
                    help="Stop after processing N battles (test runs).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-export battles whose .jsonl already exists.")
    ap.add_argument("--workers", "-w", type=int, default=1,
                    help="Parallel parser processes (1 = in-process serial).")
    ap.add_argument("--tp-only", action="store_true",
                    help="Write only the FIRST transition per perspective (2 lines/"
                         "battle, post-action state stripped) — the team-preview "
                         "training slice. Makes the 77.8k MA-BO3 file tractable for "
                         "the TP retrain. ⚠ tp-only folders are TP corpora ONLY — "
                         "never pass them to battle-BC --data or corpus_qa.")
    args = ap.parse_args(argv)

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ots_open = args.ots_mode == "open"

    n_written = n_skipped_exist = n_skipped_fmt = n_skipped_unrated = 0
    n_skipped_nowin = n_empty = 0
    total_transitions = 0
    errors: list = []
    n_processed = 0
    t0 = time.time()

    # Pre-flight: every era present in the inputs must have its prior on disk
    # BEFORE hours of parsing start (fail-loud up front, not 30%-in).
    for f in args.input:
        name = Path(f).name
        m = re.search(r"regm([ab])", name)
        if m and not _pika_path_for_era(m.group(1)).exists():
            print(f"[FATAL] {name}: era regm{m.group(1)} prior missing: "
                  f"{_pika_path_for_era(m.group(1))}", file=sys.stderr)
            return 2

    pool = None
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=args.workers)

    # Sweep stray .tmp files from a previously-killed run (atomic-write leftovers).
    for stale in out_dir.glob("*.jsonl.tmp"):
        try:
            stale.unlink()
        except OSError:
            pass

    try:
        for in_file in args.input:
            if args.limit is not None and n_processed >= args.limit:
                break        # limit already exhausted — don't load further (large!) files
            in_path = Path(in_file).resolve()
            if not in_path.is_file():
                print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
                return 2
            print(f"[load] {in_path.name} ...")
            data = json.loads(in_path.read_text(encoding="utf-8"))
            print(f"[load] {len(data)} battle logs")

            work: list = []
            for rid in sorted(data.keys()):
                if args.limit is not None and n_processed + len(work) >= args.limit:
                    break
                m = _KEY_RE.match(rid)
                if not m:
                    n_skipped_fmt += 1
                    continue
                rec = data[rid]
                log = rec[1] if isinstance(rec, (list, tuple)) and len(rec) > 1 else None
                if not log or not _WIN_RE.search(log):
                    n_skipped_nowin += 1
                    continue
                if args.rated_only and not _RATED_RE.search(log):
                    n_skipped_unrated += 1
                    continue
                out_name = f"{rid}.jsonl" if ots_open else f"{rid}__closed.jsonl"
                if not args.overwrite and (out_dir / out_name).exists():
                    n_skipped_exist += 1
                    continue
                work.append((rid, log, m.group(1), str(out_dir), ots_open,
                             bool(args.tp_only)))

            # Release the big dict before parsing (RAM-bound machine).
            del data
            gc.collect()

            print(f"[plan] {in_path.name}: {len(work)} to parse "
                  f"(ots={args.ots_mode}, workers={args.workers})")
            results = pool.map(_process_one, work, chunksize=8) if pool \
                else map(_process_one, work)
            t_file = time.time()          # per-file rate (t0 spans loads + prior files)
            for i, (rid, n_tr, err) in enumerate(results, 1):
                n_processed += 1
                if err is not None:
                    errors.append((rid, err))
                elif n_tr == 0:
                    n_empty += 1
                else:
                    n_written += 1
                    total_transitions += n_tr
                if i % 200 == 0 or i == len(work):
                    dt = max(time.time() - t_file, 1e-6)
                    print(f"  [{i}/{len(work)}] written={n_written} "
                          f"errors={len(errors)} ({i/dt:.1f} battles/s)")
    except KeyboardInterrupt:
        # Ctrl-C (scan 2026-07-02): Executor.map has ALREADY submitted every
        # battle in `work`, so the default shutdown(wait=True) below would
        # block until thousands of queued parses drain — reads as a hang.
        # Cancel the queue and exit promptly: the atomic per-battle writes
        # guarantee every finished file is whole, and a rerun resumes.
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            pool = None
        print("\n[ingest] interrupted (Ctrl-C) — pending battles cancelled; "
              "finished files are complete (atomic); rerun to resume.",
              file=sys.stderr)
        return 130
    finally:
        if pool is not None:
            pool.shutdown()

    dt = time.time() - t0
    print("\n" + "─" * 60)
    print(f"Done in {dt:.1f}s ({args.ots_mode} team sheets)")
    print(f"  written        : {n_written} file(s), {total_transitions} transitions")
    print(f"  skipped-exists : {n_skipped_exist}")
    print(f"  skipped-format : {n_skipped_fmt} (non-Champions battle id)")
    print(f"  skipped-no-win : {n_skipped_nowin} (incomplete log)")
    if args.rated_only:
        print(f"  skipped-unrated: {n_skipped_unrated}")
    print(f"  empty          : {n_empty} (0 turns)")
    print(f"  errored        : {len(errors)}")
    for rid, msg in errors[:20]:
        print(f"      - {rid}: {msg}")
    if len(errors) > 20:
        print(f"      … and {len(errors) - 20} more")
    print(f"Output folder: {out_dir}")
    print("─" * 60)
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
