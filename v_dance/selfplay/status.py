"""Live run status for the self-play dashboard (3c.6e).

The self-play loop (``generation.py`` / ``game_runner.py``) writes a small
``status.json`` continuously as it runs; the dashboard server serves it and the
browser polls it every couple of seconds to show a LIVE badge, an in-generation
progress bar, the running win-rate, the latest PPO/critic health, and the list of
**active battle rooms** so the Spectate tab can embed the local Showdown client.

This module is the single source of truth for that file's schema. It is pure
(no torch / poke-env), writes ATOMICALLY (tmp + ``os.replace`` so the dashboard
never reads a half-written file), and takes an injectable ``clock`` so it is
deterministically unit-testable.

Schema (``status.json``)::

    {
      "live": bool,                     # a run is in progress
      "updated_at": float,              # unix seconds of the last write
      "showdown_url": "http://localhost:8000",
      "run": {
        "phase": "starting|collecting|updating|evaluating|idle|done",
        "generation": int,
        "n_generations": int|null,      # target (null = until stopped)
        "games_done": int, "games_total": int,
        "running_p1_winrate": float|null,
        "started_at": float|null, "hours_budget": float|null,
        "last_verdict": str|null
      },
      "update": { ...numeric PPO health (loss, kl_to_bc, explained_variance, ...) },
      "active_battles": [ {"tag": "battle-...-123", "p1": "SP1", "p2": "SP2", "turn": int} ]
    }
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

PHASES = ("starting", "collecting", "updating", "evaluating", "idle", "done")


def _numeric(d: dict) -> dict:
    """Keep only chartable scalars (bool -> 0/1) from a PPO update_stats dict. audit: DROP non-finite floats
    (NaN/inf) — e.g. opp_ce is NaN for a generation whose minibatches carried no aligned opp-action targets,
    and a bare `NaN` token in status.json is INVALID JSON that crashes the browser dashboard's JSON.parse."""
    import math
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)) and math.isfinite(v):
            out[k] = float(v)
    return out


def _blank() -> dict:
    return {
        "phase": "idle", "generation": 0, "n_generations": None,
        "games_done": 0, "games_total": 0, "running_p1_winrate": None,
        "started_at": None, "hours_budget": None, "last_verdict": None,
    }


def _atomic_write(path: Path, text: str, retries: int = 3) -> bool:
    """Write ``text`` to ``path`` atomically (UNIQUE tmp + ``os.replace``). Returns True on
    success. RESILIENT by design: under parallel collection (3c.8c) the status file is
    rewritten many times a second, and on Windows a rapidly-churned file transiently fails to
    open (antivirus scanning it / the dashboard reading it) → retry a few times, then GIVE UP
    SILENTLY. The status feed is COSMETIC and must NEVER crash the training run (a dropped
    frame just means the dashboard misses one poll). The pid-stamped tmp name avoids any
    same-name collision between writers."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    for attempt in range(max(1, retries)):
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)            # atomic on the same filesystem
            return True
        except OSError:                       # PermissionError (Windows lock) is an OSError
            time.sleep(0.01 * (attempt + 1))
    try:
        tmp.unlink()
    except OSError:
        pass
    return False


class LiveStatus:
    """Accumulates run state and writes ``status.json`` atomically on every change.

    Every mutator persists immediately so the dashboard always reads the latest
    state. Cheap (a few-hundred-byte JSON), so per-game writes during collection
    are fine.
    """

    def __init__(self, path, showdown_url: str = "http://localhost:8000", clock=time.time,
                 min_interval: float = 0.0):
        self.path = Path(path)
        self.clock = clock
        # min_interval > 0 THROTTLES the high-frequency writes (games / active-battles /
        # live-log) to at most one per this many seconds — cuts the file churn that triggers
        # the Windows lock failures under parallel collection. 0 = write every time (default,
        # so unit tests with a fake clock still observe every mutation). The live run sets 0.5.
        self.min_interval = float(min_interval)
        self._last = 0.0
        self._last_log = 0.0
        self.data = {
            "live": False, "updated_at": None, "showdown_url": showdown_url,
            "run": _blank(), "update": {}, "active_battles": [],
        }

    # ── persistence ──────────────────────────────────────────────────────────
    def _write(self, force: bool = False) -> None:
        now = self.clock()
        # throttle high-frequency writes (collection progress) unless forced (phase changes,
        # PPO health, finish — rare + important). Crash-proof: never propagates a write error.
        if not force and self.min_interval > 0 and (now - self._last) < self.min_interval:
            return
        self._last = now
        self.data["updated_at"] = now
        _atomic_write(self.path, json.dumps(self.data, indent=2))

    # ── mutators (each persists) ───────────────────────────────────────────────
    def start_run(self, n_generations: Optional[int], hours: Optional[float] = None) -> None:
        self.data["live"] = True
        self.data["run"] = _blank()
        self.data["run"].update({"phase": "starting", "n_generations": n_generations,
                                 "started_at": self.clock(), "hours_budget": hours})
        self.data["active_battles"] = []
        self._write(force=True)

    def phase(self, phase: str, *, generation: Optional[int] = None,
              games_total: Optional[int] = None) -> None:
        assert phase in PHASES, f"unknown phase {phase!r}"
        r = self.data["run"]
        r["phase"] = phase
        if generation is not None:
            r["generation"] = generation
        if games_total is not None:
            r["games_total"] = games_total
            r["games_done"] = 0
            r["running_p1_winrate"] = None
        # leaving collection/eval clears the live battle list
        if phase in ("updating", "idle", "done"):
            self.data["active_battles"] = []
        self._write(force=True)

    def games(self, games_done: int, running_p1_winrate: Optional[float] = None) -> None:
        self.data["run"]["games_done"] = int(games_done)
        if running_p1_winrate is not None:
            self.data["run"]["running_p1_winrate"] = float(running_p1_winrate)
        self._write()

    def set_active_battles(self, battles: List[dict]) -> None:
        """``battles`` = list of {tag, p1, p2, turn}. The dashboard iframes
        ``{showdown_url}/{tag}`` for each to spectate live."""
        clean = []
        for b in battles or []:
            if not b.get("tag"):
                continue
            clean.append({"tag": str(b["tag"]), "p1": b.get("p1", "p1"),
                          "p2": b.get("p2", "p2"), "turn": int(b.get("turn", 0) or 0)})
        self.data["active_battles"] = clean
        self._write()

    def set_update(self, stats: dict, last_verdict: Optional[str] = None) -> None:
        self.data["update"] = _numeric(stats)
        if last_verdict is not None:
            self.data["run"]["last_verdict"] = last_verdict
        self._write(force=True)

    def set_gen_counts(self, counts: dict) -> None:
        """fs-monitor: record THIS generation's tally of force-switch / forced-default / forfeit
        edge events (the times the model did NOT drive a decision) + accumulate a run-wide total,
        so the dashboard can confirm these stay rare. ``counts`` = {event: int}. Per-gen, rare +
        important → forced write."""
        counts = {str(k): int(v) for k, v in (counts or {}).items()}
        r = self.data["run"]
        r["fs_monitor"] = counts                                  # this generation's tally
        total = dict(r.get("fs_monitor_total") or {})
        for k, v in counts.items():
            total[k] = int(total.get(k, 0)) + v                  # run-wide cumulative
        r["fs_monitor_total"] = total
        self._write(force=True)

    def finish_run(self) -> None:
        self.data["live"] = False
        self.data["run"]["phase"] = "done"
        self.data["active_battles"] = []
        self._write(force=True)

    def write_live_log(self, tag: str, lines, turn: int = 0) -> None:
        """Write the active battle's growing ``|``-protocol log to ``live_log.json``
        (a sibling of status.json) — the data source for the dashboard's in-page turn
        viewer (3c.6f). Throttled (min_interval) + crash-proof (``_atomic_write`` swallows
        transient Windows lock failures) so the cosmetic feed never crashes collection."""
        now = self.clock()
        if self.min_interval > 0 and (now - self._last_log) < self.min_interval:
            return
        self._last_log = now
        p = self.path.with_name("live_log.json")
        data = {"tag": tag, "turn": int(turn or 0), "updated_at": now,
                "n_lines": len(lines or []), "log": list(lines or [])}
        _atomic_write(p, json.dumps(data))


def read_status(path) -> Optional[dict]:
    """Read status.json (None if missing / unreadable) — used by the server/tests."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ── multi-battle spectate: file-per-active-battle (#18) ────────────────────────
def _safe_tag(tag: str) -> str:
    """Filesystem-safe filename stem from a battle tag (``battle-gen9...-123`` → same, but any
    odd char → ``_``)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(tag)) or "battle"


# ── structured saved-replay layout (#18b): live/<run-stamp>/gen_<N>/{replays,eval} ─────────────
_GEN_RE = re.compile(r"^gen_(\d+)$")


def run_stamp(dt=None) -> str:
    """Timestamp for a run's spectate folder (``2026-06-18_14-30-05``). ``dt`` is injectable for
    tests; the live loop stamps it once at training start."""
    import datetime as _dt
    return (dt or _dt.datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")


def gen_kind_dir(run_dir, generation: int, kind: str) -> Path:
    """Per-generation spectate dir ``<run_dir>/gen_<N>/<kind>`` where ``kind`` is ``"replays"``
    (collection self-play) or ``"eval"`` (eval matches)."""
    assert kind in ("replays", "eval"), f"unknown kind {kind!r}"
    return Path(run_dir) / f"gen_{int(generation)}" / kind


def _gen_from_path(p: Path):
    for part in p.parts:
        m = _GEN_RE.match(part)
        if m:
            return int(m.group(1))
    return None


class LiveBattles:
    """File-per-active-battle spectate feed (#18). Each CONCURRENT battle writes its own
    ``<live_dir>/<tag>.json`` ({tag, p1, p2, turn, updated_at, n_lines, log}), so the dashboard
    can show SEVERAL battles at once — and, crucially, battles from SEPARATE worker processes
    (multiprocess collection #14) land in the same dir with NO shared in-memory state. Crash-proof
    + per-battle throttled like ``LiveStatus`` (the feed is cosmetic — never crash collection)."""

    def __init__(self, live_dir, clock=time.time, min_interval: float = 0.0):
        self.dir = Path(live_dir)
        self.clock = clock
        self.min_interval = float(min_interval)
        self._last: dict = {}        # tag -> last write time (per-battle throttle)

    def update(self, tag: str, *, p1: str = "p1", p2: str = "p2", turn: int = 0, log=None,
               done: bool = False) -> None:
        """Publish/refresh one battle's room + growing ``|``-log. Throttled per battle.
        ``done`` marks a finished (saved) replay so it drops out of the LIVE list."""
        if not tag:
            return
        now = self.clock()
        if self.min_interval > 0 and (now - self._last.get(tag, 0.0)) < self.min_interval:
            return
        self._last[tag] = now
        data = {"tag": str(tag), "p1": str(p1), "p2": str(p2), "turn": int(turn or 0),
                "updated_at": now, "n_lines": len(log or []), "log": list(log or []),
                "done": bool(done)}
        _atomic_write(self.dir / f"{_safe_tag(tag)}.json", json.dumps(data))

    def finalize(self, tag: str, *, p1: str = "p1", p2: str = "p2", turn: int = 0, log=None) -> None:
        """Write the battle's FINAL snapshot with ``done=True`` (a SAVED replay) — bypasses the
        throttle so the last write always lands. Used instead of ``remove()`` when saving."""
        self._last.pop(tag, None)            # clear throttle so the final write isn't skipped
        self.update(tag, p1=p1, p2=p2, turn=turn, log=log, done=True)

    def remove(self, tag: str) -> None:
        """Drop a finished battle's file so it stops showing (used when NOT saving)."""
        try:
            (self.dir / f"{_safe_tag(tag)}.json").unlink()
        except OSError:
            pass
        self._last.pop(tag, None)

    def save_html_replay(self, tag: str, battle, *, out_dir=None, label: Optional[str] = None,
                         copy_dir=None) -> Optional[Path]:
        """Save a finished battle as a real, playable Showdown **HTML** replay and DROP the transient
        live JSON. This is what ``--save-replays`` keeps (the artifact you can open in a browser),
        instead of the cosmetic live-feed JSON.

        The replay is rendered by OUR Showdown-native emitter (``replay_html.render_replay_html``):
        the battle's RAW ``|``-protocol log (the server's ground truth, read from
        ``battle._replay_data``) wrapped in the canonical Showdown replay template — the SAME format
        the client's Download-replay button / the Type_B corpus produce. This replaces poke-env's
        ``battle.save_replay`` so the saved replays are Showdown-native, not poke-env's rendering
        (the user's call: less poke-env, more Showdown). It animates via the Showdown CDN client.

        The file is ``<out_dir or self.dir>/<label_>?<tag>.html`` — ``out_dir`` (task E) lets the
        eval gauntlet file replays into per-opponent sub-folders (``eval/<kind>/``, ``eval/league/``)
        while the LIVE spectate JSON stays in ``self.dir``; ``label`` prefixes the name so it reads
        ``gen<N>_vs_<kind>_<tag>.html``. ``battle`` is DUCK-TYPED (any object exposing
        ``_replay_data``). Crash-proof: a failed save never breaks teardown, and the live JSON
        (in ``self.dir``) is removed regardless. Returns the written path, or None on failure."""
        from v_dance.selfplay.replay_html import battle_replay_lines, render_replay_html
        d = Path(out_dir) if out_dir else self.dir
        stem = f"{label}_{_safe_tag(tag)}" if label else _safe_tag(tag)
        out = d / f"{stem}.html"
        try:
            d.mkdir(parents=True, exist_ok=True)
            html = render_replay_html(battle_replay_lines(battle), replayid=_safe_tag(tag))
            out.write_text(html, encoding="utf-8")
            # Training-corpus copy (2026-07-10, USER request): a second copy of the SAME
            # Showdown-native HTML into ``copy_dir`` (data/vods/Type_C) so real games feed the
            # ingest pipeline later. The copy's name is prefixed with out_dir's basename (the
            # harness SESSION id) — local battle tags restart numbering with the server, so the
            # bare tag is not collision-safe across sessions. Copy failure never breaks the save.
            if copy_dir is not None:
                try:
                    cd = Path(copy_dir)
                    cd.mkdir(parents=True, exist_ok=True)
                    prefix = f"{d.name}_" if out_dir else ""
                    (cd / f"{prefix}{stem}.html").write_text(html, encoding="utf-8")
                except Exception:
                    pass
            return out
        except Exception:
            return None
        finally:
            self.remove(tag)            # the live JSON (in self.dir) was a LIVE-only feed — drop it


def current_live_dir(live_root):
    """The CURRENT run's LATEST-generation spectate dir — the only place live battles are. The
    dashboard scans THIS (≈ one generation of files) instead of rglob-ing every saved replay ever
    written under ``live/`` (review #18b: with --save-replays the tree grows without bound, so a
    full-tree scan every poll is a stat-storm). Falls back to ``live_root`` when there's no
    ``<stamp>/gen_<N>`` structure yet (nothing collected, or a flat layout)."""
    root = Path(live_root)
    if not root.is_dir():
        return root
    runs = sorted(p for p in root.iterdir() if p.is_dir())   # run-stamps sort chronologically
    if not runs:
        return root
    run = runs[-1]
    gens = sorted((p for p in run.iterdir() if p.is_dir() and _GEN_RE.match(p.name)),
                  key=lambda p: int(_GEN_RE.match(p.name).group(1)))
    return gens[-1] if gens else run


def latest_run_dir(live_root):
    """The most-recent run-stamp dir under ``live/`` (run stamps sort chronologically), or None."""
    root = Path(live_root)
    if not root.is_dir():
        return None
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    return runs[-1] if runs else None


# the gen-vs-gen 'league' section sorts FIRST (it's the promotion mirror + HoF the user cares about);
# the scripted sections follow alphabetically.
_SECTION_RANK = {"league": 0}


def list_saved_eval_replays(live_root, generation) -> List[dict]:
    """Saved eval replay HTMLs for ``generation`` under the CURRENT run, as an ORDERED LIST of
    ``{"section": str, "files": [{"name","mtime"}, ...]}`` — section = the ``eval/`` sub-folder
    (``league`` / ``random`` / ``max_damage`` / ``heuristic``, task E). ``league`` (the gen-vs-gen
    promotion mirror + HoF) is ordered FIRST, then scripted alpha; within a section, newest first.
    A LIST (not a dict) so the league-first order survives JSON serialization (Flask's jsonify sorts
    dict keys). Empty list if the generation has none yet. Read-only; used by the dashboard's
    saved-replay browser (#H)."""
    run = latest_run_dir(live_root)
    if run is None:
        return []
    evald = run / f"gen_{int(generation)}" / "eval"
    if not evald.is_dir():
        return []
    sections: dict = {}
    for kd in evald.iterdir():
        if not kd.is_dir():
            continue
        files = []
        for f in kd.glob("*.html"):
            try:
                files.append({"name": f.name, "mtime": f.stat().st_mtime})
            except OSError:
                pass
        if files:
            files.sort(key=lambda x: -x["mtime"])           # newest first
            sections[kd.name] = files
    ordered = sorted(sections, key=lambda k: (_SECTION_RANK.get(k, 1), k))
    return [{"section": k, "files": sections[k]} for k in ordered]


def read_live_battles(live_dir, *, stale_after: float = 45.0, clock=time.time) -> List[dict]:
    """Currently-LIVE battles under ``live_dir`` (RECURSIVELY — the structured tree
    ``<run>/gen_<N>/<replays|eval>/<tag>.json``). A cheap ``mtime`` pre-filter skips the (possibly
    thousands of) finished SAVED replays without parsing them; the rest are read, ``done`` ones
    (saved, not live) are dropped, and each is annotated with its ``gen`` + ``kind`` (replays/eval)
    from the path. Used by the dashboard server."""
    d = Path(live_dir)
    if not d.is_dir():
        return []
    now = clock()
    out: List[dict] = []
    for p in d.rglob("*.json"):
        try:
            if stale_after and (now - p.stat().st_mtime) > stale_after:   # cheap stat; skip saved/old
                continue
            b = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if b.get("done"):                                # finished (saved) replay — not live
            continue
        b["gen"] = _gen_from_path(p)
        b["kind"] = p.parent.name if p.parent.name in ("replays", "eval") else None
        out.append(b)
    out.sort(key=lambda b: ((b.get("kind") or ""), str(b.get("tag", ""))))
    return out
