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
import time
from pathlib import Path
from typing import List, Optional

PHASES = ("starting", "collecting", "updating", "evaluating", "idle", "done")


def _numeric(d: dict) -> dict:
    """Keep only chartable scalars (bool -> 0/1) from a PPO update_stats dict."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _blank() -> dict:
    return {
        "phase": "idle", "generation": 0, "n_generations": None,
        "games_done": 0, "games_total": 0, "running_p1_winrate": None,
        "started_at": None, "hours_budget": None, "last_verdict": None,
    }


class LiveStatus:
    """Accumulates run state and writes ``status.json`` atomically on every change.

    Every mutator persists immediately so the dashboard always reads the latest
    state. Cheap (a few-hundred-byte JSON), so per-game writes during collection
    are fine.
    """

    def __init__(self, path, showdown_url: str = "http://localhost:8000", clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.data = {
            "live": False, "updated_at": None, "showdown_url": showdown_url,
            "run": _blank(), "update": {}, "active_battles": [],
        }

    # ── persistence ──────────────────────────────────────────────────────────
    def _write(self) -> None:
        self.data["updated_at"] = self.clock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)   # atomic on the same filesystem

    # ── mutators (each persists) ───────────────────────────────────────────────
    def start_run(self, n_generations: Optional[int], hours: Optional[float] = None) -> None:
        self.data["live"] = True
        self.data["run"] = _blank()
        self.data["run"].update({"phase": "starting", "n_generations": n_generations,
                                 "started_at": self.clock(), "hours_budget": hours})
        self.data["active_battles"] = []
        self._write()

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
        self._write()

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
        self._write()

    def finish_run(self) -> None:
        self.data["live"] = False
        self.data["run"]["phase"] = "done"
        self.data["active_battles"] = []
        self._write()

    def write_live_log(self, tag: str, lines, turn: int = 0) -> None:
        """Write the active battle's growing ``|``-protocol log to ``live_log.json``
        (a sibling of status.json) — the data source for the dashboard's in-page
        turn viewer (3c.6f). Atomic; compact (no indent — the log can be long)."""
        p = self.path.with_name("live_log.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"tag": tag, "turn": int(turn or 0), "updated_at": self.clock(),
                "n_lines": len(lines or []), "log": list(lines or [])}
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, p)


def read_status(path) -> Optional[dict]:
    """Read status.json (None if missing / unreadable) — used by the server/tests."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
