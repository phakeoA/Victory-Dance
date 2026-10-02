"""Ladder ACCOUNTS — which Showdown login an online-bot process plays as (2026-10-02, USER: a second account for
more rated games per night, and per-account experiments — the control on one account, the experiment on the other).

One bot process = one account = one browser = one panel port range, chosen with ``--account N``:

* slot 1 = the original ``PS_USERNAME`` / ``PS_PASSWORD`` / ``PS_AVATAR``; slot N >= 2 = ``PSN_USERNAME`` /
  ``PSN_PASSWORD`` / ``PSN_AVATAR`` (the USER's spelling) or ``PS_USERNAME_N`` … (``cred_keys``). Panels: slot 1
  binds in 8777-8786, slot 2 in 8787-8796.
* Both processes share the box and its IP. Showdown matches two same-IP accounts against each other NEVER
  (``ladders.ts``: "users must have different IPs"), but its prep cap — 12 battles per 3 min — is PER IP, so the
  panels share a search ledger (``PrepLedger``) and the box as a whole stays under it.
* Every ``human_bench.jsonl`` row carries ``account`` (the userid). Rows written before 2026-10-02 have none and
  belong to slot 1 (``row_account``).
* Each account keeps its OWN bandit state, and may have its own bandit CONFIG: ``VD_BANDIT_CONFIG_N``, else
  ``config/serve_bandit_N.json`` when it exists, else the shared ``config/serve_bandit.json``.
* Per-account launch knobs: ``VD_BANDIT_PIN_N`` / ``VD_LADDER_LANES_N`` / ``VD_BANDIT_CONFIG_N`` /
  ``VD_DEFAULT_TEAM_N`` override the plain key for slot N (``knob_overrides``) — e.g. one TEAM per account to
  ladder-test a team change (USER 2026-10-02).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

_REPO = Path(__file__).resolve().parents[2]

MAX_SLOTS = 2
PANEL_PORT_BASE = 8777
PANEL_PORT_SPAN = 10                     # a panel binds the first free port in [start, start + 9]
PREP_DIR = _REPO / "artifacts" / "online" / "prep"
BANDIT_CONFIG_DIR = _REPO / "config"
ACCOUNT_KNOBS = ("VD_BANDIT_CONFIG", "VD_BANDIT_PIN", "VD_LADDER_LANES", "VD_DEFAULT_TEAM")


def read_env(path: Path = _REPO / ".env") -> dict:
    """The .env as a dict (the bot's own parser, for tools that must not import the bot); {} when missing."""
    env: dict = {}
    try:
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, _, v = ln.partition("=")
                env[k.strip()] = v.strip()
    except OSError:
        pass
    return env


def userid(name: Optional[str]) -> str:
    """Showdown's own id rule: lowercase, letters + digits only."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def slot_key(key: str, slot: int) -> str:
    return key if int(slot) == 1 else f"{key}_{int(slot)}"


def panel_ports(slot: int) -> range:
    start = PANEL_PORT_BASE + (int(slot) - 1) * PANEL_PORT_SPAN
    return range(start, start + PANEL_PORT_SPAN)


def all_panel_ports() -> range:
    return range(PANEL_PORT_BASE, PANEL_PORT_BASE + MAX_SLOTS * PANEL_PORT_SPAN)


def slot_of_port(port: int) -> Optional[int]:
    for s in range(1, MAX_SLOTS + 1):
        if int(port) in panel_ports(s):
            return s
    return None


@dataclass(frozen=True)
class Account:
    slot: int
    username: str
    password: str = field(repr=False)
    avatar: str = ""
    avatar_key: str = "PS_AVATAR"        # the .env key an in-browser avatar pick is saved under

    @property
    def userid(self) -> str:
        return userid(self.username)

    @property
    def panel_port(self) -> int:
        return panel_ports(self.slot).start

    @property
    def label(self) -> str:
        return f"account {self.slot} ({self.username})"


def cred_keys(slot: int, name: str) -> list:
    """The .env keys a login field may live under. Account 1: ``PS_<NAME>``. Account N: ``PS<N>_<NAME>`` (the
    USER's own spelling, 2026-10-02) or ``PS_<NAME>_<N>`` — the first one set wins."""
    slot = int(slot)
    return [f"PS_{name}"] if slot == 1 else [f"PS{slot}_{name}", f"PS_{name}_{slot}"]


def _cred(env: Mapping[str, str], slot: int, name: str) -> tuple:
    keys = cred_keys(slot, name)
    for k in keys:
        if (env.get(k) or "").strip():
            return k, env[k]
    return keys[0], ""


def load_account(env: Mapping[str, str], slot: int) -> Account:
    """The login for ``slot`` from the .env mapping; ValueError names the missing keys."""
    slot = int(slot)
    if not 1 <= slot <= MAX_SLOTS:
        raise ValueError(f"--account must be 1..{MAX_SLOTS} (got {slot})")
    uk, user = _cred(env, slot, "USERNAME")
    _, pw = _cred(env, slot, "PASSWORD")
    if not (user.strip() and pw):
        want = " or ".join(f"{u} / {p}" for u, p in zip(cred_keys(slot, "USERNAME"), cred_keys(slot, "PASSWORD")))
        raise ValueError(f"{want} missing from .env (account {slot})")
    ak, avatar = _cred(env, slot, "AVATAR")
    if not avatar and slot > 1:                       # not set yet: save under the username's spelling
        ak = "PS_AVATAR_%d" % slot if uk.startswith("PS_") else f"PS{slot}_AVATAR"
    return Account(slot, user.strip(), pw, avatar.strip(), ak)


def configured_slots(env: Mapping[str, str]) -> list:
    """Slots whose username AND password are set — what Mission Control offers to launch."""
    return [s for s in range(1, MAX_SLOTS + 1)
            if _cred(env, s, "USERNAME")[1].strip() and _cred(env, s, "PASSWORD")[1]]


def primary_userid(env: Mapping[str, str]) -> str:
    return userid(env.get("PS_USERNAME"))


def row_account(row: Mapping, primary: str) -> str:
    """The account a bench row belongs to — rows from before 2026-10-02 carry none = slot 1's."""
    return userid(row.get("account") or "") or primary


def knob_overrides(slot: int, lookup: Callable[[str], Optional[str]]) -> dict:
    """{plain key: value} for every ``<KNOB>_N`` set for slot N >= 2 (``lookup`` = real env, then .env)."""
    if int(slot) == 1:
        return {}
    out = {}
    for k in ACCOUNT_KNOBS:
        v = lookup(slot_key(k, slot))
        if v not in (None, ""):
            out[k] = str(v).strip()
    return out


def bandit_config_path(slot: int, lookup: Callable[[str], Optional[str]], default: Path,
                       config_dir: Path = BANDIT_CONFIG_DIR) -> Path:
    """Slot N >= 2: ``VD_BANDIT_CONFIG_N`` > ``config/serve_bandit_N.json`` (when it exists) > the shared
    config. Slot 1: ``VD_BANDIT_CONFIG`` > the shared default — exactly as before 2026-10-02."""
    slot = int(slot)
    if slot > 1:
        own = lookup(slot_key("VD_BANDIT_CONFIG", slot))
        if own:
            return _resolve(own)
        per = Path(config_dir) / f"serve_bandit_{slot}.json"
        if per.is_file():
            return per
    plain = lookup("VD_BANDIT_CONFIG")
    return _resolve(plain) if plain else Path(default)


def bandit_state_path(account: Account, fmt: str, state_dir: Path) -> Path:
    """Slot 1 keeps the pre-2026-10-02 file (``<fmt>.json``); every other account has its own, keyed by userid so a
    credential swap on slot 2 never inherits another account's arm statistics."""
    if account.slot == 1:
        return Path(state_dir) / f"{fmt}.json"
    return Path(state_dir) / f"{fmt}_{account.userid}.json"


def _resolve(p) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else (_REPO / p)


# ── shared files ─────────────────────────────────────────────────────────────
def append_jsonl(path: Path, row: dict, *, timeout: float = 5.0) -> None:
    """Append one JSON line under a cross-process lock. Two bot processes append to ``human_bench.jsonl``; Windows
    emulates O_APPEND with seek-then-write, so an unlocked append from each can land on the same offset and one
    row overwrites the other. ``filelock`` ships with torch; a lock timeout still writes (a row is never dropped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row) + "\n"
    try:
        from filelock import FileLock, Timeout
    except ImportError:                                   # pragma: no cover — torch always brings it
        FileLock = None
    if FileLock is None:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        return
    lock = FileLock(str(path) + ".lock")
    try:
        with lock.acquire(timeout=timeout):
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
            return
    except Timeout:
        pass
    with path.open("a", encoding="utf-8") as f:           # a stuck lock must not lose the row
        f.write(line)


class PrepLedger:
    """Ladder searches by EVERY bot process on this box. Showdown caps battle preps per IP (12 per 3 min) and every
    account here shares the IP, so a panel counts the other accounts' recent searches next to its own. One
    append-only file per account (``<userid>.log``, one wall-clock timestamp per line): a process writes only its
    own file, so there is no write race, and a reader that catches a half-written last line skips it. A
    check-then-send race can overshoot the shared budget by one search per extra process — the panel's ceiling
    (10) sits 2 under the server's 12 for that."""

    TAIL_BYTES = 4096                                     # ~280 timestamps — far more than one window holds

    def __init__(self, account_id: str, directory: Path = PREP_DIR, clock: Callable[[], float] = time.time):
        self.account_id = userid(account_id) or "unknown"
        self.dir = Path(directory)
        self.clock = clock

    @property
    def path(self) -> Path:
        return self.dir / f"{self.account_id}.log"

    def record(self, t: Optional[float] = None) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(f"{self.clock() if t is None else t:.3f}\n")
        except OSError:
            pass                                          # the ledger is advisory: never break a search

    def others(self, window_s: float, now: Optional[float] = None) -> int:
        """Searches the OTHER accounts sent in the last ``window_s`` seconds."""
        now = self.clock() if now is None else now
        n = 0
        try:
            files = list(self.dir.glob("*.log"))
        except OSError:
            return 0
        for p in files:
            if p.stem == self.account_id:
                continue
            try:
                if now - p.stat().st_mtime > window_s:
                    continue                              # untouched for a whole window: nothing recent
                with p.open("rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - self.TAIL_BYTES))
                    tail = f.read().decode("utf-8", errors="replace")
            except OSError:
                continue
            for ln in tail.splitlines():
                try:
                    t = float(ln)
                except ValueError:
                    continue                              # the cut first line / a half-written last one
                if 0.0 <= now - t < window_s:
                    n += 1
        return n
