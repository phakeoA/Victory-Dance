"""Serve-side bandit — the ladder as the arbiter (era-5 design §3.0, 2026-09-01).

Every candidate we want to compare live (a battle checkpoint, a team-preview checkpoint, a serve
knob such as sampling temperature or the TP near-tie epsilon) is an ARM. Before each ladder game
the panel asks the bandit for an arm, the arm's models/knobs are applied to the served player, the
game is played, and the server's rating line (OLD → NEW for our account) is the reward: one
observation of per-game rating delta for that arm. Arms are interleaved PER GAME — every blocked
comparison this project ran was confounded by band equilibration, and 2a's regression took 50
games plus a manual rollback to catch; here a −15 pp arm dies at ~40 games automatically.

Allocation = Thompson sampling over each arm's mean per-game rating delta (Normal posterior with a
weak prior), after a short round-robin warm-up so no arm is judged on zero games. Retirement = the
arm's Wilson upper bound on win rate sits ≥ ``retire_margin`` under the incumbent's win rate after
``retire_min_games`` games. The incumbent is never retired by the rule (only replaced by a human
promotion). Promotion is a HUMAN decision informed by the report (≥ 200 games per arm ≈ ±5 pp).

Byte-identical serve when disabled: ``VD_BANDIT=0`` (or no config file) → no arm is ever applied.
State persists in ``artifacts/bandit/<format>.json`` so a restart keeps the evidence.

Serve-mode PIN (2026-09-02, USER: "a toggle in Mission Control / a separate mode to ladder with a
frozen-weight version"): ``pinned`` names ONE arm that plays every game until unpinned — the frozen
mode (nothing ever trains live; the bandit only swaps FIXED checkpoints / serve knobs between games).
Set from the panel at runtime (takes effect at the next battle) or at launch via ``VD_BANDIT_PIN``
(an arm name; ``explore`` / ``0`` / ``none`` / ``off`` clears; unset keeps the persisted pin).
Pinned games still credit that arm's stats (they are real games of that arm) and carry ``pinned``
in the bench rows so a clean block can be read on its own (e.g. pin ``era5a_big6`` for 50 games).

Ladder LANES (2026-09-02, USER: "design for the 5-games-per-account limit so live sessions are
quicker"): several rated games run at once, each under the arm BOUND to its tag — every decision
swaps that arm's bundle in for its duration (``arm_scope``; the decision path is synchronous), the
warm-up counts games IN FLIGHT so five simultaneous searches still interleave the arms, and rewards
arrive per tag in any order.
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _REPO / "config" / "serve_bandit.json"
STATE_DIR = _REPO / "artifacts" / "bandit"


# ── arms ─────────────────────────────────────────────────────────────────────
@dataclass
class Arm:
    name: str
    battle_ckpt: Optional[str] = None      # None / "default" = the deployed checkpoint
    tp_ckpt: Optional[str] = None          # None / "default" = the deployed TP checkpoint
    tau: float = 0.0                       # battle-policy sampling temperature (0 = argmax)
    top_p: float = 1.0
    tp_tie_eps: Optional[float] = None     # None = leave VD_TP_TIE_EPS as launched
    incumbent: bool = False
    note: str = ""

    def uses_default(self, which: str) -> bool:
        v = self.battle_ckpt if which == "battle" else self.tp_ckpt
        return v is None or str(v).strip().lower() in ("", "default")


@dataclass
class ArmStats:
    n: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    sum_delta: float = 0.0
    sumsq_delta: float = 0.0
    retired: bool = False
    retired_reason: str = ""
    last_played: float = 0.0

    def mean_delta(self) -> float:
        return self.sum_delta / self.n if self.n else 0.0

    def win_rate(self) -> Optional[float]:
        d = self.wins + self.losses
        return (self.wins / d) if d else None


def load_arms(path: Path, *, exists=None) -> List[Arm]:
    """Read the arms config. Arms whose checkpoint file is missing are DROPPED with a note (a
    typo'd path must not silently become 'default'). ``exists`` is injectable for tests."""
    exists = exists or (lambda p: Path(p).is_file())
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    arms: List[Arm] = []
    for raw in cfg.get("arms") or []:
        a = Arm(name=str(raw["name"]), battle_ckpt=raw.get("battle_ckpt"), tp_ckpt=raw.get("tp_ckpt"),
                tau=float(raw.get("tau", 0.0) or 0.0), top_p=float(raw.get("top_p", 1.0) or 1.0),
                tp_tie_eps=(None if raw.get("tp_tie_eps") is None else float(raw["tp_tie_eps"])),
                incumbent=bool(raw.get("incumbent", False)), note=str(raw.get("note", "")))
        missing = [w for w, v in (("battle", a.battle_ckpt), ("tp", a.tp_ckpt))
                   if not a.uses_default(w) and not exists(_resolve(v))]
        if missing:
            print(f"[bandit] arm {a.name!r} DROPPED — missing {missing} checkpoint file(s)")
            continue
        arms.append(a)
    if arms and not any(a.incumbent for a in arms):
        arms[0].incumbent = True
    return arms


def config_params(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: cfg[k] for k in ("prior_games", "prior_sd", "min_games", "retire_min_games",
                                "retire_margin") if k in cfg}


def _resolve(p) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else (_REPO / p)


# ── applying an arm to the served player ─────────────────────────────────────
def load_bundle(arm: Arm, cache: dict, *, default_battle, default_tp, device: str = "cpu",
                seed: int = 0, loader=None) -> dict:
    """Everything an arm needs at decision time — the battle net + heads, the team chooser (+ vocab
    / cfg) and the serve knobs — loaded ONCE per checkpoint path (``cache``). 2026-09-02 (lanes):
    bundles are also resolved PER BATTLE TAG while several games run at once, so a bundle is
    self-contained (its own sampling RNG included). ``loader`` is injectable: ``(kind, path) ->
    loaded tuple`` (tests)."""
    import numpy as np
    if loader is None:
        from v_dance.play import model_io as _M

        def loader(kind, path):
            return (_M.load_bc_policy(path, device) if kind == "battle"
                    else _M.load_team_chooser(path, device))
    bpath = Path(default_battle) if arm.uses_default("battle") else _resolve(arm.battle_ckpt)
    tpath = Path(default_tp) if arm.uses_default("tp") else _resolve(arm.tp_ckpt)
    bkey, tkey = ("battle", str(bpath)), ("tp", str(tpath))
    if bkey not in cache:
        cache[bkey] = loader("battle", bpath)
    if tkey not in cache:
        cache[tkey] = loader("tp", tpath)
    model, heads = cache[bkey]
    chooser, vocab, cfg = cache[tkey]
    return {"name": arm.name, "model": model, "heads": heads, "chooser": chooser, "vocab": vocab,
            "cfg": cfg, "tau": float(arm.tau), "top_p": float(arm.top_p),
            "tp_tie_eps": arm.tp_tie_eps,
            "rng": (np.random.default_rng(seed) if arm.tau > 0.0 else None)}


_PLAYER_FIELDS = ("_model", "_model_heads", "_team_chooser", "_tc_vocab", "_tc_cfg",
                  "_temperature", "_top_p", "_rng", "_arm_name")


def _bundle_values(b: dict) -> tuple:
    return (b["model"], b["heads"], b["chooser"], b["vocab"], b["cfg"], b["tau"], b["top_p"],
            b["rng"], b["name"])


def apply_bundle(player, bundle: dict) -> None:
    """Make ``bundle`` the player's DEFAULT serve stack (what a game without a per-tag resolution
    plays, and what the player reports as its current arm)."""
    for k, v in zip(_PLAYER_FIELDS, _bundle_values(bundle)):
        setattr(player, k, v)
    if bundle["tp_tie_eps"] is not None:              # model_io reads it per decision
        os.environ["VD_TP_TIE_EPS"] = str(bundle["tp_tie_eps"])


def apply_arm(host, arm: Arm, cache: dict, *, default_battle, default_tp, device: str = "cpu",
              seed: int = 0, loader=None) -> None:
    """Swap the served ``VGCPlayer``'s model handles + serve knobs to ``arm`` — the one-lane path,
    and the DEFAULT stack under lanes (live games resolve their own arm via ``arm_scope``).
    Models load ONCE per checkpoint path (``cache``)."""
    apply_bundle(host.player, load_bundle(arm, cache, default_battle=default_battle,
                                         default_tp=default_tp, device=device, seed=seed,
                                         loader=loader))


@contextmanager
def arm_scope(player, tag: str):
    """2026-09-02 (lanes): decide THIS battle under the arm bound to its tag. The online harness
    installs ``player._arm_resolver`` (``tag -> bundle | None``) when the bandit is on; with
    several games open at once the player's default stack may belong to another game, so each
    decision swaps the bound bundle in for its duration and restores afterwards (the decision
    path is synchronous — nothing interleaves inside the scope). No resolver / no bundle → no-op,
    byte-identical to the one-lane serve."""
    resolver = getattr(player, "_arm_resolver", None)
    bundle = None
    if resolver is not None:
        try:
            bundle = resolver(tag)
        except Exception:
            bundle = None
    if not bundle:
        yield None
        return
    saved = tuple(getattr(player, k, None) for k in _PLAYER_FIELDS)
    saved_eps = os.environ.get("VD_TP_TIE_EPS")
    apply_bundle(player, bundle)
    try:
        yield bundle
    finally:
        for k, v in zip(_PLAYER_FIELDS, saved):
            setattr(player, k, v)
        if bundle["tp_tie_eps"] is not None:
            if saved_eps is None:
                os.environ.pop("VD_TP_TIE_EPS", None)
            else:
                os.environ["VD_TP_TIE_EPS"] = saved_eps


# ── the bandit ───────────────────────────────────────────────────────────────
class ServeBandit:
    def __init__(self, arms: List[Arm], *, fmt: str, state_path: Optional[Path] = None,
                 applier: Optional[Callable[[Arm], None]] = None, seed: int = 0,
                 prior_games: float = 5.0, prior_sd: float = 25.0, min_games: int = 8,
                 retire_min_games: int = 40, retire_margin: float = 0.10, now=None):
        import numpy as np
        if not arms:
            raise ValueError("ServeBandit needs at least one arm")
        self.arms = list(arms)
        self.by_name = {a.name: a for a in self.arms}
        if len(self.by_name) != len(self.arms):
            raise ValueError("duplicate arm names")
        self.fmt = fmt
        self.state_path = Path(state_path) if state_path else (STATE_DIR / f"{fmt}.json")
        self.applier = applier
        self.rng = np.random.default_rng(seed)
        self.prior_games = float(prior_games)
        self.prior_sd = float(prior_sd)
        self.min_games = int(min_games)
        self.retire_min_games = int(retire_min_games)
        self.retire_margin = float(retire_margin)
        self._now = now or time.time
        self.stats: Dict[str, ArmStats] = {a.name: ArmStats() for a in self.arms}
        self.pending: Optional[str] = None          # arm applied for the NEXT battle
        self.by_tag: Dict[str, str] = {}            # base tag -> arm name
        self.current: Optional[str] = None          # arm currently applied to the player
        self.pinned: Optional[str] = None           # serve-mode pin: this arm plays EVERY game (frozen)
        # 2026-09-02 (lanes): games bound but not yet rewarded, per arm — the warm-up counts them so
        # five simultaneous searches still rotate the arms; the tags that were pinned when bound.
        self.in_flight: Dict[str, int] = {a.name: 0 for a in self.arms}
        self.pinned_tags: List[str] = []
        self.load()

    # -- persistence --
    def load(self) -> None:
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for name, s in (d.get("stats") or {}).items():
            if name in self.stats:
                self.stats[name] = ArmStats(**{k: s.get(k, getattr(ArmStats(), k))
                                               for k in ArmStats.__dataclass_fields__})
        self.by_tag = dict(list((d.get("by_tag") or {}).items())[-200:])
        pin = d.get("pinned")                       # 2026-09-02: the frozen mode survives a restart
        self.pinned = pin if (isinstance(pin, str) and pin in self.by_name) else None
        self.pinned_tags = [t for t in (d.get("pinned_tags") or []) if isinstance(t, str)][-200:]

    def save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"fmt": self.fmt, "saved": self._now(),
                                       "stats": {k: asdict(v) for k, v in self.stats.items()},
                                       "by_tag": dict(list(self.by_tag.items())[-200:]),
                                       "pinned": self.pinned,
                                       "pinned_tags": self.pinned_tags[-200:]},
                                      indent=1), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except Exception:
            pass                                    # persistence must never break play

    # -- queries --
    @property
    def incumbent(self) -> Arm:
        return next((a for a in self.arms if a.incumbent), self.arms[0])

    def active_arms(self) -> List[Arm]:
        return [a for a in self.arms if not self.stats[a.name].retired]

    @staticmethod
    def base_tag(tag: str) -> str:
        parts = (tag or "").lstrip(">").split("-")
        return "-".join(parts[:3]) if len(parts) >= 3 else (tag or "")

    # -- allocation --
    def choose(self) -> Arm:
        """The arm for the NEXT battle. Idempotent while that battle has not started (a second
        call before ``bind`` returns the same arm — the search and challenge paths both ask)."""
        if self.pinned and self.pinned in self.by_name:   # FROZEN mode: the pin beats everything,
            self.pending = self.pinned                     # a retired flag included (human choice)
            return self.by_name[self.pinned]
        if self.pending and self.pending in self.by_name and not self.stats[self.pending].retired:
            return self.by_name[self.pending]
        active = self.active_arms() or [self.incumbent]

        def played(a):                              # rewarded + in flight (lanes, 2026-09-02)
            return self.stats[a.name].n + self.in_flight.get(a.name, 0)
        under = [a for a in active if played(a) < self.min_games]
        if under:                                   # warm-up: fewest games first, config order
            arm = min(under, key=lambda a: (played(a), self.arms.index(a)))
        else:                                       # Thompson over mean per-game rating delta
            best, best_s = None, -math.inf
            for a in active:
                s = self.stats[a.name]
                n_eff = self.prior_games + s.n
                mean = s.sum_delta / n_eff
                sd = self.prior_sd / math.sqrt(n_eff)
                sample = float(self.rng.normal(mean, sd))
                if sample > best_s:
                    best, best_s = a, sample
            arm = best or self.incumbent
        self.pending = arm.name
        return arm

    def apply_pending(self) -> Optional[Arm]:
        """Apply the pending arm to the served player (via ``applier``); no-op without one."""
        arm = self.choose()
        if self.applier is not None and self.current != arm.name:
            self.applier(arm)
        self.current = arm.name
        return arm

    def pin(self, name: Optional[str]) -> Optional[str]:
        """Serve-mode pin (2026-09-02). ``name`` = an arm → that arm plays every game from the NEXT
        battle on (frozen mode); ``None`` / "" → unpin (Thompson explores again). Unknown names
        raise — a typo must never silently become 'explore'. A change drops the pending choice so
        the next ``apply_pending`` re-picks; persisted with the state."""
        name = (name or "").strip() or None
        if name is not None and name not in self.by_name:
            raise ValueError(f"unknown arm {name!r} — arms: {', '.join(self.by_name)}")
        if name == self.pinned:
            return self.pinned
        self.pinned = name
        self.pending = None
        self.save()
        return self.pinned

    def bind(self, tag: str) -> Optional[str]:
        """A battle started: attribute it to the pending arm (else to the current one)."""
        name = self.pending or self.current
        if not name:
            return None
        base = self.base_tag(tag)
        if base not in self.by_tag:
            self.by_tag[base] = name
            if len(self.by_tag) > 512:
                for k in list(self.by_tag)[:-256]:
                    self.by_tag.pop(k, None)
            self.in_flight[name] = self.in_flight.get(name, 0) + 1
            if self.pinned == name:                 # remember: this game was a frozen-mode game
                self.pinned_tags.append(base)
                del self.pinned_tags[:-200]
            self.save()                             # by_tag + pinned_tags survive a restart mid-game
        self.pending = None
        return self.by_tag[base]

    def arm_for(self, tag: str) -> Optional[str]:
        return self.by_tag.get(self.base_tag(tag))

    def pinned_for(self, tag: str) -> bool:
        """Was this battle bound while its arm was pinned (a frozen-mode game)?"""
        return self.base_tag(tag) in self.pinned_tags

    # -- reward --
    def observe(self, tag: str, delta: float, won: Optional[bool] = None) -> Optional[str]:
        """Our post-game rating change for ``tag`` → one observation for its arm."""
        name = self.arm_for(tag)
        if name is None or name not in self.stats:
            return None
        s = self.stats[name]
        s.n += 1
        self.in_flight[name] = max(0, self.in_flight.get(name, 0) - 1)
        s.sum_delta += float(delta)
        s.sumsq_delta += float(delta) ** 2
        if won is None:
            won = (delta > 0) if delta != 0 else None
        if won is True:
            s.wins += 1
        elif won is False:
            s.losses += 1
        else:
            s.draws += 1
        s.last_played = self._now()
        self._maybe_retire()
        self.save()
        return name

    def _maybe_retire(self) -> None:
        from v_dance.selfplay.gate import wilson_upper_bound
        inc = self.stats[self.incumbent.name]
        inc_wr = inc.win_rate()
        if inc_wr is None or (inc.wins + inc.losses) < self.retire_min_games:
            return                                  # nothing to compare against yet
        for a in self.arms:
            s = self.stats[a.name]
            if a.incumbent or s.retired:
                continue
            g = s.wins + s.losses
            if g < self.retire_min_games:
                continue
            ub = wilson_upper_bound(s.wins, g, z=1.645)
            if ub <= inc_wr - self.retire_margin:
                s.retired = True
                s.retired_reason = (f"WR upper bound {ub:.2f} ≤ incumbent {inc_wr:.2f} − "
                                    f"{self.retire_margin:.2f} after {g} games")

    # -- reporting --
    def summary(self) -> List[dict]:
        out = []
        for a in self.arms:
            s = self.stats[a.name]
            wr = s.win_rate()
            out.append({"name": a.name, "incumbent": a.incumbent, "n": s.n, "wins": s.wins,
                        "losses": s.losses, "draws": s.draws,
                        "mean_delta": round(s.mean_delta(), 1), "win_rate": (None if wr is None else round(wr, 3)),
                        "retired": s.retired, "reason": s.retired_reason,
                        "pending": (self.pending == a.name), "current": (self.current == a.name),
                        "pinned": (self.pinned == a.name),
                        "in_flight": self.in_flight.get(a.name, 0),
                        "tau": a.tau, "tp_tie_eps": a.tp_tie_eps, "note": a.note})
        return out

    def banner(self) -> str:
        active = [a.name for a in self.active_arms()]
        pin = f"; PINNED → {self.pinned} (frozen: every game until unpinned)" if self.pinned else ""
        return (f"[online] serve BANDIT ACTIVE — {len(active)} arm(s), incumbent {self.incumbent.name}: "
                f"{', '.join(active)}; warm-up {self.min_games} g/arm, retire at ≥{self.retire_min_games} g "
                f"if WR upper bound ≤ incumbent − {self.retire_margin:.2f}; state {self.state_path.name}{pin}")


# ── launch-time pin from the environment (VD_BANDIT_PIN) ─────────────────────
_ENV_PIN_CLEAR = {"0", "none", "off", "explore", "bandit", "unpin"}


def apply_env_pin(bandit: "ServeBandit", raw: Optional[str]) -> str:
    """``VD_BANDIT_PIN`` at launch: unset/blank → keep whatever the state file holds; ``explore`` /
    ``0`` / ``none`` / ``off`` → clear the pin; an arm name → pin it. An unknown name is refused
    LOUDLY and the persisted pin stays (a typo must not silently become 'explore'). Returns a
    one-line note for the launch log ('' = nothing to say)."""
    raw = (raw or "").strip()
    if not raw:
        return (f"[bandit] pin restored from state: {bandit.pinned} (frozen — every game)"
                if bandit.pinned else "")
    if raw.lower() in _ENV_PIN_CLEAR:
        had = bandit.pinned
        bandit.pin(None)
        return f"[bandit] VD_BANDIT_PIN={raw}: pin cleared (was {had}) — exploring" if had else ""
    try:
        bandit.pin(raw)
    except ValueError as exc:
        return f"[bandit] VD_BANDIT_PIN={raw!r} IGNORED — {exc}"
    return f"[bandit] VD_BANDIT_PIN={raw}: PINNED (frozen — every game until unpinned)"


# ── the site's official numbers (the true "elo reflector") ───────────────────
SITE_USER_AGENT = "Mozilla/5.0 (compatible; Victory-Dance/1.0; VGC ladder bot; profile-rating poll)"


def fetch_official_ratings(userid: str, timeout: float = 6.0, opener=None) -> dict:
    """``https://pokemonshowdown.com/users/<userid>.json`` → {format_id: {elo, gxe, glicko,
    glicko_dev, w, l}}. The site's own numbers (Elo shown on the profile, GXE, Glicko-1 with its
    deviation, lifetime W/L per format). ``opener`` injectable for tests."""
    import urllib.request
    url = f"https://pokemonshowdown.com/users/{userid}.json"
    if opener is None:
        def opener(u):
            # 2026-09-02 live check: the site answers 403 to the default "Python-urllib/x.y"
            # User-Agent (Cloudflare bot rule) and 200 to anything else — send our own.
            req = urllib.request.Request(u, headers={"User-Agent": SITE_USER_AGENT,
                                                     "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:    # noqa: S310 (fixed host)
                return r.read().decode("utf-8")
    d = json.loads(opener(url))
    out = {}
    for fmt, r in (d.get("ratings") or {}).items():
        try:
            out[fmt] = {"elo": int(round(float(r.get("elo") or 0))),
                        "gxe": (None if r.get("gxe") is None else float(r["gxe"])),
                        "glicko": (None if r.get("rpr") is None else int(round(float(r["rpr"])))),
                        "glicko_dev": (None if r.get("rprd") is None else int(round(float(r["rprd"])))),
                        "w": int(r.get("w") or 0), "l": int(r.get("l") or 0)}
        except (TypeError, ValueError):
            continue
    return out
