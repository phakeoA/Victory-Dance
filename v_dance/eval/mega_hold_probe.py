"""MEGA-HOLD PROBE (2026-10-03, USER: "a check first: measure how often the bot holds Tyranitar's mega when the
opponent has a rain setter in the back").

Delaying Tyranitar's mega against rain is standard play: Mega Tyranitar re-triggers Sand Stream when it evolves, so a
mega held until THEIR rain is up wipes the rain (the 10-03 rain-leak report: Archaludon's Electro Shot KO'd one of
ours 99 times under rain vs 4 under our sand). This probe asks every checkpoint what it would do in the REAL ladder
situations the bot met.

Source: the ladder recorder's games (``artifacts/ladder_rl/<fmt>/*.jsonl`` — every MODEL decision's encoded state +
its gimmick legal masks) joined to their replays (``data/vods/Type_C``). Every TURN decision where ``--mon`` (our
Tyranitar) is on the field and may mega (its gimmick mask) is labelled from the replay at the start of that turn:

  · hold        — a ``--setters`` mon (Pelipper / Politoed / Kyogre) is in their six, alive and NOT on the field, and
                  the weather is not ``--weather`` (rain) → the standard play HOLDS the mega;
                  ``hold:seen`` = that setter already took the field once, ``hold:unseen`` = it may not even be brought;
  · reset       — the weather IS rain → mega NOW (the mega's Sand Stream wipes the rain);
  · setter_out  — the setter is on the field and the weather is not rain (we re-set sand over it) — reported apart;
  · free        — no living setter in their six → nothing to wait for.

Per checkpoint it reports P(mega) (masked softmax at tau 1 over the recorded gimmick mask) and the share of
decisions whose ARGMAX is mega (what a tau-0 serve would do); ``recorded`` = what the served arm actually played.
A checkpoint that learned the play shows LOW P(mega) under ``hold`` and HIGH under ``reset``. Descriptive, read-only.

GUARD (USER 10-03: "I'm afraid if the bot learns that, it will mess up with other megas. Such as mega salamence"):
``--guard salamence`` adds ``guard:<mon>`` rows — every decision where that mon is active and may mega, whatever
the weather. Its P(mega) must NOT fall from the baseline: holding is a weather-mega play, not a Salamence one.

STATE LAYOUT: a recorded state is only usable in the layout it was encoded in. The live encoder grew from 5,057 to
6,377 dims on 2026-10-02 (sessions from 14:09 on); every checkpoint is asked only about the states of ITS
``state_dim`` (the rest are counted as skipped), so the usable sample grows as the bots play.

    .venv/Scripts/python.exe -X utf8 -m v_dance.eval.mega_hold_probe --ckpt g50=<battle ckpt> --ckpt gen39=<…>
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
GIMMICK_NONE, GIMMICK_MEGA = 0, 1
SITUATIONS = ("hold", "hold:seen", "hold:unseen", "reset", "setter_out", "free")
WEATHER_KIND = {"sandstorm": "sand", "raindance": "rain", "sunnyday": "sun", "snowscape": "snow", "snow": "snow",
                "hail": "snow", "desolateland": "sun", "primordialsea": "rain"}


def _id(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _sp(details: str) -> str:
    """'Tyranitar-Mega, L50, M' → 'tyranitar' (base species)."""
    return _id(str(details).split(",")[0].split("-")[0])


def turn_snapshots(text: str, our_ids: set) -> Optional[dict]:
    """The board at the START of every turn (when that turn's decision is made), from a replay:
    ``{"side": "p1", "turns": {N: {"active": {0: sp, 1: sp}, "weather": kind|None, "opp_active": set,
    "opp_fainted": set, "opp_seen": set}}, "opp_six": [...]}`` — None when our side is not found."""
    lines = text.splitlines()
    side = None
    for ln in lines:
        m = re.match(r"\|player\|(p[12])\|([^|]*)\|", ln)
        if m and _id(m.group(2)) in our_ids:
            side = m.group(1)
            break
    if side is None:
        return None
    opp = "p2" if side == "p1" else "p1"
    opp_six: List[str] = []
    active: Dict[str, str] = {}
    fainted_opp, seen_opp = set(), set()
    weather = None
    turns: Dict[int, dict] = {}
    for ln in lines:
        p = ln.split("|")
        if len(p) < 2:
            continue
        k = p[1]
        if k == "poke" and len(p) > 3 and p[2] == opp:
            opp_six.append(_sp(p[3]))
        elif k in ("switch", "drag") and len(p) > 3:
            pos = p[2][:3]
            active[pos] = _sp(p[3])
            if pos.startswith(opp):
                seen_opp.add(active[pos])
        elif k == "faint" and len(p) > 2:
            pos = p[2][:3]
            if pos.startswith(opp) and pos in active:
                fainted_opp.add(active[pos])
            active.pop(pos, None)
        elif k == "-weather" and len(p) > 2:
            if "[upkeep]" in ln:
                continue
            w = _id(p[2])
            weather = None if w in ("none", "") else WEATHER_KIND.get(w, w)
        elif k == "turn" and len(p) > 2:
            n = int(re.sub(r"\D", "", p[2]) or 0)
            turns[n] = {"active": {0: active.get(side + "a"), 1: active.get(side + "b")}, "weather": weather,
                        "opp_active": {active.get(opp + "a"), active.get(opp + "b")} - {None},
                        "opp_fainted": set(fainted_opp), "opp_seen": set(seen_opp)}
    return {"side": side, "turns": turns, "opp_six": opp_six}


def situation(snap: dict, opp_six: Iterable[str], setters: set, weather_kind: str) -> str:
    """One decision's label (module docstring)."""
    if snap.get("weather") == weather_kind:
        return "reset"
    alive = [s for s in opp_six if s in setters and s not in snap["opp_fainted"]]
    if not alive:
        return "free"
    if any(s in snap["opp_active"] for s in alive):
        return "setter_out"
    return "hold:seen" if any(s in snap["opp_seen"] for s in alive) else "hold:unseen"


def iter_recorded_games(rl_dir: Path, mon: str):
    """Yield (meta, transitions) for every recorded game whose own team carries ``mon`` (the meta is decoded first so
    other games skip the heavy transition parse)."""
    dec = json.JSONDecoder()
    for f in sorted(Path(rl_dir).glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for ln in fh:
                i = ln.find('"meta":')
                if i < 0:
                    continue
                try:
                    j = i + len('"meta":')
                    while ln[j] == " ":
                        j += 1
                    meta, _ = dec.raw_decode(ln, j)
                except (ValueError, IndexError):
                    continue
                if mon not in {_id(s) for s in meta.get("own_team") or []}:
                    continue
                try:
                    obj = json.loads(ln)
                except ValueError:
                    continue
                yield meta, obj.get("transitions") or []


def collect_decisions(rl_dir: Path, replay_dir: Path, our_ids: set, *, mon: str = "tyranitar",
                      setters: Iterable[str] = ("pelipper", "politoed", "kyogre"), weather_kind: str = "rain",
                      base_tag=None, guards: Iterable[str] = ()) -> Tuple[List[dict], dict]:
    """Every TURN decision where ``mon`` is active and may mega, labelled — plus, for each ``guards`` mon, every
    decision where IT is active and may mega (label ``guard:<mon>``). Returns (rows, counts) — a row carries the
    state, the slot, the gimmick mask, the recorded pick, the label and the game's arm / result."""
    guards = {_id(g) for g in guards if g}
    base_tag = base_tag or (lambda t: "-".join((t or "").lstrip(">").split("-")[:3]))
    files = {}
    for f in Path(replay_dir).glob("*.html"):
        m = re.search(r"(battle-gen9[a-z0-9]+-\d+)", f.name)
        if m:
            files[m.group(1)] = f
    setters = {_id(s) for s in setters}
    rows, counts = [], collections.Counter()
    for meta, trans in iter_recorded_games(rl_dir, mon):
        counts["games"] += 1
        f = files.get(base_tag(meta.get("battle_id") or ""))
        if f is None:
            counts["no_replay"] += 1
            continue
        snaps = turn_snapshots(f.read_text(encoding="utf-8", errors="replace"), our_ids)
        if snaps is None:
            counts["side_not_found"] += 1
            continue
        arm = (meta.get("sampling") or {}).get("arm")
        for t in trans:
            if t.get("decision_type", "turn") != "turn":
                continue
            snap = snaps["turns"].get(int(t.get("turn", -1)))
            if snap is None:
                counts["turn_not_in_replay"] += 1
                continue
            for slot in (0, 1):
                gm = t.get(f"gmask_s{slot}")
                if not gm or len(gm) <= GIMMICK_MEGA or not gm[GIMMICK_MEGA]:
                    continue
                who = snap["active"].get(slot)
                if who == mon:
                    label = situation(snap, snaps["opp_six"], setters, weather_kind)
                elif who in guards:
                    label = f"guard:{who}"
                else:
                    continue
                rows.append({"state": np.asarray(t["state"], np.float32), "slot": slot, "gmask": list(gm),
                             "recorded": int(t.get(f"gimmick_s{slot}", 0)), "label": label,
                             "arm": arm, "won": meta.get("won"), "battle": meta.get("battle_id"),
                             "turn": int(t.get("turn", 0))})
        counts["games_used"] += 1
    return rows, dict(counts)


def mega_probs(model, head_names, rows: List[dict], batch: int = 256, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """(P(mega) at tau 1 over each row's gimmick mask, argmax-is-mega) for one battle checkpoint."""
    import torch
    pm, am = np.zeros(len(rows), np.float64), np.zeros(len(rows), bool)
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        x = torch.as_tensor(np.stack([r["state"] for r in chunk]), device=device)
        with torch.no_grad():
            out = model(x)
        g = out[1]
        for i, r in enumerate(chunk):
            head = head_names[r["slot"]]
            logits = (g[head] if isinstance(g, dict) else g[r["slot"]])[i].detach().cpu().numpy().astype(np.float64)
            legal = np.asarray(r["gmask"], bool)[:len(logits)]
            z = np.where(legal, logits, -np.inf)
            z = z - z[legal].max()
            p = np.exp(z)
            p = p / p.sum()
            pm[s + i] = p[GIMMICK_MEGA]
            am[s + i] = int(np.argmax(np.where(legal, logits, -np.inf))) == GIMMICK_MEGA
    return pm, am


def summarize(rows: List[dict], results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[dict]:
    """Per label: n, the recorded mega rate, and per checkpoint the mean P(mega) and argmax-mega share. ``rows`` are
    the rows every checkpoint in ``results`` was evaluated on (same order as their arrays)."""
    out = []
    guard_labels = sorted({r["label"] for r in rows if str(r["label"]).startswith("guard:")})
    for lab in SITUATIONS + tuple(guard_labels):
        idx = [i for i, r in enumerate(rows) if (r["label"] == lab or (lab == "hold" and r["label"].startswith("hold:")))]
        if not idx:
            out.append({"label": lab, "n": 0})
            continue
        row = {"label": lab, "n": len(idx),
               "recorded_mega": float(np.mean([rows[i]["recorded"] == GIMMICK_MEGA for i in idx]))}
        for name, (pm, am) in results.items():
            row[f"{name}:p_mega"] = float(np.mean(pm[idx]))
            row[f"{name}:argmax_mega"] = float(np.mean(am[idx]))
        out.append(row)
    return out


def usable_rows(rows: List[dict], state_dim: int) -> List[dict]:
    """The rows whose recorded state has the checkpoint's input size (module docstring: STATE LAYOUT)."""
    return [r for r in rows if len(r["state"]) == int(state_dim)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=BATTLE_CKPT")
    ap.add_argument("--fmt", default="gen9championsvgc2026regmc")
    ap.add_argument("--mon", default="tyranitar")
    ap.add_argument("--setters", default="pelipper,politoed,kyogre")
    ap.add_argument("--weather", default="rain")
    ap.add_argument("--guard", default="salamence",
                    help="'+'-joined megas whose P(mega) must NOT fall (USER 10-03); '' = none")
    ap.add_argument("--rl-dir", default=None)
    ap.add_argument("--replays", default=str(_REPO / "data" / "vods" / "Type_C"))
    ap.add_argument("--json", default=None, help="also write the summary rows here")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(_REPO))
    from v_dance.online.accounts import our_userids
    from v_dance.play import model_io as M
    rl_dir = Path(a.rl_dir) if a.rl_dir else _REPO / "artifacts" / "ladder_rl" / a.fmt
    rows, counts = collect_decisions(rl_dir, Path(a.replays), our_userids() | {"victoriousdancing"},
                                     mon=_id(a.mon), setters=a.setters.split(","), weather_kind=a.weather,
                                     guards=[g for g in str(a.guard or "").split("+") if g])
    print(f"[mega-probe] {counts.get('games', 0)} recorded games with {a.mon} · {counts.get('games_used', 0)} joined "
          f"to a replay · {len(rows)} turn decisions where {a.mon} could mega · counts {counts}")
    if not rows:
        return 1
    dims = collections.Counter(len(r["state"]) for r in rows)
    models, use_dim = {}, None
    for item in a.ckpt:
        name, _, path = item.partition("=")
        model, heads = M.load_bc_policy(path, "cpu")
        model.eval()
        sd = int(getattr(model, "state_dim", 0) or 0)
        if use_dim is not None and sd != use_dim:
            print(f"[mega-probe] {name}: state_dim {sd} ≠ {use_dim} of the first checkpoint — skipped (one table, "
                  f"one state set)")
            continue
        use_dim = sd
        models[name] = (model, heads)
    rows = usable_rows(rows, use_dim or 0)
    print(f"[mega-probe] recorded state sizes {dict(dims)} → {len(rows)} decisions in the checkpoints' layout "
          f"({use_dim} dims; older layouts skipped)")
    if not rows:
        return 1
    results = {name: mega_probs(model, heads, rows) for name, (model, heads) in models.items()}
    summary = summarize(rows, results)
    names = list(results)
    print(f"\n  {'situation':12s} {'n':>5s} {'served':>8s}" + "".join(f" {n[:14]:>16s}" for n in names))
    print(f"  {'':12s} {'':>5s} {'mega %':>8s}" + "".join(f" {'P(mega) / argmax':>16s}" for _ in names))
    for r in summary:
        if not r["n"]:
            print(f"  {r['label']:12s} {0:5d}")
            continue
        cells = "".join(f" {r[f'{n}:p_mega'] * 100:7.0f}% /{r[f'{n}:argmax_mega'] * 100:5.0f}%" for n in names)
        print(f"  {r['label']:12s} {r['n']:5d} {r['recorded_mega'] * 100:7.0f}%" + cells)
    print("\n  hold = their rain setter alive in the back (standard play: HOLD, LOW is right) · reset = rain is up "
          "(mega NOW re-sets sand, HIGH is right) · free = no living setter · guard:<mon> = another mega, must NOT "
          "fall")
    by_arm = collections.defaultdict(list)
    for r in rows:
        if r["label"].startswith("hold"):
            by_arm[r["arm"]].append(r["recorded"] == GIMMICK_MEGA)
    if by_arm:
        print("  served behaviour under 'hold', by arm: " + ", ".join(
            f"{k} {np.mean(v) * 100:.0f}% mega ({len(v)})" for k, v in sorted(by_arm.items(), key=lambda kv: -len(kv[1]))))
    if a.json:
        Path(a.json).write_text(json.dumps({"counts": counts, "summary": summary}, indent=1), encoding="utf-8")
        print(f"[mega-probe] summary → {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
