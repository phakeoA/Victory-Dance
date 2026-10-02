"""Where does the bot lose? A matchup breakdown of its LADDER games with one team (2026-10-01).

Sources: the opponent dossiers (``artifacts/dossiers/*.json``: per game the opponent's brought species, megas, the
moves they revealed, our result, turns) joined to the saved replays (``data/vods/Type_C``: OUR leads, brought four
and mega). Only games with ``--team`` since ``--since`` (default: the team file's last edit, so an edited sheet is
never mixed with its old version).

Every row is compared with the team's OVERALL win rate: ``z`` = (wins − n·p) / sqrt(n·p·(1−p)). Many rows are tested
at once, so ~1 in 20 reaches |z| ≥ 2 by chance alone — treat a flag as a lead to look at, not a verdict, and
remember co-occurrence (a Trick Room setter and its abuser arrive together).

    .venv/Scripts/python.exe -X utf8 -m v_dance.eval.loss_breakdown --team Baltimore_Sand_Psy
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
BOT = "victoriousdancing"

# opponent game plans, read off the moves / abilities / species their mons REVEALED
PLANS = {
    "Trick Room": lambda mv, ab, sp: "trickroom" in mv,
    "Tailwind": lambda mv, ab, sp: "tailwind" in mv,
    "Rain": lambda mv, ab, sp: "drizzle" in ab or "raindance" in mv or sp & {"pelipper", "politoed"},
    "Sun": lambda mv, ab, sp: "drought" in ab or "sunnyday" in mv or sp & {"torkoal", "ninetales", "charizardmegay"},
    "Fake Out": lambda mv, ab, sp: "fakeout" in mv,
    "Redirection": lambda mv, ab, sp: bool(mv & {"followme", "ragepowder"}),
    "Perish Song": lambda mv, ab, sp: "perishsong" in mv,
    "Intimidate": lambda mv, ab, sp: "intimidate" in ab,
    "Spore/sleep": lambda mv, ab, sp: bool(mv & {"spore", "sleeppowder", "yawn"}),
    "Wide Guard": lambda mv, ab, sp: "wideguard" in mv,
}


def _id(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _z(w: int, n: int, p: float) -> float:
    return (w - n * p) / math.sqrt(max(1e-9, n * p * (1 - p)))


def load_games(team: str, since_iso: str, dossier_dir: Path) -> dict:
    games = {}
    for f in dossier_dir.glob("*.json"):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        mons = doc.get("mons") or {}
        for g in doc.get("games") or []:
            tag, ts = g.get("battle_tag") or "", g.get("ts") or ""
            if g.get("our_team") != team or "regmc" not in tag or ts < since_iso or g.get("result") not in ("ai", "human"):
                continue
            rev = [_id(s) for s in g.get("revealed") or []]
            mv, ab = set(), set()
            for s in rev:
                r = mons.get(s) or {}
                mv |= {_id(m) for m in r.get("moves") or []}
                ab |= {_id(r.get("ability")), _id(r.get("mega_ability"))}
            games[tag] = {"won": g["result"] == "ai", "turns": g.get("turns") or 0, "opp": rev,
                          "megas": [_id(m) for m in g.get("megas") or []], "moves": mv, "abilities": ab - {""},
                          "opponent": doc.get("opponent") or f.stem}
    return games


def _our_ids() -> set:
    """Every ladder account in .env (2026-10-02: a second account — its games were invisible to the side lookup)."""
    try:
        from v_dance.online.accounts import our_userids
        ids = our_userids()
    except Exception:
        ids = set()
    return ids | {BOT}


def join_replays(games: dict, replay_dir: Path) -> None:
    """Our leads / brought four / mega, from the saved replays (side found by our accounts' player names)."""
    ours = _our_ids()
    for f in glob.glob(str(replay_dir / "*.html")):
        m = re.search(r"(battle-gen9[a-z0-9]+-\d+)", f)
        if not m or m.group(1) not in games:
            continue
        lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
        side = next((mm.group(1) for ln in lines for mm in [re.match(r"\|player\|(p[12])\|([^|]*)\|", ln)]
                     if mm and _id(mm.group(2)) in ours), None)
        if not side:
            continue
        g, leads, brought, mega, started = games[m.group(1)], [], [], None, False
        for ln in lines:
            p = ln.split("|")
            if ln.strip() == "|turn|1":
                started = True                     # leads = our switch-ins BEFORE turn 1 begins
            elif len(p) > 3 and p[1] in ("switch", "drag") and p[2].startswith(side):
                sp = _id(re.sub(r",.*", "", p[3]).split("-")[0])
                if sp not in brought:
                    brought.append(sp)
                if not started:
                    leads.append(sp)
            elif len(p) > 3 and p[1] == "detailschange" and p[2].startswith(side) and "Mega" in p[3]:
                mega = _id(re.sub(r",.*", "", p[3]))
        g.update(leads=tuple(sorted(leads[:2])), brought=frozenset(brought), our_mega=mega or "none")


def table(title: str, rows: dict, p: float, min_n: int, top: int = 0) -> list:
    out = []
    for key, (w, n) in rows.items():
        if n >= min_n:
            out.append((key, w, n, _z(w, n, p)))
    out.sort(key=lambda r: r[3])
    if top:
        out = out[:top] + ([None] if len(out) > 2 * top else []) + out[-top:] if len(out) > 2 * top else out
    print(f"\n{title}  (n ≥ {min_n}; overall {p * 100:.1f} %)")
    for r in out:
        if r is None:
            print("    …")
            continue
        key, w, n, z = r
        flag = "  ◀ LOSES MORE" if z <= -2 else "  ▶ wins more" if z >= 2 else ""
        print(f"    {str(key):34s} {w:4d}/{n:<4d} = {w / n * 100:5.1f}%   z {z:+5.2f}{flag}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True)
    ap.add_argument("--since", default=None, help="ISO time (UTC); default = the team file's last edit")
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--dossiers", default=str(_REPO / "artifacts" / "dossiers"))
    ap.add_argument("--replays", default=str(_REPO / "data" / "vods" / "Type_C"))
    ap.add_argument("--arm", default=None, help="only games served by this bandit arm (from human_bench.jsonl) — "
                                                "read an A/B, e.g. lg2_g50_rule vs lg2_g50")
    ap.add_argument("--account", default=None,
                    help="2026-10-02 (two ladder accounts): only games played by this account — 1 / 2 (the .env "
                         "slot) or a username; games logged before 2026-10-02 belong to account 1")
    ap.add_argument("--bench", default=str(_REPO / "artifacts" / "human_benchmark" / "human_bench.jsonl"))
    a = ap.parse_args(argv)
    if a.since is None:
        import v_dance.play.run_local_battle as R
        mt = os.path.getmtime(R.resolve_team_path(a.team))
        a.since = dt.datetime.fromtimestamp(mt, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    games = load_games(a.team, a.since, Path(a.dossiers))
    if not games:
        print(f"[loss] no {a.team} games since {a.since}")
        return 1
    if a.arm or a.account:
        from v_dance.online import accounts as _ACC
        from v_dance.ui.rating_chart import env_accounts
        primary, _ = env_accounts()
        want_acct = None
        if a.account:
            if str(a.account).isdigit():
                want_acct = _ACC.load_account(_ACC.read_env(), int(a.account)).userid
            else:
                want_acct = _ACC.userid(a.account)
        arm_of, acct_of = {}, {}
        for ln in open(a.bench, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if "result" in r and r.get("battle_tag"):
                arm_of[r["battle_tag"]] = r.get("arm")
                acct_of[r["battle_tag"]] = _ACC.row_account(r, primary)
        games = {t: g for t, g in games.items()
                 if (not a.arm or arm_of.get(t) == a.arm) and (not want_acct or acct_of.get(t) == want_acct)}
        if not games:
            print(f"[loss] no {a.team} games{f' served by arm {a.arm!r}' if a.arm else ''}"
                  f"{f' on account {want_acct}' if want_acct else ''} since {a.since}")
            return 1
    join_replays(games, Path(a.replays))
    n = len(games)
    w = sum(g["won"] for g in games.values())
    p = w / n
    print(f"[loss] {a.team}{f' / arm {a.arm}' if a.arm else ''}{f' / account {a.account}' if a.account else ''}: "
          f"{n} ladder games since {a.since} — "
          f"{w}W-{n - w}L = {p * 100:.1f} %")

    sp = collections.defaultdict(lambda: [0, 0])
    for g in games.values():
        for s in set(g["opp"]):
            sp[s][0] += g["won"]
            sp[s][1] += 1
    table("OPPONENT POKÉMON (brought + seen), worst and best", sp, p, a.min_n, top=12)

    mg = collections.defaultdict(lambda: [0, 0])
    for g in games.values():
        for s in (g["megas"] or ["(no mega)"]):
            mg[s][0] += g["won"]
            mg[s][1] += 1
    table("OPPONENT MEGA", mg, p, max(8, a.min_n // 2))

    pl = collections.defaultdict(lambda: [0, 0])
    for g in games.values():
        spset = set(g["opp"])
        for name, fn in PLANS.items():
            if fn(g["moves"], g["abilities"], spset):
                pl[name][0] += g["won"]
                pl[name][1] += 1
    table("OPPONENT GAME PLAN (from revealed moves / abilities)", pl, p, a.min_n)

    ln = collections.defaultdict(lambda: [0, 0])
    om = collections.defaultdict(lambda: [0, 0])
    for g in games.values():
        if g.get("leads"):
            ln[" + ".join(g["leads"])][0] += g["won"]
            ln[" + ".join(g["leads"])][1] += 1
        if g.get("our_mega"):
            om[g["our_mega"]][0] += g["won"]
            om[g["our_mega"]][1] += 1
    table("OUR LEADS", ln, p, a.min_n)
    table("OUR MEGA", om, p, max(8, a.min_n // 2))

    tl = collections.defaultdict(lambda: [0, 0])
    for g in games.values():
        t = g["turns"]
        b = "1-4" if t <= 4 else "5-7" if t <= 7 else "8-11" if t <= 11 else "12+"
        tl[f"{b} turns"][0] += g["won"]
        tl[f"{b} turns"][1] += 1
    table("GAME LENGTH", tl, p, 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
