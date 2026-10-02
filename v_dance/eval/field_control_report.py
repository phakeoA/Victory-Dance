"""Weather and terrain CONTROL in the bot's ladder games (USER 2026-10-02: "terrain wars and weather wars are concepts
it doesn't understand … I don't know how to get it to learn it").

Rebuilds, from the saved replays (``data/vods/Type_C``), WHO held the weather and the terrain at the start of every
turn, and what happened after the bot LOST control — so the leak can be named before anything is trained:
  · picks   — the setter not brought / not led, or the lead fight lost at turn 1
  · reclaim — control lost while our setter was alive, and the bot never switched it back in
  · value   — the setter knocked out early (it was not protected or kept healthy)
Control is by KIND: "ours" = a weather / terrain our TEAM sets (from the team sheet's abilities: Baltimore = sand +
Psychic Terrain), whoever set it; "theirs" = any other kind. Games: ``human_bench.jsonl`` rows for ``--team`` since
``--since`` (default: the team file's last edit), joined to their replays by battle tag.

Every number here is DESCRIPTIVE: holding control and winning go together partly because a winning position makes
control easy to keep. A row says where to look, not what causes what.

    .venv/Scripts/python.exe -X utf8 -m v_dance.eval.field_control_report --team Baltimore_Sand_Psy
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

WEATHER_KIND = {"sandstorm": "sand", "raindance": "rain", "sunnyday": "sun", "snowscape": "snow", "snow": "snow",
                "hail": "snow", "desolateland": "sun", "primordialsea": "rain", "deltastream": "wind"}
ABILITY_WEATHER = {"sandstream": "sand", "drizzle": "rain", "drought": "sun", "snowwarning": "snow",
                   "orichalcumpulse": "sun", "desolateland": "sun", "primordialsea": "rain"}
ABILITY_TERRAIN = {"psychicsurge": "psychic", "grassysurge": "grassy", "electricsurge": "electric",
                   "mistysurge": "misty", "hadronengine": "electric", "seedsower": "grassy"}
# a terrain set ON SWITCH-IN (the drill pressure: "switch the setter back in") — Seed Sower sets it only when HIT
SWITCH_IN_TERRAIN = {k: v for k, v in ABILITY_TERRAIN.items() if k != "seedsower"}
# MEGA STONES that make their holder a weather / terrain setter AS ITS MEGA — the paste's Ability line shows the BASE
# ability (Charizard 'Blaze', Froslass 'Cursed Body'), so a paste read by ability alone missed them. Derived from the
# project dex (each mega forme's requiredItem -> its ability); Charizardite Y kept as a floor if the dex is missing.
# (review 10-02: Froslassite = Snow Warning and Raichunite X = Electric Surge were missed — 4 Froslass pool teams.)
def _mega_item_setters():
    import json
    weather, terrain = {"charizarditey": "sun"}, {}
    try:
        dex = json.loads((_REPO / "data" / "pokedex.json").read_text(encoding="utf-8"))
        dex = dex.get("pokedex", dex) if isinstance(dex, dict) else {}
    except (OSError, ValueError):
        return weather, terrain
    for e in dex.values():
        if not isinstance(e, dict) or not e.get("requiredItem"):
            continue
        item = _id(e["requiredItem"])
        for ab in (e.get("abilities") or {}).values():
            a = _id(ab)
            if a in ABILITY_WEATHER:
                weather[item] = ABILITY_WEATHER[a]
            if a in ABILITY_TERRAIN and a != "seedsower":
                terrain[item] = ABILITY_TERRAIN[a]
    return weather, terrain


ITEM_WEATHER: dict = {}
ITEM_TERRAIN: dict = {}
DOMAINS = ("weather", "terrain")
RECLAIM_TURNS = 2           # "reclaimed" = our kind is back within this many turns of losing it


def _id(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


ITEM_WEATHER, ITEM_TERRAIN = _mega_item_setters()


def _base_tag(tag: str) -> str:
    m = re.search(r"(battle-gen9[a-z0-9]+-\d+)", tag or "")
    return m.group(1) if m else (tag or "")


def _paste_header_species(ln: str) -> str:
    """The forme-free species id of a paste's header line: 'Nick (Species) (F) @ Item' / 'Species (F) @ Item'."""
    head = ln.split("@")[0].strip()
    m = re.match(r"^(.*?)\s*\(([^)]+)\)", head)
    name = m.group(2) if m and len(m.group(2)) > 1 else re.sub(r"\s*\([MF]\)", "", head)
    return _id(name.split("-")[0])


def _is_header(ln: str) -> bool:
    return bool(ln) and not ln.startswith(("Ability:", "EVs:", "IVs:", "Level:", "Tera", "-")) \
        and not ln.endswith("Nature") and ":" not in ln


def paste_species(paste: str) -> list:
    """Every species (forme-free ids) in a Showdown paste, in order."""
    return [_paste_header_species(ln.strip()) for ln in paste.splitlines() if _is_header(ln.strip())]


def team_setters(paste: str) -> dict:
    """{domain: {kind: species id}} from a Showdown paste (``Ability: Sand Stream`` under ``Tyranitar @ …``)."""
    out = {"weather": {}, "terrain": {}}
    species = None
    for ln in paste.splitlines():
        ln = ln.strip()
        if _is_header(ln):
            species = _paste_header_species(ln)                    # forme-free, like the replay's switch lines
            it = _id(ln.split("@", 1)[1]) if "@" in ln else ""
            if it in ITEM_WEATHER:                                 # a mega stone → the mega's weather (Drought, …)
                out["weather"][ITEM_WEATHER[it]] = species
            if it in ITEM_TERRAIN:                                 # … or terrain (Raichunite X → Electric Surge)
                out["terrain"][ITEM_TERRAIN[it]] = species
        elif ln.startswith("Ability:") and species:
            ab = _id(ln.split(":", 1)[1])
            if ab in ABILITY_WEATHER:
                out["weather"][ABILITY_WEATHER[ab]] = species
            if ab in ABILITY_TERRAIN:
                out["terrain"][ABILITY_TERRAIN[ab]] = species
    return out


def parse_replay(text: str, our_ids: set, ours: dict) -> dict | None:
    """One game's control timeline. ``ours`` = {domain: {kind: setter species}} for OUR team."""
    lines = text.splitlines()
    side = None
    for ln in lines:
        m = re.match(r"\|player\|(p[12])\|([^|]*)\|", ln)
        if m and _id(m.group(2)) in our_ids:
            side = m.group(1)
            break
    if side is None:
        return None
    our_kinds = {d: set(ours[d]) for d in DOMAINS}
    setter_species = {d: set(ours[d].values()) for d in DOMAINS}
    cur = {"weather": None, "terrain": None}                # current kind per domain
    active, fainted, entered = {}, set(), []                 # our slot -> species; our fainted / entered species
    last_mover = None
    turn, per_turn = 0, []                                   # per_turn: (turn, {domain: 'ours'/'theirs'/None})
    events = []                                              # {domain, turn, kind, setter_alive, setter_active}
    sets = {"ours": {d: [] for d in DOMAINS}, "theirs": {d: [] for d in DOMAINS}}
    setter_faint_turn = {d: None for d in DOMAINS}
    winner = None

    def owner(d):
        k = cur[d]
        return None if k is None else ("ours" if k in our_kinds[d] else "theirs")

    def set_kind(d, kind, by_side):
        before = owner(d)
        cur[d] = kind
        who = "ours" if by_side == side else "theirs"
        sets[who][d].append((turn, kind))
        after = owner(d)
        if before == "ours" and after == "theirs":
            alive = [s for s in setter_species[d] if s not in fainted]
            events.append({"domain": d, "turn": turn, "kind": kind,
                           "setter_alive": bool(alive),
                           "setter_active": any(s in active.values() for s in alive),
                           "reclaimed": False})
        if after == "ours":                                  # closes every open loss of this domain in the window
            for e in events:
                if e["domain"] == d and not e["reclaimed"] and turn - e["turn"] <= RECLAIM_TURNS \
                        and "reclaim_turn" not in e:
                    e["reclaimed"], e["reclaim_turn"] = True, turn

    for ln in lines:
        p = ln.split("|")
        if len(p) < 2:
            continue
        kind = p[1]
        if kind == "turn":
            turn = int(re.sub(r"\D", "", p[2]) or 0)
            per_turn.append((turn, {d: owner(d) for d in DOMAINS}))
        elif kind in ("switch", "drag") and len(p) > 3:
            pos, sp = p[2][:3], _id(p[3].split(",")[0].split("-")[0])
            if pos.startswith(side):
                active[pos] = sp
                if sp not in entered:
                    entered.append(sp)
        elif kind == "move" and len(p) > 2:
            last_mover = p[2][:2]
        elif kind == "faint" and len(p) > 2 and p[2].startswith(side):
            sp = active.get(p[2][:3]) or _id(p[2].split(":", 1)[-1])
            fainted.add(sp)
            for d in DOMAINS:
                if sp in setter_species[d] and setter_faint_turn[d] is None:
                    setter_faint_turn[d] = turn
        elif kind == "-weather" and len(p) > 2:
            if "[upkeep]" in ln:
                continue
            w = _id(p[2])
            if w in ("none", ""):
                cur["weather"] = None
                continue
            m = re.search(r"\[of\] (p[12])", ln)
            by = m.group(1) if m else last_mover
            if by:
                set_kind("weather", WEATHER_KIND.get(w, w), by)
        elif kind == "-fieldstart" and len(p) > 2 and "Terrain" in p[2]:
            t = _id(p[2].replace("move:", "").replace("Terrain", ""))
            m = re.search(r"\[of\] (p[12])", ln)
            by = m.group(1) if m else last_mover
            if by:
                set_kind("terrain", t, by)
        elif kind == "-fieldend" and len(p) > 2 and "Terrain" in p[2]:
            cur["terrain"] = None
        elif kind == "win" and len(p) > 2:
            winner = _id(re.sub(r"<.*", "", p[2]))
        elif kind == "tie":
            winner = "tie"
    if winner is None or winner == "tie" or turn == 0:
        return None
    held = {d: sum(1 for _, o in per_turn if o[d] == "ours") for d in DOMAINS}
    lost = {d: sum(1 for _, o in per_turn if o[d] == "theirs") for d in DOMAINS}
    lead = per_turn[0][1] if per_turn else {d: None for d in DOMAINS}
    setter_entered = {d: any(s in entered for s in setter_species[d]) for d in DOMAINS}
    for e in events:                     # brought for sure = it took the field at some point this game
        e["setter_brought"] = setter_entered[e["domain"]]
    # a FIGHT = the opponent set a kind that is not ours (their own sand / Psychic Terrain is no contest)
    opp_other = {d: sorted({k for _, k in sets["theirs"][d] if k not in our_kinds[d]}) for d in DOMAINS}
    return {"won": winner in our_ids, "turns": turn, "held": held, "lost": lost, "lead": lead,
            "sets": sets, "events": events, "entered": entered, "led": entered[:2],
            "setter_faint_turn": setter_faint_turn, "setter_entered": setter_entered,
            "opp_kinds": opp_other, "contested": {d: bool(opp_other[d]) for d in DOMAINS}}


def load_games(bench: Path, team: str, since: str, account: str | None, arm: str | None, primary: str) -> dict:
    from v_dance.online.accounts import row_account
    games = {}
    for ln in open(bench, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        tag = _base_tag(r.get("battle_tag") or "")
        if "result" not in r or r.get("ai_team") != team or "regmc" not in tag or (r.get("ts") or "") < since \
                or r.get("self_match"):
            continue
        if account and row_account(r, primary) != account:
            continue
        if arm and r.get("arm") != arm:
            continue
        games[tag] = {"arm": r.get("arm"), "account": row_account(r, primary)}
    return games


def _pct(w, n):
    return f"{100 * w / n:5.1f}%" if n else "   — "


def report(rows: list, ours: dict) -> None:
    n = len(rows)
    w = sum(g["won"] for g in rows)
    print(f"\n{n} games · {w}W-{n - w}L = {_pct(w, n)}   (our weather: {', '.join(ours['weather']) or '—'} · "
          f"our terrain: {', '.join(ours['terrain']) or '—'})")
    for d in DOMAINS:
        if not ours[d]:
            continue
        print(f"\n━━ {d.upper()} ━━")
        contested = [g for g in rows if g["contested"][d]]
        cw = sum(g["won"] for g in contested)
        print(f"  opponent set a DIFFERENT {d} in {len(contested)}/{n} games ({_pct(len(contested), n).strip()}) — "
              f"we won {_pct(cw, len(contested)).strip()} of those vs {_pct(w - cw, n - len(contested)).strip()} "
              f"of the rest")
        # by the opponent's kind
        by_kind = collections.defaultdict(list)
        for g in contested:
            for k in g["opp_kinds"][d]:
                by_kind[k].append(g)
        print(f"  {'their ' + d:16s} {'games':>6s} {'win':>7s} {'our share of turns':>19s} {'their share':>12s}")
        for k, gs in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
            tt = sum(g["turns"] for g in gs) or 1
            print(f"  {k:16s} {len(gs):6d} {_pct(sum(g['won'] for g in gs), len(gs)):>7s} "
                  f"{100 * sum(g['held'][d] for g in gs) / tt:18.0f}% {100 * sum(g['lost'][d] for g in gs) / tt:11.0f}%")
        # control share → win rate (contested games)
        print(f"  win rate by OUR share of turns (contested games):")
        for lo, hi, lab in ((0.0, 1 / 3, "under 1/3"), (1 / 3, 2 / 3, "1/3 – 2/3"), (2 / 3, 1.01, "over 2/3")):
            gs = [g for g in contested if lo <= g["held"][d] / max(1, g["turns"]) < hi]
            print(f"    {lab:12s} {len(gs):5d} games  {_pct(sum(g['won'] for g in gs), len(gs))}")
        # the lead fight (turn 1)
        lead_ours = [g for g in contested if g["lead"][d] == "ours"]
        lead_theirs = [g for g in contested if g["lead"][d] == "theirs"]
        lead_none = [g for g in contested if g["lead"][d] is None]
        print(f"  at turn 1 (the leads): ours {len(lead_ours)} ({_pct(sum(g['won'] for g in lead_ours), len(lead_ours)).strip()} won)"
              f" · theirs {len(lead_theirs)} ({_pct(sum(g['won'] for g in lead_theirs), len(lead_theirs)).strip()})"
              f" · none {len(lead_none)} ({_pct(sum(g['won'] for g in lead_none), len(lead_none)).strip()})")
        # picks: did our setter ever come in?
        never = [g for g in contested if not g["setter_entered"][d]]
        print(f"  our setter never entered: {len(never)}/{len(contested)} contested games "
              f"({_pct(sum(g['won'] for g in never), len(never)).strip()} won)")
        # reclaim
        ev = [(g, e) for g in contested for e in g["events"] if e["domain"] == d]
        alive = [(g, e) for g, e in ev if e["setter_alive"] and e["setter_brought"]]
        bench = [(g, e) for g, e in alive if not e["setter_active"]]
        rec = [(g, e) for g, e in alive if e["reclaimed"]]
        rec_b = [(g, e) for g, e in bench if e["reclaimed"]]
        rec_p = [(g, e) for g, e in alive if e["setter_active"] and e["reclaimed"]]
        print(f"  times we LOST {d} control: {len(ev)} — our setter was alive (and brought) {len(alive)}× "
              f"(on the bench {len(bench)}×, still on the field {len(alive) - len(bench)}×), dead {sum(1 for _, e in ev if not e['setter_alive'])}×")
        print(f"    reclaimed within {RECLAIM_TURNS} turns: {len(rec)}/{len(alive)} = {_pct(len(rec), len(alive)).strip()} "
              f"(setter on the bench: {len(rec_b)}/{len(bench)} · on the field, so it had to switch out and back: "
              f"{len(rec_p)}/{len(alive) - len(bench)})")
        g_rec = {id(g) for g, e in rec}
        g_norec = {id(g) for g, e in alive if id(g) not in g_rec}
        gr = [g for g in contested if id(g) in g_rec]
        gn = [g for g in contested if id(g) in g_norec]
        print(f"    games with a reclaim {len(gr)} ({_pct(sum(g['won'] for g in gr), len(gr)).strip()} won) · "
              f"games where the setter was alive but never reclaimed {len(gn)} ({_pct(sum(g['won'] for g in gn), len(gn)).strip()} won)")
        # value: the setter knocked out early
        early = [g for g in contested if g["setter_faint_turn"][d] is not None and g["setter_faint_turn"][d] <= 3]
        later = [g for g in contested if g not in early and g["setter_entered"][d]]
        print(f"  setter KO'd by turn 3: {len(early)} contested games ({_pct(sum(g['won'] for g in early), len(early)).strip()} won)"
              f" vs {_pct(sum(g['won'] for g in later), len(later)).strip()} when it survived turn 3")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="Baltimore_Sand_Psy")
    ap.add_argument("--since", default=None, help="ISO time (UTC); default = the team file's last edit")
    ap.add_argument("--account", default=None, help="1 / 2 (the .env slot) or a username; default = every account")
    ap.add_argument("--arm", default=None, help="only games served by this bandit arm")
    ap.add_argument("--replays", default=str(_REPO / "data" / "vods" / "Type_C"))
    ap.add_argument("--bench", default=str(_REPO / "artifacts" / "human_benchmark" / "human_bench.jsonl"))
    ap.add_argument("--json", default=None, help="also write the per-game rows here")
    a = ap.parse_args(argv)
    from v_dance.online import accounts as A
    import v_dance.play.run_local_battle as R
    env = A.read_env()
    our_ids = A.our_userids(env)
    primary = A.primary_userid(env)
    team_path = Path(R.resolve_team_path(a.team))
    if a.since is None:
        a.since = dt.datetime.fromtimestamp(os.path.getmtime(team_path), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    acct = None
    if a.account:
        acct = A.load_account(env, int(a.account)).userid if str(a.account).isdigit() else A.userid(a.account)
    paste = team_path.read_text(encoding="utf-8") if team_path.is_file() else \
        "\n".join(p.read_text(encoding="utf-8") for p in sorted(team_path.glob("*")) if p.is_file())
    ours = team_setters(paste)
    games = load_games(Path(a.bench), a.team, a.since, acct, a.arm, primary)
    rows, missing = [], 0
    by_tag = {}
    for f in Path(a.replays).glob("*.html"):
        t = _base_tag(f.name)
        if t in games:
            by_tag[t] = f
    for tag, meta in games.items():
        f = by_tag.get(tag)
        if f is None:
            missing += 1
            continue
        g = parse_replay(f.read_text(encoding="utf-8", errors="replace"), our_ids, ours)
        if g is None:
            missing += 1
            continue
        g.update(tag=tag, **meta)
        rows.append(g)
    print(f"[field] {a.team}{f' / account {a.account}' if a.account else ''}{f' / arm {a.arm}' if a.arm else ''} "
          f"since {a.since}: {len(games)} logged games, {len(rows)} with a readable replay ({missing} without)")
    if not rows:
        return 1
    report(rows, ours)
    print("\nDescriptive only: holding control and winning feed each other (a won position keeps control easily).")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=list), encoding="utf-8")
        print(f"[field] per-game rows → {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
