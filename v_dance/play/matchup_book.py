"""Matchup book — which opponent Pokémon we face and how we do against them (USER 2026-09-02).

Two books, both keyed by the battle FORMAT (a regulation's ladder is its own meta):

  * ALL-TIME per (format, OUR team) — M-B/The_Big_6, M-B/maw_zard and M-A/maw_zard are three
    DIFFERENT tables (USER ruling 09-02: "the big 6 would have different match up data than maw
    zard, and m-a maw zard different than m-b"). Seeded from every opponent dossier at startup
    (``artifacts/dossiers/*.json`` — per game the species SEEN in battle = the brought four, our
    result, our team) and fed by every finished game from then on.
  * SESSION per (format, session id) — this bot process only.

Per species: the games it appeared in, our W-L-D in those games, the items / abilities it
REVEALED (poke-env reveals an item only when it acts and this ladder has no open team sheets,
so most rows fall back to the belief's top item — marked "(belief)" in the UI, as the USER asked).
An item counts ONCE per opponent per species (a dossier holds one last-seen item per opponent,
so the live hook mirrors that: "7 seen" = seven opponents ran it).

Torch-free, pure bookkeeping: nothing here can change play. Every public method is defensive —
a bad dossier line is skipped, never raised.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_DOSSIER_DIR = _REPO / "artifacts" / "dossiers"
UNKNOWN_TEAM = "(unknown team)"
ALL_TEAMS = "*"                         # summary(team=ALL_TEAMS) merges every team of the format
_TAG_RE = re.compile(r"^(battle-([a-z0-9]+)-\d+)")
_RESULTS = {"ai": "wins", "human": "losses", "draw": "draws"}   # bench-row convention: "ai" = WE won
_UNKNOWN_ITEMS = (None, "", "unknown_item")   # "" = confirmed itemless (Knock Off) — not a run item


def toid(s) -> str:
    return "".join(c for c in str(s or "").lower() if c.isalnum())


def base_tag(tag) -> str:
    """``battle-<fmt>-<n>`` without a private-room suffix (the bench / bandit / panel key)."""
    m = _TAG_RE.match((tag or "").lstrip(">"))
    return m.group(1) if m else (tag or "")


def fmt_of_tag(tag) -> Optional[str]:
    m = _TAG_RE.match((tag or "").lstrip(">"))
    return m.group(2) if m else None


def _title(s: str) -> str:
    return "-".join(w[:1].upper() + w[1:] for w in str(s or "").split("-"))


def display_species(species: str, belief=None) -> str:
    """'ninetalesalola' -> 'Ninetales-Alola': the belief's own key first (the reg's display names),
    then the pokedex name, else the id as-is. Never raises."""
    sp = species or ""
    if belief is not None:
        try:
            key = belief._resolve(sp)        # noqa: SLF001 — the belief's display key
            if key:
                return str(key)
        except Exception:
            pass
    try:
        from v_dance.parser.vod_parser.pokedex import get_pokedex
        dx = get_pokedex()
        e = dx.entry(sp) if dx is not None else None
        name = (e or {}).get("name")
        if name:
            return _title(str(name))
    except Exception:
        pass
    return sp


def _display(kind: str, raw: str) -> str:
    """poke-env id ('lifeorb') -> display name ('Life Orb') via the team-sheet maps; title-case
    fallback when the data files are unavailable (tests without data/)."""
    try:
        from v_dance.parser.vod_parser.team_sheet import _packed_display
        d = _packed_display(kind, raw)
        if d and str(d) != str(raw):          # a map miss echoes the id back ('latiosite')
            return str(d)
    except Exception:
        pass
    return _title(str(raw or ""))


def _dex():
    try:
        from v_dance.parser.vod_parser.pokedex import get_pokedex
        return get_pokedex()
    except Exception:
        return None


def mega_of(species, ability=None, *, is_mega: Optional[bool] = None) -> Optional[dict]:
    """The mega forme this mon is PROVEN to be, or None (USER bug report 09-02: a Gardevoir row
    read "Choice Scarf (belief) · Pixilate" — Pixilate IS the mega).

    poke-env keeps ``mon.species`` at the BASE forme after a mega and never sets the stone, but it
    rewrites the ability to the mega forme's. So the evidence is either the live detector
    (``is_mega``: base stats = a mega dex entry) or an ability ONLY a mega forme of this species
    can have. A shared base/mega ability (Latios Levitate, Scizor Technician, Abomasnow Snow
    Warning) proves nothing by itself. Returns ``{"forme": "Gardevoir-Mega", "ability":
    "Pixilate", "stone": "Gardevoirite"}`` — the stone is a REVEALED item (a mega holds it)."""
    dx = _dex()
    if dx is None or not species:
        return None
    try:
        e0 = dx.entry(species) or {}
        if (e0.get("forme") or "").startswith("Mega"):         # already a mega forme id
            is_mega = True
            species = e0.get("baseSpecies") or species
        megas = dx.mega_formes_for(species)
        if not megas:
            return None
        base_abilities = {toid(a) for a in dx.abilities_for(species)}
        ab = toid(ability) if ability else ""
        hit = None
        if ab:
            cands = [m for m in megas if toid(m.get("ability")) == ab]
            if len(cands) == 1 and (ab not in base_abilities or is_mega):
                hit = cands[0]
        if hit is None and is_mega and len(megas) == 1:
            hit = megas[0]
        if hit is None:
            return None
        fe = dx.entry(hit["forme"]) or {}
        return {"forme": hit["forme"], "ability": hit.get("ability"), "stone": fe.get("requiredItem")}
    except Exception:
        return None


def live_is_mega(mon) -> Optional[bool]:
    """The encoder's mega detector (base stats = a mega dex entry) on a live poke-env mon; None
    when unavailable (the ability inference in ``mega_of`` still applies)."""
    try:
        from v_dance.encoders.live_state_encoder import is_mega_forme_live
        return bool(is_mega_forme_live(mon))
    except Exception:
        return None


def _new_stat() -> dict:
    # items / abilities: ``*_by_opp`` = (opponent, value) pairs (one count per opponent, so a
    # per-team table and the merged all-teams table agree); ``*_anon`` = counts from games with no
    # opponent name (nothing to dedupe by). ``mega`` = games this species was seen mega-evolved.
    return {"games": 0, "wins": 0, "losses": 0, "draws": 0, "items_anon": Counter(),
            "abilities_anon": Counter(), "opponents": set(), "item_by_opp": set(),
            "ability_by_opp": set(), "mega": 0, "last_seen": None}


def _counts(st: dict, kind: str) -> Counter:
    c = Counter(st[f"{kind}_anon"] if kind == "items" else st["abilities_anon"])
    for _opp, val in (st["item_by_opp"] if kind == "items" else st["ability_by_opp"]):
        c[val] += 1
    return c


def _prior(belief, sp: str, kind: str) -> list:
    """The belief's distribution as [(display name, p)] — items (top 12) or abilities (top 6;
    ``top_ability`` alone when the belief has no distribution). [] = no belief for the species."""
    if belief is None:
        return []
    try:
        if kind == "items":
            dist = belief.item_distribution(sp, 12) or []
        else:
            fn = getattr(belief, "ability_distribution", None)
            dist = (fn(sp, 6) if fn is not None else None) or []
            if not dist:
                top = belief.top_ability(sp)
                dist = [{"name": top, "p": 1.0}] if top else []
        return [(str(d["name"]), float(d.get("p") or 0.0)) for d in dist
                if isinstance(d, dict) and d.get("name")]
    except Exception:
        return []


def _blend(seen: Counter, n_pop: int, prior: list, *, kind: str, species: str = "",
           top: int = 3) -> tuple:
    """USER 09-02 ("one-offs shown as fact"): the population estimate for one species —
    revealed sightings are HARD evidence for the opponents who showed them, the belief prior
    fills the unrevealed rest. 60 Sneasler opponents with ONE Life Orb seen and a belief of
    Focus Sash 45 % / Life Orb 20 % / Choice Band 15 % → Focus Sash 44 % · Life Orb 21 % (1 seen)
    · Choice Band 15 %. With no belief the rest is "?" (unknown). Returns
    ``([{"name", "p", "seen"(, "mega")}], known)`` — the top entries by share (a seen entry is
    never dropped), ``known`` = sightings counted."""
    n_pop = max(int(n_pop or 0), 1)
    known = min(int(sum(seen.values())), n_pop)
    rest = n_pop - known
    mass: dict = {}
    disp: dict = {}
    seen_by: dict = {}
    for raw, c in seen.items():
        k = toid(raw)
        mass[k] = mass.get(k, 0.0) + float(c)
        seen_by[k] = seen_by.get(k, 0) + int(c)
        disp.setdefault(k, _display(kind, raw))
    if rest > 0:
        tot = sum(p for _n, p in prior)
        if prior and tot > 0:
            for name, p in prior:
                k = toid(name)
                mass[k] = mass.get(k, 0.0) + rest * p / tot
                # the belief's display name wins — unless the blend data carries a bare id
                # ('sandforce', 'drought' in the 07-20 blend): map that like a sighting
                disp[k] = name if name != k else _display(kind, name)
        else:
            mass["?"] = mass.get("?", 0.0) + rest
            disp["?"] = "?"
    entries = []
    for k, m in mass.items():
        e = {"name": disp.get(k, k), "p": int(round(100.0 * m / n_pop)), "seen": seen_by.get(k, 0)}
        if kind == "abilities" and k != "?":
            e["mega"] = mega_of(species, k) is not None
        entries.append(e)
    entries.sort(key=lambda e: (-e["p"], -e["seen"], e["name"]))
    keep = entries[:top] + [e for e in entries[top:] if e["seen"]][:max(0, 4 - top)]
    return keep, known


def _merge_into(dst: dict, src: dict) -> None:
    for sp, st in src.items():
        d = dst.setdefault(sp, _new_stat())
        for k in ("games", "wins", "losses", "draws", "mega"):
            d[k] += st[k]
        d["items_anon"].update(st["items_anon"])
        d["abilities_anon"].update(st["abilities_anon"])
        d["item_by_opp"] |= st["item_by_opp"]
        d["ability_by_opp"] |= st["ability_by_opp"]
        d["opponents"] |= st["opponents"]
        if st["last_seen"] and (d["last_seen"] is None or st["last_seen"] > d["last_seen"]):
            d["last_seen"] = st["last_seen"]


class MatchupBook:
    def __init__(self) -> None:
        self._alltime: dict = {}        # (fmt, team) -> {species_id: stat}
        self._session: dict = {}        # (fmt, session_id) -> {species_id: stat}
        self._games: Counter = Counter()  # (scope, fmt, key) -> games
        self._opps: dict = {}           # (scope, fmt, key) -> set of opponent ids
        self._seen_tags: set = set()
        self.loaded_games = 0
        self.loaded_files = 0
        self.skipped = 0
        self.live_games = 0

    # ── recording ────────────────────────────────────────────────────────────
    @staticmethod
    def _clean_mons(opp_mons: Iterable) -> list:
        out: dict = {}
        for m in (opp_mons or []):
            if isinstance(m, str):
                m = {"species": m}
            if not isinstance(m, dict):
                continue
            sp = toid(m.get("species"))
            if not sp:
                continue
            item = m.get("item")
            item = None if item in _UNKNOWN_ITEMS else (toid(item) or None)
            ability = toid(m.get("ability")) or None
            # A proven mega (mega-only ability, or the live detector) holds its stone: that is the
            # revealed item, and the ability is the mega forme's (poke-env never sets the stone).
            mg = mega_of(sp, ability, is_mega=m.get("mega"))
            mega = False
            if mg:
                mega = True
                item = toid(mg.get("stone")) or item
                ability = toid(mg.get("ability")) or ability
            cur = out.setdefault(sp, {"species": sp, "item": None, "ability": None, "mega": False})
            cur["item"] = cur["item"] or item
            cur["ability"] = cur["ability"] or ability
            cur["mega"] = cur["mega"] or mega
        return list(out.values())

    def record_game(self, fmt: Optional[str], our_team: Optional[str], result: Optional[str],
                    opp_mons: Iterable, *, ts: Optional[str] = None, tag: Optional[str] = None,
                    session_id: Optional[str] = None, opponent: Optional[str] = None,
                    live: bool = False) -> bool:
        """One finished game. ``result`` = our bench-row result ("ai" = we won, "human" = we lost,
        "draw"); ``opp_mons`` = [{"species", "item", "ability"}] for the opponent's mons SEEN in the
        game (poke-env ids; unknown item = None / "unknown_item"). Returns False when skipped (no
        format, or a battle tag already recorded)."""
        fmt = (fmt or "").strip() or fmt_of_tag(tag)
        if not fmt:
            return False
        base = base_tag(tag) if tag else None
        if base:
            if base in self._seen_tags:
                return False
            self._seen_tags.add(base)
        rkey = _RESULTS.get(result)
        team = (our_team or "").strip() or UNKNOWN_TEAM
        mons = self._clean_mons(opp_mons)
        opp = toid(opponent) if opponent else None
        targets = [("alltime", fmt, team)]
        if session_id:
            targets.append(("session", fmt, str(session_id)))
        for scope, f, k in targets:
            store = (self._alltime if scope == "alltime" else self._session).setdefault((f, k), {})
            self._games[(scope, f, k)] += 1
            if opp:
                self._opps.setdefault((scope, f, k), set()).add(opp)
            for m in mons:
                st = store.setdefault(m["species"], _new_stat())
                st["games"] += 1
                if rkey:
                    st[rkey] += 1
                if m.get("mega"):
                    st["mega"] += 1
                if m["item"]:
                    if opp:
                        st["item_by_opp"].add((opp, m["item"]))     # once per opponent
                    else:
                        st["items_anon"][m["item"]] += 1
                if m["ability"]:
                    if opp:
                        st["ability_by_opp"].add((opp, m["ability"]))
                    else:
                        st["abilities_anon"][m["ability"]] += 1
                if opp:
                    st["opponents"].add(opp)
                if ts and (st["last_seen"] is None or str(ts) > str(st["last_seen"])):
                    st["last_seen"] = str(ts)
        if live:
            self.live_games += 1
        return True

    def record_battle(self, battle, row: dict, *, session_id: Optional[str] = None) -> bool:
        """The live hook: a finished poke-env battle + its bench row (``ai_team`` / ``result`` /
        ``opponent`` / ``session_id`` / ``battle_tag``). ``opponent_team`` holds the mons that
        APPEARED (poke-env falls back to the preview six only before anyone switched in)."""
        try:
            row = row or {}
            mons = []
            for mon in (getattr(battle, "opponent_team", None) or {}).values():
                mons.append({"species": getattr(mon, "species", None),
                             "item": getattr(mon, "item", None),
                             "ability": getattr(mon, "ability", None),
                             "mega": live_is_mega(mon)})
            tag = row.get("battle_tag") or getattr(battle, "battle_tag", None)
            return self.record_game(fmt_of_tag(tag), row.get("ai_team"), row.get("result"), mons,
                                    ts=row.get("ts"), tag=tag,
                                    session_id=row.get("session_id") or session_id,
                                    opponent=(row.get("opponent")
                                              or getattr(battle, "opponent_username", None)),
                                    live=True)
        except Exception:
            return False

    # ── the dossier seed ─────────────────────────────────────────────────────
    def load_dossiers(self, dossier_dir=DEFAULT_DOSSIER_DIR) -> tuple:
        """Seed the ALL-TIME book from every ``<opp>.json`` (games -> revealed species; the
        opponent's last-seen item / ability per species as the item proxy). Returns
        ``(games_loaded, files_loaded)``; unreadable files / tag-less games are counted in
        ``skipped``."""
        d = Path(dossier_dir)
        if not d.is_dir():
            return self.loaded_games, self.loaded_files
        for p in sorted(d.glob("*.json")):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                self.skipped += 1
                continue
            if not isinstance(doc, dict):
                self.skipped += 1
                continue
            mons = doc.get("mons") if isinstance(doc.get("mons"), dict) else {}
            opp = doc.get("opponent") or p.stem
            for g in (doc.get("games") or []):
                if not isinstance(g, dict):
                    continue
                tag = g.get("battle_tag")
                if not fmt_of_tag(tag):
                    self.skipped += 1
                    continue
                opp_mons = []
                megas_g = g.get("megas") if isinstance(g.get("megas"), list) else None
                for sp in (g.get("revealed") or []):
                    rec = mons.get(sp) if isinstance(mons.get(sp), dict) else {}
                    # New-shape dossiers (09-02): the mega forme's ability sits under
                    # ``mega_ability``, ``ability`` is the base's, and each game lists its
                    # ``megas``. A game that logged no mega for this species reads the base
                    # ability so the inference does not re-flag it. Old files (mega ability in
                    # ``ability``, no ``megas``) still resolve through ``mega_of``.
                    is_mega = (sp in megas_g) if megas_g is not None else None
                    ability = (rec.get("ability") if is_mega is False
                               else (rec.get("mega_ability") or rec.get("ability")))
                    opp_mons.append({"species": sp, "item": rec.get("item"),
                                     "ability": ability, "mega": is_mega})
                if self.record_game(None, g.get("our_team"), g.get("result"), opp_mons,
                                    ts=g.get("ts"), tag=tag, opponent=opp):
                    self.loaded_games += 1
            self.loaded_files += 1
        return self.loaded_games, self.loaded_files

    # ── queries ──────────────────────────────────────────────────────────────
    def teams(self, fmt: str) -> list:
        """Teams with all-time games in ``fmt``, most games first: [{"team", "games"}]."""
        out = [{"team": t, "games": self._games[("alltime", f, t)]}
               for (f, t) in self._alltime if f == fmt]
        out.sort(key=lambda d: (-d["games"], d["team"]))
        return out

    def formats(self) -> list:
        return sorted({f for (f, _t) in self._alltime})

    @staticmethod
    def _legacy(entries: list, prior: list) -> tuple:
        """(name, src, seen, pct) of the TOP blended entry — the single-value view the tests and
        the activity log use; None / "none" when the unknown remainder leads."""
        if not entries or entries[0]["name"] == "?":
            return None, "none", 0, None
        e = entries[0]
        return e["name"], ("seen" if e["seen"] else ("belief" if prior else "none")), int(e["seen"]), e["p"]

    def summary(self, fmt: str, *, team: Optional[str] = ALL_TEAMS,
                session_id: Optional[str] = None, top: int = 12, belief=None) -> dict:
        """Rows for one table. ``session_id`` set -> the session book (every team of that
        session); else the all-time book for ``team`` (``ALL_TEAMS`` / None = every team of the
        format)."""
        if session_id is not None:
            keys = [("session", fmt, str(session_id))]
            stores = [self._session.get((fmt, str(session_id)), {})]
        elif team in (None, ALL_TEAMS, ""):
            keys = [("alltime", f, t) for (f, t) in self._alltime if f == fmt]
            stores = [self._alltime[(f, t)] for (_s, f, t) in keys]
        else:
            keys = [("alltime", fmt, team)]
            stores = [self._alltime.get((fmt, team), {})]
        merged: dict = {}
        for s in stores:
            _merge_into(merged, s)
        rows = []
        for sp, st in merged.items():
            decided = st["wins"] + st["losses"]
            # population = distinct opponents who used the species (games when no names)
            n_pop = len(st["opponents"]) or st["games"]
            ip, ap = _prior(belief, sp, "items"), _prior(belief, sp, "abilities")
            items, item_known = _blend(_counts(st, "items"), n_pop, ip, kind="items")
            abilities, ability_known = _blend(_counts(st, "abilities"), n_pop, ap,
                                              kind="abilities", species=sp)
            item, item_src, item_n, item_pct = self._legacy(items, ip)
            ability, ability_src, ability_n, _apct = self._legacy(abilities, ap)
            ability_mega = bool(abilities and abilities[0].get("mega"))
            rows.append({"species": sp, "display": display_species(sp, belief),
                         "games": st["games"], "wins": st["wins"], "losses": st["losses"],
                         "draws": st["draws"],
                         "win_pct": (int(round(100.0 * st["wins"] / decided)) if decided else None),
                         # the blended distributions (USER 09-02) + the single-value legacy view
                         "items": items, "item_known": item_known,
                         "abilities": abilities, "ability_known": ability_known, "pop": n_pop,
                         "item": item, "item_src": item_src, "item_n": item_n, "item_pct": item_pct,
                         "ability": ability, "ability_src": ability_src, "ability_n": ability_n,
                         "ability_mega": ability_mega, "mega": st["mega"],
                         "opponents": len(st["opponents"]), "last_seen": st["last_seen"]})
        rows.sort(key=lambda r: (-r["games"], -r["wins"], r["species"]))
        games = sum(self._games[k] for k in keys)
        opps: set = set()
        for k in keys:
            opps |= self._opps.get(k, set())
        footer = {"games": games, "opponents": len(opps), "species": len(rows),
                  "teams": (sorted({t for (_s, _f, t) in keys}) if session_id is None else [])}
        return {"rows": rows[:max(0, int(top))], "footer": footer}

    def banner(self) -> str:
        fmts = ", ".join(self.formats()) or "no games"
        return (f"[panel] matchup book: {self.loaded_games} game(s) from {self.loaded_files} "
                f"dossier(s) — formats {fmts}"
                + (f" · {self.skipped} skipped" if self.skipped else ""))
