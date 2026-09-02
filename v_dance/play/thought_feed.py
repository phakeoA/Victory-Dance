"""Thought feed — "what the nets are thinking", in words (USER 2026-09-02).

A per-battle ring buffer of verbose narration for the Online tab: what the TEAM-PREVIEW net scored
(set head / joint / greedy path, runner-up set, lead pair, near-tie deviation), what the BATTLE net
put on each slot's legal actions (top-3 with names and percentages, the 2b decode order and its
confidence, the conditional second slot, τ / top-p, futility drops), the GIMMICK head's mega margin,
the VALUE head's win probability, the opponent actives with their revealed-or-belief item / ability,
and a ⚠ line whenever what EXECUTED differs from the net's pick (a Showdown rejection, a forced
move, a missing model). So a decision that does not fit the board is explainable from the panel.

Design rules:
  * ZERO extra forwards — the taps read ``model_io.LAST_DECODE`` / ``LAST_TP``, the numbers the
    decode already computed. Formatting only.
  * Every tap is guarded by its caller; the feed itself never raises into play (``failed`` counts).
  * ``install(player)`` sets ``player._thoughts``; every tap is a no-op when that attribute is
    missing, so the local harnesses and tests are byte-identical without a feed.
  * The pure text builders (``turn_text`` / ``tp_text`` / ``replacement_text`` / ``action_label``)
    take plain dicts — torch- and poke-env-free, unit-tested without a model.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional, Sequence

SWITCH_OFFSET = 12          # encoder_layout: actions 0-11 = (move, target), 12-15 = bench switch
_MOVES_PER_MON = 4
_TARGETS_PER_MOVE = 3       # 0 = opp slot a, 1 = opp slot b, 2 = ally
_TAG_SPLIT = "-"


def base_tag(tag: str) -> str:
    parts = (tag or "").lstrip(">").split(_TAG_SPLIT)
    return _TAG_SPLIT.join(parts[:3]) if len(parts) >= 3 else (tag or "")


def _pct(p) -> str:
    try:
        return f"{int(round(100.0 * float(p)))}%"
    except Exception:
        return "?%"


def _sgn(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}"


def _title(s: str) -> str:
    return "-".join(w[:1].upper() + w[1:] for w in str(s or "").split("-"))


def _display(kind: str, raw) -> str:
    """poke-env id -> display name via the team-sheet maps; title-case fallback."""
    raw = str(raw or "")
    try:
        from v_dance.parser.vod_parser.team_sheet import _packed_display
        d = _packed_display(kind, raw)
        if d and str(d) != raw:               # a map miss echoes the id back ('latiosite')
            return str(d)
    except Exception:
        pass
    return _title(raw)


# ── pure text builders ───────────────────────────────────────────────────────
def action_label(a: Optional[int], moves: Sequence[dict], targets: Sequence[str],
                 bench: Sequence[str]) -> str:
    """One action index in words. ``moves`` = [{"name", "target"}] for the slot's mon (``target``
    ∈ normal | spread | field | ally), ``targets`` = [opp slot a, opp slot b, partner] names,
    ``bench`` = switch-in names in encoder bench order."""
    if a is None:
        return "pass (no legal action)"
    a = int(a)
    if a >= SWITCH_OFFSET:
        i = a - SWITCH_OFFSET
        return f"switch→{bench[i]}" if 0 <= i < len(bench) else f"switch→bench {i + 1} (?)"
    m, t = divmod(a, _TARGETS_PER_MOVE)
    mv = moves[m] if m < len(moves) else None
    name = (mv or {}).get("name") or f"move {m + 1}"
    kind = (mv or {}).get("target") or "normal"
    if kind == "spread":
        return f"{name}→spread"
    if kind == "field":
        return name
    if kind == "ally" or t == 2:
        return f"{name}→partner" + ("" if len(targets) < 3 or not targets[2] else f" ({targets[2]})")
    tgt = targets[t] if t < len(targets) and targets[t] else f"opp slot {t + 1}"
    return f"{name}→{tgt}"


def _top_actions(probs, mask, moves, targets, bench, pick, k: int = 3) -> str:
    n = len(probs or [])
    legal = [i for i in range(n) if (i < len(mask) and mask[i]) and float(probs[i]) > 0.0]
    legal.sort(key=lambda i: -float(probs[i]))
    parts = []
    for i in legal[:k]:
        lab = action_label(i, moves, targets, bench)
        parts.append(("✔ " if pick == i else "") + f"{lab} {_pct(probs[i])}")
    if pick is not None and pick not in legal[:k] and pick < n:
        parts.append(f"picked: {action_label(pick, moves, targets, bench)} {_pct(probs[pick])}")
    return " · ".join(parts) if parts else "no legal action"


def opp_line(opp: Sequence[Optional[dict]]) -> str:
    """'Incineroar 72% (item ? → Leftovers 60% belief · Intimidate belief) · Garchomp 100% (Life Orb
    revealed · Rough Skin revealed)'."""
    out = []
    for o in opp or []:
        if not o:
            continue
        bits = []
        if o.get("item_src") == "seen":
            bits.append(f"{o.get('item')} revealed")
        elif o.get("item_src") == "none_item":
            bits.append("no item (knocked off / consumed)")
        elif o.get("item"):
            pct = f" {o['item_pct']}%" if o.get("item_pct") is not None else ""
            bits.append(f"item ? → {o['item']}{pct} belief")
        else:
            bits.append("item ?")
        if o.get("ability"):
            bits.append(f"{o['ability']} {'revealed' if o.get('ability_src') == 'seen' else 'belief'}")
        hp = f" {o['hp']}%" if o.get("hp") is not None else ""
        out.append(f"{o.get('species', '?')}{hp} ({' · '.join(bits)})")
    return " · ".join(out) if out else "no opponent active visible"


def turn_text(ctx: dict, decode: dict, *, picks, wp=None, arm=None) -> str:
    """The battle-net narration for one normal turn.

    ``ctx`` (from ``battle_context``): own = [{"species", "moves"} | None] ×2, bench = [names],
    opp = [{"species", "hp", "item", "item_src", "item_pct", "ability", "ability_src"} | None] ×2.
    ``decode`` = ``model_io.LAST_DECODE``: pair, first, conf, probs (per slot), masks, tau,
    top_p, dropped, cond_on. ``picks`` = the FINAL (a0, a1) after the cross-slot dedup."""
    own = list(ctx.get("own") or [None, None]) + [None, None]
    opp = list(ctx.get("opp") or [None, None]) + [None, None]
    bench = list(ctx.get("bench") or [])
    targets = [(opp[0] or {}).get("species") or "", (opp[1] or {}).get("species") or "",
               ((own[1] or {}).get("species") or "")]
    head = f"turn {ctx.get('turn', '?')}"
    if wp is not None:
        try:
            w = float(wp)
            head += f" · win-prob {w:.2f} ({'winning' if w >= 0.55 else 'losing' if w <= 0.45 else 'even'})"
        except Exception:
            pass
    if arm:
        head += f" · arm {arm}"
    lines = [head, "opp: " + opp_line(opp[:2])]
    probs = decode.get("probs") or (None, None)
    masks = decode.get("masks") or ([], [])
    pair = bool(decode.get("pair"))
    first = decode.get("first")
    conf = decode.get("conf") or (None, None)
    for s in (0, 1):
        mon = own[s]
        if mon is None:
            lines.append(f"slot {s + 1}: empty / fainted")
            continue
        partner = own[1 - s]
        tg = [targets[0], targets[1], (partner or {}).get("species") or ""]
        tag = ""
        if pair and first is not None:
            if s == first:
                c = conf[s] if s < len(conf) and conf[s] is not None else None
                tag = " [decoded first" + (f", conf {_pct(c)}" if c is not None else "") + "]"
            else:
                cond = decode.get("cond_on")
                fmoves = (own[first] or {}).get("moves") or []
                ftg = [targets[0], targets[1], (own[1 - first] or {}).get("species") or ""]
                tag = (f" [given {action_label(cond, fmoves, ftg, bench)}]" if cond is not None
                       else " [zero-conditioned]")
        p = probs[s] if s < len(probs) else None
        m = masks[s] if s < len(masks) else []
        if p is None:
            lines.append(f"slot {s + 1} {mon.get('species', '?')}{tag}: (no decode numbers)")
            continue
        lines.append(f"slot {s + 1} {mon.get('species', '?')}{tag}: "
                     + _top_actions(p, m, mon.get("moves") or [], tg, bench, picks[s] if s < len(picks) else None))
    knobs = []
    tau = decode.get("tau")
    if tau:
        knobs.append(f"sampled at τ={float(tau):g}"
                     + (f" (top-p {float(decode['top_p']):g})" if decode.get("top_p") not in (None, 1.0, 1) else ""))
    dropped = decode.get("dropped") or ()
    if dropped and pair and first is not None:
        smoves = (own[1 - first] or {}).get("moves") or []
        stg = [targets[0], targets[1], (own[first] or {}).get("species") or ""]
        knobs.append("futility dropped: " + ", ".join(action_label(i, smoves, stg, bench) for i in dropped))
    if knobs:
        lines.append(" · ".join(knobs))
    return "\n".join(lines)


def tp_text(our: Sequence[str], opp: Sequence[str], picks: Sequence[int], lead_k: int,
            stash: dict, *, arm=None, source: str = "NET") -> str:
    """The team-preview narration from ``model_io.LAST_TP`` (set_head / joint / greedy)."""
    def names(idx):
        return " + ".join(our[i] if 0 <= i < len(our) else f"#{i}" for i in idx)
    lines = ["TEAM PREVIEW vs " + (", ".join(s for s in opp if s) or "?")
             + (f" · arm {arm}" if arm else "")]
    picks = list(picks)
    lines.append(f"{source}: bring {names(picks)} — leads {names(picks[:lead_k])}")
    path = (stash or {}).get("path")
    if path in ("set_head", "joint") and stash.get("subsets") and stash.get("scores"):
        subsets, scores = stash["subsets"], [float(x) for x in stash["scores"]]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        best = order[0]
        chosen = stash.get("set") or []
        ci = next((i for i, s in enumerate(subsets) if sorted(s) == sorted(chosen)), best)
        line = f"{'set head' if path == 'set_head' else 'joint decode'}: {names(sorted(chosen))} {scores[ci]:.2f}"
        line += " (best)" if ci == best else f" (best was {names(sorted(subsets[best]))} {scores[best]:.2f})"
        runner = next((i for i in order if i != ci), None)
        if runner is not None:
            line += (f" · runner-up {names(sorted(subsets[runner]))} {scores[runner]:.2f}"
                     f" (Δ{_sgn(scores[runner] - scores[ci])})")
        lines.append(line)
        ll = stash.get("lead_logits")
        leads = list(stash.get("leads") or picks[:lead_k])
        if ll and chosen:
            try:
                lead_scores = sorted(((float(ll[i]), i) for i in chosen), reverse=True)
                chosen_sum = sum(float(ll[i]) for i in leads)
                best_pair = lead_scores[:lead_k]
                best_sum = sum(v for v, _ in best_pair)
                lline = f"leads {names(leads)} {chosen_sum:.2f}"
                if sorted(i for _, i in best_pair) != sorted(leads):
                    lline += f" (argmax pair {names(sorted(i for _, i in best_pair))} {best_sum:.2f}, Δ{_sgn(chosen_sum - best_sum)})"
                else:
                    alt = [i for _, i in lead_scores[lead_k:]]
                    if alt:
                        alt_sum = sum(v for v, _ in lead_scores[1:lead_k + 1])
                        lline += f" (next pair {names(sorted([i for _, i in lead_scores[1:lead_k + 1]]))} {alt_sum:.2f}, Δ{_sgn(alt_sum - chosen_sum)})"
                lines.append(lline)
            except Exception:
                pass
        eps = stash.get("eps")
        if eps:
            sd, ld = bool(stash.get("set_dev")), bool(stash.get("lead_dev"))
            lines.append(f"near-tie sampling (eps {float(eps):g}): "
                         + ("SET DEVIATED from the argmax" if sd else "set kept")
                         + " · " + ("LEADS DEVIATED from the argmax" if ld else "leads kept"))
    elif path == "greedy" and stash.get("bring_logits"):
        bl = [float(x) for x in stash["bring_logits"]]
        ll = [float(x) for x in (stash.get("lead_logits") or [])]
        lines.append("greedy: bring logits " + " · ".join(
            f"{our[i] if i < len(our) else i} {bl[i]:.2f}" for i in sorted(range(len(bl)), key=lambda i: -bl[i])))
        if ll:
            lines.append("lead logits " + " · ".join(
                f"{our[i] if i < len(our) else i} {ll[i]:.2f}" for i in picks if i < len(ll)))
    elif path:
        lines.append(f"decode path: {path}")
    return "\n".join(lines)


def replacement_text(slots: Sequence[dict], bench: Sequence[str], *, arm=None) -> str:
    """``slots`` = [{"slot", "species", "probs", "mask", "pick"}] for the forced slots."""
    parts = []
    for s in slots:
        p, m = s.get("probs") or [], s.get("mask") or []
        parts.append(f"slot {int(s.get('slot', 0)) + 1} ({s.get('species') or 'fainted'}) → "
                     + _top_actions(p, m, [], ["", "", ""], bench, s.get("pick")))
    return "forced replacement" + (f" · arm {arm}" if arm else "") + ": " + (" · ".join(parts) or "nothing to replace")


# ── live-object adapter (duck-typed; injectable helpers for tests) ───────────
_SPREAD = ("ALL_ADJACENT_FOES", "ALL_ADJACENT", "ALL", "RANDOM_NORMAL")
_FIELD = ("SELF", "ALLY_SIDE", "FOE_SIDE", "ALLY_TEAM", "ALLIES")
_ALLY = ("ADJACENT_ALLY", "ADJACENT_ALLY_OR_SELF")


def _move_kind(move) -> str:
    t = getattr(move, "target", None)
    name = str(getattr(t, "name", None) or t or "").upper().replace("-", "_")
    if name in _SPREAD:
        return "spread"
    if name in _FIELD:
        return "field"
    if name in _ALLY:
        return "ally"
    return "normal"


def _opp_view(mon, belief) -> Optional[dict]:
    if mon is None:
        return None
    sp = getattr(mon, "species", None) or "?"
    out = {"species": _display("species", sp) if "-" in sp else _title(sp), "hp": None,
           "item": None, "item_src": "none", "item_pct": None, "ability": None, "ability_src": "none"}
    try:
        from v_dance.play.matchup_book import display_species
        out["species"] = display_species(sp, belief)
    except Exception:
        pass
    try:
        hp = getattr(mon, "current_hp_fraction", None)
        if hp is not None:
            out["hp"] = int(round(100.0 * float(hp)))
    except Exception:
        pass
    it = getattr(mon, "item", None)
    if it == "":
        out["item_src"] = "none_item"
    elif it and it != "unknown_item":
        out["item"], out["item_src"] = _display("items", it), "seen"
    elif belief is not None:
        try:
            top = belief.top_item(sp)
            if top:
                out["item"], out["item_src"] = str(top), "belief"
                for d in (belief.item_distribution(sp, 8) or []):
                    if d.get("name") == top:
                        out["item_pct"] = int(round(100.0 * float(d.get("p") or 0.0)))
                        break
        except Exception:
            pass
    ab = getattr(mon, "ability", None)
    if ab:
        out["ability"], out["ability_src"] = _display("abilities", ab), "seen"
    elif belief is not None:
        try:
            top = belief.top_ability(sp)
            if top:
                out["ability"], out["ability_src"] = str(top), "belief"
        except Exception:
            pass
    # A proven mega (mega-only ability / the live detector): the stone is the revealed item and
    # the ability is the mega forme's — poke-env keeps the base species and never sets the stone
    # (USER bug report 09-02: "Choice Scarf (belief) · Pixilate").
    try:
        from v_dance.play.matchup_book import live_is_mega, mega_of
        mg = mega_of(sp, ab, is_mega=live_is_mega(mon))
    except Exception:
        mg = None
    if mg:
        out["species"] = f"{out['species']} (Mega)"
        if mg.get("stone"):
            out["item"], out["item_src"], out["item_pct"] = str(mg["stone"]), "seen", None
        if mg.get("ability"):
            out["ability"], out["ability_src"] = str(mg["ability"]), "seen"
    return out


def battle_context(battle, *, belief=None, move_list_fn=None, bench_fn=None) -> dict:
    """The names the narration needs, read off a live poke-env DoubleBattle. ``move_list_fn`` /
    ``bench_fn`` default to the encoder's own helpers (the SAME lists the codec decodes with)."""
    if move_list_fn is None or bench_fn is None:
        from v_dance.encoders.live_state_encoder import own_active_move_list, own_bench_mons
        move_list_fn = move_list_fn or own_active_move_list
        bench_fn = bench_fn or own_bench_mons
    own = []
    try:
        actives = list(battle.active_pokemon)
    except Exception:
        actives = [None, None]
    for slot in (0, 1):
        mon = actives[slot] if slot < len(actives) else None
        if mon is None or getattr(mon, "fainted", False):
            own.append(None)
            continue
        moves = []
        try:
            for mv in move_list_fn(battle, slot, mon)[:_MOVES_PER_MON]:
                moves.append({"name": _display("moves", getattr(mv, "id", None) or str(mv)),
                              "target": _move_kind(mv)})
        except Exception:
            pass
        own.append({"species": _title(getattr(mon, "species", "?") or "?"), "moves": moves})
    bench = []
    try:
        bench = [_title(getattr(m, "species", "?") or "?") for m in bench_fn(battle)]
    except Exception:
        pass
    opp = []
    try:
        oa = list(battle.opponent_active_pokemon)
    except Exception:
        oa = [None, None]
    for slot in (0, 1):
        mon = oa[slot] if slot < len(oa) else None
        opp.append(_opp_view(mon, belief) if mon is not None and not getattr(mon, "fainted", False) else None)
    return {"own": own, "bench": bench, "opp": opp, "turn": getattr(battle, "turn", None)}


# ── the taps (module functions, so a stub ``player`` can never break a decision) ──────────
def _feed_of(player):
    return getattr(player, "_thoughts", None)


def _fail(feed) -> None:
    try:
        feed.failed += 1
    except Exception:
        pass


def note(player, battle, text: str) -> None:
    """A one-line note (why the net is NOT driving: rejection / forced move / missing model)."""
    feed = _feed_of(player)
    if feed is None:
        return
    try:
        feed.add(getattr(battle, "battle_tag", "?"), "note", text,
                 turn=getattr(battle, "turn", None), arm=getattr(player, "_arm_name", None),
                 opponent=getattr(battle, "opponent_username", None))
    except Exception:
        _fail(feed)


def tap_turn(player, battle, a0, a1, wp, *, decode: dict) -> None:
    """The battle-net narration for the decode just made (``decode`` = model_io.LAST_DECODE)."""
    feed = _feed_of(player)
    if feed is None:
        return
    try:
        belief = getattr(getattr(player, "_encoder", None), "belief", None)
        ctx = battle_context(battle, belief=belief)
        arm = getattr(player, "_arm_name", None)
        feed.add(battle.battle_tag, "turn",
                 turn_text(ctx, dict(decode or {}), picks=(a0, a1), wp=wp, arm=arm),
                 turn=getattr(battle, "turn", None), arm=arm,
                 opponent=getattr(battle, "opponent_username", None))
    except Exception:
        _fail(feed)


def tap_gimmick(player, battle, out, glog, *, gmask_fn, mega: int, none: int) -> None:
    """'gimmick: Mega Charizard NOW (margin +1.30)' appended to the turn block — only for slots
    where a gimmick is legal right now (nothing to say otherwise)."""
    feed = _feed_of(player)
    if feed is None or glog is None:
        return
    try:
        actives = list(battle.active_pokemon)
        bits = []
        for slot in (0, 1):
            mon = actives[slot] if slot < len(actives) else None
            if mon is None or slot >= len(glog):
                continue
            gmask = list(gmask_fn(battle, slot))
            if not any(bool(x) for x in gmask[1:]):
                continue
            row = glog[slot]
            margin = float(row[mega]) - float(row[none]) if len(row) > mega else 0.0
            sp = _title(getattr(mon, "species", "?") or "?")
            bits.append(f"Mega {sp} NOW (margin {margin:+.2f})" if out[slot] == mega
                        else f"no mega on {sp} (margin {margin:+.2f})")
        if bits:
            feed.amend(battle.battle_tag, "gimmick: " + " · ".join(bits),
                       turn=getattr(battle, "turn", None))
    except Exception:
        _fail(feed)


def tap_replacement(player, battle, logits, rep_masks, out, *, softmax) -> None:
    feed = _feed_of(player)
    if feed is None:
        return
    try:
        ctx = battle_context(battle)
        slots = []
        for slot in (0, 1):
            m = rep_masks[slot]
            if m is None:
                continue
            own = ctx["own"][slot] if slot < len(ctx["own"]) else None
            slots.append({"slot": slot, "species": (own or {}).get("species"),
                          "probs": list(softmax(logits[slot], m)), "mask": list(m),
                          "pick": out[slot]})
        arm = getattr(player, "_arm_name", None)
        feed.add(battle.battle_tag, "replacement", replacement_text(slots, ctx["bench"], arm=arm),
                 turn=getattr(battle, "turn", None), arm=arm,
                 opponent=getattr(battle, "opponent_username", None))
    except Exception:
        _fail(feed)


def tap_tp(player, battle, our_species, opp_species, picks, lead_k, *, stash: dict) -> None:
    feed = _feed_of(player)
    if feed is None:
        return
    try:
        from v_dance.play.matchup_book import display_species
        belief = getattr(getattr(player, "_encoder", None), "belief", None)
        our = [display_species(s, belief) if s else "?" for s in our_species]
        opp = [display_species(s, belief) if s else "" for s in opp_species]
        arm = getattr(player, "_arm_name", None)
        feed.add(battle.battle_tag, "tp", tp_text(our, opp, picks, lead_k, dict(stash or {}), arm=arm),
                 turn=0, arm=arm, opponent=getattr(battle, "opponent_username", None))
    except Exception:
        _fail(feed)


# ── the feed ─────────────────────────────────────────────────────────────────
class ThoughtFeed:
    def __init__(self, maxlen: int = 60, max_battles: int = 8, now=None) -> None:
        self.maxlen, self.max_battles = int(maxlen), int(max_battles)
        self._now = now or time.time
        self._by_tag: dict = {}         # base tag -> deque of entries (oldest first)
        self._meta: dict = {}           # base tag -> {arm, opponent, turn, started}
        self._finished: set = set()
        self.entries = 0
        self.failed = 0

    def install(self, player) -> "ThoughtFeed":
        try:
            player._thoughts = self
        except Exception:
            pass
        return self

    def banner(self) -> str:
        return (f"thought feed ACTIVE — TP + battle-net narration per decision "
                f"(last {self.maxlen} per battle, {self.max_battles} battles)")

    def _bucket(self, tag: str) -> deque:
        base = base_tag(tag)
        d = self._by_tag.get(base)
        if d is None:
            while len(self._by_tag) >= self.max_battles:
                victim = next((t for t in self._by_tag if t in self._finished), None) \
                    or next(iter(self._by_tag))
                self._by_tag.pop(victim, None)
                self._meta.pop(victim, None)
                self._finished.discard(victim)
            d = self._by_tag[base] = deque(maxlen=self.maxlen)
            self._meta[base] = {"arm": None, "opponent": None, "turn": None, "started": self._now()}
        return d

    def add(self, tag: str, kind: str, text: str, *, turn=None, arm=None, opponent=None) -> dict:
        d = self._bucket(tag)
        base = base_tag(tag)
        meta = self._meta[base]
        if arm:
            meta["arm"] = arm
        if opponent:
            meta["opponent"] = opponent
        if turn is not None:
            meta["turn"] = turn
        e = {"ts": time.strftime("%H:%M:%S", time.localtime(self._now())), "tag": base,
             "turn": turn, "kind": kind, "text": str(text)}
        d.append(e)
        self.entries += 1
        return e

    def amend(self, tag: str, text: str, *, turn=None) -> dict:
        """Append a line to the tag's LAST entry when it belongs to the same turn (the gimmick line
        rides the turn block); else a fresh note."""
        d = self._by_tag.get(base_tag(tag))
        if d and (turn is None or d[-1].get("turn") == turn):
            d[-1]["text"] = d[-1]["text"] + "\n" + str(text)
            return d[-1]
        return self.add(tag, "note", text, turn=turn)

    def mark_finished(self, tag: str) -> None:
        base = base_tag(tag)
        if base in self._by_tag:
            self._finished.add(base)

    def summary(self, live_tags=(), limit: int = 120) -> dict:
        """Live battles first (newest activity first), then finished ones; within a battle the
        newest entry first. ``limit`` caps the flattened entry list."""
        live = {base_tag(t) for t in (live_tags or ())}
        order = sorted(self._by_tag, key=lambda t: (t not in live, t in self._finished,
                                                    -(self._meta.get(t) or {}).get("started", 0.0)))
        battles, entries = [], []
        for t in order:
            d = self._by_tag[t]
            m = self._meta.get(t) or {}
            battles.append({"tag": t, "live": t in live, "finished": t in self._finished,
                            "arm": m.get("arm"), "opponent": m.get("opponent"),
                            "turn": m.get("turn"), "n": len(d)})
            for e in reversed(d):
                if len(entries) >= limit:
                    break
                entries.append(e)
        return {"battles": battles, "entries": entries, "total": self.entries, "failed": self.failed}
