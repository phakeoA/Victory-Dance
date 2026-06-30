"""
transitions.py
==============
Converts parsed Showdown replays into JSONL-ready training transitions.

Public surface
--------------
    parse_replay_for_preview    — used by server.py /parse endpoint
    replay_to_transitions       — used by server.py /export endpoint
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

from v_dance.parser.vod_parser.pokedex import get_pokedex, norm_species
from v_dance.parser.vod_parser.replay_parser import (
    ShowdownReplayParser,
    extract_log_from_html,
    extract_replay_id_from_html,
)


def _merge_known_moves(injected: Optional[list], revealed: Optional[list]) -> list:
    """Combine user-injected move slots with replay-revealed moves.

    The UI sends fixed 4-slot arrays where untouched slots are '' — those
    placeholders must never reach the export (the encoder treats a non-empty
    ``known_moves`` as the complete authoritative moveset).  Injected order
    wins (it is stable across turns), revealed moves the user didn't type are
    appended, duplicates are matched loosely ("Close Combat" == "closecombat"),
    and the result is capped at the 4 real move slots.

    When the union exceeds 4 the typed data contradicts the log (a mon only
    has 4 moves) — replay reveals are ground truth and always survive the
    cap; unconfirmed typed moves are dropped first, last-typed first.
    """
    revealed_keys = {norm_species(str(mv)) for mv in (revealed or [])
                     if mv and str(mv).strip()}
    merged: list = []
    seen: set = set()
    for mv in list(injected or []) + list(revealed or []):
        if not mv or not str(mv).strip():
            continue
        key = norm_species(str(mv))
        if key in seen:
            continue
        seen.add(key)
        merged.append(str(mv).strip())
    i = len(merged) - 1
    while len(merged) > 4 and i >= 0:
        if norm_species(merged[i]) not in revealed_keys:
            merged.pop(i)
        i -= 1
    return merged[:4]   # >4 revealed (called-move artifacts) → hard cap


def _inject_known_stats(mon_dict: dict, inj: dict) -> None:
    """Merge a user-supplied known_teams entry into one mon's state dict.

    Bug 8 (mega ability split): the user-injected ``ability`` always refers
    to the BASE forme's ability — that is the only free choice a player makes.
    A mega forme's ability is fixed and fully determined by the species, so:

      * non-mega mon  → injected ability fills known_ability (if not already
                        revealed in the replay) and pre_mega_ability.
      * mega'd mon    → injected ability only fills pre_mega_ability;
                        known_ability stays/becomes the pokedex mega ability.
    """
    if not inj:
        return
    if inj.get("ev_spread") is not None:
        mon_dict["ev_spread"] = inj["ev_spread"]
    if inj.get("nature") is not None:
        mon_dict["nature"] = inj["nature"]
    # IVs default to 31 across the board in this format; the UI never asks
    # for them, so stamp the default whenever the user vouched for the
    # mon's stats at all (previously iv_spread stayed null forever).
    if (inj.get("ev_spread") or inj.get("nature")) and not mon_dict.get("iv_spread"):
        mon_dict["iv_spread"] = [31] * 6
    mon_dict["known_item"] = mon_dict.get("known_item") or inj.get("item")
    # Only an actual user-typed move makes the moveset "known" — a merge of
    # revealed moves alone stays in revealed_moves (it may be incomplete).
    if any(m and str(m).strip() for m in (inj.get("moves") or [])):
        mon_dict["known_moves"] = _merge_known_moves(
            inj["moves"], mon_dict.get("revealed_moves")
        )

    inj_ability = inj.get("ability")
    if mon_dict.get("is_mega"):
        # Base ability is historical context only — never the active one.
        if inj_ability:
            mon_dict["pre_mega_ability"] = mon_dict.get("pre_mega_ability") or inj_ability
        # Guarantee the active ability is the fixed mega ability.
        if not mon_dict.get("known_ability"):
            dex = get_pokedex()
            mega_ab = (
                mon_dict.get("mega_ability")
                or (dex.mega_ability_for(mon_dict.get("species")) if dex else None)
            )
            if mega_ab:
                mon_dict["mega_ability"]  = mega_ab
                mon_dict["known_ability"] = mega_ab
    elif inj_ability:
        mon_dict["pre_mega_ability"] = mon_dict.get("pre_mega_ability") or inj_ability
        mon_dict["known_ability"]    = mon_dict.get("known_ability") or inj_ability


def _apply_known_injection(snaps, side_inj: dict) -> None:
    """Inject a side's user-supplied known stats into each snapshot's own mons.

    Shared by the normal per-turn path and the post-faint replacement path.
    Mons already enriched as ``exact`` (sheet-complete inject cards) are left
    alone — a raw re-inject would clobber their computed stats.
    """
    def _lookup(mon_dict: dict) -> dict:
        base = mon_dict.get("base_species") or mon_dict.get("species", "")
        return side_inj.get(base) or side_inj.get(mon_dict.get("species", "")) or {}

    for snap in snaps:
        for active_dict in (snap.get("our_active") or {}).values():
            if not active_dict.get("exact"):
                _inject_known_stats(active_dict, _lookup(active_dict))
        for bench_mon in (snap.get("our_bench") or []):
            if not bench_mon.get("exact"):
                _inject_known_stats(bench_mon, _lookup(bench_mon))


def _retrofit_own_side_knowledge(battle: dict) -> None:
    """Rebuild each side's OWN decision-time knowledge into its snapshots.

    The parser is strictly log-faithful: a turn-N snapshot only contains what
    an observer had seen by turn N.  But the ACTING player always knew their
    own moveset and which mons they brought — without that, early-game
    decisions are not expressible in the action space (the first use of every
    move, and every switch to a not-yet-fielded mon, would encode as None).

    Two retroactive merges per physical side, applied only to that side's own
    half of its own-perspective snapshots (the opponent half keeps the
    progressive view — merging there would leak future information):

      · every own mon's revealed_moves becomes the battle-END reveal list
        (the live list is always a prefix of it, so per-mon move indices are
        stable across the whole battle);
      · our_bench gains seen=False stubs for brought mons that have not
        entered yet — the own-side mirror of opp_bench's unseen roster stubs.

    Mons brought but never fielded stay unknown (the log cannot name them);
    masks under-approximate switch options in that rare case.
    """
    revealed_info = battle.get("revealed_info") or {}
    players = battle.get("players") or {}

    def _final_moves(pid: str, mon: dict) -> list:
        base = mon.get("base_species") or mon.get("species")
        final = (revealed_info.get(f"{pid}:{base}") or {}).get("revealed_moves") or []
        cur = mon.get("revealed_moves") or []
        # final is an append-only superset of cur; keep cur if the lookup
        # misses (forme/nickname edge) rather than dropping live reveals
        return list(final) if len(final) >= len(cur) else list(cur)

    def _process(pid: str, snap: dict) -> None:
        if not isinstance(snap, dict):
            return
        own_mons = (list((snap.get("our_active") or {}).values())
                    + list(snap.get("our_bench") or []))
        present = set()
        for mon in own_mons:
            mon["revealed_moves"] = _final_moves(pid, mon)
            present.add(norm_species(
                mon.get("base_species") or mon.get("species") or ""))
        for sp in (players.get(pid) or {}).get("brought") or []:
            if norm_species(sp) in present:
                continue
            info = revealed_info.get(f"{pid}:{sp}") or {}
            snap.setdefault("our_bench", []).append({
                "species": sp, "base_species": sp, "nickname": None,
                "player": pid, "slot": None,
                "hp_pct": None, "status": None, "boosts": {},
                "is_mega": False, "is_fainted": False,
                "revealed_moves": list(info.get("revealed_moves") or []),
                "known_item": None, "known_tera_type": None,
                "is_terastallized": False,
                "known_ability": None, "pre_mega_ability": None,
                "mega_ability": None,
                "can_have_choice_item": info.get("can_have_choice_item", True),
                "is_transformed": False, "transformed_into": None,
                "ev_spread": None, "iv_spread": None, "nature": None,
                "seen": False,
            })

    def _process_sides(sides) -> None:
        if not isinstance(sides, dict) or "our_active" in sides:
            return              # flattened preview shape — not this path
        for pid, snap in sides.items():
            _process(pid, snap)

    for turn in battle.get("turns") or []:
        for key in ("state_before_actions", "state_after_actions"):
            _process_sides(turn.get(key))
        # Post-faint replacement decisions carry their own both-perspective state.
        for repl in turn.get("replacements") or []:
            _process_sides(repl.get("state"))


def _known_entry_side_to_sheet(side_dict: Optional[dict]) -> list[dict]:
    """Convert one side of a UI known_teams entry into team-sheet entries
    (the shape belief_state.fill_blanks expects for its "exact" fill mode).

    Only mons whose stats are actually pinned down — nature AND ev_spread
    both filled in — are promoted to sheet entries; computing "exact" stats
    from a half-filled inject card would bake wrong numbers into the export.
    Partially-filled mons keep the legacy per-field injection path and get a
    "not found on team sheet" warning from fill_blanks.
    """
    sheet: list[dict] = []
    for species, inj in (side_dict or {}).items():
        if not inj or not (inj.get("nature") and inj.get("ev_spread")):
            continue
        sheet.append({
            "species": species,
            "nickname": None,
            "item": inj.get("item"),
            "ability": inj.get("ability"),
            "moves": [m for m in (inj.get("moves") or []) if m and str(m).strip()],
            "nature": inj.get("nature"),
            "evs": inj.get("ev_spread") or {},
            "ivs": {},          # UI never asks for IVs → default 31s
            "tera_type": None,
            "level": 50,
        })
    return sheet


def parse_replay_for_preview(
    html_content: str,
    belief=None,
    known_entry: Optional[dict] = None,
) -> dict:
    """
    Parse a Showdown replay HTML string and return the structured battle dict
    that the team-builder UI expects.

    Parameters
    ----------
    html_content : str
        Raw contents of the .html replay file.
    belief : BeliefState, optional
        Pikalytics belief state (unused for now; accepted so the signature
        stays stable when belief inference is wired in).
    known_entry : dict, optional
        A partial known_teams entry the user has already filled in.  Shape::

            {
              "p1": { "Kingambit": { "nature": "Adamant", ... } },
              "p2": { ... },
              "_meta": { "yourSide": "p1", "winner": "p1", ... }
            }

        When supplied, the ``_meta`` fields are merged into the preview's
        top-level ``players`` dict and the per-species dicts are surfaced as
        ``known_team_overrides`` so the UI can pre-populate injection cards.

    Returns
    -------
    dict
        Full battle object matching the shape produced by the JS
        ``parseShowdownLog()`` function.
    """
    log = extract_log_from_html(html_content)
    replay_id = extract_replay_id_from_html(html_content)

    # Determine our_player from known_entry if present
    our_player = "p1"
    if known_entry:
        meta = known_entry.get("_meta", {})
        our_player = meta.get("yourSide", "p1") or "p1"

    parser = ShowdownReplayParser(log, our_player=our_player)
    result = parser.parse()
    result["replay_id"] = replay_id
    result["known_team_overrides"] = {}

    # The parser now stores state_before_actions / state_after_actions as
    # {"p1": <snap>, "p2": <snap>} to support both-perspective transitions.
    # The preview UI expects a single flat state object (our_player's view),
    # so flatten each turn's snapshots here before handing off to the UI.
    for turn in result.get("turns", []):
        if isinstance(turn.get("state_before_actions"), dict) and our_player in turn["state_before_actions"]:
            turn["state_before_actions"] = turn["state_before_actions"][our_player]
        if isinstance(turn.get("state_after_actions"), dict) and our_player in turn["state_after_actions"]:
            turn["state_after_actions"] = turn["state_after_actions"][our_player]

    # Merge known_entry into the result
    if known_entry:
        meta = known_entry.get("_meta", {})

        if meta.get("winner"):
            result["winner"] = meta["winner"]

        # Surface per-species data as known_team_overrides
        for pid in ("p1", "p2"):
            for species, data in (known_entry.get(pid) or {}).items():
                key = f"{pid}:{species}"
                if any(v is not None for v in data.values()):
                    result["known_team_overrides"][key] = data

    return result


def replay_to_transitions(
    replay_path,
    belief=None,
    encoder=None,
    players: Optional[list] = None,
    known_teams: Optional[dict] = None,
    source_type: Optional[str] = None,
) -> list:
    """
    Parse a replay and convert every turn into a JSONL-ready transition dict.

    Each transition records the state visible to one player at decision time,
    the actions taken that turn, the resulting state, and a decomposed reward
    block ready for RL training.

    Schema per transition
    ---------------------
    {
      "source_type": "own_vod" | "ranked_player_vod" | "live_bot_battle"
                     | "self_play",      # canonical Type A/B/C/D token
      "replay_id": str,
      "format": str | null,
      "perspective": "p1" | "p2",
      "turn": int,
      "winner": str | null,              # username of winning player
      "state_before_actions": {
        "field": {...},
        "side_conditions": {"our_side": {...}, "opp_side": {...}},
        "our_active":  {"our_a": {...}, "our_b": {...}},   # perspective-relative keys
        "opp_active":  {"opp_a": {...}, "opp_b": {...}},   # same for both perspectives
        "our_bench":   [{...seen:true}, ...],
        "opp_bench":   [{...seen:bool}, ...],              # seen=false = unrevealed
        "known_team":  {"p1": [...], "p2": [...]},
      },
      # Mon dicts in BOTH snapshots are enriched per the vod type's fill
      # modes (belief_state.fill_blanks): distribution sides carry "belief"
      # blocks (EV-spread/item/ability/move distributions) + stats_estimate
      # mode "distribution"; sheet-complete exact sides carry "exact"
      # (computed stats, bucket EV lists, IVs) + stats_estimate mode "exact".
      "state_after_actions": { ...same shape... },
      "our_actions": [
        # action_index = fixed 0–15 policy index (state_encoder action codec):
        #   move m × target t → m*3+t (t: 0=opp_a 1=opp_b 2=ally;
        #   un-choosable targets canonicalise to 0), switch → 12+bench_slot.
        #   None when inexpressible (move outside the 4 encoded slots /
        #   mid-turn replacement from a mon that was active at turn start).
        {"slot": "our_a", "action": "move",   "move": "Calm Mind", "target_slot": null,  "action_index": 0, ...},
        {"slot": "our_b", "action": "move",   "move": "Sucker Punch", "target_slot": "opp_b", "action_index": 4, ...},
        {"slot": "our_b", "action": "switch", "species": "Incineroar", "action_index": 12},
      ],
      "opp_actions_predicted": null,     # null in VOD data; filled at inference time
      "opp_actions_actual": [
        {"slot": "opp_a", "action": "move", "move": "Rock Slide", "target_slot": null, ...},
      ],
      "reward": {
        "hp_delta_ours":   float | null,  # sum of hp_pct_delta for our active mons
        "hp_delta_theirs": float | null,  # sum of hp_pct_delta for opp active mons
        "tempo":           float | null,  # (hp_delta_ours - hp_delta_theirs) / 100
        "prediction_correct": null,       # null in VOD data; computed at inference time
        "win":             1 | -1 | null, # +1 on final turn if we win, -1 if we lose
      },
      "damage_events": [...],            # full event log; slots normalised to our_*/opp_*
      "actions":       [...],            # full flat event log; slots normalised to our_*/opp_*
      "belief_fill":   {...},            # fill_blanks audit metadata: vod_type,
                                         # fill_modes, warnings, pikalytics source
                                         # (per-turn validation details trimmed)
      "players": {"our_side": "p1"|"p2", "p1": {...}, "p2": {...}},
                                         # each side carries roster (all 6
                                         # teampreview mons) AND brought (the
                                         # ≤4 that entered, switch-in order,
                                         # leads first) — teampreview-choice
                                         # training target

      "state_vector":  null,             # encoded at TRAINING time (raw JSON
                                         # ages better than baked vectors)
      "action_mask":   {                 # decision-time legality per slot,
        "our_a": [0/1 × 16],             # from state_before_actions (see
        "our_b": [0/1 × 16],             # state_encoder.build_action_mask)
      },
    }

    Parameters
    ----------
    replay_path : path-like
    belief : BeliefState, optional
    encoder : StateEncoder, optional
    players : list of str, optional   e.g. ["p1"]  — defaults to ["p1", "p2"]
    known_teams : dict, optional
    source_type : str, optional
        The UI's Type A/B/C/D selector value ("own_vod", "ranked_player_vod",
        "bot_vod", "self_play", or a bare letter).  Canonicalised via
        belief_state.VodType and stamped on every transition.  Defaults to
        Type B ("ranked_player_vod") when omitted.
    """
    html_content = Path(replay_path).read_text(encoding="utf-8")
    replay_id = extract_replay_id_from_html(html_content) or Path(replay_path).stem

    known_entry: dict = {}
    if known_teams:
        known_entry = known_teams.get(replay_id, {})
        # server.py wraps a single user-approved entry under the UI's
        # battle_id, which may differ from the replayid embedded in the HTML
        # (e.g. manually created battles).  If the id lookup misses and there
        # is exactly one entry, it is unambiguous — use it rather than
        # silently dropping every injected stat.
        if not known_entry and len(known_teams) == 1:
            known_entry = next(iter(known_teams.values())) or {}

    our_player = known_entry.get("_meta", {}).get("yourSide", "p1") or "p1"

    log = extract_log_from_html(html_content)
    parser = ShowdownReplayParser(log, our_player=our_player)
    battle = parser.parse()
    battle["replay_id"] = replay_id

    # ── Canonicalise the source type (UI selector → spec token) ──────────
    # Lazy import: belief_state imports _inject_known_stats from this module
    # at top level, so importing it here avoids a circular import.
    from v_dance.parser.belief_state import VodType, fill_blanks

    vt = VodType.coerce(source_type or battle.get("source_type") or "B")

    # Each side's own half of its snapshots gets its full own-team knowledge
    # (final movesets + brought-but-unentered bench stubs) BEFORE enrichment,
    # so belief blocks and the action codec see the acting player's view.
    _retrofit_own_side_knowledge(battle)

    # ── Belief / exact enrichment (the Type A–D fill modes) ──────────────
    # fill_blanks walks BOTH state_before_actions and state_after_actions of
    # every turn and, per the vod type, attaches Pikalytics belief blocks +
    # stats_estimate to distribution sides and computed exact stats to sheet
    # sides.  Sheets are derived from the user's inject panel; fill_blanks
    # ignores them on sides whose fill mode is "distribution", so passing
    # them unconditionally is safe.  This also stamps battle["source_type"]
    # with the canonical token and battle["belief_fill"] with the audit
    # metadata (warnings, fill modes, pikalytics source).
    opp_player = "p2" if our_player == "p1" else "p1"
    battle = fill_blanks(
        battle, vt,
        belief=belief,
        team_sheet=_known_entry_side_to_sheet(known_entry.get(our_player)) or None,
        opp_team_sheet=_known_entry_side_to_sheet(known_entry.get(opp_player)) or None,
        our_side=our_player,
    )

    # Action codec (optional — state_encoder needs numpy).  Annotates every
    # transition with action_index per our_actions entry + the decision-time
    # action_mask.  Lazy for the same circular-import reason as belief_state.
    try:
        from v_dance.encoders.state_encoder import (
            annotate_transition_actions, action_to_index, _living_bench,
            SWITCH_OFFSET, ACTIONS_PER_SLOT, GIMMICK_DIM, GIMMICK_NONE,
        )
    except ImportError:
        annotate_transition_actions = None

    # Per-transition audit copy — drop the bulky per-turn details, keep the
    # warnings (they are the "this export is underfilled" signal).
    raw_fill = battle.get("belief_fill") or {}
    fill_meta = {k: v for k, v in raw_fill.items()
                 if k not in ("back_calc", "validation")}
    if "validation" in raw_fill:
        v = raw_fill["validation"] or {}
        fill_meta["validation"] = {
            "complete": bool(v.get("complete")),
            "turns_with_missing": len(v.get("missing_by_turn") or {}),
        }

    if players is None:
        players = ["p1", "p2"]

    transitions = []
    total_turns = len(battle["turns"])

    for turn_idx, turn in enumerate(battle["turns"]):
        is_final_turn = (turn_idx == total_turns - 1)

        for perspective in players:
            opp_perspective = "p2" if perspective == "p1" else "p1"

            # ── Pull the perspective-correct state snapshots ──────────────
            # state_before_actions and state_after_actions are now dicts keyed
            # by "p1"/"p2" — always index by perspective here so both sides
            # get the right our_active/opp_active/bench split.
            state_before = copy.deepcopy(turn["state_before_actions"][perspective])
            after = copy.deepcopy(turn["state_after_actions"][perspective])

            # ── Inject known stats for our side — into BOTH snapshots ─────
            # state_before_actions is the state the model decides from; only
            # injecting into state_after (the old behaviour) starved the
            # model's input of every user-supplied EV/nature/item.
            # Lookup uses BASE species (Bug 7/8): known_teams entries are
            # keyed by roster names ("Floette-Eternal"), while a mega'd
            # active's `species` is the mega forme ("Floette-Mega").
            side_inj = known_entry.get(perspective) or {}
            _apply_known_injection((state_before, after), side_inj)

            # ── Reward computation ────────────────────────────────────────
            # hp_delta_{ours,theirs}: sum of hp_pct_delta events for each side.
            # Damage events use negative deltas; heals use positive.
            # NOTE: damage_events still use absolute slot notation (p1a, p2a)
            # because they come straight from the parser log.  We resolve them
            # against perspective here for the reward calc, then normalise the
            # slots before writing into the transition below.
            hp_delta_ours   = 0.0
            hp_delta_theirs = 0.0
            has_damage      = False
            for ev in turn.get("damage_events", []):
                delta = ev.get("hp_pct_delta")
                if delta is None:
                    continue
                has_damage = True
                slot = ev.get("slot", "")
                if slot.startswith(perspective):
                    hp_delta_ours   += delta
                elif slot.startswith(opp_perspective):
                    hp_delta_theirs += delta

            # Tempo = net HP swing in our favour, normalised to [-1, 1]
            tempo = None
            if has_damage:
                tempo = round((hp_delta_ours - hp_delta_theirs) / 100.0, 4)
                hp_delta_ours   = round(hp_delta_ours, 2)
                hp_delta_theirs = round(hp_delta_theirs, 2)
            else:
                hp_delta_ours = hp_delta_theirs = None

            # Win/loss signal on the final turn only
            win_signal = None
            if is_final_turn and battle.get("winner"):
                our_username = (battle["players"].get(perspective) or {}).get("username")
                # only credit when our side is identifiable by name; a missing/blank username must stay
                # UNKNOWN (None), not silently flip the actual winner to a loss (corpus_qa guards identically).
                if our_username:
                    win_signal = 1 if battle["winner"] == our_username else -1

            # ── Perspective-aware action lists ────────────────────────────
            # our_actions / opp_actions_actual were extracted relative to the
            # parser's our_player.  Swap if perspective differs from our_player.
            if perspective == our_player:
                our_actions_raw        = turn.get("our_actions", [])
                opp_actions_actual_raw = turn.get("opp_actions_actual", [])
            else:
                our_actions_raw        = turn.get("opp_actions_actual", [])
                opp_actions_actual_raw = turn.get("our_actions", [])

            # ── Normalise all absolute slot strings to perspective-relative ─
            # After this point every "slot"/"target_slot"/"source_slot" field
            # reads "our_a", "our_b", "opp_a", "opp_b" regardless of which
            # physical player the transition was generated for.
            normalize = lambda evs: ShowdownReplayParser.normalize_slots(evs, perspective)

            our_actions        = normalize(our_actions_raw)
            opp_actions_actual = normalize(opp_actions_actual_raw)
            damage_events      = normalize(turn.get("damage_events", []))
            actions            = normalize(turn.get("actions", []))

            t = {
                "source_type": battle.get("source_type", "ranked_player_vod"),
                "replay_id":   replay_id,
                "format":      battle.get("format"),
                "perspective": perspective,
                "turn":        turn["turn"],
                "total_turns": total_turns,
                "winner":      battle.get("winner"),
                "decision_type": "turn",   # vs "replacement" (post-faint switch)
                "state_before_actions": state_before,
                "state_after_actions":  after,
                "our_actions":           our_actions,
                "opp_actions_predicted": None,
                "opp_actions_actual":    opp_actions_actual,
                "reward": {
                    "hp_delta_ours":      hp_delta_ours,
                    "hp_delta_theirs":    hp_delta_theirs,
                    "tempo":              tempo,
                    "prediction_correct": None,   # filled at inference time
                    "win":                win_signal,
                },
                "damage_events": damage_events,
                "actions":       actions,
                "belief_fill":   fill_meta,
                # Placeholders for future encoder output
                "state_vector": None,
                "action_mask":  None,
                "gimmick_mask": None,   # filled by annotate_transition_actions
                "players": {
                    "our_side": perspective,
                    "p1": battle["players"]["p1"],
                    "p2": battle["players"]["p2"],
                },
            }
            if annotate_transition_actions is not None:
                annotate_transition_actions(t)
            transitions.append(t)

        # ── Post-faint replacement decisions (the #1 gap) ─────────────────
        # Each is its own (board-after-faint → which bench mon) example, from
        # the replacing player's perspective only (the ally is not deciding).
        # Needs the action codec to build the switch-only mask + index.
        if annotate_transition_actions is not None:
            for repl in turn.get("replacements") or []:
                rp = repl.get("player")
                if rp not in players:
                    continue
                snap_src = (repl.get("state") or {}).get(rp)
                if not snap_src:
                    continue
                state_before = copy.deepcopy(snap_src)
                after        = copy.deepcopy(snap_src)   # pure switch → reward null
                _apply_known_injection((state_before, after),
                                       known_entry.get(rp) or {})

                rel_slot = "our_" + (repl.get("slot") or "")[2:]   # p1a → our_a
                switch_action = {
                    "slot": rel_slot, "action": "switch",
                    "species": repl.get("species"),
                }

                # Switch-only mask: the fainted slot must pick a living-bench
                # mon; the ally slot is not deciding (all-zero).  build_action_mask
                # gives an EMPTY active slot an all-zero row, so we build it here.
                bench = _living_bench(state_before)
                row = [0] * ACTIONS_PER_SLOT
                for i in range(len(bench)):
                    row[SWITCH_OFFSET + i] = 1
                idx = action_to_index(switch_action, None, state_before)
                if idx is None or idx >= ACTIONS_PER_SLOT or row[idx] != 1:
                    idx = None     # replacement not on the living bench (rare)
                switch_action["action_index"] = idx
                mask = {"our_a": [0] * ACTIONS_PER_SLOT,
                        "our_b": [0] * ACTIONS_PER_SLOT}
                if rel_slot in mask:
                    mask[rel_slot] = row

                # A replacement is switch-only — it NEVER gimmicks.  The deciding
                # (fainted) slot may only pick "no gimmick" (bucket 0); mega is
                # never legal on a replacement.  The ally slot is not deciding →
                # all-zero, mirroring the switch-only action mask above.
                gimmick_mask = {"our_a": [0] * GIMMICK_DIM,
                                "our_b": [0] * GIMMICK_DIM}
                if rel_slot in gimmick_mask:
                    gimmick_mask[rel_slot][GIMMICK_NONE] = 1
                switch_action["gimmick_index"] = GIMMICK_NONE if idx is not None else None

                transitions.append({
                    "source_type": battle.get("source_type", "ranked_player_vod"),
                    "replay_id":   replay_id,
                    "format":      battle.get("format"),
                    "perspective": rp,
                    "turn":        turn["turn"],
                    "total_turns": total_turns,
                    "winner":      battle.get("winner"),
                    "decision_type": "replacement",
                    "state_before_actions": state_before,
                    "state_after_actions":  after,
                    "our_actions":           [switch_action],
                    "opp_actions_predicted": None,
                    "opp_actions_actual":    [],
                    "reward": {
                        "hp_delta_ours": None, "hp_delta_theirs": None,
                        "tempo": None, "prediction_correct": None, "win": None,
                    },
                    "damage_events": [],
                    "actions":       [],
                    "belief_fill":   fill_meta,
                    "state_vector":  None,
                    "action_mask":   mask,
                    "gimmick_mask":  gimmick_mask,
                    "players": {
                        "our_side": rp,
                        "p1": battle["players"]["p1"],
                        "p2": battle["players"]["p2"],
                    },
                })

    return transitions
