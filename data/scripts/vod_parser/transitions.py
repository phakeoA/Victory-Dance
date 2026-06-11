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

from vod_parser.replay_parser import (
    ShowdownReplayParser,
    extract_log_from_html,
    extract_replay_id_from_html,
)


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
) -> list:
    """
    Parse a replay and convert every turn into a JSONL-ready transition dict.

    Each transition records the state visible to one player at decision time,
    the actions taken that turn, the resulting state, and a decomposed reward
    block ready for RL training.

    Schema per transition
    ---------------------
    {
      "source_type": "ranked_player_vod",
      "replay_id": str,
      "format": str | null,
      "perspective": "p1" | "p2",
      "turn": int,
      "winner": str | null,              # username of winning player
      "state_before_actions": {
        "field": {...},
        "side_conditions": {"p1": {...}, "p2": {...}},
        "our_active":  {"p1a": {...}, "p1b": {...}},   # or p2a/p2b
        "opp_active":  {"p2a": {...}, "p2b": {...}},
        "our_bench":   [{...seen:true}, ...],
        "opp_bench":   [{...seen:bool}, ...],          # seen=false = unrevealed
        "known_team":  {"p1": [...], "p2": [...]},
      },
      "state_after_actions": { ...same shape... },
      "our_actions": [
        {"slot": "p1a", "action": "move",   "move": "Calm Mind", "target_slot": null,  ...},
        {"slot": "p1b", "action": "move",   "move": "Sucker Punch", "target_slot": "p2b", ...},
        {"slot": "p1b", "action": "switch", "species": "Incineroar"},
      ],
      "opp_actions_predicted": null,     # null in VOD data; filled at inference time
      "opp_actions_actual": [
        {"slot": "p2a", "action": "move", "move": "Rock Slide", "target_slot": null, ...},
      ],
      "reward": {
        "hp_delta_ours":   float | null,  # sum of hp_pct_delta for our active mons
        "hp_delta_theirs": float | null,  # sum of hp_pct_delta for opp active mons
        "tempo":           float | null,  # (hp_delta_ours - hp_delta_theirs) / 100
        "prediction_correct": null,       # null in VOD data; computed at inference time
        "win":             1 | -1 | null, # +1 on final turn if we win, -1 if we lose
      },
      "damage_events": [...],            # full event log for back-calculation
      "actions":       [...],            # full flat event log (moves, boosts, faints…)
      "players": {"our_side": "p1", "p1": {...}, "p2": {...}},
      "state_vector":  null,             # reserved for StateEncoder output
      "action_mask":   null,             # reserved for legal-action masking
    }

    Parameters
    ----------
    replay_path : path-like
    belief : BeliefState, optional
    encoder : StateEncoder, optional
    players : list of str, optional   e.g. ["p1"]  — defaults to ["p1", "p2"]
    known_teams : dict, optional
    """
    html_content = Path(replay_path).read_text(encoding="utf-8")
    replay_id = extract_replay_id_from_html(html_content) or Path(replay_path).stem

    known_entry: dict = {}
    if known_teams:
        known_entry = known_teams.get(replay_id, {})

    our_player = known_entry.get("_meta", {}).get("yourSide", "p1") or "p1"

    log = extract_log_from_html(html_content)
    parser = ShowdownReplayParser(log, our_player=our_player)
    battle = parser.parse()
    battle["replay_id"] = replay_id

    if players is None:
        players = ["p1", "p2"]

    transitions = []
    total_turns = len(battle["turns"])

    for turn_idx, turn in enumerate(battle["turns"]):
        is_final_turn = (turn_idx == total_turns - 1)

        for perspective in players:
            opp_perspective = "p2" if perspective == "p1" else "p1"

            # ── Inject known stats into state_after for our side ──────────
            after = copy.deepcopy(turn["state_after_actions"])
            for active_dict in (after.get("our_active") or {}).values():
                species = active_dict.get("species", "")
                inj = (known_entry.get(perspective) or {}).get(species, {})
                if inj:
                    active_dict["ev_spread"]  = inj.get("ev_spread")
                    active_dict["nature"]     = inj.get("nature")
                    active_dict["known_item"] = active_dict.get("known_item") or inj.get("item")
                    if inj.get("moves"):
                        active_dict["known_moves"] = inj["moves"]
            for bench_mon in (after.get("our_bench") or []):
                species = bench_mon.get("species", "")
                inj = (known_entry.get(perspective) or {}).get(species, {})
                if inj:
                    bench_mon["ev_spread"]  = inj.get("ev_spread")
                    bench_mon["nature"]     = inj.get("nature")
                    bench_mon["known_item"] = bench_mon.get("known_item") or inj.get("item")
                    if inj.get("moves"):
                        bench_mon["known_moves"] = inj["moves"]

            # ── Reward computation ────────────────────────────────────────
            # hp_delta_{ours,theirs}: sum of hp_pct_delta events for each side.
            # Damage events use negative deltas; heals use positive.
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
                our_username = battle["players"][perspective]["username"]
                win_signal   = 1 if battle["winner"] == our_username else -1

            # ── Perspective-aware action lists ────────────────────────────
            # our_actions / opp_actions_actual were extracted relative to the
            # parser's our_player.  We may need to swap if perspective != our_player.
            if perspective == our_player:
                our_actions        = turn.get("our_actions", [])
                opp_actions_actual = turn.get("opp_actions_actual", [])
            else:
                our_actions        = turn.get("opp_actions_actual", [])
                opp_actions_actual = turn.get("our_actions", [])

            t = {
                "source_type": battle.get("source_type", "ranked_player_vod"),
                "replay_id":   replay_id,
                "format":      battle.get("format"),
                "perspective": perspective,
                "turn":        turn["turn"],
                "winner":      battle.get("winner"),
                "state_before_actions": turn["state_before_actions"],
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
                "damage_events": turn.get("damage_events", []),
                "actions":       turn.get("actions", []),
                # Placeholders for future encoder output
                "state_vector": None,
                "action_mask":  None,
                "players": {
                    "our_side": our_player,
                    "p1": battle["players"]["p1"],
                    "p2": battle["players"]["p2"],
                },
            }
            transitions.append(t)

    return transitions
