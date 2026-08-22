"""v11 Phase C.3 — Fake-Out first-turn legality (first_turn per-mon volatile channel).

Fake Out / First Impression / Mat Block are usable ONLY on a mon's first acting turn since switch-in
(Showdown onTry: activeMoveActions > 1 → fail). The encoder gets ONE per-mon ``first_turn`` bool (12th
volatile channel), mirroring poke-env ``Pokemon.first_turn = (_active_turns == 1)``: the offline parser
tracks ``PokemonSlot.active_turns`` (reset 0 on switch-IN, +1 per |turn| while active, BEFORE the snapshot).
Representation chosen via design workflow wf_2bed90d9-e22 (Option A, per-mon, not per-move). Layout v15.
"""
from __future__ import annotations

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
from v_dance.encoders.state_encoder import (
    StateEncoder, get_state_dim, get_state_layout_version, POKEMON_FEATURES, VOLATILE_FEATURES, NUM_TYPES,
)

# the 3 first-turn-gated moves (pinned for documentation; the channel is per-mon, not per-move)
GATED_MOVES = frozenset({"fakeout", "firstimpression", "matblock"})

# first_turn = 12th volatile channel (index +11); volatile block sits at POKEMON_FEATURES-4-VOLATILE_FEATURES.
FIRST_TURN_CH = POKEMON_FEATURES - 4 - NUM_TYPES - VOLATILE_FEATURES + 11   # v11 Phase D D9: tera one-hot before flags

HEADER = """|player|p1|Alice|1|
|player|p2|Bob|1|
|poke|p1|Incineroar, L50, M|
|poke|p1|Rillaboom, L50, M|
|poke|p2|Garchomp, L50, M|
|poke|p2|Rotom-Wash, L50|
|teamsize|p1|2
|teamsize|p2|2
|start
|switch|p1a: Incineroar|Incineroar, L50, M|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
"""


def _parse(body):
    return ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()


def _turn(result, n):
    for t in result["turns"]:
        if t["turn"] == n:
            return t
    raise AssertionError(f"turn {n} not found")


def _ft(result, n, rel="our_a", side="p1"):
    st = _turn(result, n)["state_before_actions"][side]
    grp = "our_active" if rel.startswith("our") else "opp_active"
    mon = st[grp].get(rel)
    return mon["volatiles"]["first_turn"] if mon else None


# ════════════════════════════ parser tracking ════════════════════════════
def test_lead_first_turn_then_false():
    r = _parse("|turn|1\n|move|p1a: Incineroar|Fake Out|p2a: Garchomp\n"
               "|move|p2a: Garchomp|Earthquake|p1a: Incineroar\n|turn|2\n")
    assert _ft(r, 1, "our_a") is True              # pre-game lead → first acting turn
    assert _ft(r, 1, "opp_a") is True
    assert _ft(r, 2, "our_a") is False             # second turn on the field → Fake Out now illegal


def test_switch_in_resets_first_turn():
    # Rillaboom switches in during turn 1; at turn 2 (its first acting turn) first_turn flips True,
    # then False at turn 3.
    r = _parse("|turn|1\n|switch|p1a: Rillaboom|Rillaboom, L50, M|100/100\n|turn|2\n"
               "|move|p1a: Rillaboom|Grassy Glide|p2a: Garchomp\n|turn|3\n")
    assert _ft(r, 2, "our_a") is True
    assert _ft(r, 3, "our_a") is False


def test_faint_replacement_first_turn():
    # Incineroar faints turn 1, Rillaboom replaces it; the replacement is NOT first_turn at the forced-
    # replacement snapshot, but IS at the next |turn|.
    r = _parse("|turn|1\n|move|p2a: Garchomp|Close Combat|p1a: Incineroar\n"
               "|faint|p1a: Incineroar\n|switch|p1a: Rillaboom|Rillaboom, L50, M|100/100\n"
               "|turn|2\n|move|p1a: Rillaboom|Grassy Glide|p2a: Garchomp\n|turn|3\n")
    assert _ft(r, 2, "our_a") is True              # Rillaboom's first acting turn
    assert _ft(r, 3, "our_a") is False


def test_drag_resets_like_switch():
    # A phazed-in (dragged) mon resets like a voluntary switch → first_turn True the next turn.
    r = _parse("|turn|1\n|drag|p1a: Rillaboom|Rillaboom, L50, M|100/100\n|turn|2\n"
               "|move|p1a: Rillaboom|Grassy Glide|p2a: Garchomp\n|turn|3\n")
    assert _ft(r, 2, "our_a") is True
    assert _ft(r, 3, "our_a") is False


def test_stays_on_field_not_first_turn():
    # A mon that never leaves stays first_turn=False after its opening turn.
    r = _parse("|turn|1\n|move|p1a: Incineroar|Fake Out|p2a: Garchomp\n|turn|2\n"
               "|move|p1a: Incineroar|Knock Off|p2a: Garchomp\n|turn|3\n"
               "|move|p1a: Incineroar|Knock Off|p2a: Garchomp\n|turn|4\n")
    assert _ft(r, 1, "our_a") is True
    assert _ft(r, 2, "our_a") is False
    assert _ft(r, 3, "our_a") is False
    assert _ft(r, 4, "our_a") is False


# ════════════════════════════ encoder channel ════════════════════════════
def test_encoder_channel_reflects_first_turn():
    r = _parse("|turn|1\n|move|p1a: Incineroar|Fake Out|p2a: Garchomp\n"
               "|move|p2a: Garchomp|Earthquake|p1a: Incineroar\n|turn|2\n")
    enc = StateEncoder()
    v1 = enc.encode_snapshot(_turn(r, 1)["state_before_actions"]["p1"], turn=1)
    v2 = enc.encode_snapshot(_turn(r, 2)["state_before_actions"]["p1"], turn=2)
    assert v1[FIRST_TURN_CH] == 1.0                # our_a (slot 0) first turn
    assert v2[FIRST_TURN_CH] == 0.0


# ════════════════════════════ layout pins ════════════════════════════
def test_layout_v15():
    assert VOLATILE_FEATURES == 12
    assert POKEMON_FEATURES == 413       # v16 (B2b: +2 per-move hit-chance channels)
    assert get_state_dim() == 5057
    assert get_state_layout_version() == 19
