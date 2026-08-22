"""Items reveal themselves by their EFFECT via a ``[from] item:`` tag (no |-item| line).

Life Orb recoil / Leftovers heal / Black Sludge / Rocky Helmet contact / Flame Orb status all carry a
``[from] item: X`` tag on a -damage/-heal/-status line. The parser learns the HELD item from it (the holder
is the ``[of]`` mon for reactive items like Rocky Helmet, else the line subject), WITHOUT marking it consumed
(|-enditem| owns consumption). This makes the opponent item belief ground-truth where the replay shows it.
"""
from __future__ import annotations

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser

_HEAD = """|gen|9
|player|p1|Me|1|
|player|p2|Foe|2|
|teamsize|p1|2
|teamsize|p2|2
|poke|p1|Garchomp, L50, M|
|poke|p1|Corviknight, L50, M|
|poke|p2|Snorlax, L50, M|
|poke|p2|Ferrothorn, L50, M|
|start
|switch|p1a: Garchomp|Garchomp, L50, M|100/100
|switch|p1b: Corviknight|Corviknight, L50, M|100/100
|switch|p2a: Snorlax|Snorlax, L50, M|100/100
|switch|p2b: Ferrothorn|Ferrothorn, L50, M|100/100
|turn|1
"""


def _parse(body: str):
    p = ShowdownReplayParser(_HEAD + body, our_player="p1")
    p.parse()
    return p


def test_life_orb_recoil_reveals_life_orb():
    p = _parse("|move|p1a: Garchomp|Earthquake|p2a: Snorlax\n"
               "|-damage|p1a: Garchomp|85/100|[from] item: Life Orb\n|upkeep\n|turn|2\n")
    mon = p.active_slots["p1a"]
    assert mon.known_item == "Life Orb"
    assert mon.item_consumed is False          # still held, not consumed


def test_leftovers_heal_reveals_leftovers():
    p = _parse("|-heal|p2a: Snorlax|66/100|[from] item: Leftovers\n|upkeep\n|turn|2\n")
    assert p.active_slots["p2a"].known_item == "Leftovers"
    assert p.active_slots["p2a"].item_consumed is False


def test_reactive_item_uses_of_mon_as_holder():
    # Rocky Helmet hurts the ATTACKER; the [of] mon (defender) is the holder
    p = _parse("|move|p1a: Garchomp|Tackle|p2b: Ferrothorn\n"
               "|-damage|p1a: Garchomp|70/100|[from] item: Rocky Helmet|[of] p2b: Ferrothorn\n"
               "|upkeep\n|turn|2\n")
    assert p.active_slots["p2b"].known_item == "Rocky Helmet"   # defender holds it
    assert p.active_slots["p1a"].known_item != "Rocky Helmet"   # attacker does NOT


def test_flame_orb_status_reveals_item():
    p = _parse("|-status|p2a: Snorlax|brn|[from] item: Flame Orb\n|upkeep\n|turn|2\n")
    assert p.active_slots["p2a"].known_item == "Flame Orb"


def test_no_tag_no_inference():
    # a plain -damage with no [from] item: tag must not invent an item
    p = _parse("|move|p1a: Garchomp|Earthquake|p2a: Snorlax\n"
               "|-damage|p2a: Snorlax|60/100\n|upkeep\n|turn|2\n")
    assert p.active_slots["p2a"].known_item is None


# ── explicit anti-confusion guards (the cases that ride on the SAME line type) ────
def test_life_orb_not_confused_with_rocky_helmet():
    # the attacker holds Life Orb AND hits a Rocky-Helmet mon → TWO -damage lines on the attacker.
    # Life Orb (no [of]) → attacker; Rocky Helmet ([of] defender) → defender. Never crossed.
    p = _parse("|move|p1a: Garchomp|Iron Head|p2b: Ferrothorn\n"
               "|-damage|p2b: Ferrothorn|70/100\n"
               "|-damage|p1a: Garchomp|85/100|[from] item: Life Orb\n"
               "|-damage|p1a: Garchomp|70/100|[from] item: Rocky Helmet|[of] p2b: Ferrothorn\n"
               "|upkeep\n|turn|2\n")
    assert p.active_slots["p1a"].known_item == "Life Orb"        # attacker = Life Orb
    assert p.active_slots["p2b"].known_item == "Rocky Helmet"    # defender = Rocky Helmet
    assert p.active_slots["p1a"].known_item != "Rocky Helmet"


def test_rocky_helmet_item_not_confused_with_rough_skin_ability():
    # Rough Skin is an ABILITY ([from] ability:), not an item — the item handler must ignore it
    p = _parse("|move|p1a: Garchomp|Iron Head|p2b: Ferrothorn\n"
               "|-damage|p2b: Ferrothorn|70/100\n"
               "|-damage|p1a: Garchomp|85/100|[from] ability: Rough Skin|[of] p2b: Ferrothorn\n"
               "|upkeep\n|turn|2\n")
    assert p.active_slots["p2b"].known_ability == "Rough Skin"
    assert p.active_slots["p2b"].known_item is None              # NOT mistaken for Rocky Helmet
    assert p.active_slots["p1a"].known_item is None


def test_leftovers_held_vs_sitrus_consumed():
    # both heal via [from] item:, but a Sitrus is eaten (|-enditem| → consumed) while Leftovers is held
    p = _parse("|-enditem|p1a: Garchomp|Sitrus Berry|[eat]\n"
               "|-heal|p1a: Garchomp|75/100|[from] item: Sitrus Berry\n"
               "|-heal|p2a: Snorlax|66/100|[from] item: Leftovers\n|upkeep\n|turn|2\n")
    assert p.active_slots["p1a"].known_item == "Sitrus Berry"
    assert p.active_slots["p1a"].item_consumed is True           # eaten → not held
    assert p.active_slots["p2a"].known_item == "Leftovers"
    assert p.active_slots["p2a"].item_consumed is False          # held


# ── silent item TRANSFERS must not leave a stale 1.0 known_item (adversarial scan finds) ──
def test_sticky_barb_silent_transfer_clears_former_holder():
    # p2a holds Sticky Barb; an itemless p1a contact-hits it → barb transfers SILENTLY to p1a.
    # The new holder's residual self-damage reveals it; the former holder must be cleared.
    p = _parse("|-damage|p2a: Snorlax|88/100|[from] item: Sticky Barb\n|upkeep\n|turn|2\n"
               "|move|p1a: Garchomp|Close Combat|p2a: Snorlax\n|-damage|p2a: Snorlax|40/100\n"
               "|-damage|p1a: Garchomp|88/100|[from] item: Sticky Barb\n|upkeep\n|turn|3\n")
    assert p.active_slots["p1a"].known_item == "Sticky Barb"     # new holder
    assert p.active_slots["p2a"].known_item is None              # former holder cleared, no stale 1.0


def test_symbiosis_pass_moves_item_off_giver_onto_receiver():
    # Florges (Symbiosis) passes Leftovers to Flapple; the -activate line names the hand-off
    p = _parse("|-heal|p2a: Snorlax|90/100|[from] item: Leftovers\n|upkeep\n|turn|2\n"
               "|-enditem|p2b: Ferrothorn|Sitrus Berry|[eat]\n"
               "|-heal|p2b: Ferrothorn|75/100|[from] item: Sitrus Berry\n"
               "|-activate|p2a: Snorlax|ability: Symbiosis|Leftovers|[of] p2b: Ferrothorn\n"
               "|upkeep\n|turn|3\n")
    assert p.active_slots["p2a"].item_consumed is True           # giver passed it away → itemless
    assert p.active_slots["p2b"].known_item == "Leftovers"       # receiver now holds it
    assert p.active_slots["p2b"].item_consumed is False


# ── a -heal item heals its OWN holder; [of] is the VICTIM, not the holder (Shell Bell) ──
def test_shell_bell_heal_attributes_item_to_holder_not_of_victim():
    # Shell Bell heals its HOLDER (the line subject) for a fraction of the damage that holder just
    # dealt; Showdown emits the damaged FOE as [of]. So [of] is the victim — trusting it as the holder
    # (the pre-fix behaviour) would brand the OPPONENT with Shell Bell while the real holder gets nothing.
    p = _parse("|move|p2a: Snorlax|Body Slam|p1a: Garchomp\n|-damage|p1a: Garchomp|60/100\n"
               "|-heal|p2a: Snorlax|95/100|[from] item: Shell Bell|[of] p1a: Garchomp\n"
               "|upkeep\n|turn|2\n")
    assert p.active_slots["p2a"].known_item == "Shell Bell"      # holder = line subject
    assert p.active_slots["p1a"].known_item != "Shell Bell"      # [of] victim is NOT the holder
    assert p.active_slots["p1a"].known_item is None


def test_reactive_status_item_still_uses_of_holder():
    # a -status reactor (hypothetical Flame Orb thrown by an [of] holder) keeps [of]-as-holder on
    # -damage/-status lines — guards that the heal-only carve-out did not break the damage/status path.
    p = _parse("|-damage|p1a: Garchomp|70/100|[from] item: Rocky Helmet|[of] p2b: Ferrothorn\n"
               "|upkeep\n|turn|2\n")
    assert p.active_slots["p2b"].known_item == "Rocky Helmet"    # -damage line: [of] is still the holder
    assert p.active_slots["p1a"].known_item != "Rocky Helmet"
