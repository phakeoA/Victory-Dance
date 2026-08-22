"""B1-mechanics step 2a: per-move effect tags derived from the pinned moves.ts data fields. Asserts the
representative moves get the right tags (the mechanics the user called out: phazing, setup, speed control,
status, charge/recharge, Body Press/Foul Play/Psyshock, variable BP, hazards, pivot)."""
from __future__ import annotations

import pytest

from v_dance.encoders import mechanic_tags as MT
from v_dance.encoders.mechanic_vocab import _DEX

pytestmark = pytest.mark.skipif(not (_DEX / "moves.ts").exists(), reason="pinned moves.ts not present")


def has(move, tag):
    return tag in MT.move_tags(move)


def test_taxonomy_shape():
    assert MT.NUM_MOVE_TAGS == 29 and len(set(MT.MOVE_TAG_NAMES)) == 29
    assert MT.move_tags(None) == [] and MT.move_tags("not_a_move") == []
    # indices line up with names
    assert MT.move_tag_indices("roar") == sorted(MT.MOVE_TAG_NAMES.index(t) for t in MT.move_tags("roar"))


def test_setup_self_boosts():
    assert has("swordsdance", "self_boost_off")
    assert has("nastyplot", "self_boost_off")
    assert has("irondefense", "self_boost_def")
    assert has("agility", "self_boost_spe")
    assert has("dragondance", "self_boost_off") and has("dragondance", "self_boost_spe")
    assert has("shellsmash", "self_boost_off") and has("shellsmash", "self_boost_spe")
    # a self-DEBUFF damage move must NOT read as a self-boost
    assert not has("closecombat", "self_boost_def")
    assert not has("dracometeor", "self_boost_off")


def test_target_drops_and_speed_control():
    assert has("growl", "stat_drop_target")
    assert has("icywind", "stat_drop_target") and has("icywind", "speed_control")
    assert has("stringshot", "speed_control")
    assert has("tailwind", "speed_control")
    assert has("trickroom", "speed_control")
    assert has("thunderwave", "status_par") and has("thunderwave", "speed_control")
    assert has("stickyweb", "speed_control") and has("stickyweb", "hazard_set")
    # PLURAL secondaries:[] target drop — Triple Arrows' {def:-1} was previously unparsed (audit 2026-06-30)
    assert has("triplearrows", "stat_drop_target")
    # a self-DEBUFF damaging move (top-level self:{boosts}) must NOT be read as a TARGET drop
    for mv in ("closecombat", "dracometeor", "overheat", "superpower", "leafstorm"):
        assert not has(mv, "stat_drop_target"), mv


def test_switching_and_tempo():
    for m in ("roar", "whirlwind", "dragontail", "circlethrow"):
        assert has(m, "force_switch")
    for m in ("uturn", "voltswitch", "flipturn", "partingshot", "teleport"):
        assert has(m, "pivot")
    for m in ("hyperbeam", "gigaimpact", "blastburn", "hydrocannon"):
        assert has(m, "recharge")
    assert has("solarbeam", "two_turn_charge")
    assert has("fly", "two_turn_charge") and has("fly", "semi_invulnerable")


def test_damage_mechanics():
    assert has("bodypress", "dmg_uses_def")
    assert has("foulplay", "dmg_uses_target_atk")
    assert has("psyshock", "dmg_hits_def")
    for m in ("lowkick", "grassknot", "gyroball", "heavyslam", "eruption", "storedpower"):
        assert has(m, "bp_variable")


def test_status_hazards_restrict_heal():
    assert has("willowisp", "status_brn")
    assert has("spore", "status_slp")
    assert has("toxic", "status_psn")
    assert has("confuseray", "status_confuse")
    assert has("fakeout", "flinch") and has("ironhead", "flinch")
    for m in ("stealthrock", "spikes", "toxicspikes"):
        assert has(m, "hazard_set")
    for m in ("defog", "rapidspin"):
        assert has(m, "hazard_clear")
    for m in ("taunt", "encore", "disable"):
        assert has(m, "move_restrict")
    assert has("leechseed", "residual_chip")
    for m in ("recover", "roost", "gigadrain", "drainpunch"):
        assert has(m, "heal")
    assert has("followme", "redirect")
    assert has("knockoff", "item_interact")
    assert has("skillswap", "ability_interact")


def test_counts_match_parse_ballpark():
    # vs scratch/verify_mechanic_coverage tallies (>= to allow the speed_control/heal supersets)
    n = lambda tag: sum(1 for tags in MT.MOVE_TAGS.values() if tag in tags)
    assert n("force_switch") >= 4
    assert n("recharge") >= 8
    assert n("bp_variable") >= 40
    assert n("flinch") >= 20
    assert n("pivot") >= 7


# ── ability tags (step 2b) ────────────────────────────────────────────────────────
def ahas(ab, tag):
    return tag in MT.ability_tags(ab)


def test_ability_taxonomy_shape():
    assert MT.NUM_ABILITY_TAGS == 43 and len(set(MT.ABILITY_TAG_NAMES)) == 43
    assert MT.ability_tags(None) == [] and MT.ability_tags("not_an_ability") == []
    # ability_suppressed is runtime-only: never returned from static membership
    assert all("ability_suppressed" not in MT.ability_tags(a) for a in MT._ABILITY_TAGS)


def test_trapping_and_typechange_are_data_grounded():
    assert MT._DATA_TRAPPING == {"shadowtag", "arenatrap", "magnetpull"}
    for trap in ("shadowtag", "arenatrap", "magnetpull"):
        assert ahas(trap, "trapping")
    for tc in ("aerilate", "dragonize", "galvanize", "liquidvoice", "normalize", "pixilate", "refrigerate"):
        assert tc in MT._DATA_MOVE_TYPECHANGE and ahas(tc, "type_change")


def test_representative_ability_tags_incl_multitag():
    assert ahas("shadowtag", "trapping")
    assert ahas("unaware", "unaware") and ahas("contrary", "contrary") and ahas("simple", "contrary")
    assert ahas("parentalbond", "parental_bond")
    assert ahas("moxie", "ko_snowball") and ahas("beastboost", "ko_snowball")
    assert ahas("magicguard", "magic_guard") and ahas("disguise", "disguise")
    assert ahas("suctioncups", "force_switch_block") and ahas("guarddog", "force_switch_block")
    assert ahas("unseenfist", "protect_bypass")
    assert ahas("queenlymajesty", "priority_block") and ahas("prankster", "priority_grant")
    assert ahas("clearbody", "stat_drop_immune") and ahas("poisonheal", "poison_heal")
    assert ahas("trace", "ability_change") and ahas("frisk", "info_on_entry")
    assert ahas("roughskin", "contact_punish") and ahas("static", "contact_status")
    assert ahas("cloudnine", "weather_negate") and ahas("noguard", "accuracy_mod")
    # multi-tag: an -ate is both damage_boost AND type_change; guts is guts_boost AND damage_boost
    assert ahas("pixilate", "damage_boost") and ahas("pixilate", "type_change")
    assert ahas("guts", "guts_boost") and ahas("guts", "damage_boost")
    assert ahas("prankster", "prankster") and ahas("prankster", "priority_grant")


def test_every_tagged_ability_is_in_vocab_no_typos():
    from v_dance.encoders.mechanic_vocab import ABILITY_VOCAB
    bad = sorted(i for i in MT._ABILITY_TAGS if i not in ABILITY_VOCAB)
    assert bad == [], f"tagged ability ids not in the dex vocab (typos / not-in-format): {bad}"


def test_mega_uses_active_ability_for_tags():
    """v9 step 6: a mega'd mon's ability block uses the ACTIVE mega ability (Mega Gengar -> Shadow Tag),
    so trapping/parental_bond fire — NOT the base ability. Conf 1.0 on both paths (parity-clean)."""
    from v_dance.encoders.state_encoder import resolve_active_ability_json, resolve_ability_json
    mega = {"species": "Gengar-Mega", "base_species": "Gengar", "is_mega": True,
            "pre_mega_ability": "Cursed Body", "known_ability": "Shadow Tag", "mega_ability": "Shadow Tag"}
    assert resolve_active_ability_json(mega) == ("shadowtag", 1.0)
    assert "trapping" in MT.ability_tags(resolve_active_ability_json(mega)[0])
    kanga = {"species": "Kangaskhan-Mega", "base_species": "Kangaskhan", "is_mega": True,
             "mega_ability": "Parental Bond"}
    assert "parental_bond" in MT.ability_tags(resolve_active_ability_json(kanga)[0])
    # the base resolver is unchanged (still returns the user-chosen base ability)
    assert resolve_ability_json(mega)[0] == "cursedbody"
    # non-mega passes straight through
    nm = {"species": "Gengar", "base_species": "Gengar", "known_ability": "Cursed Body"}
    assert resolve_active_ability_json(nm) == resolve_ability_json(nm)


# ── item tags (step 2c) ────────────────────────────────────────────────────────────
def ihas(it, tag):
    return tag in MT.item_tags(it)


def test_item_taxonomy_shape():
    assert MT.NUM_ITEM_TAGS == 23 and len(set(MT.ITEM_TAG_NAMES)) == 23
    assert MT.item_tags("") == [] and MT.item_tags(None) == [] and MT.item_tags("nothing") == []
    # any concrete item sets has_item
    assert ihas("leftovers", "has_item")


def test_item_existing_categories_preserved():
    assert ihas("choicescarf", "choice") and ihas("choicescarf", "choice_speed")
    assert ihas("choiceband", "choice") and not ihas("choiceband", "choice_speed")
    assert ihas("leftovers", "passive_recovery")
    assert ihas("sitrusberry", "heal_berry") and ihas("occaberry", "resist_berry")
    assert ihas("lumberry", "status_cure")
    assert ihas("charcoal", "type_boost") and ihas("splashplate", "type_boost")   # plate via suffix
    assert ihas("rockyhelmet", "contact_punish") and ihas("focussash", "focus_sash")
    assert ihas("lifeorb", "life_orb") and ihas("assaultvest", "assault_vest")
    assert ihas("scopelens", "luck_item") and ihas("clearamulet", "drop_guard")


def test_item_v9_new_tags():
    for r in ("damprock", "heatrock", "icyrock", "smoothrock"):
        assert ihas(r, "weather_rock")
    assert ihas("lightclay", "screen_extend")
    assert ihas("shedshell", "escape_trap")          # escapes the trapping abilities
    assert ihas("expertbelt", "se_boost")
    assert ihas("muscleband", "category_boost") and ihas("wiseglasses", "category_boost")
    assert ihas("terrainextender", "terrain_extend")


def test_mega_stones_data_grounded():
    assert 80 <= len(MT._DATA_MEGA_STONES) <= 110          # ~93 in the dex
    for stone in ("gengarite", "charizarditey", "kangaskhanite", "metagrossite", "lucarionite"):
        assert ihas(stone, "mega_stone") and ihas(stone, "has_item")
    assert not ihas("leftovers", "mega_stone")
