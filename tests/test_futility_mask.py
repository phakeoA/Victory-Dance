"""Futility-mask batch (2026-07-24) — rules-exact always-fail target buckets.

Covers the pure core (futile_target_buckets) per rule class + the offline
build_action_mask integration + the FAIL-OPEN contract (unknown ability/typing
must never mask). Serve-adapter and decode-level tests live with their layers.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from v_dance.encoders.action_codec import (      # noqa: E402
    build_action_mask,
    futile_target_buckets,
)


def _foe(types=("normal",), status=None, encored=False, confused=False, grounded=True):
    return {"types": tuple(types) if types is not None else None, "status": status,
            "encored": encored, "confused": confused, "grounded": grounded}


def _ftb(move, **kw):
    base = dict(user_ability=None, foes={0: _foe(), 1: None})
    base.update(kw)
    return futile_target_buckets(move, **base)


# ── prankster → dark ──────────────────────────────────────────────────────────
def test_prankster_status_into_dark():
    assert _ftb("Encore", user_ability="prankster",
                foes={0: _foe(types=("dark", "fire")), 1: _foe(types=("water",))}) == {0}
    # fail open: unknown ability / unknown typing / suppressed ability
    assert _ftb("Encore", foes={0: _foe(types=("dark",)), 1: None}) == set()
    assert _ftb("Encore", user_ability="prankster",
                foes={0: _foe(types=None), 1: None}) == set()
    assert _ftb("Encore", user_ability="prankster", user_ability_active=False,
                foes={0: _foe(types=("dark",)), 1: None}) == set()
    # damaging moves unaffected (Moonblast vs Dark is fine)
    assert _ftb("Moonblast", user_ability="prankster",
                foes={0: _foe(types=("dark",)), 1: None}) == set()


# ── status type-immunities + already-statused + safeguard ─────────────────────
def test_status_inflict_immunities():
    assert _ftb("Will-O-Wisp", foes={0: _foe(types=("fire",)), 1: None}) == {0}
    assert _ftb("Thunder Wave", foes={0: _foe(types=("electric",)), 1: None}) == {0}
    assert _ftb("Thunder Wave", foes={0: _foe(types=("ground",)), 1: None}) == {0}
    assert _ftb("Toxic", foes={0: _foe(types=("steel",)), 1: _foe(types=("poison",))}) == {0, 1}
    assert _ftb("Toxic", user_ability="corrosion",
                foes={0: _foe(types=("steel",)), 1: None}) == set()      # our Corrosion pierces
    assert _ftb("Will-O-Wisp", foes={0: _foe(types=("water",), status="par"), 1: None}) == {0}
    assert _ftb("Will-O-Wisp", opp_safeguard=True,
                foes={0: _foe(types=("water",)), 1: None}) == {0}
    assert _ftb("Will-O-Wisp", opp_safeguard=True, user_ability="infiltrator",
                foes={0: _foe(types=("water",)), 1: None}) == set()
    assert _ftb("Will-O-Wisp", foes={0: _foe(types=None), 1: None}) == set()   # fail open


def test_powder_confusion_encore_rules():
    assert _ftb("Sleep Powder", foes={0: _foe(types=("grass", "poison")), 1: None}) == {0}
    assert _ftb("Spore", foes={0: _foe(types=("water",)), 1: None}) == set()
    assert _ftb("Confuse Ray", foes={0: _foe(confused=True), 1: None}) == {0}
    assert _ftb("Encore", foes={0: _foe(encored=True), 1: None}) == {0}


# ── duplicate side / field state ──────────────────────────────────────────────
def test_duplicate_side_and_field():
    assert _ftb("Tailwind", our_sideconds=frozenset({"tailwind"})) == {0}
    assert _ftb("Tailwind") == set()
    assert _ftb("Reflect", our_sideconds=frozenset({"reflect"})) == {0}
    assert _ftb("Rain Dance", weather="RainDance") == {0}
    assert _ftb("Rain Dance", weather="Sandstorm") == set()
    assert _ftb("Psychic Terrain", terrain="psychic") == {0}
    assert _ftb("Grassy Terrain", terrain="psychic") == set()      # overwrite = legal
    assert _ftb("Gravity", gravity_on=True) == {0}
    assert _ftb("Trick Room", **{}) == set()                       # toggle — never masked


# ── psychic terrain effective-priority block ──────────────────────────────────
def test_psychic_terrain_priority_block():
    kw = dict(terrain="psychic", foes={0: _foe(grounded=True), 1: _foe(grounded=False)})
    assert _ftb("Sucker Punch", **kw) == {0}                       # grounded foe blocked only
    assert _ftb("Aqua Jet", **kw) == {0}
    assert _ftb("Tackle", **kw) == set()                           # no priority
    # grassy glide has NO priority under psychic terrain (terrains exclusive)
    assert _ftb("Grassy Glide", **kw) == set()
    # prankster status gains priority; gale wings only at full HP
    assert _ftb("Encore", user_ability="prankster", **kw) == {0}
    assert _ftb("Brave Bird", user_ability="galewings", user_hp_full=True, **kw) == {0}
    assert _ftb("Brave Bird", user_ability="galewings", user_hp_full=False, **kw) == set()
    # unknown grounding fails open
    assert _ftb("Sucker Punch", terrain="psychic",
                foes={0: _foe(grounded=None), 1: None}) == set()


# ── offline integration through build_action_mask ─────────────────────────────
def _snap(our_mon, opp_a, field=None, sides=None):
    return {
        "our_active": {"our_a": our_mon},
        "opp_active": {"opp_a": opp_a} if opp_a else {},
        "our_bench": [], "opp_bench": [],
        "field": field or {},
        "side_conditions": sides or {"our_side": {}, "opp_side": {}},
    }


def test_build_action_mask_masks_prankster_encore_into_dark():
    whims = {"species": "Whimsicott", "known_ability": "Prankster",
             "revealed_moves": ["Encore", "Moonblast", "Tailwind", "Protect"],
             "volatiles": {}}
    dark = {"species": "Kingambit", "runtime_types": ["dark", "steel"],
            "known_ability": "Defiant", "volatiles": {}}
    row = build_action_mask(_snap(whims, dark))["our_a"]
    assert row[0 * 3 + 0] == 0                       # Encore -> foe_a masked (dark)
    assert row[1 * 3 + 0] == 1                       # Moonblast -> foe_a legal
    assert row[2 * 3 + 0] == 1                       # Tailwind legal (not yet up)
    # same board with our tailwind active -> tailwind re-click masked
    sides = {"our_side": {"tailwind_turns_remaining": 3, "screens": {}},
             "opp_side": {}}
    row2 = build_action_mask(_snap(whims, dark, sides=sides))["our_a"]
    assert row2[2 * 3 + 0] == 0


def test_build_action_mask_fail_open_without_ability():
    whims = {"species": "Whimsicott",                # Type-B row: ability never revealed
             "revealed_moves": ["Encore"], "volatiles": {}}
    dark = {"species": "Kingambit", "runtime_types": ["dark", "steel"], "volatiles": {}}
    row = build_action_mask(_snap(whims, dark))["our_a"]
    assert row[0] == 1                               # unknown ability -> Encore stays legal


# ── serve adapter (mock poke-env battle) ──────────────────────────────────────
class _FMove:
    def __init__(self, mid):
        self.id = mid


class _FMon:
    def __init__(self, species, types=(), ability=None, moves=(), item=None):
        self.species, self.fainted = species, False
        self.types = tuple(types)
        self.ability = ability
        self.item = item
        self.effects = {}
        self.current_hp_fraction = 1.0
        self.moves = {m: _FMove(m) for m in moves}
        self.status = None


class _FBattle:
    def __init__(self, actives, opps):
        self.active_pokemon = actives
        self.opponent_active_pokemon = opps
        self.available_moves = [list(m.moves.values()) for m in actives]
        self.side_conditions = {}
        self.opponent_side_conditions = {}
        self.fields = {}
        self.weather = {}
        self.trapped = [False, False]
        self.reviving = False


def test_serve_adapter_prankster_dark_and_kill_switch(monkeypatch):
    import v_dance.play.vgc_base as vb
    whims = _FMon("whimsicott", ("grass", "fairy"), ability="prankster",
                  moves=("encore", "moonblast"))
    partner = _FMon("farigiraf", ("normal", "psychic"), ability="armortail",
                    moves=("trickroom",))
    dark = _FMon("kingambit", ("dark", "steel"), ability="defiant")
    battle = _FBattle([whims, partner], [dark, None])
    monkeypatch.setattr(vb, "own_bench_mons", lambda b: [])
    mask = vb.build_legal_action_mask(battle, 0)
    assert mask[0 * 3 + 0] is False or mask[0] == 0    # Encore -> dark foe masked
    assert mask[1 * 3 + 0]                             # Moonblast legal
    monkeypatch.setenv("VD_FUTILITY_MASK", "0")        # kill switch restores legality
    mask_off = vb.build_legal_action_mask(battle, 0)
    assert mask_off[0]


def test_hh_pair_futility_hook_blocks_hh_on_status_partner(monkeypatch):
    import v_dance.play.vgc_base as vb
    fari = _FMon("farigiraf", ("normal", "psychic"), ability="armortail",
                 moves=("helpinghand", "trickroom", "psychic", "thunderbolt"))
    chomp = _FMon("garchomp", ("dragon", "ground"), ability="roughskin",
                  moves=("protect", "earthquake", "dragonclaw", "rockslide"))
    foe = _FMon("kingambit", ("dark", "steel"))
    battle = _FBattle([chomp, fari], [foe, None])
    hook = vb.hh_pair_futility_hook(battle)
    assert hook is not None
    # partner (slot 0, garchomp) chose Protect (move 0, bucket 0 = action 0) -> HH dropped
    dropped = hook(1, 0, 0)
    assert 0 * 3 + 2 in dropped                        # fari's HH = move slot 0, ally bucket
    # partner chose Earthquake (move 1 -> action 3): HH stays
    assert hook(1, 0, 3) == set()
    # partner switches (action 12): HH dropped
    assert 0 * 3 + 2 in hook(1, 0, 12)
    # kill switch -> no hook at all
    monkeypatch.setenv("VD_FUTILITY_MASK", "0")
    assert vb.hh_pair_futility_hook(battle) is None


def test_transition_futile_label_stat():
    from collections import Counter
    from v_dance.training.bc_dataset import transition_to_example
    from v_dance.encoders.state_encoder import StateEncoder
    whims = {"species": "Whimsicott", "known_ability": "Prankster",
             "revealed_moves": ["Encore", "Moonblast", "Tailwind", "Protect"],
             "volatiles": {}}
    dark = {"species": "Kingambit", "runtime_types": ["dark", "steel"], "volatiles": {}}
    snap = _snap(whims, dark)
    stale_row = [0] * 16
    stale_row[0] = 1                                   # export-era mask: Encore->foe_a legal
    stale_row[3] = 1
    t = {"state_before_actions": snap, "replay_id": "r-fut", "perspective": "p1",
         "action_mask": {"our_a": stale_row, "our_b": [0] * 16},
         "gimmick_mask": {},
         "our_actions": [{"slot": "our_a", "action_index": 0, "gimmick_index": None}]}
    stats = Counter()
    transition_to_example(t, StateEncoder(), stats=stats)
    assert stats["skipped_futile_target"] == 1         # the human Encore->Dark label masked, LOUDLY
