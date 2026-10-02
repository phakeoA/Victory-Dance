"""Drill PRESSURE (2026-10-02): the opponent-only field-fighter logit bias.

Covers v_dance/play/field_fight.py (the bias itself), the two VGCPlayer decision paths that consume it
(``_select_actions`` and ``_select_replacement_actions``), and the kwarg plumbing that keeps it opponent-only
(SplicingVGCPlayerBase pops it, SelfPlayVGCPlayer refuses it, run_local_battle.make_player forwards it to the
model branch only). Offline: SimpleNamespace battles carry the REAL poke-env Weather / Field enums, and
``field_fight.own_bench_mons`` is monkeypatched so the bench order is explicit.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("poke_env")

from poke_env.battle import Field, Weather  # noqa: E402

import v_dance.play.field_fight as FF  # noqa: E402
from v_dance.encoders.encoder_layout import ACTIONS_PER_SLOT, BENCH_SLOTS, SWITCH_OFFSET  # noqa: E402

B = 2.5


# ── fixtures / builders ─────────────────────────────────────────────────────────
def _mon(species, ability=None, item=None):
    return SimpleNamespace(species=species, ability=ability, item=item, fainted=False)


def _battle(team, *, weather=None, terrain=None, fields=None, force_switch=(False, False), **extra):
    w = {weather: 1} if weather is not None else {}
    f = dict(fields or {})
    if terrain is not None:
        f[terrain] = 1
    return SimpleNamespace(team={f"p1: {m.species}": m for m in team}, weather=w, fields=f,
                           force_switch=list(force_switch), battle_tag="battle-gen9test-1", turn=3,
                           player_role="p1", **extra)


@pytest.fixture
def bench(monkeypatch):
    """bench(mons) -> installs own_bench_mons returning exactly that list (in that order)."""
    def _set(mons):
        monkeypatch.setattr(FF, "own_bench_mons", lambda battle: list(mons))
    return _set


def _only(arr, idx, val=B):
    """arr is a (16,) float32 vector that is ``val`` at the given indices and 0 everywhere else."""
    idx = [idx] if isinstance(idx, int) else list(idx)
    assert isinstance(arr, np.ndarray) and arr.shape == (ACTIONS_PER_SLOT,) and arr.dtype == np.float32
    want = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
    for i in idx:
        want[i] = val
    np.testing.assert_array_equal(arr, want)


# Common cast: two plain actives, a plain bench mon at index 0 and a Drizzle setter at index 1.
ACT_A, ACT_B = _mon("incineroar", "Intimidate", "Sitrus Berry"), _mon("amoonguss", "Regenerator", "Rocky Helmet")
PLAIN = _mon("kingambit", "Defiant", "Black Glasses")
PELIPPER = _mon("pelipper", "Drizzle", "Damp Rock")
RILLA = _mon("rillaboom", "Grassy Surge", "Miracle Seed")
TORKOAL = _mon("torkoal", "Drought", "Charcoal")
ARBOLIVA = _mon("arboliva", "Seed Sower", "Leftovers")
ZARD_Y = _mon("charizard", "Blaze", "Charizardite Y")


# ══════════════════════════════════════════════════════════════════════════════
# setter_kinds / current_kinds
# ══════════════════════════════════════════════════════════════════════════════
def test_setter_kinds_reads_ability_and_weather_item():
    assert FF.setter_kinds(PELIPPER) == {"weather": "rain"}
    assert FF.setter_kinds(_mon("tyranitar", "sandstream")) == {"weather": "sand"}   # id-form too
    assert FF.setter_kinds(RILLA) == {"terrain": "grassy"}
    assert FF.setter_kinds(_mon("indeedee", "Psychic Surge")) == {"terrain": "psychic"}
    assert FF.setter_kinds(ZARD_Y) == {"weather": "sun"}                          # item-based mega setter
    assert FF.setter_kinds(_mon("charizard", "Blaze", "charizarditey")) == {"weather": "sun"}
    assert FF.setter_kinds(ARBOLIVA) == {}                                        # Seed Sower: on-hit only
    assert FF.setter_kinds(PLAIN) == {}
    assert FF.setter_kinds(_mon("x", None, None)) == {}
    assert FF.setter_kinds(SimpleNamespace()) == {}                               # no attrs at all


def test_setter_kinds_ability_wins_over_weather_item():
    assert FF.setter_kinds(_mon("charizard", "Drizzle", "Charizardite Y")) == {"weather": "rain"}


@pytest.mark.parametrize("w, kind", [
    (Weather.SANDSTORM, "sand"), (Weather.RAINDANCE, "rain"), (Weather.SUNNYDAY, "sun"),
    (Weather.SNOWSCAPE, "snow"), (Weather.HAIL, "snow"), (Weather.DESOLATELAND, "sun"),
    (Weather.PRIMORDIALSEA, "rain"), (Weather.DELTASTREAM, "wind"), (Weather.UNKNOWN, None),
])
def test_current_kinds_weather_real_enum(w, kind):
    assert FF.current_kinds(_battle([], weather=w))["weather"] == kind


@pytest.mark.parametrize("f, kind", [
    (Field.PSYCHIC_TERRAIN, "psychic"), (Field.GRASSY_TERRAIN, "grassy"),
    (Field.ELECTRIC_TERRAIN, "electric"), (Field.MISTY_TERRAIN, "misty"),
])
def test_current_kinds_terrain_real_enum(f, kind):
    assert FF.current_kinds(_battle([], terrain=f))["terrain"] == kind


def test_current_kinds_ignores_non_terrain_fields_and_empty_field():
    b = _battle([], fields={Field.TRICK_ROOM: 1, Field.GRAVITY: 2})
    assert FF.current_kinds(b) == {"weather": None, "terrain": None}
    b = _battle([], fields={Field.TRICK_ROOM: 1}, terrain=Field.ELECTRIC_TERRAIN)
    assert FF.current_kinds(b)["terrain"] == "electric"
    assert FF.current_kinds(SimpleNamespace()) == {"weather": None, "terrain": None}   # no attrs


# ══════════════════════════════════════════════════════════════════════════════
# field_biases
# ══════════════════════════════════════════════════════════════════════════════
def test_bench_setter_not_on_field_gets_bias_on_its_switch_index(bench):
    """Drizzle mon at bench index 1, sand up (not one of OUR kinds) -> +B on SWITCH_OFFSET+1 on BOTH slots."""
    bench([PLAIN, PELIPPER])
    b0, b1 = FF.field_biases(_battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.SANDSTORM), B)
    assert SWITCH_OFFSET + 1 == 13
    _only(b0, 13)
    _only(b1, 13)
    assert b0 is not b1                       # independent copies: a per-slot edit must not leak
    b0[13] = 0.0
    assert b1[13] == np.float32(B)


def test_no_bias_when_field_is_already_ours(bench):
    bench([PLAIN, PELIPPER])
    assert FF.field_biases(_battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.RAINDANCE), B) == (None, None)


def test_any_of_our_kinds_counts_as_ours(bench):
    """Team sets rain AND sun; sun is up (our Torkoal's) -> the benched Drizzle mon is NOT pushed in."""
    bench([PELIPPER])
    battle = _battle([TORKOAL, ACT_B, PELIPPER], weather=Weather.SUNNYDAY, active_pokemon=[TORKOAL, ACT_B])
    assert FF.field_biases(battle, B) == (None, None)


def test_a_setter_left_at_home_does_not_make_the_field_ours(bench):
    """Review F5: 'ours' = the kinds the side can still FIELD (on the field + alive on the bench). Torkoal is
    in the 6-mon team but was not brought (not active, not on the bench) -> sun is THEIRS -> push Pelipper."""
    bench([PELIPPER])
    battle = _battle([TORKOAL, ACT_A, ACT_B, PELIPPER], weather=Weather.SUNNYDAY, active_pokemon=[ACT_A, ACT_B])
    b0, b1 = FF.field_biases(battle, B)
    _only(b0, SWITCH_OFFSET + 0)


def test_weather_ours_but_terrain_theirs_fires_terrain_only(bench):
    """Rain (ours) -> no weather push; Psychic Terrain (theirs) -> Grassy Surge at bench idx 1 still fires."""
    bench([PELIPPER, RILLA])
    battle = _battle([ACT_A, ACT_B, PELIPPER, RILLA], weather=Weather.RAINDANCE, terrain=Field.PSYCHIC_TERRAIN)
    b0, b1 = FF.field_biases(battle, B)
    _only(b0, SWITCH_OFFSET + 1)
    _only(b1, SWITCH_OFFSET + 1)


def test_both_domains_theirs_fires_both_setters(bench):
    bench([PELIPPER, PLAIN, RILLA])
    battle = _battle([ACT_A, ACT_B, PELIPPER, PLAIN, RILLA],
                     weather=Weather.SANDSTORM, terrain=Field.PSYCHIC_TERRAIN)
    b0, _ = FF.field_biases(battle, B)
    _only(b0, [SWITCH_OFFSET + 0, SWITCH_OFFSET + 2])


def test_fire_on_empty(bench):
    bench([PLAIN, PELIPPER])
    battle = _battle([ACT_A, ACT_B, PLAIN, PELIPPER])            # clear field
    b0, b1 = FF.field_biases(battle, B)                          # default: fire on an empty field
    _only(b0, 13)
    _only(b1, 13)
    assert FF.field_biases(battle, B, fire_on_empty=False) == (None, None)


def test_fire_on_empty_false_still_fires_against_their_weather(bench):
    bench([PLAIN, PELIPPER])
    battle = _battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.SANDSTORM)
    b0, _ = FF.field_biases(battle, B, fire_on_empty=False)
    _only(b0, 13)


def test_weather_unknown_counts_as_clear(bench):
    """poke-env's Weather.UNKNOWN is neither ours nor theirs: it behaves exactly like an empty field."""
    bench([PLAIN, PELIPPER])
    battle = _battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.UNKNOWN)
    b0, _ = FF.field_biases(battle, B)
    _only(b0, 13)
    assert FF.field_biases(battle, B, fire_on_empty=False) == (None, None)


def test_seed_sower_gives_no_bias(bench):
    bench([PLAIN, ARBOLIVA])
    battle = _battle([ACT_A, ACT_B, PLAIN, ARBOLIVA], terrain=Field.PSYCHIC_TERRAIN)
    assert FF.field_biases(battle, B) == (None, None)


def test_charizardite_y_gives_sun_bias(bench):
    bench([ZARD_Y, PLAIN])
    battle = _battle([ACT_A, ACT_B, ZARD_Y, PLAIN], weather=Weather.RAINDANCE)
    b0, b1 = FF.field_biases(battle, B)
    _only(b0, SWITCH_OFFSET + 0)
    _only(b1, SWITCH_OFFSET + 0)


def test_active_setter_is_never_biased(bench):
    """The bias goes on BENCH switch indices only: an on-field setter has no switch action to push."""
    bench([PLAIN])
    battle = _battle([PELIPPER, ACT_B, PLAIN], weather=Weather.SANDSTORM)
    assert FF.field_biases(battle, B) == (None, None)


def test_no_setters_on_team_returns_none(bench):
    bench([PLAIN])
    assert FF.field_biases(_battle([ACT_A, ACT_B, PLAIN], weather=Weather.SANDSTORM), B) == (None, None)


def test_bench_beyond_bench_slots_is_ignored(bench):
    """Only the first BENCH_SLOTS bench mons map to switch actions; a 5th setter must not index past 15."""
    extra = [_mon(f"plain{i}", "Defiant") for i in range(BENCH_SLOTS)]
    bench(extra + [PELIPPER])
    battle = _battle([ACT_A, ACT_B, *extra, PELIPPER], weather=Weather.SANDSTORM)
    assert FF.field_biases(battle, B) == (None, None)


def test_replacement_biases_only_forced_slots(bench):
    bench([PLAIN, PELIPPER])
    team = [ACT_A, ACT_B, PLAIN, PELIPPER]
    r0, r1 = FF.field_biases(_battle(team, weather=Weather.SANDSTORM, force_switch=(False, True)), B,
                             replacement=True)
    assert r0 is None
    _only(r1, 13)

    r0, r1 = FF.field_biases(_battle(team, weather=Weather.SANDSTORM, force_switch=(True, False)), B,
                             replacement=True)
    _only(r0, 13)
    assert r1 is None

    r0, r1 = FF.field_biases(_battle(team, weather=Weather.SANDSTORM, force_switch=(True, True)), B,
                             replacement=True)
    _only(r0, 13)
    _only(r1, 13)
    assert r0 is not r1

    assert FF.field_biases(_battle(team, weather=Weather.SANDSTORM, force_switch=(False, False)), B,
                           replacement=True) == (None, None)
    nofs = _battle(team, weather=Weather.SANDSTORM)
    del nofs.force_switch
    assert FF.field_biases(nofs, B, replacement=True) == (None, None)


def test_normal_turn_ignores_force_switch(bench):
    bench([PLAIN, PELIPPER])
    battle = _battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.SANDSTORM, force_switch=(False, False))
    b0, b1 = FF.field_biases(battle, B)
    _only(b0, 13)
    _only(b1, 13)


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, -B, float("nan"), float("inf"), float("-inf"), None, "abc"])
def test_bad_bias_returns_none(bench, bad):
    bench([PLAIN, PELIPPER])
    battle = _battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.SANDSTORM)
    assert FF.field_biases(battle, bad) == (None, None)
    assert FF.pressure_biases("field", battle, bad) == (None, None)
    assert FF.pressure_biases("field", battle, bad, replacement=True) == (None, None)


def test_bias_value_is_the_requested_magnitude(bench):
    bench([PLAIN, PELIPPER])
    battle = _battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.SANDSTORM)
    b0, _ = FF.field_biases(battle, "0.75")              # numeric string is accepted (float())
    _only(b0, 13, 0.75)


# ══════════════════════════════════════════════════════════════════════════════
# pressure_biases / add_bias / note_pressure
# ══════════════════════════════════════════════════════════════════════════════
def test_biases_registry_and_dispatch(bench):
    assert FF.BIASES["field"] is FF.field_biases
    bench([PLAIN, PELIPPER])
    battle = _battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.SANDSTORM, force_switch=(False, True))
    p0, p1 = FF.pressure_biases("field", battle, B)
    _only(p0, 13)
    _only(p1, 13)
    r0, r1 = FF.pressure_biases("field", battle, B, replacement=True)    # replacement forwarded
    assert r0 is None
    _only(r1, 13)


@pytest.mark.parametrize("name", ["nope", "Field", "", None, "focus"])
def test_unknown_pressure_name_returns_none(bench, name):
    bench([PLAIN, PELIPPER])
    battle = _battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.SANDSTORM)
    assert FF.pressure_biases(name, battle, B) == (None, None)


def test_add_bias_none_safe():
    a = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
    a[13] = B
    b = [0.0] * ACTIONS_PER_SLOT
    b[0], b[13] = 1.0, 0.5
    a_before, b_before = a.copy(), list(b)

    assert FF.add_bias(None, None) is None

    r = FF.add_bias(a, None)
    assert isinstance(r, np.ndarray) and r.dtype == np.float32
    np.testing.assert_array_equal(r, a)

    r = FF.add_bias(None, b)                         # a plain list comes back as a float32 array
    assert isinstance(r, np.ndarray) and r.dtype == np.float32
    np.testing.assert_allclose(r, np.asarray(b, dtype=np.float32))

    r = FF.add_bias(a, b)
    assert r.dtype == np.float32
    assert r[13] == pytest.approx(3.0) and r[0] == pytest.approx(1.0) and float(r.sum()) == pytest.approx(4.0)

    np.testing.assert_array_equal(a, a_before)       # inputs are never mutated
    assert b == b_before


def _vec(*idx, val=B):
    v = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
    for i in idx:
        v[i] = val
    return v


def _mask(*legal):
    return [i in legal for i in range(ACTIONS_PER_SLOT)]


def test_note_pressure_counts_fired_and_taken_on_legal_biased_indices():
    p = SimpleNamespace()                             # no _pressure_stats yet -> created
    FF.note_pressure(p, (_vec(13), _vec(13)), (_mask(0, 13), _mask(0, 12)), (13, 0))
    # slot 0: 13 legal + biased -> fired, picked 13 -> taken; slot 1: 13 ILLEGAL -> not fired
    assert p._pressure_stats == {"fired": 1, "taken": 1}

    FF.note_pressure(p, (_vec(13), None), (_mask(0, 13), _mask(13)), (0, 13))
    # slot 0 fired but the model chose 0; slot 1 has no bias -> skipped even though it picked 13
    assert p._pressure_stats == {"fired": 2, "taken": 1}

    FF.note_pressure(p, (_vec(13), _vec(13)), (None, _mask(13)), (13, None))
    # slot 0 has no mask -> skipped; slot 1 fired, pick None (pass) -> not taken
    assert p._pressure_stats == {"fired": 3, "taken": 1}


def test_note_pressure_ignores_zero_and_illegal_biases():
    p = SimpleNamespace(_pressure_stats={"fired": 5, "taken": 2})     # accumulates onto existing stats
    FF.note_pressure(p, (np.zeros(ACTIONS_PER_SLOT, dtype=np.float32), _vec(14)),
                     (_mask(*range(16)), _mask(12, 13)), (0, 12))
    assert p._pressure_stats == {"fired": 5, "taken": 2}
    FF.note_pressure(p, (_vec(12, 14), _vec(15)), (_mask(14), _mask(15)), (np.int64(14), 15))
    assert p._pressure_stats == {"fired": 7, "taken": 4}


# ══════════════════════════════════════════════════════════════════════════════
# VGCPlayer._select_actions (normal turn) — stub self
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def turn_env(monkeypatch, bench):
    """Wire _select_actions to fakes. Returns a namespace with the recorded bc_action_indices calls."""
    import v_dance.play.model_io as M
    import v_dance.play.player as P
    import v_dance.play.thought_feed as TF
    import v_dance.play.vgc_base as VB

    env = SimpleNamespace(calls=[], picks=[(13, 0)], masks={0: _mask(0, 1, 13), 1: _mask(0, 1)}, taps=[])

    def fake_bc(model, heads, state_vec, mask0, mask1, device, **kw):
        env.calls.append({"mask0": list(mask0), "mask1": list(mask1), **kw})
        return env.picks[min(len(env.calls) - 1, len(env.picks) - 1)]

    monkeypatch.setattr(P, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(P, "build_legal_action_mask", lambda battle, slot: list(env.masks[slot]))
    monkeypatch.setattr(M, "bc_action_indices", fake_bc)
    monkeypatch.setattr(M, "value_trained", lambda model: False)
    monkeypatch.setattr(M, "decode_record", lambda: {})
    monkeypatch.setattr(TF, "tap_turn", lambda *a, **k: env.taps.append((a, k)))
    monkeypatch.setattr(VB, "hh_pair_futility_hook", lambda battle: None)
    bench([PLAIN, PELIPPER])
    env.battle = _battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.SANDSTORM)
    return env


def _stub_player(*, pressure="field", bias=B, adapt=False):
    from v_dance.play.player import VGCPlayer
    p = VGCPlayer.__new__(VGCPlayer)
    p._model = object()
    p._model_heads = ("our_a", "our_b")
    p._device = "cpu"
    p._temperature = 0.0
    p._top_p = 1.0
    p._rng = None
    p._adapt_rules = adapt
    p._pressure = pressure
    p._pressure_bias = bias
    p._pressure_stats = {"fired": 0, "taken": 0}
    return p


def _state():
    return np.zeros(8, dtype=np.float32)          # never read: bc_action_indices is faked


def test_select_actions_pressure_reaches_decode(turn_env):
    p = _stub_player()
    a0, a1, src = p._select_actions(turn_env.battle, _state())
    assert (a0, a1, src) == (13, 0, "model")
    assert len(turn_env.calls) == 1
    kw = turn_env.calls[0]
    _only(kw["bias0"], 13)
    _only(kw["bias1"], 13)
    # slot 0: 13 legal + picked -> fired/taken; slot 1: 13 illegal -> not fired
    assert p._pressure_stats == {"fired": 1, "taken": 1}
    assert len(turn_env.taps) == 1                 # the thought-feed tap still runs once


def test_select_actions_pressure_sums_with_adapt_rules_bias(turn_env, monkeypatch):
    import v_dance.play.adapt_rules as AR
    ab0 = _vec(0, val=1.0)
    ab0[13] = 0.5
    monkeypatch.setattr(AR, "action_biases", lambda player, battle: (ab0, None))
    p = _stub_player(adapt=True)
    p._select_actions(turn_env.battle, _state())
    kw = turn_env.calls[0]
    want0 = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
    want0[0], want0[13] = 1.0, 0.5 + B
    np.testing.assert_allclose(kw["bias0"], want0)
    _only(kw["bias1"], 13)                          # adapt gave None on slot 1 -> pressure alone
    assert ab0[13] == np.float32(0.5)               # the adapt-rules array was not mutated in place


def test_select_actions_adapt_only_when_no_pressure(turn_env, monkeypatch):
    import v_dance.play.adapt_rules as AR
    ab0 = _vec(0, val=1.0)
    monkeypatch.setattr(AR, "action_biases", lambda player, battle: (ab0, None))
    p = _stub_player(pressure=None, adapt=True)
    p._select_actions(turn_env.battle, _state())
    kw = turn_env.calls[0]
    np.testing.assert_array_equal(kw["bias0"], ab0)
    assert kw["bias1"] is None


def test_select_actions_without_pressure_has_no_bias(turn_env):
    p = _stub_player(pressure=None)
    assert p._select_actions(turn_env.battle, _state()) == (13, 0, "model")
    kw = turn_env.calls[0]
    assert kw["bias0"] is None and kw["bias1"] is None
    assert p._pressure_stats == {"fired": 0, "taken": 0}


def test_select_actions_pressure_attr_missing_is_off(turn_env):
    """A player built before the drill existed (no _pressure attribute) decodes unbiased."""
    p = _stub_player()
    del p._pressure, p._pressure_bias, p._pressure_stats
    p._select_actions(turn_env.battle, _state())
    kw = turn_env.calls[0]
    assert kw["bias0"] is None and kw["bias1"] is None


def test_select_actions_pressure_that_does_not_fire_leaves_bias_none(turn_env):
    turn_env.battle = _battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.RAINDANCE)   # already ours
    p = _stub_player()
    p._select_actions(turn_env.battle, _state())
    kw = turn_env.calls[0]
    assert kw["bias0"] is None and kw["bias1"] is None
    assert p._pressure_stats == {"fired": 0, "taken": 0}


def test_select_actions_pressure_reaches_dedup_redecode(turn_env):
    """Both heads pick the same switch (13) -> slot 1 is re-decoded with 13 masked out; the re-decode
    must carry the SAME summed bias, and the fire count uses slot 1's deduped mask."""
    turn_env.masks = {0: _mask(0, 13), 1: _mask(0, 13)}
    turn_env.picks = [(13, 13), (13, 0)]
    p = _stub_player()
    a0, a1, src = p._select_actions(turn_env.battle, _state())
    assert (a0, a1, src) == (13, 0, "model")
    assert len(turn_env.calls) == 2
    for kw in turn_env.calls:
        _only(kw["bias0"], 13)
        _only(kw["bias1"], 13)
    assert turn_env.calls[1]["mask1"][13] is False
    assert p._pressure_stats == {"fired": 1, "taken": 1}


def test_select_actions_pressure_failure_is_non_fatal(turn_env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("pressure exploded")
    monkeypatch.setattr(FF, "pressure_biases", boom)
    p = _stub_player()
    assert p._select_actions(turn_env.battle, _state()) == (13, 0, "model")
    kw = turn_env.calls[0]
    assert kw["bias0"] is None and kw["bias1"] is None
    assert p._pressure_stats == {"fired": 0, "taken": 0}


# ══════════════════════════════════════════════════════════════════════════════
# VGCPlayer._select_replacement_actions — stub self
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def rep_env(monkeypatch, bench):
    import v_dance.play.model_io as M
    import v_dance.play.player as P
    import v_dance.play.thought_feed as TF

    zeros = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
    monkeypatch.setattr(P, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(M, "head_logits", lambda model, heads, sv, device: (zeros.copy(), zeros.copy()))
    rep_mask = {0: _mask(12, 13), 1: _mask()}
    monkeypatch.setattr(P, "build_replacement_mask", lambda battle, slot: list(rep_mask[slot]))
    monkeypatch.setattr(TF, "tap_replacement", lambda *a, **k: None)
    bench([PLAIN, PELIPPER])
    return _battle([ACT_A, ACT_B, PLAIN, PELIPPER], weather=Weather.SANDSTORM, force_switch=(True, False))


def test_replacement_unbiased_is_first_wins(rep_env):
    p = _stub_player(pressure=None)
    assert p._select_replacement_actions(rep_env, _state()) == (SWITCH_OFFSET + 0, None, "forced_switch_model")
    assert p._pressure_stats == {"fired": 0, "taken": 0}


def test_replacement_pressure_sends_the_setter_back_in(rep_env):
    """Zero logits, slot 0 forced, mask {12, 13}: argmax first-wins would bring bench 0 (index 12);
    pressure on the Drizzle mon (bench 1 -> index 13) flips it to 13."""
    p = _stub_player()
    a0, a1, src = p._select_replacement_actions(rep_env, _state())
    assert (a0, a1, src) == (13, None, "forced_switch_model")
    assert p._pressure_stats == {"fired": 1, "taken": 1}


def test_replacement_pressure_ignored_when_field_is_ours(rep_env):
    rep_env.weather = {Weather.RAINDANCE: 1}
    p = _stub_player()
    assert p._select_replacement_actions(rep_env, _state())[0] == SWITCH_OFFSET + 0
    assert p._pressure_stats == {"fired": 0, "taken": 0}


def test_replacement_pressure_failure_is_non_fatal(rep_env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("pressure exploded")
    monkeypatch.setattr(FF, "pressure_biases", boom)
    p = _stub_player()
    assert p._select_replacement_actions(rep_env, _state()) == (SWITCH_OFFSET + 0, None, "forced_switch_model")


# ══════════════════════════════════════════════════════════════════════════════
# kwarg plumbing — opponent-only by construction
# ══════════════════════════════════════════════════════════════════════════════
def _real_player(name, **kw):
    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    from v_dance.play.player import VGCPlayer
    team = R.load_team(R.resolve_team_path("WolfeGlick"))
    return VGCPlayer(model_path=None, account_configuration=AccountConfiguration(name, None),
                     battle_format=R.BATTLE_FORMAT, team=team, start_listening=False, **kw)


def test_vgcplayer_accepts_pressure_kwargs():
    p = _real_player("FFpress1", pressure="field", pressure_bias=2.5)
    try:
        assert p._pressure == "field"
        assert p._pressure_bias == 2.5
        assert p._pressure_stats == {"fired": 0, "taken": 0}
    finally:
        p.close()


def test_vgcplayer_pressure_defaults_off():
    p = _real_player("FFpress2")
    try:
        assert p._pressure is None and p._pressure_bias == 0.0
        assert p._pressure_stats == {"fired": 0, "taken": 0}
    finally:
        p.close()


@pytest.mark.parametrize("bias", [0.0, -1.0, float("nan"), None, "abc"])
def test_vgcplayer_pressure_without_positive_bias_is_off(bias):
    p = _real_player("FFpress3", pressure="field", pressure_bias=bias)
    try:
        assert p._pressure is None
        if not (isinstance(bias, float) and math.isnan(bias)):
            assert p._pressure_bias == (float(bias) if isinstance(bias, float) else 0.0)
    finally:
        p.close()


@pytest.mark.parametrize("kw", [
    {"pressure": "field", "pressure_bias": 1.0},
    {"pressure": "field"},
    {"pressure_bias": 1.0},
    {"pressure": "field", "pressure_bias": 0.0},
])
def test_selfplay_player_refuses_pressure(kw):
    from v_dance.selfplay.game_runner import SelfPlayVGCPlayer
    with pytest.raises(ValueError, match="opponent-only"):
        SelfPlayVGCPlayer(object(), **kw)


class _Recorder:
    made: list = []

    def __init__(self, *args, **kwargs):
        type(self).made.append((type(self).__name__, args, kwargs))


@pytest.fixture
def make_player_env(monkeypatch):
    import v_dance.play.run_local_battle as R

    class RecVGC(_Recorder):
        made = []

    class RecRandom(_Recorder):
        made = []

    monkeypatch.setattr(R, "VGCPlayer", RecVGC)
    monkeypatch.setattr(R, "RandomVGCPlayer", RecRandom)
    return R, RecVGC, RecRandom


def test_make_player_forwards_pressure_to_model_branch(make_player_env):
    R, RecVGC, RecRandom = make_player_env
    p = R.make_player("FFmk1", "team-text", model_path=Path("fake_battle.pt"), pressure="field", pressure_bias=2.5)
    assert isinstance(p, RecVGC) and not RecRandom.made
    (_, _, kw), = RecVGC.made
    assert kw["pressure"] == "field" and kw["pressure_bias"] == 2.5
    assert kw["model_path"] == Path("fake_battle.pt")


def test_make_player_model_branch_defaults_pressure_off(make_player_env):
    R, RecVGC, _ = make_player_env
    R.make_player("FFmk2", "team-text", model_path=Path("fake_battle.pt"))
    (_, _, kw), = RecVGC.made
    assert kw["pressure"] is None and kw["pressure_bias"] == 0.0


def test_make_player_random_branch_never_gets_pressure(make_player_env):
    R, RecVGC, RecRandom = make_player_env
    p = R.make_player("FFmk3", "team-text", model_path=None, pressure="field", pressure_bias=2.5)
    assert isinstance(p, RecRandom) and not RecVGC.made
    (_, _, kw), = RecRandom.made
    assert "pressure" not in kw and "pressure_bias" not in kw
