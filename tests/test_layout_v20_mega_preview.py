"""Layout v20 — the MEGA PREVIEW block (mega audit gap 2, 2026-10-02; USER "fix 1-5").

Every v19 feature of a mega-capable mon that has not mega'd yet described its BASE forme, while the mega resolves
before anyone moves (55 % of opponent megas on turn 1). v20 adds, per mon, what it is about to become + how each
move fares INTO the enemy's mega and AS our own mega. Locks: the layout, the values on the audit's own example
(Arcanine into a pre-mega Golisopod), the gating, live ↔ offline parity on a real M-C replay, the move-slot
augmentation, and the v19 → v20 AUTO-UPGRADE (the served checkpoints keep their exact policy).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from v_dance.encoders import mega_preview as MP
from v_dance.encoders.encoder_layout import (
    MEGA_PREVIEW_FEATURES, MEGA_PREVIEW_MON_FEATURES, MEGA_PREVIEW_PER_MOVE, MEGA_PREVIEW_REL, NUM_TYPES,
    POKEMON_FEATURES, STATE_DIM, STATE_LAYOUT_VERSION, V19_POKEMON_FEATURES, V19_STATE_DIM, _TYPE_IDX,
)
from v_dance.encoders.state_encoder import MOVE_FEATURES, StateEncoder, _MOVE_BLOCK_REL

_REPO = Path(__file__).resolve().parents[1]
_MC = "gen9championsvgc2026regmc"


def _mon(sp, ab, moves=(), stats=None, item=None, exact=False, **kw):
    d = {"species": sp, "base_species": sp, "hp_pct": 100.0, "seen": True, "is_fainted": False,
         "known_moves": list(moves), "revealed_moves": [], "boosts": {}, "status": None, "known_ability": ab,
         "stats_estimate": {"mode": "exact", "stats": stats or {"atk": 150, "spa": 100, "def": 100, "spd": 100,
                                                                 "hp": 180, "spe": 100}}}
    if item:
        d["known_item"] = item
    if exact:
        d["exact"] = True
    d.update(kw)
    return d


def _encode(our_a, opp_a, fmt=_MC, opp_bench=(), our_bench=()):
    snap = {"our_active": {"our_a": our_a, "our_b": None}, "opp_active": {"opp_a": opp_a, "opp_b": None},
            "our_bench": list(our_bench), "opp_bench": list(opp_bench), "field": {}, "side_conditions": {}}
    return StateEncoder().encode_snapshot(snap, turn=1, fmt=fmt)


def _row(v, slot):
    return v[slot * POKEMON_FEATURES:(slot + 1) * POKEMON_FEATURES]


def _mon_part(row):
    return row[MEGA_PREVIEW_REL:MEGA_PREVIEW_REL + MEGA_PREVIEW_MON_FEATURES]


def _move_part(row, m):
    b = MEGA_PREVIEW_REL + MEGA_PREVIEW_MON_FEATURES + m * MEGA_PREVIEW_PER_MOVE
    return row[b:b + MEGA_PREVIEW_PER_MOVE]


_ARC = _mon("Arcanine", "Intimidate", ["Flare Blitz", "Thunder Fang", "Extreme Speed", "Protect"],
            item="Sitrus Berry", exact=True)
_GOLI = _mon("Golisopod", "Emergency Exit", ["Leech Life", "First Impression"],
             stats={"atk": 160, "spa": 70, "def": 160, "spd": 110, "hp": 180, "spe": 60})


# ── layout ─────────────────────────────────────────────────────────────────────
def test_layout_v20():
    assert STATE_LAYOUT_VERSION == 20
    assert (MEGA_PREVIEW_MON_FEATURES, MEGA_PREVIEW_PER_MOVE, MEGA_PREVIEW_FEATURES) == (70, 10, 110)
    assert POKEMON_FEATURES == V19_POKEMON_FEATURES + 110 == 523
    assert STATE_DIM == V19_STATE_DIM + 12 * 110 == 6377
    assert MEGA_PREVIEW_REL == V19_POKEMON_FEATURES - 4 == POKEMON_FEATURES - 4 - 110


# ── values: the audit's own example ──────────────────────────────────────────
def test_opponent_pre_mega_golisopod_reads_its_mega_forme():
    v = _encode(_ARC, _GOLI)
    mp = _mon_part(_row(v, 2))
    assert mp[0] == pytest.approx(0.984, abs=1e-3)                       # Golisopite share (M-C)
    assert mp[1 + _TYPE_IDX["BUG"]] == pytest.approx(1.0) and mp[1 + _TYPE_IDX["STEEL"]] == pytest.approx(1.0)
    assert mp[1 + _TYPE_IDX["WATER"]] == 0.0
    arc = _row(v, 0)
    flare, fang = _move_part(arc, 0), _move_part(arc, 1)
    base = lambda m, k: float(arc[_MOVE_BLOCK_REL + m * MOVE_FEATURES + k])
    assert base(0, 9) == 0.0 and flare[0] == 1.0                         # Flare Blitz: neutral → 4× into the mega
    assert flare[1] > base(0, 14)                                        # … and a bigger damage band
    assert base(1, 9) == 0.5 and fang[0] == 0.0                          # Thunder Fang: 2× → neutral
    assert _mon_part(arc)[0] == 0.0                                      # Arcanine (Sitrus) never megas


def test_the_regulation_decides_the_stone_mix():
    chomp = _mon("Garchomp", "Rough Skin", ["Earthquake"])
    mc = _mon_part(_row(_encode(_ARC, chomp, fmt=_MC), 2))
    assert mc[0] == pytest.approx(0.473, abs=1e-3)                       # Garchompite Z 0.452 + Garchompite 0.021
    # conditional typing: mostly the pure-Dragon Z forme
    assert mc[1 + _TYPE_IDX["DRAGON"]] == pytest.approx(1.0) and mc[1 + _TYPE_IDX["GROUND"]] < 0.1
    mb = _mon_part(_row(_encode(_ARC, chomp, fmt="[Gen 9 Champions] VGC 2026 Reg M-B"), 2))
    assert mb[0] == 0.0 or mb[1 + _TYPE_IDX["GROUND"]] == pytest.approx(1.0)   # no Z stone existed in M-B


def test_gating_known_item_side_megaed_and_mega_already():
    scarf = dict(_GOLI, known_item="Choice Scarf")
    assert _mon_part(_row(_encode(_ARC, scarf), 2))[0] == 0.0           # a revealed non-stone item
    knocked = dict(_GOLI, item_consumed=True)
    assert _mon_part(_row(_encode(_ARC, knocked), 2))[0] == 0.0          # itemless: no stone
    megaed_mate = _mon("Charizard-Mega-Y", "Drought", is_mega=True)
    v = _encode(_ARC, _GOLI, opp_bench=[megaed_mate])
    assert _mon_part(_row(v, 2))[0] == 0.0                               # the opponent already used its mega
    assert float(np.abs(_move_part(_row(v, 0), 0)).sum()) == 0.0         # … so nothing to preview into
    mega_now = _mon("Golisopod-Mega", "Tough Claws", is_mega=True)
    assert float(np.abs(_mon_part(_row(_encode(_ARC, mega_now), 2))).sum()) == 0.0


def test_our_own_known_stone_previews_our_mega():
    chomp = _mon("Garchomp", "Rough Skin", ["Earthquake", "Dragon Claw"], item="Garchompite Z", exact=True,
                 stats={"atk": 182, "spa": 100, "def": 115, "spd": 105, "hp": 183, "spe": 154})
    tank = _mon("Snorlax", "Thick Fat", stats={"atk": 80, "spa": 80, "def": 300, "spd": 300, "hp": 700, "spe": 40})
    v = _encode(chomp, tank)
    row = _row(v, 0)
    mp = _mon_part(row)
    assert mp[0] == 1.0 and mp[1 + _TYPE_IDX["DRAGON"]] == 1.0 and mp[1 + _TYPE_IDX["GROUND"]] == 0.0
    assert mp[1 + NUM_TYPES + 5] == pytest.approx(49 / 255, abs=1e-6)    # Spe +49 (102 → 151)
    eq, claw = _move_part(row, 0), _move_part(row, 1)
    base_band = lambda m: float(row[_MOVE_BLOCK_REL + m * MOVE_FEATURES + 14])
    assert eq[6] < base_band(0)                                          # EQ loses its STAB as Garchomp-Mega-Z
    assert claw[6] == pytest.approx(base_band(1), rel=1e-5)              # Dragon Claw: STAB kept, Atk 130 both


def test_mega_options_and_forme_resolution():
    assert MP.mega_options("garchomp", "garchompitez", _MC) == [("Garchomp-Mega-Z", 1.0)]
    assert MP.mega_options("garchomp", "", _MC) == []
    assert MP.mega_options("incineroar", None, _MC) == []
    assert MP._forme_for_stone("meowsticf", "meowsticite") == "Meowstic-F-Mega"
    assert MP._forme_for_stone("meowstic", "meowsticite") == "Meowstic-M-Mega"
    assert MP.best_forme([("Charizard-Mega-X", 0.05), ("Charizard-Mega-Y", 0.93)]) == "Charizard-Mega-Y"


# ── live ↔ offline parity on a real M-C replay (opp rows, raw poke-env live path) ──────────────────────
_REPLAY = _REPO / "data/vods/Type_C/20260929-080543-15680_online_battle-gen9championsvgc2026regmc-2689836031.html"


@pytest.mark.skipif(not _REPLAY.exists(), reason="M-C replay corpus not present")
def test_live_matches_offline_mon_part_on_a_real_game():
    import tests._parity_harness as H
    from v_dance.parser.vod_parser.replay_parser import extract_log_from_html
    log = extract_log_from_html(_REPLAY.read_text(encoding="utf-8"))
    persp = "p1"
    user = H.player_usernames(log)[persp]
    live = H.live_vectors_per_turn(log, user)
    off = H.offline_vectors_per_turn(log, persp)
    checked = 0
    for turn in sorted(set(live) & set(off)):
        for slot in (2, 3, 8, 9, 10, 11):                     # opponent actives + bench
            lv, ov = _row(live[turn], slot), _row(off[turn], slot)
            if not lv.any() or not ov.any():
                continue
            blk = slice(MEGA_PREVIEW_REL, MEGA_PREVIEW_REL + MEGA_PREVIEW_FEATURES)   # mon AND per-move parts
            assert np.allclose(lv[blk], ov[blk], atol=1e-6), (turn, slot)
            checked += int(ov[MEGA_PREVIEW_REL] > 0)
    assert checked > 0                                        # the opponent's Golisopod / Garchomp previewed


# ── move-slot augmentation carries the per-move preview ─────────────────────────────────────────────
def test_permute_move_slots_moves_the_preview_sub_blocks():
    from v_dance.encoders.action_codec import permute_move_slots
    v = np.zeros(STATE_DIM, np.float32)
    base = MEGA_PREVIEW_REL + MEGA_PREVIEW_MON_FEATURES
    for m in range(4):
        v[_MOVE_BLOCK_REL + m * MOVE_FEATURES] = m + 1
        v[base + m * MEGA_PREVIEW_PER_MOVE] = 10 * (m + 1)
    permute_move_slots(v, 0, [2, 0, 3, 1])
    assert [v[_MOVE_BLOCK_REL + m * MOVE_FEATURES] for m in range(4)] == [3, 1, 4, 2]
    assert [v[base + m * MEGA_PREVIEW_PER_MOVE] for m in range(4)] == [30, 10, 40, 20]
    from v_dance.datatools.policy_analysis import permute_move_slots as pa_perm
    w = np.zeros(STATE_DIM, np.float32)
    for m in range(4):
        w[base + m * MEGA_PREVIEW_PER_MOVE] = 10 * (m + 1)
    assert [pa_perm(w, 0, [2, 0, 3, 1])[base + m * MEGA_PREVIEW_PER_MOVE] for m in range(4)] == [30, 10, 40, 20]


# ── v19 → v20 auto-upgrade ──────────────────────────────────────────────────────────────────────────
torch = pytest.importorskip("torch")
from v_dance.models import layout_upgrade as LU  # noqa: E402


def test_pad_states_inserts_the_block_in_every_mon_row():
    x = np.arange(V19_STATE_DIM, dtype=np.float32) + 1.0
    y = LU.pad_states(x)
    assert y.shape == (STATE_DIM,)
    for s in range(12):
        r19, r20 = x[s * 413:(s + 1) * 413], y[s * POKEMON_FEATURES:(s + 1) * POKEMON_FEATURES]
        assert np.array_equal(r20[:MEGA_PREVIEW_REL], r19[:MEGA_PREVIEW_REL])
        assert not r20[MEGA_PREVIEW_REL:MEGA_PREVIEW_REL + 110].any()
        assert np.array_equal(r20[MEGA_PREVIEW_REL + 110:], r19[MEGA_PREVIEW_REL:])
    assert np.array_equal(y[12 * POKEMON_FEATURES:], x[12 * 413:])
    assert LU.pad_states(y) is y                                         # v20 passes through
    assert LU.pad_states(np.stack([x, x])).shape == (2, STATE_DIM)


def test_widened_mon_enc_is_exact():
    torch.manual_seed(0)
    w19 = torch.randn(256, 517)
    sd = {"policy.mon_enc.0.weight": w19.clone(), "other": torch.randn(3, 517)}
    assert LU.upgrade_state_dict(sd) == 1 and sd["policy.mon_enc.0.weight"].shape == (256, 627)
    assert sd["other"].shape == (3, 517)                                 # only mon_enc.0.weight keys
    row19 = torch.randn(517)
    row20 = torch.cat([row19[:MEGA_PREVIEW_REL], torch.randn(110), row19[MEGA_PREVIEW_REL:]])   # any preview
    assert torch.allclose(sd["policy.mon_enc.0.weight"] @ row20, w19 @ row19, atol=1e-5)
    opt = {"state": {0: {"exp_avg": torch.ones(256, 517), "exp_avg_sq": torch.ones(256, 517), "step": torch.tensor(3.)}}}
    assert LU.upgrade_optimizer_state(opt) == 2 and opt["state"][0]["exp_avg"].shape == (256, 627)


_SERVED = _REPO / "ai_train_scripts/BC_model/checkpoints_attn_lg2_g50_20261001/battle_base.pt"


@pytest.mark.skipif(not _SERVED.exists(), reason="served checkpoint not present")
def test_the_served_v19_checkpoint_loads_and_ignores_the_preview_until_trained():
    from v_dance.play.model_io import load_bc_policy
    model, heads = load_bc_policy(str(_SERVED))
    raw = torch.load(_SERVED, map_location="cpu", weights_only=False)
    w19 = raw["model_state"]["mon_enc.0.weight"]
    assert w19.shape[1] == 517 and model.mon_enc[0].weight.shape[1] == 627
    w20 = model.mon_enc[0].weight.detach()
    assert torch.equal(w20[:, :MEGA_PREVIEW_REL], w19[:, :MEGA_PREVIEW_REL])
    assert torch.equal(w20[:, MEGA_PREVIEW_REL + 110:], w19[:, MEGA_PREVIEW_REL:])
    assert not w20[:, MEGA_PREVIEW_REL:MEGA_PREVIEW_REL + 110].any()
    # the preview block cannot move the policy / value until training gives it weight
    model.eval()
    v = _encode(_ARC, _GOLI)
    blank = v.copy()
    for s in range(12):
        blank[s * POKEMON_FEATURES + MEGA_PREVIEW_REL: s * POKEMON_FEATURES + MEGA_PREVIEW_REL + 110] = 0.0
    assert not np.array_equal(v, blank)
    with torch.no_grad():
        a1, g1, v1 = model(torch.tensor(v))
        a0, g0, v0 = model(torch.tensor(blank))
    assert all(torch.allclose(a1[k], a0[k], atol=1e-6) for k in a1)
    assert math.isclose(float(v1), float(v0), abs_tol=1e-6)


# ── review 10-02 fixes ───────────────────────────────────────────────────────────────────────────────
def _band(row, m, e=0):
    return float(_move_part(row, m)[3 * e + 1])


def _first(row, m, e=0):
    return float(_move_part(row, m)[3 * e + 2])


def test_a_believed_held_item_does_not_follow_the_mon_into_its_mega():
    # an unrevealed Charizard's belief top item is a Choice Scarf — impossible while it holds Charizardite Y
    fast = _mon("Garchomp", "Rough Skin", ["Dragon Claw"], item="Life Orb", exact=True,
                stats={"atk": 182, "spa": 100, "def": 115, "spd": 105, "hp": 183, "spe": 200})
    zard = _mon("Charizard", "Blaze", stats={"atk": 100, "spa": 160, "def": 100, "spd": 105, "hp": 160, "spe": 152})
    scarf = dict(zard, belief={"items": [{"name": "Choice Scarf", "p": 0.7}]})
    a = _first(_row(_encode(fast, zard), 0), 0)
    b = _first(_row(_encode(fast, scarf), 0), 0)
    assert a == pytest.approx(b)                      # the believed Scarf (×1.5 → 228 > 200) no longer leaks in


def test_a_mega_weather_setter_changes_the_field_for_the_preview():
    # Mega Charizard Y sets SUN on mega-evolving: our Water move into it is halved, not neutral-weather
    pel = _mon("Pelipper", "Drizzle", ["Hydro Pump"], item="Focus Sash", exact=True)
    zard = _mon("Charizard", "Blaze", stats={"atk": 100, "spa": 160, "def": 100, "spd": 105, "hp": 160, "spe": 100})
    plain = _mon("Charizard", "Blaze", stats=zard["stats_estimate"]["stats"], known_item="Charcoal")
    row = _row(_encode(pel, zard), 0)
    assert _band(row, 0) > 0
    # same Charizard WITHOUT a mega option: the normal band is computed in clear weather
    base = float(_row(_encode(pel, plain), 0)[_MOVE_BLOCK_REL + 14])
    from v_dance.encoders.mega_preview import forme_view
    d_spd = forme_view("charizard", "Charizard-Mega-Y")["delta"]["spd"]          # +30 SpD
    # Water into Fire/Flying stays 2× (Y keeps the typing) → the sun ×0.5 and the SpD gain are the only changes
    assert _band(row, 0) / base == pytest.approx(0.5 * 105 / (105 + d_spd), rel=0.05)


def test_the_partner_still_blocks_priority_in_the_preview():
    kanga = _mon("Kangaskhan", "Scrappy", stats={"atk": 125, "spa": 60, "def": 100, "spd": 100, "hp": 180, "spe": 110})
    farig = _mon("Farigiraf", "Armor Tail")
    us = _mon("Incineroar", "Intimidate", ["Fake Out"], item="Sitrus Berry", exact=True)
    snap = {"our_active": {"our_a": us, "our_b": None}, "opp_active": {"opp_a": kanga, "opp_b": farig},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    v = StateEncoder().encode_snapshot(snap, turn=1, fmt=_MC)
    row = _row(v, 0)
    assert float(row[_MOVE_BLOCK_REL + 14]) == 0.0               # Armor Tail: Fake Out blocked
    assert _band(row, 0) == 0.0                                    # … and still blocked into Mega Kangaskhan


def test_mega_floette_fairy_aura_reaches_its_own_moves():
    flo = _mon("Floette-Eternal", "Flower Veil", ["Moonblast"], item="Floettite", exact=True,
               stats={"atk": 70, "spa": 150, "def": 90, "spd": 160, "hp": 150, "spe": 100})
    tank = _mon("Snorlax", "Thick Fat", stats={"atk": 80, "spa": 80, "def": 300, "spd": 300, "hp": 700, "spe": 40})
    row = _row(_encode(flo, tank), 0)
    from v_dance.encoders.mega_preview import forme_view
    d_spa = forme_view("floetteeternal", "Floette-Mega")["delta"]["spa"]
    ratio = float(_move_part(row, 0)[6]) / float(row[_MOVE_BLOCK_REL + 14])
    assert ratio == pytest.approx((150 + d_spa) / 150 * 5448 / 4096, rel=0.03)


def test_only_the_dex_pre_mega_forme_can_mega():
    assert MP.mega_options("raichualola", None, _MC) == []
    assert MP.mega_options("floette", None, _MC) == []                 # only Floette-ETERNAL megas
    assert MP.mega_options("floetteeternal", None, _MC) == [("Floette-Mega", pytest.approx(0.984))]
    assert {f for f, _ in MP.mega_options("raichu", None, _MC)} == {"Raichu-Mega-X", "Raichu-Mega-Y"}


def test_reg_ma_stones_filed_under_the_mega_entries_are_read():
    opts = dict(MP.mega_options("charizard", None, "[Gen 9 Champions] VGC 2026 Reg M-A"))
    assert opts.get("Charizard-Mega-Y", 0) > 0.5


def test_our_item_reaches_the_reconstruction_through_a_preview_only_key():
    m = {"species": "Garchomp", "item_consumed": True}
    assert MP.offline_known_item(m) == ""
    assert MP.offline_known_item(dict(m, preview_item="garchompitez")) == "garchompitez"
    assert MP.offline_known_item({"species": "Garchomp", "preview_item": ""}) == ""


def test_an_empty_slot_stays_all_zero():
    # the model's presence mask is "any nonzero in the row" — the preview must never light an empty slot up
    v = _encode(_ARC, _GOLI)
    assert not _row(v, 1).any() and not _row(v, 3).any()
