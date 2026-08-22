"""Tests for the VGC-Bench HF-log / open-team-sheet (OTS) ingestion path:

  * team_sheet.parse_packed_team / parse_showteam_sides — Showdown packed
    format (the |showteam| payload), golden strings from the real HF sample.
  * team_sheet.packed_team_to_known_side — EVs stay None (stripped, unknown).
  * belief_state.spread_distribution revealed_nature narrowing + fallback.
  * transitions.log_to_transitions — open vs closed regimes end-to-end on a
    synthetic Champions log carrying |showteam| lines.
"""

from __future__ import annotations

import json

import pytest

from v_dance.parser.vod_parser.team_sheet import (
    packed_team_to_known_side,
    parse_packed_team,
    parse_showteam_sides,
)
from v_dance.parser.vod_parser.transitions import log_to_transitions

# Verbatim from logs_gen9championsvgc2026regmb.json (2026-07-01 download).
_PACKED_P1 = (
    "Blaziken||FocusSash|SpeedBoost|HeatWave,AuraSphere,Coaching,Protect|Timid||F|||50|"
    "]Metagross||Metagrossite|ClearBody|IronHead,PsychicFangs,IcePunch,Protect|Jolly|||||50|"
    "]Floette-Eternal||Floettite|FlowerVeil|Protect,DrainingKiss,CalmMind,DazzlingGleam|Timid||F|||50|"
    "]Rotom-Wash||SitrusBerry|Levitate|HydroPump,Thunderbolt,WillOWisp,Protect|Modest|||||50|"
    "]Vivillon-Tundra||ChoiceScarf|CompoundEyes|Hurricane,RagePowder,SleepPowder,Uturn|Timid||M|||50|"
    "]Kommo-o||Leftovers|Soundproof|ClangorousSoul,ClangingScales,Protect,AuraSphere|Modest||F|||50|"
)


# ── parse_packed_team ──────────────────────────────────────────────────────

def test_packed_team_basic_fields():
    mons = parse_packed_team(_PACKED_P1)
    assert len(mons) == 6
    blaziken = mons[0]
    assert blaziken["species"] == "Blaziken"
    assert blaziken["nickname"] is None
    assert blaziken["item"] == "Focus Sash"
    assert blaziken["ability"] == "Speed Boost"
    assert blaziken["nature"] == "Timid"
    assert blaziken["gender"] == "F"
    assert blaziken["level"] == 50
    # OTS strips EVs/IVs — unknown, NOT zero
    assert blaziken["evs"] == {}
    assert blaziken["ivs"] == {}
    assert blaziken["moves"] == ["Heat Wave", "Aura Sphere", "Coaching", "Protect"]


def test_packed_team_display_name_restore():
    mons = {m["species"]: m for m in parse_packed_team(_PACKED_P1)}
    # hyphenated species survive verbatim
    assert "Floette-Eternal" in mons and "Kommo-o" in mons
    # packName'd moves map back through moves.json ("Uturn" → "U-turn")
    assert "U-turn" in mons["Vivillon-Tundra"]["moves"]
    assert "Will-O-Wisp" in mons["Rotom-Wash"]["moves"]
    # items restore their display spacing
    assert mons["Rotom-Wash"]["item"] == "Sitrus Berry"
    assert mons["Vivillon-Tundra"]["item"] == "Choice Scarf"
    # mega stones pass through (single word, no split needed)
    assert mons["Metagross"]["item"] == "Metagrossite"


def test_packed_team_evs_parsed_when_present():
    # Defensive path: a NON-stripped packed mon (standard Showdown pack)
    mons = parse_packed_team("Kingambit||AssaultVest|Defiant|KowtowCleave|Adamant|252,252,,,,4|M|||50|")
    assert mons[0]["evs"] == {"hp": 252, "atk": 252, "spe": 4}
    assert mons[0]["item"] == "Assault Vest"


def test_packed_team_empty_and_garbage():
    assert parse_packed_team("") == []
    assert parse_packed_team("]]") == []


# ── parse_showteam_sides ───────────────────────────────────────────────────

def test_parse_showteam_sides_extracts_both():
    log = "\n".join([
        "|player|p1|alice|101|",
        f"|showteam|p1|{_PACKED_P1}",
        "|showteam|p2|Garchomp||LifeOrb|RoughSkin|Earthquake,StompingTantrum,Protect,DragonClaw|Jolly||F|||50|",
        "|start",
    ])
    sides = parse_showteam_sides(log)
    assert set(sides) == {"p1", "p2"}
    assert len(sides["p1"]) == 6
    assert sides["p2"][0]["species"] == "Garchomp"
    assert sides["p2"][0]["item"] == "Life Orb"


def test_parse_showteam_sides_closed_log():
    assert parse_showteam_sides("|player|p1|alice|101|\n|start\n|turn|1") == {}


# ── packed_team_to_known_side ──────────────────────────────────────────────

def test_packed_known_side_ev_spread_stays_none():
    side = packed_team_to_known_side(parse_packed_team(_PACKED_P1))
    assert side["Blaziken"]["ev_spread"] is None      # stripped ≠ zero
    assert side["Blaziken"]["nature"] == "Timid"
    assert side["Blaziken"]["item"] == "Focus Sash"
    assert side["Blaziken"]["moves"] == ["Heat Wave", "Aura Sphere", "Coaching", "Protect"]
    # mega holder keys under its (base) sheet species
    assert "Metagross" in side


# ── spread_distribution revealed_nature narrowing ──────────────────────────

@pytest.fixture()
def tiny_belief(tmp_path):
    from v_dance.parser.belief_state import BeliefState
    data = {"pokemon": {"Garchomp": {
        "usage_pct": 40.0,
        "moves": [{"name": "Earthquake", "pct": 90.0}],
        "items": [{"name": "Life Orb", "pct": 50.0}, {"name": "Choice Scarf", "pct": 30.0}],
        "abilities": [{"name": "Rough Skin", "pct": 99.0}],
        "spreads": [
            {"nature": "Jolly",   "evs": [0, 32, 0, 0, 0, 32], "pct": 50.0},
            {"nature": "Adamant", "evs": [4, 32, 0, 0, 0, 28], "pct": 30.0},
            {"nature": "Jolly",   "evs": [8, 28, 0, 0, 0, 32], "pct": 20.0},
        ],
        "natures": [{"nature": "Jolly", "pct": 70.0}, {"nature": "Adamant", "pct": 30.0}],
    }}}
    p = tmp_path / "pika.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return BeliefState(p)


def test_spread_distribution_nature_filter(tiny_belief):
    spreads = tiny_belief.spread_distribution("Garchomp", revealed_nature="Adamant")
    assert [s["nature"] for s in spreads] == ["Adamant"]
    assert spreads[0]["p"] == 1.0                     # renormalised over the subset


def test_spread_distribution_nature_fallback(tiny_belief):
    # No listed spread runs Modest → keep the broad prior, not an empty dist
    spreads = tiny_belief.spread_distribution("Garchomp", revealed_nature="Modest")
    assert len(spreads) == 3


def test_spread_distribution_no_reveal_unchanged(tiny_belief):
    spreads = tiny_belief.spread_distribution("Garchomp")
    assert [s["nature"] for s in spreads] == ["Jolly", "Adamant", "Jolly"]


def test_belief_block_nature_narrows_expected_stats(tiny_belief):
    broad = tiny_belief.belief_block("Garchomp")
    narrowed = tiny_belief.belief_block("Garchomp", revealed_nature="Adamant")
    assert {s["nature"] for s in narrowed["spreads"]} == {"Adamant"}
    assert broad["expected_stats"] != narrowed["expected_stats"]


# ── log_to_transitions: open vs closed end-to-end ──────────────────────────

_OTS_LOG = "\n".join([
    "|player|p1|alice|101|",
    "|player|p2|bob|102|",
    "|gen|9",
    "|tier|[Gen 9 Champions] VGC 2026 Reg M-B",
    "|rated|Tournament battle",
    "|clearpoke",
    "|poke|p1|Blaziken, L50, F|",
    "|poke|p1|Metagross, L50|",
    "|poke|p1|Rotom-Wash, L50|",
    "|poke|p1|Kommo-o, L50, F|",
    "|poke|p2|Garchomp, L50, F|",
    "|poke|p2|Aerodactyl, L50, M|",
    "|poke|p2|Whimsicott, L50, F|",
    "|poke|p2|Kingambit, L50, F|",
    "|teampreview|4",
    f"|showteam|p1|{_PACKED_P1}",
    "|showteam|p2|Garchomp||LifeOrb|RoughSkin|Earthquake,StompingTantrum,Protect,DragonClaw|Jolly||F|||50|"
    "]Aerodactyl||FocusSash|Unnerve|RockSlide,Protect,Tailwind,WideGuard|Jolly||M|||50|"
    "]Whimsicott||CovertCloak|Prankster|Tailwind,Moonblast,Encore,Protect|Timid||F|||50|"
    "]Kingambit||BlackGlasses|Defiant|KowtowCleave,SuckerPunch,SwordsDance,Protect|Adamant||F|||50|",
    "|teamsize|p1|4",
    "|teamsize|p2|4",
    "|start",
    "|switch|p1a: Blaziken|Blaziken, L50, F|100/100",
    "|switch|p1b: Metagross|Metagross, L50|100/100",
    "|switch|p2a: Garchomp|Garchomp, L50, F|100/100",
    "|switch|p2b: Whimsicott|Whimsicott, L50, F|100/100",
    "|turn|1",
    "|move|p2b: Whimsicott|Tailwind|p2b: Whimsicott",
    "|-sidestart|p2: bob|move: Tailwind",
    "|move|p1a: Blaziken|Heat Wave|p2a: Garchomp",
    "|-damage|p2a: Garchomp|55/100",
    "|move|p2a: Garchomp|Earthquake|p1a: Blaziken",
    "|-damage|p1a: Blaziken|20/100",
    "|move|p1b: Metagross|Iron Head|p2b: Whimsicott",
    "|-damage|p2b: Whimsicott|30/100",
    "|turn|2",
    "|move|p1a: Blaziken|Heat Wave|p2a: Garchomp",
    "|-damage|p2a: Garchomp|0 fnt",
    "|faint|p2a: Garchomp",
    "|win|alice",
])


def _opp_active_of(transition, species):
    for mon in (transition["state_before_actions"].get("opp_active") or {}).values():
        if mon and mon.get("species") == species:
            return mon
    return None


def test_ots_open_stamps_opponent_knowledge():
    ts = log_to_transitions(_OTS_LOG, "test-ots-1", players=["p1"], source_type="B")
    assert ts, "no transitions parsed"
    t1 = ts[0]
    assert t1["ots"] is True
    assert t1["rated"] is True
    assert t1["bo3_set_id"] is None
    # p1's view of the OPPONENT at turn 1 carries the full revealed sheet
    chomp = _opp_active_of(t1, "Garchomp")
    assert chomp is not None
    assert chomp["known_item"] == "Life Orb"
    assert chomp["known_ability"] == "Rough Skin"
    assert chomp["nature"] == "Jolly"
    assert chomp["known_moves"] == ["Earthquake", "Stomping Tantrum", "Protect", "Dragon Claw"]
    # our own side gets its sheet too (nature was never log-visible before)
    our = list(t1["state_before_actions"]["our_active"].values())[0]
    assert our["nature"] in ("Timid", "Jolly")
    # EVs must NOT be pinned — OTS strips them
    assert chomp.get("ev_spread") is None
    assert not chomp.get("exact")


def test_ots_closed_mode_matches_ladder_regime():
    ts = log_to_transitions(_OTS_LOG, "test-ots-1c", players=["p1"],
                            source_type="B", ots=False)
    t1 = ts[0]
    assert t1["ots"] is False
    chomp = _opp_active_of(t1, "Garchomp")
    assert chomp is not None
    # nothing revealed at decision time of turn 1 in the closed regime
    assert chomp.get("known_item") is None
    assert chomp.get("known_ability") is None
    assert not chomp.get("known_moves")
    assert chomp.get("nature") is None


def test_ots_auto_detect_on_closed_log_is_noop():
    closed_log = "\n".join(
        ln for ln in _OTS_LOG.splitlines() if not ln.startswith("|showteam|")
    )
    ts = log_to_transitions(closed_log, "test-closed-log", players=["p1"])
    assert ts[0]["ots"] is False
    chomp = _opp_active_of(ts[0], "Garchomp")
    assert chomp.get("known_item") is None


def test_ots_required_raises_on_closed_log():
    closed_log = "\n".join(
        ln for ln in _OTS_LOG.splitlines() if not ln.startswith("|showteam|")
    )
    with pytest.raises(ValueError, match="showteam"):
        log_to_transitions(closed_log, "test-closed-log", players=["p1"], ots=True)


def test_ots_final_turn_win_signal_and_bench_stamped():
    ts = log_to_transitions(_OTS_LOG, "test-ots-2", players=["p1", "p2"])
    last_p1 = [t for t in ts if t["perspective"] == "p1"][-1]
    assert last_p1["reward"]["win"] == 1
    # unrevealed opp bench stubs (Aerodactyl/Kingambit never entered) still
    # carry the sheet — OTS knowledge exists from team preview on
    t1 = ts[0]
    bench = {m.get("species"): m for m in t1["state_before_actions"].get("opp_bench") or []}
    entered = {"Garchomp", "Whimsicott"}
    stubs = [m for sp, m in bench.items() if sp not in entered]
    assert stubs, "expected unentered opp roster stubs on the bench"
    assert any(m.get("known_item") for m in stubs)


def test_ots_sheet_moves_survive_struggle_reveal():
    """Scan 2026-07-02 (MED): a revealed Struggle / called-move artifact must not
    evict a real sheet move — the server |showteam| sheet is strictly authoritative."""
    from v_dance.parser.vod_parser.transitions import _inject_known_stats
    sheet = {"moves": ["Earthquake", "Stomping Tantrum", "Protect", "Dragon Claw"],
             "item": "Life Orb", "ability": "Rough Skin", "nature": "Jolly",
             "ev_spread": None}
    mon = {"species": "Garchomp", "revealed_moves": ["Struggle", "Earthquake"]}
    _inject_known_stats(mon, sheet, sheet_authoritative=True)
    assert mon["known_moves"] == ["Earthquake", "Stomping Tantrum", "Protect",
                                  "Dragon Claw"]
    # the user-typed (non-authoritative) path keeps its reveal-wins merge: typed
    # moves are unconfirmed against replay ground truth, so the reveal survives
    mon2 = {"species": "Garchomp", "revealed_moves": ["Struggle", "Earthquake"]}
    _inject_known_stats(mon2, dict(sheet), sheet_authoritative=False)
    assert "Struggle" in mon2["known_moves"]


def test_ots_stamp_norm_keyed_lookup_nicknamed_forme():
    """Scan 2026-07-02 (claims 2+3, CONFIRMED): a NICKNAMED mon's packed species
    field is packName'd ("ChienPao") and the CamelCase restore ("Chien Pao") is
    only norm-identical to the parser's display species ("Chien-Pao") — the old
    exact-string lookup silently skipped the sheet stamp for such mons."""
    from v_dance.parser.vod_parser.team_sheet import (
        parse_packed_team, packed_team_to_known_side)
    from v_dance.parser.vod_parser.transitions import _apply_ots_knowledge

    packed = ("Pao|ChienPao|HeavyDutyBoots|SwordofRuin|"
              "IcicleCrash,SuckerPunch,SacredSword,Protect|Hasty|||||50|")
    mons = parse_packed_team(packed)
    assert mons and mons[0]["nickname"] == "Pao"
    assert mons[0]["species"] == "Chien Pao"          # the lossy restore
    side = packed_team_to_known_side(mons)
    assert "Chien Pao" in side                        # display-string key

    mon = {"species": "Chien-Pao"}                    # parser display species
    battle = {"turns": [{
        "state_before_actions": {"p1": {
            "our_active": {"our_a": mon}, "our_bench": [],
            "opp_active": {}, "opp_bench": []}},
    }]}
    _apply_ots_knowledge(battle, {"p1": side})
    assert mon.get("known_moves") == ["Icicle Crash", "Sucker Punch",
                                      "Sacred Sword", "Protect"]
    assert mon.get("nature") == "Hasty"
    assert mon.get("known_ability")                   # sheet ability landed


def test_user_inject_cannot_clobber_sheet_authoritative_moves():
    """Scan 2026-07-02 (claim 5, interplay guard): a later per-perspective
    user-typed inject must not re-run the reveal-wins merge over a server-sheet
    stamp (the Struggle eviction would sneak back in through that path)."""
    from v_dance.parser.vod_parser.transitions import _inject_known_stats
    sheet_moves = ["Earthquake", "Stomping Tantrum", "Protect", "Dragon Claw"]
    mon = {"species": "Garchomp", "revealed_moves": ["Struggle"]}
    _inject_known_stats(mon, {"moves": sheet_moves}, sheet_authoritative=True)
    _inject_known_stats(mon, {"moves": ["Earthquake", "Scale Shot"]},
                        sheet_authoritative=False)
    assert mon["known_moves"] == sheet_moves          # the sheet stamp survives
    # a mon with no sheet stamp still takes the user-typed path unchanged
    mon2 = {"species": "Garchomp"}
    _inject_known_stats(mon2, {"moves": ["Earthquake"]}, sheet_authoritative=False)
    assert mon2["known_moves"] == ["Earthquake"]


def test_ots_itemless_mon_confirmed_not_phantom():
    """Scan 2026-07-02 (LOW): a sheet with NO item is a CONFIRMED itemless mon —
    without the sentinel the encoder's belief fallback hands it a phantom item."""
    from v_dance.parser.vod_parser.transitions import _inject_known_stats
    from v_dance.encoders.battle_mechanics import resolve_item_json
    mon = {"species": "Whimsicott",
           "belief": {"items": [{"name": "Focus Sash", "pct": 40.0}]}}
    _inject_known_stats(mon, {"moves": ["Tailwind"], "item": None, "nature": "Timid"},
                        sheet_authoritative=True)
    assert mon.get("sheet_itemless") is True
    assert resolve_item_json(mon) == ("", 1.0)   # confirmed itemless, no phantom
    # a sheet WITH an item must not stamp the sentinel
    mon2 = {"species": "Garchomp"}
    _inject_known_stats(mon2, {"item": "Life Orb"}, sheet_authoritative=True)
    assert not mon2.get("sheet_itemless")
    assert resolve_item_json(mon2) == ("lifeorb", 1.0)
    # the user-typed path never stamps it (no-item there just means "not typed")
    mon3 = {"species": "Whimsicott"}
    _inject_known_stats(mon3, {"moves": ["Tailwind"]}, sheet_authoritative=False)
    assert not mon3.get("sheet_itemless")


def test_replay_html_wrapper_still_works(vod_path):
    from v_dance.parser.vod_parser.transitions import replay_to_transitions
    ts = replay_to_transitions(vod_path, players=["p1"])
    assert ts
    # the ladder VOD has no |showteam| → closed regime, new keys present
    assert ts[0]["ots"] is False
    assert "bo3_set_id" in ts[0]


# ── split_with_reference (pinned-val data-expansion split) ─────────────────

def test_split_with_reference_pins_val_and_dedupes():
    from v_dance.training.bc_dataset import split_by_replay, split_with_reference
    ref = [{"replay_id": f"r{i}", "x": i} for i in range(10) for _ in range(3)]
    # extra corpus: 5 new replays + 2 that duplicate reference replays
    extra = ([{"replay_id": f"h{i}", "x": 100 + i} for i in range(5)]
             + [{"replay_id": "r0", "x": 999}, {"replay_id": "r9", "x": 998}])
    ref_train, ref_val = split_by_replay(ref, val_frac=0.2, seed=0)
    train, val = split_with_reference(extra, ref, val_frac=0.2, seed=0)
    # val is EXACTLY the reference split's val — extra data can't move it
    assert {e["replay_id"] for e in val} == {e["replay_id"] for e in ref_val}
    # duplicated replay_ids from extra are dropped entirely
    train_ids = {e["replay_id"] for e in train}
    assert not any(e["x"] in (999, 998) for e in train)
    # all genuinely-new replays land in train
    assert {f"h{i}" for i in range(5)} <= train_ids
    # reference train half is preserved
    assert {e["replay_id"] for e in ref_train} <= train_ids


# ── ingest driver: atomic per-battle writes (audit 2026-07-02) ──────────────

def test_ingest_process_one_atomic_write(tmp_path):
    from v_dance.datatools.ingest_hf_logs import _process_one
    rid = "gen9championsvgc2026regmb-99000001"
    rid2, n_tr, err = _process_one((rid, _OTS_LOG, "b", str(tmp_path), True, False))
    assert err is None and n_tr > 0 and rid2 == rid
    out = tmp_path / f"{rid}.jsonl"
    assert out.exists()
    # no .tmp remnants — the final name only ever appears via atomic os.replace
    assert list(tmp_path.glob("*.tmp")) == []
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_tr
    first = json.loads(lines[0])
    assert first["replay_id"] == rid and first["ots"] is True


def test_ingest_tp_only_writes_first_transitions(tmp_path):
    """--tp-only (2026-07-02): the team-preview slice keeps ONE transition per
    perspective, strips the post-action state, and stays TP-dataset-compatible."""
    from v_dance.datatools.ingest_hf_logs import _process_one
    rid = "gen9championsvgc2026regmb-99000002"
    rid2, n_tr, err = _process_one((rid, _OTS_LOG, "b", str(tmp_path), True, True))
    assert err is None and rid2 == rid
    lines = (tmp_path / f"{rid}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == n_tr <= 2                      # one per perspective
    ts = [json.loads(ln) for ln in lines]
    assert {t["perspective"] for t in ts} == {"p1", "p2"}
    for t in ts:
        assert t["turn"] == 1
        assert "state_after_actions" not in t           # bulk stripped
        assert "players" in t and t["ots"] is True      # TP labels intact
        assert "state_before_actions" in t              # sheet map source intact
    # the TP dataset reads the sliced file directly (this fixture's 4-mon teams
    # are below the 6-mon TEAM_SIZE, so the examples are skipped as incomplete —
    # the point here is that the tp-only shape parses cleanly end-to-end)
    from v_dance.training.teampreview_dataset import build_examples
    exs, stats = build_examples([str(tmp_path / f"{rid}.jsonl")])
    assert stats["files"] == 1 and stats["bad_files"] == 0
    assert stats["skipped_incomplete"] == 2 and len(exs) == 0


# ── era-retrain 2026-07-11: per-era belief OVERRIDE (spawn-safe via env) ─────────────────────────
def test_pika_path_era_override(monkeypatch):
    """--pikalytics-regmb must redirect ONLY the regmb belief (the era blend snapshot), leaving
    regma untouched; unset restores the live-file default. Env-based so worker processes inherit."""
    from v_dance.datatools.ingest_hf_logs import _pika_path_for_era
    assert _pika_path_for_era("b").name == "pikalytics_regmb.json"
    # forward slash: Path().name treats "\" as a separator only on Windows — this test runs on CI's Linux too
    monkeypatch.setenv("VD_PIKA_REGMB_OVERRIDE", "data/pikalytics_regmb_blend_2026-07-11.json")
    assert _pika_path_for_era("b").name == "pikalytics_regmb_blend_2026-07-11.json"
    assert _pika_path_for_era("a").name == "pikalytics_regma.json"
