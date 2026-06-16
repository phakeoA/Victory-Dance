"""Train/serve parity tests for state_encoder.

These feed ONE replay log to BOTH encode paths (live poke-env DoubleBattle vs
offline vod_parser) and diff the resulting 1398-dim vectors block by block.
See tests/_parity_harness.py for the driver and the scope notes (the OWN side
is not faithfully reproducible from a replay without |request| lines, so the
assertions target the publicly-derived OPPONENT side and globals).

Gap #1 — opponent bench [B2]: the live encode() must fill its 4 opp-bench
slots from the opponent's full teampreview roster (seen mons + unseen stubs),
ordered seen-alive → seen-fainted → unseen, exactly like encode_snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("poke_env")
import numpy as np

# tests/ is not a package; make the sibling harness importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _parity_harness as H  # noqa: E402

from live_state_encoder import LiveStateEncoder as StateEncoder  # noqa: E402
from vod_parser.replay_parser import extract_log_from_html  # noqa: E402

_VODS = Path(__file__).resolve().parents[2] / "vods"

# A clean replay (no Zoroark / Ditto / transform) — every turn's opponent side
# is unambiguous, so opp-bench parity must hold for all turns, both sides.
_CLEAN_NAME = "Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html"
# The dual-edge fixture: BOTH players run Zoroark-Hisui + Ditto.
_DUAL_NAME = "Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html"


def _find(name: str) -> Path:
    hit = next(_VODS.rglob(name), None)
    assert hit is not None, f"replay fixture not found: {name}"
    return hit


@pytest.fixture(scope="module")
def clean_log() -> str:
    return extract_log_from_html(_find(_CLEAN_NAME).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dual_log() -> str:
    return extract_log_from_html(_find(_DUAL_NAME).read_text(encoding="utf-8"))


# The dual fixture is recorded in REAL HP (kronomono = p2), so from p1's
# perspective the opponent's HP is in real units (X/Y, Y!=100) — the gap-#5
# HP-scale + mega edge case.  Preferred named target; falls back to a corpus
# scan so the test still runs if that exact file is absent.
_REAL_HP_NAME = _DUAL_NAME


def _real_hp_case():
    """(log, our_perspective) for a replay whose OPPONENT side uses real-HP units,
    or None.  our_perspective is the side that does NOT own the real-HP mons."""
    import re

    def real_hp_player(log: str):
        # which physical player's switch-ins carry a denominator != 100?
        for ln in log.split("\n"):
            if ln.startswith(("|switch|", "|drag|")):
                parts = ln.split("|")
                if len(parts) > 4 and parts[2][:2] in ("p1", "p2"):
                    m = re.match(r"\s*\d+\s*/\s*(\d+)", parts[4])
                    if m and m.group(1) != "100":
                        return parts[2][:2]
        return None

    hit = next(_VODS.rglob(_REAL_HP_NAME), None)
    candidates = [hit] if hit else []
    candidates += [v for v in _VODS.rglob("*.html") if v != hit]
    for v in candidates:
        if v is None:
            continue
        try:
            log = extract_log_from_html(v.read_text(encoding="utf-8"))
        except Exception:
            continue
        owner = real_hp_player(log)
        if owner:
            return log, ("p1" if owner == "p2" else "p2")
    return None


@pytest.fixture(scope="module")
def real_hp_case():
    case = _real_hp_case()
    if case is None:
        pytest.skip("no real-HP-scale replay found in the corpus")
    return case


# ── Gap #1: opponent bench roster reconstruction ─────────────────────────────
@pytest.mark.parametrize("perspective", ["p1", "p2"])
def test_opp_bench_parity_clean_replay(clean_log, perspective):
    """Live and offline must agree on the opponent-bench block (which mons fill
    the 4 slots + their identity/HP/flags) for every turn of a clean battle."""
    username = H.player_usernames(clean_log)[perspective]
    live = H.live_vectors_per_turn(clean_log, username)
    offline = H.offline_vectors_per_turn(clean_log, perspective)

    assert offline, "offline path produced no turns"
    for turn in sorted(offline):
        assert turn in live, f"turn {turn} missing from live path"
        bad = H.diff_blocks(live[turn], offline[turn], H.OPP_BENCH)
        assert not bad, (
            f"[{perspective}] turn {turn}: opp-bench mismatch at slots "
            f"{sorted(bad)}\n" +
            "\n".join(
                f"  slot {s} live={d['live']}\n  slot {s} off ={d['offline']}"
                for s, d in bad.items()
            )
        )


def test_opp_bench_populated_early(clean_log):
    """Regression guard for the gap: on turn 1 the opponent has only its two
    leads revealed, yet all 4 opp-bench slots must be filled (with unseen
    teampreview stubs).  Before the fix these slots were all-zero."""
    username = H.player_usernames(clean_log)["p1"]
    live = H.live_vectors_per_turn(clean_log, username)
    populated = [s for s in H.OPP_BENCH if not H.slot_is_empty(live[1], s)]
    assert populated == list(H.OPP_BENCH), (
        f"turn-1 opp bench not fully populated: {populated}"
    )


def test_opp_bench_unseen_stub_full_hp(clean_log):
    """Each unseen opp-bench stub carries the offline full-HP placeholder (1.0)
    and is flagged not-revealed — not HP 0 for a mon we simply haven't seen."""
    username = H.player_usernames(clean_log)["p1"]
    live = H.live_vectors_per_turn(clean_log, username)
    vec = live[1]
    for s in H.OPP_BENCH:
        b = H.slot_base(s)
        assert vec[b + H.OFF_REV] == 0.0, f"slot {s} unexpectedly revealed"
        assert vec[b + H.OFF_HP] == pytest.approx(1.0), (
            f"unseen stub slot {s} hp={vec[b + H.OFF_HP]} (want 1.0)"
        )


@pytest.mark.parametrize("perspective", ["p1", "p2"])
def test_opp_bench_parity_dualedge_all_turns(dual_log, perspective):
    """With the opponent-illusion bench fix (offline) + the broken-illusion
    phantom filter (live), opp-bench parity holds for EVERY turn of the
    dual-edge fixture (both players run Zoroark + Ditto) on both perspectives —
    including the turns where an opponent Zoroark is actively disguised and the
    turns after it has been revealed."""
    username = H.player_usernames(dual_log)[perspective]
    live = H.live_vectors_per_turn(dual_log, username)
    offline = H.offline_vectors_per_turn(dual_log, perspective)
    for turn in sorted(offline):
        if turn not in live:
            continue
        bad = H.diff_blocks(live[turn], offline[turn], H.OPP_BENCH)
        assert not bad, (
            f"[{perspective}] dual-edge turn {turn}: opp-bench mismatch at "
            f"slots {sorted(bad)}"
        )


# ── Gap #6 sub-cause 1: Ally Switch (|swap|) doubles slot tracking ───────────
# A replay where an opponent Ally-Switches its two active mons immediately
# before one of them faints and is replaced — the exact slot-pointer desync the
# |swap| parser fix targets.  poke-env applies the swap; the offline parser now
# does too, so opp active + bench stay aligned for the whole game.
_ALLY_SWITCH_NAME = "Gen9ChampionsVGC2026RegMA-2026-06-13-kronomono-orangeomar.html"


@pytest.fixture(scope="module")
def ally_switch_log() -> str:
    return extract_log_from_html(_find(_ALLY_SWITCH_NAME).read_text(encoding="utf-8"))


def test_ally_switch_opp_active_bench_parity(ally_switch_log):
    """Gap #6 sub-cause 1: an Ally Switch (|swap|) before a faint+replacement
    must keep BOTH opponent mons active, so live and offline agree on the opp
    active slots (2,3) AND the opp bench (8-11) for every turn.

    Pre-fix the offline parser ignored the |swap|, so the replacement |switch|
    overwrote the still-living ally — desyncing the opp composition and
    cascading into every per-mon feature of the affected slots."""
    persp = "p1"
    user = H.player_usernames(ally_switch_log)[persp]
    live = H.live_vectors_per_turn(ally_switch_log, user)
    offline = H.offline_vectors_per_turn(ally_switch_log, persp)
    assert offline, "offline path produced no turns"
    for turn in sorted(offline):
        if turn not in live:
            continue
        bad = H.diff_blocks(live[turn], offline[turn],
                            H.ACTIVE_OPP + H.OPP_BENCH)
        assert not bad, (
            f"turn {turn}: opp active/bench mismatch at slots {sorted(bad)}\n" +
            "\n".join(
                f"  slot {s} live={d['live']}\n  slot {s} off ={d['offline']}"
                for s, d in bad.items()
            )
        )


def test_ally_switch_both_opp_mons_active_after_replacement(ally_switch_log):
    """Direct guard: at the turn after Ally-Switch + faint + replacement the
    OFFLINE snapshot still has TWO opponent mons active (the replacement and the
    swapped-away ally), not one."""
    parser = __import__("vod_parser.replay_parser", fromlist=["ShowdownReplayParser"])
    res = parser.ShowdownReplayParser(ally_switch_log, our_player="p1").parse()
    t2 = next(t for t in res["turns"] if t["turn"] == 2)
    snap = t2["state_before_actions"]["p1"]
    assert len(snap["opp_active"]) == 2, (
        f"expected 2 opp active after replacement, got "
        f"{[d['species'] for d in snap['opp_active'].values()]}"
    )


# ── Gap #6 sub-cause 2: poke-env duplicate-species illusion MERGE ────────────
# poke-env keys opponent mons by species, so a Zoroark disguised as a brought
# teammate's species (two same-species mons on the field — Species Clause makes
# one provably the illusion) collapses into ONE Pokemon object; when the disguise
# switches out, poke-env's active list drops the REAL mon off the field too.  The
# fix (USER-chosen "port resolver to live encoder"): reconstruct the opponent
# side from the public log via the SAME vod_parser training uses, in REAL TIME
# (prefix-parse, no future lookahead), and splice it in — encode(battle,
# opp_snapshot=...).  These replays exercise that merge.

# minatopoke: opponent Zoroark disguised as a 2nd Arcanine (never |replace|-d;
#   resolved by Species Clause).  felixxd99: Zoroark disguised as a 2nd Farigiraf,
#   later revealed.  Both are FULLY clean (active + bench) on the spliced path.
_MERGE_CLEAN_NAMES = [
    "Gen9ChampionsVGC2026RegMA-2026-05-24-minatopoke-martiinusest.html",
    "Gen9ChampionsVGC2026RegMA-2026-05-15-felixxd99-lampistest.html",
]
# swirlingroses-taterstotz: Zoroark double-disguise (Excadrill then Tyranitar).
#   Active occupancy is fixed by the splice; its only residual is the bench
#   SEEN-flag of a re-disguised-after-reveal Zoroark (a train-vs-serve seam).
_MERGE_ACTIVE_ONLY_NAME = (
    "Gen9ChampionsVGC2026RegMA-2026-05-08-swirlingroses-taterstotz.html"
)


@pytest.mark.parametrize("name", _MERGE_CLEAN_NAMES)
@pytest.mark.parametrize("perspective", ["p1", "p2"])
def test_gap6_merge_replay_opp_parity_with_splice(name, perspective):
    """With the opponent splice, the live path agrees with offline on the FULL
    opponent side (active slots 2,3 + bench 8-11) for every turn of a
    duplicate-species illusion replay — the cases where poke-env's raw view
    merges the disguise with the real same-species mon."""
    log = extract_log_from_html(_find(name).read_text(encoding="utf-8"))
    username = H.player_usernames(log)[perspective]
    live = H.live_vectors_per_turn(log, username, splice=True)
    offline = H.offline_vectors_per_turn(log, perspective)
    assert offline, "offline path produced no turns"
    for turn in sorted(offline):
        if turn not in live:
            continue
        bad = H.diff_blocks(live[turn], offline[turn], H.ACTIVE_OPP + H.OPP_BENCH)
        assert not bad, (
            f"[{perspective}] {name} turn {turn}: spliced opp mismatch at slots "
            f"{sorted(bad)}"
        )


def test_gap6_merge_replay_raw_pokeenv_view_is_wrong_without_splice():
    """Guard the diagnosis: WITHOUT the splice, poke-env's raw opponent view
    diverges from offline (the merge bug); WITH the splice it does not.  Proves
    the splice is what closes the gap, not an unrelated change."""
    log = extract_log_from_html(_find(_MERGE_CLEAN_NAMES[0]).read_text(encoding="utf-8"))
    offline = H.offline_vectors_per_turn(log, "p1")
    username = H.player_usernames(log)["p1"]

    raw = H.live_vectors_per_turn(log, username, splice=False)
    raw_bad = sum(
        len(H.diff_blocks(raw[t], offline[t], H.ACTIVE_OPP + H.OPP_BENCH))
        for t in sorted(offline) if t in raw
    )
    spliced = H.live_vectors_per_turn(log, username, splice=True)
    spliced_bad = sum(
        len(H.diff_blocks(spliced[t], offline[t], H.ACTIVE_OPP + H.OPP_BENCH))
        for t in sorted(offline) if t in spliced
    )
    assert raw_bad > 0, "expected the raw poke-env view to diverge (merge bug)"
    assert spliced_bad == 0, f"splice should be clean, got {spliced_bad} diffs"


def test_gap6_active_occupancy_fixed_on_double_disguise():
    """Even on the hardest double-disguise replay (Zoroark as Excadrill then as
    Tyranitar), the splice fixes WHICH opponent mons are ACTIVE (slots 2,3) for
    every turn — the core gap #6 composition desync.  Compared on IDENTITY (type
    one-hots + base stats), so a same-mon disguise-HP seam doesn't mask the
    occupancy result."""
    log = extract_log_from_html(_find(_MERGE_ACTIVE_ONLY_NAME).read_text(encoding="utf-8"))
    for perspective in ("p1", "p2"):
        username = H.player_usernames(log)[perspective]
        live = H.live_vectors_per_turn(log, username, splice=True)
        offline = H.offline_vectors_per_turn(log, perspective)
        for turn in sorted(offline):
            if turn not in live:
                continue
            for s in H.ACTIVE_OPP:
                assert np.allclose(
                    H.identity_slot(live[turn], s),
                    H.identity_slot(offline[turn], s),
                    atol=1e-4,
                ), f"[{perspective}] turn {turn} slot {s}: active identity diverges"


def test_gap6_splice_no_regression_on_clean_replay(clean_log):
    """The opponent splice must not change a clean (no-illusion) battle: the
    spliced live opponent side equals offline for every turn, just as the raw
    poke-env view already did."""
    username = H.player_usernames(clean_log)["p1"]
    spliced = H.live_vectors_per_turn(clean_log, username, splice=True)
    offline = H.offline_vectors_per_turn(clean_log, "p1")
    for turn in sorted(offline):
        if turn not in spliced:
            continue
        bad = H.diff_blocks(spliced[turn], offline[turn], H.ACTIVE_OPP + H.OPP_BENCH)
        assert not bad, f"clean replay regressed at turn {turn}: slots {sorted(bad)}"


def test_gap6_opp_snapshot_prefix_reproduces_full_parse_on_clean_replay(clean_log):
    """opp_snapshot_from_log_prefix is the prefix machinery the live splice uses.
    On a clean (no-illusion) replay, the per-turn prefix snapshot must reproduce
    the full-parse's start-of-turn opponent state exactly — there is no
    future-only information to differ on."""
    from live_state_encoder import opp_snapshot_from_log_prefix
    from vod_parser.replay_parser import ShowdownReplayParser

    full = ShowdownReplayParser(clean_log, our_player="p1").parse()
    for turn_rec in full["turns"]:
        t = turn_rec["turn"]
        full_snap = turn_rec["state_before_actions"]["p1"]
        pre = opp_snapshot_from_log_prefix(clean_log, "p1", t)
        assert pre is not None, f"prefix snapshot missing for turn {t}"
        assert ([d["species"] for d in pre["opp_active"].values()]
                == [d["species"] for d in full_snap["opp_active"].values()]), (
            f"turn {t}: prefix opp_active != full opp_active")
        assert ([m["species"] for m in pre["opp_bench"]]
                == [m["species"] for m in full_snap["opp_bench"]]), (
            f"turn {t}: prefix opp_bench != full opp_bench")


def test_gap6_prefix_is_realtime_honest_keeps_revealed_zoroark_seen():
    """Production-honesty (positive case): in pever3ll a Zoroark is |replace|
    -revealed on turn 1, switches out, then re-disguises.  At turn 5 the live
    prefix snapshot must still mark that Zoroark as SEEN on the bench — a live
    bot does not forget it saw the reveal (the offline full-parse, with future
    lookahead, instead treats it as the secretly-active disguise; that train-
    vs-serve seam is documented for gap #6)."""
    from live_state_encoder import opp_snapshot_from_log_prefix

    name = "Gen9ChampionsVGC2026RegMA-2026-06-01-pever3ll-richardhavoc.html"
    log = extract_log_from_html(_find(name).read_text(encoding="utf-8"))
    pre = opp_snapshot_from_log_prefix(log, "p2", 5)   # opponent = p1
    assert pre is not None
    zoro = [m for m in pre["opp_bench"] if "Zoroark" in m["species"]]
    assert zoro and zoro[0].get("seen") is True, (
        f"real-time path lost the turn-1 Zoroark reveal: "
        f"{[(m['species'], m.get('seen')) for m in pre['opp_bench']]}"
    )


def test_gap6_encode_opp_snapshot_changes_only_opp_bytes():
    """encode(battle, opp_snapshot=...) splices ONLY the opponent byte ranges:
    own active/bench slots and the global block (weather/field/turn) are byte
    -identical to encode(battle) without a snapshot."""
    log = extract_log_from_html(_find(_MERGE_CLEAN_NAMES[0]).read_text(encoding="utf-8"))
    username = H.player_usernames(log)["p1"]
    from live_state_encoder import opp_snapshot_from_log_prefix

    enc = StateEncoder()
    battle = H.drive_live_battle_to_turn(log, username, 4)
    base = enc.encode(battle)
    snap = opp_snapshot_from_log_prefix(log, "p1", 4)
    spliced = enc.encode(battle, opp_snapshot=snap)

    # own active (0,1) + own bench (4-7) untouched
    for s in (*H.ACTIVE_OWN, *H.OWN_BENCH):
        b = H.slot_base(s)
        assert np.array_equal(base[b:b + H.POKEMON_FEATURES],
                              spliced[b:b + H.POKEMON_FEATURES]), f"own slot {s} changed"
    # opponent side DID change (the whole point)
    assert not np.array_equal(base, spliced), "splice had no effect"


# ── #7: live action codec / mask parity at serve time ───────────────────────
# The live serve codec (vgc_base.build_legal_action_mask / action_to_order) must
# agree with the frozen TRAINING codec (state_encoder.build_action_mask /
# action_to_index): move-target buckets keyed by DEX target kind, and the own
# bench filtered by _is_real_mon so switch index i ↔ encoder bench slot 4+i.
def _repo_root_on_path():
    import sys as _sys
    rr = Path(__file__).resolve().parents[3]   # repo root (holds vgc_base.py)
    if str(rr) not in _sys.path:
        _sys.path.insert(0, str(rr))


@pytest.mark.parametrize("name", [_CLEAN_NAME, _DUAL_NAME])
def test_live_mask_target_buckets_match_training_kinds(name):
    """build_legal_action_mask must expose move-target buckets by DEX target
    kind exactly like training's build_action_mask — spread/self/field → bucket
    0 only, ally → bucket 2, single → present foes (+ ally) — and every legal
    action must decode to a non-None order.  (Pre-fix the live mask marked ALL
    present-target buckets for EVERY move, far broader than training.)"""
    _repo_root_on_path()
    from vgc_base import build_legal_action_mask, action_to_order
    from state_encoder import (
        _move_target_kind, _CHOOSABLE_SINGLE, _ALLY_KINDS, ACTIONS_PER_SLOT,
    )
    log = extract_log_from_html(_find(name).read_text(encoding="utf-8"))
    checked = 0
    for persp in ("p1", "p2"):
        user = H.player_usernames(log)[persp]
        for turn in range(1, 12):
            battle = H.drive_live_battle_to_turn(log, user, turn)
            try:
                active = battle.active_pokemon
            except ValueError:
                continue
            for slot, mon in enumerate(active):
                if mon is None or mon.fainted:
                    continue
                mask = build_legal_action_mask(battle, slot)
                checked += 1
                ally = active[1 - slot] if (1 - slot) < len(active) else None
                ally_alive = ally is not None and not ally.fainted
                for m_idx, mv in enumerate(list(mon.moves.values())[:4]):
                    if mv.current_pp == 0:
                        continue
                    kind = _move_target_kind(mv.id)
                    buckets = {t for t in range(3) if mask[m_idx * 3 + t]}
                    if kind in _ALLY_KINDS:
                        exp = {2} if (kind == "adjacentAllyOrSelf" or ally_alive) else set()
                        assert buckets == exp, f"{mv.id} ({kind}) buckets {buckets} != {exp}"
                    elif kind not in _CHOOSABLE_SINGLE:
                        assert buckets == {0}, f"{mv.id} ({kind}) buckets {buckets} != {{0}}"
                    # single-target buckets are presence-dependent → decodability below
                for a in range(ACTIONS_PER_SLOT):
                    if mask[a]:
                        assert action_to_order(a, battle, slot) is not None, \
                            f"legal action {a} (slot {slot}) failed to decode"
    assert checked > 0, "no decisions exercised"


def test_live_codec_bench_excludes_illusion_phantom(clean_log):
    """The live codec's switch bench must (a) only offer mons Showdown will accept
    as a switch — i.e. the BROUGHT 4-of-6 in ``available_switches``, NOT every mon
    in ``battle.team`` (VGC leaves 2 roster mons un-brought) — and (b) drop a
    broken-illusion phantom (max_hp==0) via _is_real_mon, so switch index i refers
    to the SAME mon the live encoder wrote into bench slot 4+i.

    A request-less driven battle exposes no available_switches, so we synthesise a
    bench mon and toggle its membership of available_switches to exercise both the
    brought filter (the un-brought-switch bug fix) and the phantom filter."""
    _repo_root_on_path()
    from vgc_base import build_legal_action_mask
    from state_encoder import SWITCH_OFFSET
    from poke_env.battle import Pokemon

    user = H.player_usernames(clean_log)["p1"]
    battle = H.drive_live_battle_to_turn(clean_log, user, 2)

    def switch_count():
        return sum(build_legal_action_mask(battle, 0)[SWITCH_OFFSET:SWITCH_OFFSET + 4])

    base = switch_count()

    ghost = Pokemon(gen=9, species="snorlax")
    ghost.set_hp_status("100/100")
    battle.team["p1: PhantomTest"] = ghost

    # In battle.team but NOT in available_switches (an un-brought roster mon) →
    # must NOT be switchable.  This is the regression guard for the un-brought
    # switch bug: before the fix, any battle.team member was offered as a switch
    # and Showdown rejected it ("you do not have a Pokémon named X to switch to").
    assert switch_count() == base, "un-brought team mon wrongly offered as a switch"

    # Now mark it switchable (brought) for slot 0 → it must become switchable.
    battle._available_switches = [[ghost], []]
    assert switch_count() == base + 1, "brought, switchable bench mon not offered"

    # Broken-illusion phantom: switchable per poke-env but revealed disguise left
    # with no HP → _is_real_mon must still drop it out.
    ghost._max_hp = 0
    assert ghost.max_hp == 0
    assert switch_count() == base, "phantom (max_hp 0) not filtered from switch mask"


def test_live_mask_excludes_disabled_moves(clean_log):
    """Serve-time move legality: a Showdown-disabled move (one absent from
    battle.available_moves[slot] — Fake Out after turn 1, consecutive Protect,
    Choice-lock, Taunt, Encore, …) must be dropped from the legal-action mask AND
    refused by the codec, mirroring the available_switches gate.

    Before the fix the move mask only checked current_pp, so a disabled move was
    marked legal, the model argmaxed it, and Showdown rejected the order
    ('[Unavailable choice] Can't move: X is disabled') → a random fallback ran
    instead of the model — a major source of the bot "seeming random"."""
    _repo_root_on_path()
    from vgc_base import build_legal_action_mask, action_to_order
    from state_encoder import NUM_MOVES

    user = H.player_usernames(clean_log)["p1"]
    # Drive a few turns so the lead has revealed >=2 moves to work with.
    battle = None
    moves: list = []
    for tn in (4, 3, 5, 2):
        battle = H.drive_live_battle_to_turn(clean_log, user, tn)
        mon0 = battle.active_pokemon[0]
        if mon0 is not None:
            moves = list(mon0.moves.values())[:NUM_MOVES]
            if len(moves) >= 2 and moves[0].id != moves[1].id:
                break
    if not moves or len(moves) < 2 or moves[0].id == moves[1].id:
        pytest.skip("replay did not reveal >=2 distinct moves on the lead")

    # All moves available → move slot 0 exposes at least one legal target bucket.
    battle._available_moves = [list(moves), list(moves)]
    full = build_legal_action_mask(battle, 0)
    assert any(full[0:3]), "move slot 0 should be legal when all moves are available"

    # DISABLE move slot 0 (drop it from poke-env's available_moves[slot]).
    battle._available_moves = [moves[1:], moves[1:]]
    gated = build_legal_action_mask(battle, 0)
    assert not any(gated[0:3]), "disabled move (slot 0) not dropped from the mask"
    assert any(gated[3:6]), "a still-available move (slot 1) must stay legal"
    # The codec must also refuse the disabled move so a stale index can't be ordered.
    assert action_to_order(0, battle, 0) is None, "codec ordered a disabled move"
    # …and still decode an available one.
    assert action_to_order(3, battle, 0) is not None, "codec dropped an available move"


def test_live_mask_forbids_ally_target_for_damaging_moves():
    """A DAMAGING single-target move (Acrobatics, Iron Head, …) must NOT be allowed
    to target our own ally — that self-attack is the model mis-picking the ally
    (e.g. two Kingambits on the field, one ours one the opponent's).  A NON-damaging
    single-target support move (Heal Pulse) MUST still be allowed at the ally, and
    dedicated ally moves (Helping Hand/Coaching = adjacentAlly) are unaffected.

    Pure mask logic over a duck-typed board + real poke-env Move objects."""
    _repo_root_on_path()
    from poke_env.battle import Move
    from vgc_base import (
        build_legal_action_mask, action_to_order, _TARGET_OPP0, _TARGET_ALLY,
    )

    acro = Move("acrobatics", gen=9)   # 'any' target, base_power > 0  (damaging)
    heal = Move("healpulse", gen=9)    # 'any' target, base_power == 0 (support)
    assert acro.base_power > 0 and heal.base_power == 0

    class _Mon:
        def __init__(self, moves, fainted=False):
            self._mv = moves
            self.fainted = fainted
        @property
        def moves(self):
            return self._mv

    actor = _Mon({"acrobatics": acro, "healpulse": heal})
    ally  = _Mon({"acrobatics": acro})
    foe   = _Mon({"acrobatics": acro})

    class _Battle:
        active_pokemon = [actor, ally]
        opponent_active_pokemon = [foe, None]      # one foe present (opp_a)
        available_moves = [list(actor.moves.values()), []]
        available_switches = [[], []]
        team = {}
        def to_showdown_target(self, move, target_mon):   # stub for the codec
            return 1

    battle = _Battle()
    mask = build_legal_action_mask(battle, 0)

    # move slot 0 = Acrobatics (damaging): foe legal, ALLY FORBIDDEN.
    assert mask[0 * 3 + _TARGET_OPP0] is True, "damaging move can't hit the present foe"
    assert mask[0 * 3 + _TARGET_ALLY] is False, "damaging move wrongly allowed at the ally"
    # move slot 1 = Heal Pulse (support): ally ALLOWED.
    assert mask[1 * 3 + _TARGET_ALLY] is True, "support move wrongly blocked from the ally"

    # Codec mirrors the mask: it refuses to order Acrobatics at the ally (action 2),
    # but allows it at the foe (action 0).
    assert action_to_order(0 * 3 + _TARGET_ALLY, battle, 0) is None
    assert action_to_order(0 * 3 + _TARGET_OPP0, battle, 0) is not None


def test_fresh_legal_explores_untried_actions(clean_log):
    """#13 retry-escape: when a deterministic policy's order is rejected as
    '[Unavailable]', _fresh_legal returns a LEGAL action not already tried this
    turn (so the player explores instead of looping), and still returns a legal
    action once every option has been tried."""
    _repo_root_on_path()
    import sys as _sys
    lb = str(Path(__file__).resolve().parents[3] / "local_battle")
    if lb not in _sys.path:
        _sys.path.insert(0, lb)
    from live_vgc_base import SplicingVGCPlayerBase as S
    from vgc_base import build_legal_action_mask

    user = H.player_usernames(clean_log)["p1"]
    # Find a turn where slot 0 has >= 2 legal actions (enough revealed moves).
    battle = legal = None
    for turn in range(1, 16):
        b = H.drive_live_battle_to_turn(clean_log, user, turn)
        try:
            lg = {i for i, ok in enumerate(build_legal_action_mask(b, 0)) if ok}
        except Exception:
            continue
        if len(lg) >= 2:
            battle, legal = b, lg
            break
    if battle is None:
        pytest.skip("no turn with >= 2 legal slot-0 actions in this replay")

    one = next(iter(legal))
    for _ in range(25):
        a = S._fresh_legal(battle, 0, {one}, fallback=-1)
        assert a in legal and a != one, "fresh action must be legal and not the tried one"

    # All legal actions exhausted → still returns a legal action (give-up path).
    a = S._fresh_legal(battle, 0, set(legal), fallback=-1)
    assert a in legal


def test_is_seen_filters_broken_illusion_phantom():
    """When a Zoroark illusion breaks, poke-env leaves the DISGUISE mon in
    opponent_team flagged revealed=True but with no HP (max_hp 0).  _is_seen
    must reject that phantom so it never counts as a seen bench mon — keeping
    the live view aligned with the offline parser (which relabels cleanly)."""
    from poke_env.battle import Pokemon

    real = Pokemon(gen=9, species="charizard")
    real.set_hp_status("100/100")        # genuinely seen → has HP
    real._revealed = True
    assert StateEncoder._is_seen(real) is True

    phantom = Pokemon(gen=9, species="charizard")
    phantom._revealed = True              # poke-env marks the disguise revealed…
    # …but was_illusioned cleared its HP — max_hp is 0
    assert phantom.max_hp == 0
    assert StateEncoder._is_seen(phantom) is False

    unseen = Pokemon(gen=9, species="charizard")   # teampreview stub
    assert StateEncoder._is_seen(unseen) is False   # not revealed


# ── Gap #2: Ditto / transform — is_transformed flag + copied-forme stats ──────
def test_transform_flag_micro():
    """Direct guard against the attribute-name bug: poke-env exposes the
    transform state as `mon.transformed`, NOT `mon.is_transformed`, so the live
    encoder must read the former.  A transformed Ditto must (a) set the
    is_transformed flag and (b) encode the COPIED forme's types + base stats."""
    from poke_env.battle import Pokemon

    enc = StateEncoder()
    ditto = Pokemon(gen=9, species="ditto")
    garchomp = Pokemon(gen=9, species="garchomp")
    ditto.transform(garchomp)
    assert ditto.transformed is True

    vec_ditto = np.zeros(H.POKEMON_FEATURES, dtype=np.float32)
    vec_chomp = np.zeros(H.POKEMON_FEATURES, dtype=np.float32)
    enc._write_pokemon(vec_ditto, 0, ditto, is_active=True, is_own=False)
    enc._write_pokemon(vec_chomp, 0, garchomp, is_active=True, is_own=False)

    assert vec_ditto[H.OFF_TRANS] == 1.0           # flag set
    assert vec_chomp[H.OFF_TRANS] == 0.0           # untransformed Garchomp
    # copied forme: types + base stats equal a real Garchomp's
    id_ditto = vec_ditto[H.OFF_TYPE1:H.OFF_BASE + 6]
    id_chomp = vec_chomp[H.OFF_TYPE1:H.OFF_BASE + 6]
    assert np.allclose(id_ditto, id_chomp)
    assert vec_ditto[H.OFF_BASE:H.OFF_BASE + 6].sum() > 0   # Garchomp dex present


def test_transform_parity_dualedge(dual_log):
    """On the dual-edge fixture (p1's Ditto runs Imposter → copies p2's lead),
    every opp-active slot the OFFLINE path marks transformed must also be marked
    transformed by the LIVE path, with matching structural features (the copied
    forme's typing + base stats)."""
    username = H.player_usernames(dual_log)["p2"]
    live = H.live_vectors_per_turn(dual_log, username)
    offline = H.offline_vectors_per_turn(dual_log, "p2")

    transformed_seen = 0
    for turn in sorted(offline):
        for slot in H.ACTIVE_OPP:               # opp_a / opp_b
            b = H.slot_base(slot)
            if offline[turn][b + H.OFF_TRANS] != 1.0:
                continue
            transformed_seen += 1
            assert live[turn][b + H.OFF_TRANS] == 1.0, (
                f"turn {turn} slot {slot}: live is_transformed not set"
            )
            # structural (hp + types + base stats + flags) must match the copy
            bad = H.diff_blocks(live[turn], offline[turn], (slot,))
            assert not bad, (
                f"turn {turn} slot {slot}: transformed-mon structural mismatch"
            )
    assert transformed_seen > 0, "fixture exercised no transform — test inert"


# ── Gap #3: PP fraction — live must pin 1.0 to match the PP-less offline path ──
def test_pp_fraction_pinned_to_one_micro():
    """Direct guard for the gap: even a move with heavily depleted PP must
    encode pp_fraction == 1.0 on the LIVE path, because the offline parser does
    not track PP (encode_snapshot always emits 1.0).  Emitting the real
    current_pp/max_pp would be a train/serve mismatch on a feature the net only
    ever saw as 1.0 during training."""
    from poke_env.battle import Move, Pokemon

    enc = StateEncoder()
    move = Move("thunderbolt", gen=9)
    for _ in range(5):                  # burn 5 of 24 PP → real frac ≈ 0.79
        move.use()
    assert move.current_pp < move.max_pp, "move PP not actually depleted"

    user = Pokemon(gen=9, species="pikachu")
    vec = np.zeros(H.MOVE_FEATURES, dtype=np.float32)
    enc._write_move(vec, 0, move, user)

    assert vec[H.OFF_MOVE_PP] == pytest.approx(1.0), (
        f"live pp_fraction={vec[H.OFF_MOVE_PP]} (want 1.0 to match offline); "
        f"real depleted frac would be {move.current_pp / move.max_pp:.3f}"
    )


@pytest.mark.parametrize("perspective", ["p1", "p2"])
def test_pp_fraction_one_on_real_replay(clean_log, perspective):
    """Across every turn/slot of a real replay (where opponent moves have been
    used multiple times so a buggy path would emit < 1.0), every POPULATED move
    block must carry pp_fraction == 1.0 on BOTH paths — full gap-#3 parity."""
    username = H.player_usernames(clean_log)[perspective]
    live = H.live_vectors_per_turn(clean_log, username)
    offline = H.offline_vectors_per_turn(clean_log, perspective)

    all_slots = (*H.ACTIVE_OWN, *H.ACTIVE_OPP, *H.OWN_BENCH, *H.OPP_BENCH)
    live_checked = off_checked = 0
    for turn in sorted(offline):
        if turn not in live:
            continue
        for slot in all_slots:
            pp_offs = H.move_pp_offsets(slot)
            kn_offs = H.move_known_offsets(slot)
            for pp, kn in zip(pp_offs, kn_offs):
                if live[turn][kn] != 0.0:        # live wrote a move here
                    live_checked += 1
                    assert live[turn][pp] == pytest.approx(1.0), (
                        f"[{perspective}] turn {turn} slot {slot}: live "
                        f"pp_fraction={live[turn][pp]} (want 1.0)"
                    )
                if offline[turn][kn] != 0.0:     # offline wrote a move here
                    off_checked += 1
                    assert offline[turn][pp] == pytest.approx(1.0), (
                        f"[{perspective}] turn {turn} slot {slot}: offline "
                        f"pp_fraction={offline[turn][pp]} (want 1.0)"
                    )
    assert live_checked > 0, "no live move blocks exercised — test inert"
    assert off_checked > 0, "no offline move blocks exercised — test inert"


# ── Gap #4: team-count globals + per-mon is_fainted audit ─────────────────────
def test_is_real_mon_filters_own_illusion_phantom():
    """Micro guard for the gap-#4 fix: _is_real_mon must reject a broken-illusion
    phantom (max_hp==0 disguise poke-env leaves in our team) while KEEPING every
    genuine own mon — both already-seen and brought-but-unentered (revealed=False
    but carrying real HP from the private |request|).  Using _is_seen here would
    wrongly drop the brought-but-unentered mons, so the own side needs its own
    predicate."""
    from poke_env.battle import Pokemon

    phantom = Pokemon(gen=9, species="charizard")
    phantom._revealed = True              # poke-env marks the disguise revealed…
    assert phantom.max_hp == 0            # …but it never had HP — not a real mon
    assert StateEncoder._is_real_mon(phantom) is False

    seen = Pokemon(gen=9, species="charizard")
    seen.set_hp_status("100/100")
    seen._revealed = True
    assert StateEncoder._is_real_mon(seen) is True

    brought = Pokemon(gen=9, species="garchomp")   # brought-but-unentered
    brought.set_hp_status("183/183")               # real HP from the request…
    assert brought.revealed is False               # …not yet sent out
    assert StateEncoder._is_real_mon(brought) is True


@pytest.mark.parametrize("perspective", ["p1", "p2"])
def test_team_count_globals_parity_clean(clean_log, perspective):
    """All 4 team-count globals (own_live / opp_live / own_fnt / opp_fnt) must
    agree live-vs-offline on every turn of the clean replay."""
    username = H.player_usernames(clean_log)[perspective]
    live = H.live_vectors_per_turn(clean_log, username)
    offline = H.offline_vectors_per_turn(clean_log, perspective)
    assert offline, "offline path produced no turns"
    for turn in sorted(offline):
        if turn not in live:
            continue
        for name in H.TEAM_COUNT_GLOBALS:
            lv, ov = H.team_count(live[turn], name), H.team_count(offline[turn], name)
            assert lv == ov, (
                f"[{perspective}] turn {turn}: {name} live={lv} offline={ov}"
            )


@pytest.mark.parametrize("perspective", ["p1", "p2"])
def test_team_count_globals_parity_dualedge(dual_log, perspective):
    """The same parity must hold on the dual-edge fixture (both players run
    Zoroark + Ditto).  Without the fix the LIVE own_live is inflated by one on
    the turns right after an own Zoroark's illusion drops (poke-env keeps the
    disguise species in battle.team as a max_hp==0 phantom)."""
    username = H.player_usernames(dual_log)[perspective]
    live = H.live_vectors_per_turn(dual_log, username)
    offline = H.offline_vectors_per_turn(dual_log, perspective)
    for turn in sorted(offline):
        if turn not in live:
            continue
        for name in H.TEAM_COUNT_GLOBALS:
            lv, ov = H.team_count(live[turn], name), H.team_count(offline[turn], name)
            assert lv == ov, (
                f"[{perspective}] dual turn {turn}: {name} live={lv} offline={ov}"
            )


def test_own_illusion_phantom_not_counted_or_slotted(dual_log):
    """Targeted regression for the exact failing case: from p2's perspective the
    own Zoroark-Hisui leads disguised; when it is revealed poke-env leaves a
    phantom 'Charizard' (max_hp==0) in battle.team.  After the fix that phantom
    must neither inflate own_live NOR occupy an own-bench slot — matching the
    offline parser, which resolves the illusion cleanly."""
    username = H.player_usernames(dual_log)["p2"]
    live = H.live_vectors_per_turn(dual_log, username)
    offline = H.offline_vectors_per_turn(dual_log, "p2")

    checked = 0
    for turn in (2, 3):
        assert turn in live and turn in offline, f"turn {turn} missing"
        checked += 1
        assert H.team_count(live[turn], "own_live") == H.team_count(offline[turn], "own_live")
        # own-bench slot occupancy must match (no phantom-filled slot)
        for slot in H.OWN_BENCH:
            assert H.slot_is_empty(live[turn], slot) == H.slot_is_empty(offline[turn], slot), (
                f"turn {turn} own-bench slot {slot} occupancy differs "
                f"(live_empty={H.slot_is_empty(live[turn], slot)})"
            )
    assert checked == 2, "fixture did not expose the phantom turns — test inert"


@pytest.mark.parametrize("perspective", ["p1", "p2"])
def test_is_fainted_flag_parity_all_slots(clean_log, dual_log, perspective):
    """The per-mon is_fainted flag (offset 108) must agree live-vs-offline for
    EVERY one of the 12 slots, on every turn of BOTH fixtures.  Guards both the
    flag value and the upstream slot-occupancy (a fainted mon present on one path
    only would surface here)."""
    fainted_seen = 0
    for log in (clean_log, dual_log):
        username = H.player_usernames(log)[perspective]
        live = H.live_vectors_per_turn(log, username)
        offline = H.offline_vectors_per_turn(log, perspective)
        for turn in sorted(offline):
            if turn not in live:
                continue
            for slot in H.ALL_SLOTS:
                off = H.slot_base(slot) + H.OFF_FNT
                lv, ov = live[turn][off], offline[turn][off]
                assert lv == ov, (
                    f"[{perspective}] turn {turn} slot {slot}: is_fainted "
                    f"live={lv} offline={ov}"
                )
                fainted_seen += int(ov == 1.0)
    assert fainted_seen > 0, "no fainted mon encountered — test inert"


# ── Gap #5: HP scale + mega-forme parity ──────────────────────────────────────
def test_hp_scale_opp_parity_real_hp_replay(real_hp_case):
    """When the opponent recorded the replay its HP is in REAL units (175/200);
    the offline parser must normalise to a true fraction so the opp-active/bench
    HP matches poke-env's current_hp_fraction (gap #5).  Before the fix the
    offline path stored the bare numerator → clamped to 1.0."""
    log, perspective = real_hp_case
    username = H.player_usernames(log)[perspective]
    live = H.live_vectors_per_turn(log, username)
    offline = H.offline_vectors_per_turn(log, perspective)
    checked = 0
    for turn in sorted(offline):
        if turn not in live:
            continue
        for slot in (*H.ACTIVE_OPP, *H.OPP_BENCH):
            b = H.slot_base(slot)
            if H.slot_is_empty(live[turn], slot) or H.slot_is_empty(offline[turn], slot):
                continue
            lv, ov = live[turn][b + H.OFF_HP], offline[turn][b + H.OFF_HP]
            assert lv == pytest.approx(ov, abs=2e-2), (
                f"[{perspective}] turn {turn} slot {slot}: opp HP live={lv:.3f} "
                f"offline={ov:.3f}"
            )
            checked += 1
    assert checked > 0, "no opp HP compared — test inert"


def test_mega_opp_parity_real_hp_replay(real_hp_case):
    """The opponent's per-mon mega flag (offset 54) must match live-vs-offline on
    every turn — exercises a real mega-evolution (gap #5)."""
    log, perspective = real_hp_case
    username = H.player_usernames(log)[perspective]
    live = H.live_vectors_per_turn(log, username)
    offline = H.offline_vectors_per_turn(log, perspective)
    mega_turns = 0
    for turn in sorted(offline):
        if turn not in live:
            continue
        for slot in (*H.ACTIVE_OPP, *H.OPP_BENCH):
            b = H.slot_base(slot)
            lv, ov = live[turn][b + H.OFF_MEGA], offline[turn][b + H.OFF_MEGA]
            assert lv == ov, (
                f"[{perspective}] turn {turn} slot {slot}: mega live={lv} offline={ov}"
            )
            mega_turns += int(ov == 1.0)
    assert mega_turns > 0, "fixture exercised no opponent mega — test inert"


def test_is_mega_forme_no_substring_false_positive():
    """The forme-aware mega detector must NOT flag a non-mega whose species name
    merely contains 'mega' (Meganium, Yanmega) — the bug in the old
    `'mega' in mon.species` check (gap #5).  An untouched base forme is never
    mega; a mega forme's base species alone (pre-evolution) is not yet mega."""
    from poke_env.battle import Pokemon

    enc = StateEncoder()
    for sp in ("meganium", "yanmega", "charizard", "aerodactyl", "scizor"):
        mon = Pokemon(gen=9, species=sp)
        assert enc._is_mega_forme(mon) is False, f"{sp} wrongly flagged mega"


def test_is_mega_forme_detects_real_mega():
    """A mon whose current base stats match a mega-forme dex entry IS flagged —
    poke-env keeps species at the base but rewrites base_stats on mega, so the
    detector keys off the stats (gap #5)."""
    from poke_env.battle import Pokemon

    enc = StateEncoder()
    chari = Pokemon(gen=9, species="charizard")
    assert enc._is_mega_forme(chari) is False        # base forme: not mega
    chari.mega_evolve("charizarditey")               # → Charizard-Mega-Y stats
    assert chari.species == "charizard"              # poke-env keeps base species
    assert enc._is_mega_forme(chari) is True         # …but the detector catches it


@pytest.mark.parametrize("perspective", ["p1", "p2"])
def test_side_conditions_binary_parity(dual_log, perspective):
    """Side conditions are encoded as a BINARY presence bit (gap #5): the two
    global side-condition blocks must match live-vs-offline every turn, and every
    value must be 0.0 or 1.0 (no turns-remaining / activation-turn magnitude)."""
    username = H.player_usernames(dual_log)[perspective]
    live = H.live_vectors_per_turn(dual_log, username)
    offline = H.offline_vectors_per_turn(dual_log, perspective)
    active_seen = 0
    for turn in sorted(offline):
        if turn not in live:
            continue
        for lo, hi in (H.OUR_SIDE_SC, H.OPP_SIDE_SC):
            lv, ov = live[turn][lo:hi], offline[turn][lo:hi]
            assert set(np.unique(lv)) <= {0.0, 1.0}, f"live SC not binary: {lv}"
            assert set(np.unique(ov)) <= {0.0, 1.0}, f"offline SC not binary: {ov}"
            assert np.array_equal(lv, ov), (
                f"[{perspective}] turn {turn}: side-condition block differs\n"
                f"  live={list(lv)}\n  off ={list(ov)}"
            )
            active_seen += int(ov.sum() > 0)
    assert active_seen > 0, "fixture exercised no side condition — test inert"
