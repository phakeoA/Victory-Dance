"""Regression tests for the live serve-path fixes (2026-06-15).

A) Own bench / brought-team helpers must EXCLUDE the un-brought 2-of-6 roster
   mons.  VGC team-preview brings 4 of 6, but poke-env keeps all 6 in
   ``battle.team``; offering an un-brought mon as a switch made Showdown reject
   EVERY switch ("you do not have a Pokémon named X to switch to") → the live
   player thrashed through retry/random fallbacks → "the model seems random".
   The encoder bench, the team-count globals, the legal-action mask and the
   action codec must all agree on the brought-only bench (switch index i ↔
   encoder bench slot 4+i).

B) The serve path must belief-enrich the reconstructed OPPONENT snapshot so the
   gap-#6 splice reproduces the (belief-enriched) TRAINING opponent bytes —
   otherwise opp est-stats + predicted move slots sit at zero at serve time, an
   out-of-distribution input the BC net was never trained on.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

# data/scripts (parent of tests/) on path — match the sibling tests' convention.

from v_dance.encoders.live_state_encoder import (  # noqa: E402
    own_bench_mons, brought_team_mons, switchable_union, LiveStateEncoder,
)
from v_dance.encoders.state_encoder import (  # noqa: E402
    VodStateEncoder, POKEMON_FEATURES, ACTIVE_SLOTS, BENCH_SLOTS, OPP_BENCH_SLOTS,
    move_slots_for_mon,
)
from v_dance.parser.belief_state import BeliefState, _DEFAULT_PIKALYTICS_PATH  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# A) Brought-team / own-bench helpers  (pure logic — fakes, no poke-env needed)
# The helpers read only mon.species/.max_hp/.fainted/.revealed and the battle's
# .team / .active_pokemon / .available_switches.
# ══════════════════════════════════════════════════════════════════════════════
class _Mon:
    def __init__(self, species, max_hp=100, fainted=False, revealed=False):
        self.species, self.max_hp = species, max_hp
        self.fainted, self.revealed = fainted, revealed

    def __repr__(self):
        return f"<{self.species}>"


class _Battle:
    def __init__(self, team, active, available_switches):
        self.team = {f"p1: {m.species}": m for m in team}
        self._active = active
        self.available_switches = available_switches

    @property
    def active_pokemon(self):
        return self._active


def _vgc_board():
    """A VGC board: 2 active + 2 brought bench + 2 UN-brought roster mons,
    in an arbitrary team order (poke-env keeps all 6 in battle.team)."""
    a = _Mon("sneasler", revealed=True)
    b = _Mon("talonflame", revealed=True)
    bench1 = _Mon("kingambit")          # brought, on the bench (not yet entered)
    bench2 = _Mon("sylveon")            # brought, on the bench
    un1 = _Mon("charizard")             # un-brought — left behind by teampreview
    un2 = _Mon("rotomwash")             # un-brought
    team = [bench1, a, un1, bench2, un2, b]
    # Showdown lists ONLY the brought, non-active, non-fainted mons as switchable.
    avail = [[bench1, bench2], [bench1, bench2]]
    return _Battle(team, [a, b], avail), (a, b, bench1, bench2, un1, un2)


def test_own_bench_excludes_unbrought():
    battle, (a, b, bench1, bench2, un1, un2) = _vgc_board()
    bench = own_bench_mons(battle)
    assert bench == [bench1, bench2], f"bench should be the 2 brought mons, got {bench}"
    assert un1 not in bench and un2 not in bench, "un-brought mon leaked into the bench"


def test_brought_team_excludes_unbrought_and_team_counts_not_inflated():
    battle, (a, b, bench1, bench2, un1, un2) = _vgc_board()
    brought = brought_team_mons(battle)
    assert set(brought) == {a, b, bench1, bench2}, "brought set must be exactly the 4 brought"
    # own_live (brought, non-active, non-fainted) — must be 2, not 4 (un-brought
    # mons would inflate it by 2 vs the offline parser, a train/serve mismatch).
    active = {a, b}
    own_live = sum(1 for m in brought if m not in active and not m.fainted)
    assert own_live == 2, f"own_live inflated by un-brought mons: {own_live}"


def test_switchable_union_is_the_brought_bench():
    battle, (a, b, bench1, bench2, un1, un2) = _vgc_board()
    assert switchable_union(battle) == {bench1, bench2}


def test_bench_index_alignment_is_stable_team_order():
    """own_bench_mons order is the encoder/mask/codec switch index order; it must
    be the stable battle.team order of the brought bench (kingambit before
    sylveon here), so switch index i always decodes to the same mon."""
    battle, (a, b, bench1, bench2, un1, un2) = _vgc_board()
    assert own_bench_mons(battle) == [bench1, bench2]


def test_bench_drops_broken_illusion_phantom():
    battle, (a, b, bench1, bench2, un1, un2) = _vgc_board()
    bench1.max_hp = 0   # broken-illusion phantom (revealed disguise, no HP)
    assert own_bench_mons(battle) == [bench2], "phantom (max_hp 0) not filtered"


def test_fainted_brought_mon_not_on_bench_but_is_brought():
    battle, (a, b, bench1, bench2, un1, un2) = _vgc_board()
    bench1.fainted = True
    bench1.revealed = True           # a fainted mon was on the field
    assert bench1 not in own_bench_mons(battle)   # fainted → not a switch option
    assert bench1 in brought_team_mons(battle)    # but still part of the brought 4


def test_no_available_switches_yields_no_unbrought_leak():
    """If poke-env exposes no switches (both actives trapped) the bench is empty
    rather than leaking the un-brought roster mons — the un-brought mons must
    NEVER appear regardless."""
    battle, (a, b, bench1, bench2, un1, un2) = _vgc_board()
    battle.available_switches = [[], []]
    bench = own_bench_mons(battle)
    assert un1 not in bench and un2 not in bench


# ══════════════════════════════════════════════════════════════════════════════
# B) Serve-time belief enrichment of the reconstructed opponent snapshot
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def belief():
    if not _DEFAULT_PIKALYTICS_PATH.exists():
        pytest.skip("pikalytics_regma.json missing")
    return BeliefState(_DEFAULT_PIKALYTICS_PATH)


def _opp_snapshot():
    return {
        "our_active": {"our_a": {"species": "Incineroar", "base_species": "Incineroar",
                                 "seen": True}},
        "opp_active": {
            "opp_a": {"species": "Sneasler", "base_species": "Sneasler",
                      "seen": True, "revealed_moves": ["Fake Out"]},
            "opp_b": {"species": "Flutter Mane", "base_species": "Flutter Mane",
                      "seen": True},
        },
        "opp_bench": [{"species": "Kingambit", "base_species": "Kingambit", "seen": False}],
    }


def test_enrich_opp_populates_stats_and_predicted_moves(belief):
    enc = LiveStateEncoder(belief=belief)
    snap = _opp_snapshot()
    enc._enrich_opp_snapshot(snap)

    oa = snap["opp_active"]["opp_a"]
    assert (oa.get("stats_estimate") or {}).get("mode") == "distribution"
    slots = move_slots_for_mon(oa)
    assert slots and slots[0][0] == "Fake Out", "revealed move must stay in slot 0"
    assert len(slots) > 1, "unrevealed slots must be padded with predicted moves"

    # Training enriched even UNSEEN opp-bench stubs → serve must too.
    assert snap["opp_bench"][0].get("stats_estimate"), "unseen bench stub not enriched"


def test_enrich_opp_idempotent(belief):
    enc = LiveStateEncoder(belief=belief)
    snap = _opp_snapshot()
    enc._enrich_opp_snapshot(snap)
    first = copy.deepcopy(snap["opp_active"]["opp_a"]["stats_estimate"])
    enc._enrich_opp_snapshot(snap)   # a second pass must not change anything
    assert snap["opp_active"]["opp_a"]["stats_estimate"] == first


def test_enrich_opp_noop_without_belief():
    enc = LiveStateEncoder(belief=None)
    snap = _opp_snapshot()
    enc._enrich_opp_snapshot(snap)
    assert "stats_estimate" not in snap["opp_active"]["opp_a"]


def test_serve_belief_level_matches_training_default():
    """Byte-exact opp parity is CONDITIONAL on the serve encoder enriching at the
    SAME level the training export used (belief_state.DEFAULT_LEVEL).  Guard it so a
    future default-level change can't silently desync serve from the frozen data
    (the byte-parity test above would also fail, but this names the cause)."""
    from v_dance.parser.belief_state import DEFAULT_LEVEL
    assert LiveStateEncoder().level == DEFAULT_LEVEL, (
        "LiveStateEncoder default level drifted from the training export's "
        "DEFAULT_LEVEL — opponent est-stat bytes will diverge from training."
    )


@pytest.mark.parametrize("subdir,pika_name", [
    ("Regulation_MA/Jsonl_TypeB", "pikalytics_regma.json"),
    # ⚠ pinned to the EXPORT-era snapshot, not the live file: the corpus was exported under the
    # named belief, and the live pikalytics_regmb.json is allowed to move ahead for SERVING
    # (meta drift). At the next corpus re-export, re-pin this to the file used then.
    # 2026-07-11 era-2 step 2: MB corpora re-exported vs the Pikalytics×observed-meta BLEND —
    # re-pinned accordingly (2026-06-20 export kept on disk as the prior era's archive).
    ("Regulation_MB/Jsonl_TypeB", "pikalytics_regmb_blend_2026-07-11.json"),
])
def test_reenrich_reproduces_training_opp_bytes(subdir, pika_name):
    """The strongest guarantee: stripping then re-enriching the opponent of a real,
    belief-enriched TRAINING snapshot reproduces the SAME opponent bytes — i.e. the
    live gap-#6 splice matches the data the net was trained on.

    Run PER ERA with the prior the corpus was EXPORTED under (M-A<->regma,
    M-B<->regmb): byte-parity is conditional on the serve encoder enriching with the
    SAME Pikalytics prior as the training export — a mismatched prior shifts opp
    est-stats / move belief (the level analogue is the guard above). B3 trained on
    M-A-with-regma + M-B-with-regmb, each era-correct, so each era is validated."""
    import glob, json
    root = Path(__file__).resolve().parents[1]
    pika = root / "data" / pika_name
    if not pika.exists():
        pytest.skip(f"{pika_name} missing")
    files = sorted(glob.glob(
        str(root / "data/vods/Prepared_training_data" / subdir / "**/*.jsonl"),
        recursive=True))
    if not files:
        pytest.skip(f"no training data under {subdir}")

    enc = VodStateEncoder()
    live = LiveStateEncoder(belief=BeliefState(pika))
    opp_act = slice((ACTIVE_SLOTS - 2) * POKEMON_FEATURES, ACTIVE_SLOTS * POKEMON_FEATURES)
    opp_ben = slice((ACTIVE_SLOTS + BENCH_SLOTS) * POKEMON_FEATURES,
                    (ACTIVE_SLOTS + BENCH_SLOTS + OPP_BENCH_SLOTS) * POKEMON_FEATURES)

    checked = 0
    for fp in files[:5]:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                t = json.loads(line)
                snap = t.get("state_before_actions") or {}
                if not (snap.get("opp_active") or {}).get("opp_a"):
                    continue
                turn = t.get("turn") or 0
                vec_train = enc.encode_snapshot(snap, turn=turn)

                stripped = copy.deepcopy(snap)
                for k in ("opp_a", "opp_b"):
                    m = (stripped.get("opp_active") or {}).get(k)
                    if m:
                        m.pop("stats_estimate", None)
                        m.pop("belief", None)
                for m in (stripped.get("opp_bench") or []):
                    m.pop("stats_estimate", None)
                    m.pop("belief", None)
                live._enrich_opp_snapshot(stripped)
                vec_serve = enc.encode_snapshot(stripped, turn=turn)

                assert np.allclose(vec_train[opp_act], vec_serve[opp_act], atol=1e-6), \
                    f"opp ACTIVE bytes diverge from training after re-enrichment ({subdir})"
                assert np.allclose(vec_train[opp_ben], vec_serve[opp_ben], atol=1e-6), \
                    f"opp BENCH bytes diverge from training after re-enrichment ({subdir})"
                checked += 1
                if checked >= 60:
                    break
        if checked >= 60:
            break
    assert checked > 0, f"no opponent-bearing transitions found under {subdir}"
    assert checked > 0, "no opponent snapshots exercised"
