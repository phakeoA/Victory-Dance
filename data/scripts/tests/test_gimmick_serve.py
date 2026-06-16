"""Task #4: serve-side gimmick (mega) codec in vgc_base.py.

Covers the reusable live mega detection, build_gimmick_legal_mask (byte-parity
with state_encoder.build_gimmick_mask), and action_to_order's mega gating.
poke-env is required for the real-mon tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("poke_env")


def _repo_on_path():
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    for p in (str(repo), str(repo / "data" / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ── 4a: reusable live mega detection ────────────────────────────────────────
def test_is_mega_forme_live_real_charizard():
    _repo_on_path()
    from poke_env.battle import Pokemon
    from live_state_encoder import is_mega_forme_live

    chari = Pokemon(gen=9, species="charizard")
    assert is_mega_forme_live(chari) is False        # base forme: not mega
    chari.mega_evolve("charizarditey")
    assert is_mega_forme_live(chari) is True          # mega stats detected


def test_is_mega_forme_live_no_substring_false_positive():
    _repo_on_path()
    from poke_env.battle import Pokemon
    from live_state_encoder import is_mega_forme_live

    for sp in ("meganium", "yanmega", "scizor", "venusaur"):
        assert is_mega_forme_live(Pokemon(gen=9, species=sp)) is False


def test_team_has_megaed_live():
    _repo_on_path()
    from poke_env.battle import Pokemon
    from live_state_encoder import team_has_megaed_live

    base_team = {"a": Pokemon(gen=9, species="charizard"),
                 "b": Pokemon(gen=9, species="incineroar")}

    class _B:
        team = base_team

    assert team_has_megaed_live(_B()) is False
    base_team["a"].mega_evolve("charizarditey")       # one mon megas
    assert team_has_megaed_live(_B()) is True


def test_encoder_method_still_delegates():
    """The LiveStateEncoder._is_mega_forme method must keep working after the
    refactor (it now delegates to the free function)."""
    _repo_on_path()
    from poke_env.battle import Pokemon
    from state_encoder import StateEncoder  # alias of VodStateEncoder — no _is_mega
    from live_state_encoder import LiveStateEncoder

    enc = LiveStateEncoder()
    chari = Pokemon(gen=9, species="charizard")
    assert enc._is_mega_forme(chari) is False
    chari.mega_evolve("charizarditey")
    assert enc._is_mega_forme(chari) is True


# ── 4b: build_gimmick_legal_mask + byte-parity with training ────────────────
class _DMon:
    def __init__(self, species, fainted=False):
        self.species = species
        self.base_species = species
        self.fainted = fainted


class _DBattle:
    def __init__(self, active, team=None):
        self.active_pokemon = active
        self.team = team or {}


def test_gimmick_legal_mask_capable_and_noncapable():
    _repo_on_path()
    from vgc_base import build_gimmick_legal_mask
    battle = _DBattle([_DMon("charizard"), _DMon("incineroar")])
    assert build_gimmick_legal_mask(battle, 0) == [True, True]    # capable
    assert build_gimmick_legal_mask(battle, 1) == [True, False]   # not capable


def test_gimmick_legal_mask_fainted_and_empty():
    _repo_on_path()
    from vgc_base import build_gimmick_legal_mask
    battle = _DBattle([_DMon("charizard", fainted=True), None])
    assert build_gimmick_legal_mask(battle, 0) == [False, False]  # fainted
    assert build_gimmick_legal_mask(battle, 1) == [False, False]  # empty slot


def test_gimmick_legal_mask_team_already_megaed():
    _repo_on_path()
    from poke_env.battle import Pokemon
    from vgc_base import build_gimmick_legal_mask
    chari = Pokemon(gen=9, species="charizard")
    spent = Pokemon(gen=9, species="venusaur")
    spent.mega_evolve("venusaurite")                 # team's one mega is spent
    battle = _DBattle([chari, _DMon("incineroar")], team={"a": chari, "b": spent})
    # charizard is capable, but the team already used its mega → mega forbidden.
    assert build_gimmick_legal_mask(battle, 0) == [True, False]


def test_gimmick_mask_byte_parity_train_vs_serve():
    """The serve mask must equal the training mask (state_encoder.build_gimmick_mask)
    bit-for-bit on the equivalent board — the hard train/serve invariant."""
    _repo_on_path()
    import state_encoder as se
    from vgc_base import build_gimmick_legal_mask

    snap = {"our_active": {
        "our_a": {"species": "Charizard", "base_species": "Charizard",
                  "is_fainted": False, "is_mega": False},
        "our_b": {"species": "Incineroar", "base_species": "Incineroar",
                  "is_fainted": False, "is_mega": False},
    }, "our_bench": []}
    train = se.build_gimmick_mask(snap)

    battle = _DBattle([_DMon("charizard"), _DMon("incineroar")])
    assert [int(x) for x in build_gimmick_legal_mask(battle, 0)] == train["our_a"]
    assert [int(x) for x in build_gimmick_legal_mask(battle, 1)] == train["our_b"]


# ── 4c: action_to_order mega gating + _safe_order threading ──────────────────
def _move_battle(can_mega):
    """Single-foe board with a usable single-target move; can_mega is the slot-0
    can_mega_evolve value."""
    from poke_env.battle import Move
    fth = Move("flamethrower", gen=9)

    class _Mon:
        def __init__(self):
            self._mv = {"flamethrower": fth}
            self.fainted = False
            self.species = "charizard"
            self.base_species = "charizard"
        @property
        def moves(self):
            return self._mv

    actor, foe = _Mon(), _Mon()

    class _Battle:
        active_pokemon = [actor, None]
        opponent_active_pokemon = [foe, None]
        available_moves = [list(actor.moves.values()), []]
        available_switches = [[], []]
        team = {}
        can_mega_evolve = [can_mega, False]
        def to_showdown_target(self, mv, tgt):
            return 1

    return _Battle()


def test_action_to_order_megas_move_when_legal():
    _repo_on_path()
    from vgc_base import action_to_order, GIMMICK_MEGA
    order = action_to_order(0, _move_battle(can_mega=True), 0, gimmick=GIMMICK_MEGA)
    assert order is not None
    assert "mega" in order.message, f"expected a mega order: {order.message!r}"


def test_action_to_order_no_mega_when_can_mega_false():
    """Safety gate: the model picked mega but poke-env says it is illegal now
    (no stone / team already mega'd) → emit the plain move, never a rejected mega."""
    _repo_on_path()
    from vgc_base import action_to_order, GIMMICK_MEGA
    order = action_to_order(0, _move_battle(can_mega=False), 0, gimmick=GIMMICK_MEGA)
    assert order is not None
    assert "mega" not in order.message


def test_action_to_order_no_mega_when_gimmick_none():
    _repo_on_path()
    from vgc_base import action_to_order, GIMMICK_NONE
    order = action_to_order(0, _move_battle(can_mega=True), 0, gimmick=GIMMICK_NONE)
    assert order is not None
    assert "mega" not in order.message


def test_action_to_order_default_gimmick_is_none():
    """Back-compat: callers that don't pass a gimmick get a plain (non-mega) order."""
    _repo_on_path()
    from vgc_base import action_to_order
    order = action_to_order(0, _move_battle(can_mega=True), 0)
    assert order is not None and "mega" not in order.message


def test_live_can_mega_handles_shapes():
    _repo_on_path()
    from vgc_base import _live_can_mega

    class _L:  # per-slot list (DoubleBattle)
        can_mega_evolve = [True, False]

    class _B:  # plain bool (single Battle)
        can_mega_evolve = True

    class _Missing:
        pass

    assert _live_can_mega(_L(), 0) is True
    assert _live_can_mega(_L(), 1) is False
    assert _live_can_mega(_L(), 5) is False          # out of range → False
    assert _live_can_mega(_B(), 0) is True
    assert _live_can_mega(_Missing(), 0) is False     # no attr → False


def test_safe_order_threads_gimmick_to_primary_action():
    """_safe_order must pass the model's gimmick to the primary action_to_order."""
    _repo_on_path()
    from vgc_base import VGCPlayerBase, GIMMICK_MEGA, GIMMICK_NONE
    battle = _move_battle(can_mega=True)
    mega = VGCPlayerBase._safe_order(None, 0, battle, 0, GIMMICK_MEGA)
    assert "mega" in mega.message
    plain = VGCPlayerBase._safe_order(None, 0, _move_battle(can_mega=True), 0, GIMMICK_NONE)
    assert "mega" not in plain.message
