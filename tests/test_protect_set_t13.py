"""T1.3 — live `is_protect` move flag must key on the OFFLINE `_PROTECT_MOVES` set (the corpus/parser
set), NOT poke-env's `Move.is_protect_move`. The two disagree on 5 moves; before the fix an OWN mon
(whose move features are NOT opponent-spliced) carrying Wide/Quick Guard got is_protect=0 live vs 1 in
training. These lock the intended membership and assert the live writer keys on this set.
"""
from __future__ import annotations

from v_dance.encoders.encoder_layout import _PROTECT_MOVES


def test_protect_set_is_the_corpus_set():
    # INCLUDES Wide Guard / Quick Guard (staple VGC doubles support) ...
    assert {"wideguard", "quickguard"} <= _PROTECT_MOVES
    # ... and EXCLUDES Endure / King's Shield / Obstruct (which poke-env's is_protect_move includes).
    assert not ({"endure", "kingsshield", "obstruct"} & _PROTECT_MOVES)
    # the uncontested core is present on both sides.
    assert {"protect", "detect", "spikyshield", "banefulbunker",
            "burningbulwark", "silktrap", "maxguard"} <= _PROTECT_MOVES


def test_live_writer_keys_on_protect_set():
    # The live writer's predicate is `getattr(move, "id", "") in _PROTECT_MOVES`. Assert it disagrees
    # with poke-env's Move.is_protect_move on exactly the 5 known-divergent moves, so the fix is
    # load-bearing (a revert to move.is_protect_move would flip these). Skip if poke-env unavailable.
    import importlib

    Move = None
    for path in ("poke_env.battle.move", "poke_env.environment.move"):
        try:
            Move = importlib.import_module(path).Move
            break
        except Exception:
            continue
    if Move is None:
        import pytest
        pytest.skip("poke-env Move not importable")

    def _mk(mid):
        try:
            return Move(mid, gen=9)
        except TypeError:
            return Move(mid)

    # offline=1 / poke-env=0
    for mid in ("wideguard", "quickguard"):
        mv = _mk(mid)
        assert (mv.id in _PROTECT_MOVES) is True
        assert mv.is_protect_move is False, f"{mid}: poke-env excludes it → live writer must not rely on it"
    # offline=0 / poke-env=1
    for mid in ("endure", "kingsshield", "obstruct"):
        mv = _mk(mid)
        assert (mv.id in _PROTECT_MOVES) is False
        assert mv.is_protect_move is True, f"{mid}: poke-env includes it → live writer must exclude it"
    # agree on the core
    for mid in ("protect", "detect"):
        mv = _mk(mid)
        assert (mv.id in _PROTECT_MOVES) is True and mv.is_protect_move is True
