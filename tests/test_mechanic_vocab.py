"""B1-mechanics step 1: data-grounded identity vocabs (ability/move/item) parsed from the pinned Showdown
dex. Asserts coverage, determinism, PAD handling, the Champions customs, and the data-confirmed mechanic
sets — so a dex bump that shifts indices fails loud."""
from __future__ import annotations

import pytest

from v_dance.encoders import mechanic_vocab as MV

_DEX_PRESENT = MV._DEX.exists()
pytestmark = pytest.mark.skipif(not _DEX_PRESENT, reason="pinned Showdown dex not present")


def test_vocab_sizes_sane():
    # verify_mechanic_coverage parsed 320 base abilities, 953 moves, 583 items; vocab = that + PAD (+mods).
    assert 300 <= MV.ABILITY_VOCAB_SIZE <= 400
    assert 900 <= MV.MOVE_VOCAB_SIZE <= 1100
    assert 500 <= MV.ITEM_VOCAB_SIZE <= 750


def test_pad_is_zero_and_reserved():
    assert MV.PAD == 0
    assert MV.ability_index(None) == 0 and MV.ability_index("") == 0
    assert MV.ability_index("not_a_real_ability_xyz") == 0
    assert MV.move_index(None) == 0 and MV.item_index("nope") == 0
    # PAD index 0 is never assigned to a real entry
    assert 0 not in MV.ABILITY_VOCAB.values()
    assert 0 not in MV.MOVE_VOCAB.values()
    assert 0 not in MV.ITEM_VOCAB.values()


def test_known_entries_resolve_and_are_norm_insensitive():
    assert MV.ability_index("shadowtag") > 0
    assert MV.ability_index("Shadow Tag") == MV.ability_index("shadowtag")   # norm-insensitive
    assert MV.move_index("bodypress") > 0
    assert MV.move_index("Body Press") == MV.move_index("bodypress")
    assert MV.item_index("leftovers") > 0
    assert MV.item_index("Choice Scarf") > 0


def test_champions_customs_present():
    # the customs that were confirmed absent from every hand tag table must still get an embedding index
    for cust in ("dragonize", "eelevate", "firemane", "megasol", "piercingdrill", "spicyspray", "supersweetsyrup"):
        assert MV.ability_index(cust) > 0, f"custom ability {cust} missing from vocab"


def test_data_confirmed_mechanic_sets_are_in_vocab():
    # from verify_mechanic_coverage: onFoeTrapPokemon + onModifyType sets — all must be embeddable
    for trap in ("shadowtag", "arenatrap", "magnetpull"):
        assert MV.ability_index(trap) > 0
    for tc in ("aerilate", "dragonize", "galvanize", "liquidvoice", "normalize", "pixilate", "refrigerate"):
        assert MV.ability_index(tc) > 0


def test_full_coverage_no_collisions():
    # every parsed id maps to a UNIQUE index covering exactly 1..N (no gaps, no dupes)
    for vocab, n in ((MV.ABILITY_VOCAB, MV.ABILITY_VOCAB_SIZE),
                     (MV.MOVE_VOCAB, MV.MOVE_VOCAB_SIZE),
                     (MV.ITEM_VOCAB, MV.ITEM_VOCAB_SIZE)):
        idxs = sorted(vocab.values())
        assert idxs == list(range(1, n))          # contiguous 1..N, PAD(0) excluded
        assert len(set(vocab.values())) == len(vocab)   # no collisions


def test_determinism_rebuild_is_identical():
    # re-deriving from the pinned dex yields the SAME index mapping (stable embedding indices)
    assert MV._build(MV._dex_ids("abilities.ts", "mods/champions/abilities.ts")) == MV.ABILITY_VOCAB
    assert MV._build(MV._dex_ids("moves.ts", "mods/champions/moves.ts")) == MV.MOVE_VOCAB
    assert MV._build(MV._dex_ids("items.ts", "mods/champions/items.ts")) == MV.ITEM_VOCAB


def test_missing_from_vocab_guard():
    # all real for their kind -> full coverage, no gaps
    assert MV.missing_from_vocab("ability", ["shadowtag", "intimidate"]) == []
    assert MV.missing_from_vocab("move", ["bodypress", "roar"]) == []
    assert MV.missing_from_vocab("item", ["leftovers", "focussash"]) == []
    # an unknown / wrong-kind id is REPORTED (the build guard catches it)
    assert "Body Press" in MV.missing_from_vocab("ability", ["shadowtag", "Body Press"])
