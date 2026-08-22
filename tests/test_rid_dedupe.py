"""M8 (2026-07-11): canonical replay-id dedupe across the battle-/__closed boundary.

Type_C exports carry a ``battle-`` prefix and a closed-strip re-ingest a ``__closed``
suffix; the QA duplicate scan and split_with_reference's val-leakage drop compared
RAW strings and would miss those same-battle collisions.
"""
import pytest

from v_dance.datatools.corpus_qa import _canonical_rid
from v_dance.training.bc_dataset import canonical_rid, split_with_reference


@pytest.mark.parametrize("raw,canon", [
    ("gen9championsvgc2026regmb-123", "gen9championsvgc2026regmb-123"),
    ("battle-gen9championsvgc2026regmb-123", "gen9championsvgc2026regmb-123"),
    ("gen9championsvgc2026regmb-123__closed", "gen9championsvgc2026regmb-123"),
    ("battle-gen9x-9__closed", "gen9x-9"),
])
def test_canonical_rid(raw, canon):
    assert canonical_rid(raw) == canon
    assert _canonical_rid(raw) == canon          # corpus_qa's stdlib-only copy: parity


def _ex(rid):
    return {"replay_id": rid}


def test_split_with_reference_drops_prefixed_twins():
    ref = [_ex("gen9x-1"), _ex("gen9x-2"), _ex("gen9x-3")]
    extra = [_ex("battle-gen9x-1"),          # Type_C twin of a ref replay -> dropped
             _ex("gen9x-2__closed"),         # closed-strip twin -> dropped
             _ex("battle-gen9x-99")]         # genuinely new -> kept
    train, val = split_with_reference(extra, ref, val_frac=0.34, seed=0)
    kept_extra = [e["replay_id"] for e in train if e["replay_id"] not in
                  {x["replay_id"] for x in ref}]
    assert kept_extra == ["battle-gen9x-99"]
    assert len(train) + len(val) == len(ref) + 1
