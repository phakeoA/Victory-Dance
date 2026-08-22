"""15b-data.1 — the team-preview DATASET emits the SBDA tpfeat-vN per-mon features via
the SAME tp_features extractor model_io serves (the train==serve lockstep), behind a
recipe flag, with the legacy 46-dim dex path untouched and feat_dim inferred from data.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("poke_env")

from v_dance.parser.vod_parser.pokedex import norm_species  # noqa: E402
from v_dance.training import tp_features as TPF             # noqa: E402
from v_dance.training import teampreview_dataset as DS       # noqa: E402
import v_dance.play.model_io as M                            # noqa: E402


class _SandBelief:
    """Every mon 'knows' Sand Stream + Earthquake — lights up belief-derived tags so a
    test can prove the features actually flow (not zero-padded)."""
    def known(self, s):
        return True

    def ability_distribution(self, s, top_k=4):
        return [{"name": "Sand Stream", "p": 1.0}]

    def move_distribution(self, s, top_k=8):
        return [{"name": "Earthquake", "p": 1.0}]

    def usage(self, s):
        return 50.0

    def teammates(self, s, top_k=16):
        return []


_OUR = ["Tyranitar", "Charizard", "Garchomp", "Amoonguss", "Incineroar", "Rillaboom"]
_OPP = ["Miraidon", "Ironhands", "Chiyu", "Landorus", "Ogerpon", "Ironbundle"]


def _fake_transition():
    return {"players": {"p1": {"roster": _OUR, "brought": _OUR[:4]},
                        "p2": {"roster": _OPP}}, "replay_id": "r1"}


# ── feature_recipe factory ────────────────────────────────────────────────────
def test_feature_recipe_legacy():
    fn, dim, schema = DS.feature_recipe("legacy")
    assert fn is DS.mon_dex_features and dim == DS.MON_FEAT_DIM and schema is None


def test_feature_recipe_sbda():
    fn, dim, schema = DS.feature_recipe("sbda", belief=_SandBelief())
    assert dim == TPF.FEAT_DIM and schema == TPF.FEATURE_SCHEMA_VERSION
    assert fn("Tyranitar").shape == (TPF.FEAT_DIM,)


def test_feature_recipe_sbda_requires_belief():
    with pytest.raises(ValueError, match="BeliefState"):
        DS.feature_recipe("sbda")


def test_feature_recipe_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        DS.feature_recipe("bogus")


# ── THE lockstep: dataset builder == serve _pack_side == the shared extractor ──
def test_dataset_sbda_features_match_serve_and_extractor():
    b = _SandBelief()
    feat_fn = DS.sbda_feature_fn(b)
    train_row = feat_fn("Tyranitar")                                # dataset side
    extractor = TPF.opp_mon_features(norm_species("Tyranitar"), b)  # the shared extractor
    _idx, serve = M._pack_side(["Tyranitar"], {}, TPF.FEAT_DIM, belief=b, use_tp_features=True)
    np.testing.assert_array_equal(train_row, extractor)
    np.testing.assert_array_equal(train_row, serve[0])              # train == serve, byte for byte
    # and the belief-derived tags actually flowed (proves no silent zero-pad)
    assert train_row[TPF.OFF_WSETS + TPF.WEATHERS.index("sand")] == pytest.approx(1.0)


# ── _example_for_side: legacy unchanged (46) vs sbda (FEAT_DIM) ───────────────
def test_example_for_side_legacy_is_46dim_unchanged():
    ex = DS._example_for_side(_fake_transition(), "p1", "p2", None)   # default feat_fn = legacy
    assert ex["our_feat"].shape == (6, DS.MON_FEAT_DIM)
    np.testing.assert_array_equal(ex["our_feat"][0], DS.mon_dex_features("Tyranitar"))


def test_example_for_side_sbda_is_featdim():
    ex = DS._example_for_side(_fake_transition(), "p1", "p2", None,
                              DS.sbda_feature_fn(_SandBelief()))
    assert ex["our_feat"].shape == (6, TPF.FEAT_DIM)
    assert ex["opp_feat"].shape == (6, TPF.FEAT_DIM)
    # Type-B has no own build → own overlay OFF → own == opp base (the 15b parity invariant)
    np.testing.assert_array_equal(ex["our_feat"][0],
                                  TPF.opp_mon_features(norm_species("Tyranitar"), _SandBelief()))


# ── TeamPreviewDataset infers feat_dim from the examples (both recipes) ───────
def test_dataset_infers_featdim_sbda_and_legacy():
    t = _fake_transition()
    vocab = {norm_species(s): i + 1 for i, s in enumerate(_OUR + _OPP)}
    ds_sbda = DS.TeamPreviewDataset(
        [DS._example_for_side(t, "p1", "p2", None, DS.sbda_feature_fn(_SandBelief()))], vocab)
    assert ds_sbda.feat_dim == TPF.FEAT_DIM
    assert tuple(ds_sbda[0]["our_feat"].shape) == (6, TPF.FEAT_DIM)
    ds_legacy = DS.TeamPreviewDataset([DS._example_for_side(t, "p1", "p2", None)], vocab)
    assert ds_legacy.feat_dim == DS.MON_FEAT_DIM
    assert tuple(ds_legacy[0]["our_feat"].shape) == (6, DS.MON_FEAT_DIM)


# ── regression: usage_pct=null in scraped regmb data must not crash the SBDA build ──
def test_belief_usage_null_pct_is_safe_and_sbda_build_succeeds(tmp_path):
    """Caught on real pikalytics_regmb.json: some entries carry usage_pct=null
    (teammate-only rows). BeliefState.usage() must coerce null -> 0.0, not float(None)
    crash — else the 15b SBDA usage feature dies on real M-B mons."""
    import json
    from v_dance.parser.belief_state import BeliefState
    p = tmp_path / "pika.json"
    p.write_text(json.dumps({"pokemon": {"Tyranitar": {"usage_pct": None},
                                         "Garchomp": {"usage_pct": 12.5}}}),
                 encoding="utf-8")
    b = BeliefState(p)
    assert b.usage("Tyranitar") == 0.0          # null -> 0.0
    assert b.usage("Garchomp") == 12.5
    assert b.usage("Nonexistent") == 0.0        # missing entry -> 0.0
    feat = TPF.opp_mon_features(norm_species("Tyranitar"), b)   # full SBDA build end-to-end
    assert feat.shape == (TPF.FEAT_DIM,) and np.isfinite(feat).all()


# ── regression: teammates() M-B {name,rank} schema (lockstep adversarial review) ──
def test_belief_teammates_handles_mb_rank_schema(tmp_path):
    """The active pikalytics_regmb.json stores teammates as {name, rank} (no pct — the M-B
    Pikalytics page dropped the %). teammates() must not KeyError, must order by rank, and emit
    the {name, p} contract (p=1/rank) so teammate_affinity_matrix doesn't crash at serve and
    silently disable the teammate-bias SBDA net. Sibling bug to the usage_pct=null one."""
    import json
    from v_dance.parser.belief_state import BeliefState
    p = tmp_path / "pika_mb.json"
    p.write_text(json.dumps({"pokemon": {"Incineroar": {"teammates": [
        {"name": "Sinistcha", "rank": 1}, {"name": "Garchomp", "rank": 2},
        {"name": "Staraptor", "rank": 4}]}}}), encoding="utf-8")
    tm = BeliefState(p).teammates("Incineroar")
    assert [t["name"] for t in tm] == ["Sinistcha", "Garchomp", "Staraptor"]   # rank order
    assert (tm[0]["p"], tm[1]["p"], tm[2]["p"]) == (1.0, 0.5, 0.25)            # 1/rank proxy


def test_belief_teammates_handles_legacy_ma_pct_schema(tmp_path):
    """The legacy M-A {name, pct} schema still sorts by pct (desc) with p=pct/100."""
    import json
    from v_dance.parser.belief_state import BeliefState
    p = tmp_path / "pika_ma.json"
    p.write_text(json.dumps({"pokemon": {"Tyranitar": {"teammates": [
        {"name": "Excadrill", "pct": 55.0}, {"name": "Landorus", "pct": 77.0}]}}}),
        encoding="utf-8")
    tm = BeliefState(p).teammates("Tyranitar")
    assert [t["name"] for t in tm] == ["Landorus", "Excadrill"]
    assert (tm[0]["p"], tm[1]["p"]) == (0.77, 0.55)


def test_active_mb_belief_teammates_and_affinity_no_crash():
    """The review's masked gap: the only live teammates test pinned regma. Load the ACTIVE
    belief (no-arg → regmb) and sweep teammates() + teammate_affinity_matrix over real rosters —
    must not raise (the KeyError 'pct' that silently fell back to the heuristic on M-B)."""
    from v_dance.parser.belief_state import BeliefState
    from v_dance.training.tp_features import teammate_affinity_matrix
    b = BeliefState()                        # active format → pikalytics_regmb.json
    mons = b.all_pokemon()
    if not mons:
        import pytest as _pt
        _pt.skip("no active belief data on disk")
    for s in mons[:40]:
        b.teammates(s)                       # no KeyError on real M-B data
    A = teammate_affinity_matrix(mons[:6], b, n=6)
    assert A.shape == (6, 6) and np.isfinite(A).all() and (A >= 0).all()
