"""Re-export belief-load guard: a SPECIFIED-but-missing --pikalytics path must FAIL LOUD.

Regression for the wrong-era footgun found in the v10 pre-bake bug-hunt: ``_load_belief`` used to
return None on any non-existent path, after which ``fill_blanks`` silently auto-loads the ACTIVE
format's (regmb) prior. A bare-token ``--pikalytics regma`` (``Path('regma').exists()`` is False) on
an M-A re-export would therefore bake the WRONG era's distributions into the corpus with no error.
The guard now exits non-zero on a specified-but-missing path; only an omitted flag stays graceful.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from v_dance.datatools.bulk_parse_replays import _load_belief, _legacy_twin_to_clean

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_missing_specified_pikalytics_path_fails_loud():
    # the exact footgun: a bare era token resolves to a non-existent relative path
    with pytest.raises(SystemExit) as exc:
        _load_belief(Path("regma"))
    assert "does not exist" in str(exc.value)


def test_typoed_full_path_fails_loud():
    with pytest.raises(SystemExit):
        _load_belief(_PROJECT_ROOT / "data" / "pikalytics_regZZ.json")


def test_omitted_flag_stays_graceful():
    # None == flag genuinely omitted → graceful (fill_blanks falls back to the active prior)
    assert _load_belief(None) is None


def test_valid_path_loads_belief():
    real = _PROJECT_ROOT / "data" / "pikalytics_regma.json"
    if not real.exists():
        pytest.skip("pikalytics_regma.json not present")
    belief = _load_belief(real)
    assert belief is not None


# ── --overwrite must clean a stale legacy {html.stem}.jsonl twin (duplicate-replay_id guard) ──
def test_overwrite_flags_existing_legacy_twin(tmp_path):
    out = tmp_path / "gen9-12345.jsonl"          # canonical replay_id name
    legacy = tmp_path / "SomeReplay-stem.jsonl"  # older stem name, same replay
    legacy.write_text("{}", encoding="utf-8")
    assert _legacy_twin_to_clean(out, legacy, overwrite=True) == legacy


def test_no_overwrite_leaves_legacy_twin_alone(tmp_path):
    out = tmp_path / "gen9-12345.jsonl"
    legacy = tmp_path / "SomeReplay-stem.jsonl"
    legacy.write_text("{}", encoding="utf-8")
    assert _legacy_twin_to_clean(out, legacy, overwrite=False) is None


def test_no_twin_when_names_match_or_absent(tmp_path):
    out = tmp_path / "gen9-12345.jsonl"
    # same path → never clean (would delete the file we're about to write)
    assert _legacy_twin_to_clean(out, out, overwrite=True) is None
    # legacy absent → nothing to clean
    assert _legacy_twin_to_clean(out, tmp_path / "absent.jsonl", overwrite=True) is None
