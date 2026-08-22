"""Observed-meta aggregator (2026-07-10): dossiers → Pikalytics-format usage file.

Contract: record-level percentages, empty spreads/natures (not observable), display-name
mapping via the roster cache, and — the point of the schema — the output is DROP-IN loadable
by BeliefState.
"""
import json

from v_dance.datatools.observed_meta import aggregate


def _dossier(opponent, games, mons):
    return {"opponent": opponent, "games": [{"result": "ai"}] * games, "mons": mons}


def _write(tmp_path, name, d):
    (tmp_path / f"{name}.json").write_text(json.dumps(d), encoding="utf-8")


def _two_dossiers(tmp_path):
    _write(tmp_path, "alice", _dossier("alice", 2, {
        "kingambit": {"species": "kingambit", "moves": ["suckerpunch", "kowtowcleave"],
                      "item": "focussash", "ability": "defiant", "times_seen": 2},
        "pyroar": {"species": "pyroar", "moves": ["heatwave"], "item": None,
                   "ability": "moxie", "times_seen": 1},
    }))
    _write(tmp_path, "bob", _dossier("bob", 1, {
        "kingambit": {"species": "kingambit", "moves": ["suckerpunch"],
                      "item": "chopleberry", "ability": "defiant", "times_seen": 1},
    }))


def test_aggregate_counts_and_percentages(tmp_path):
    _two_dossiers(tmp_path)
    d = aggregate(tmp_path, fmt="gen9testfmt")
    assert d["n_opponents"] == 2 and d["n_games"] == 3
    kg = d["pokemon"]["kingambit"]
    assert kg["usage_pct"] == 100.0                        # seen in 3 sightings / 3 games
    assert {"name": "suckerpunch", "pct": 100.0} in kg["moves"]    # 2/2 records revealed it
    assert {"name": "kowtowcleave", "pct": 50.0} in kg["moves"]    # 1/2 records
    assert {"name": "focussash", "pct": 50.0} in kg["items"]       # 1 of 2 item-revealing records
    assert kg["abilities"][0] == {"name": "defiant", "pct": 100.0}
    assert kg["spreads"] == [] and kg["natures"] == []             # not observable from dossiers
    assert kg["teammates"] and kg["teammates"][0]["name"] == "pyroar"
    py = d["pokemon"]["pyroar"]
    assert py["usage_pct"] == round(100.0 / 3, 1)


def test_min_records_noise_floor(tmp_path):
    _two_dossiers(tmp_path)
    d = aggregate(tmp_path, fmt="f", min_records=2)
    assert "kingambit" in d["pokemon"] and "pyroar" not in d["pokemon"]


def test_display_name_mapping(tmp_path):
    _two_dossiers(tmp_path)
    d = aggregate(tmp_path, fmt="f", names={"kingambit": "Kingambit"})
    assert "Kingambit" in d["pokemon"] and "kingambit" not in d["pokemon"]


def test_output_is_drop_in_loadable_by_beliefstate(tmp_path):
    """The whole point of the schema: BeliefState must load the aggregate like a scraped file."""
    _two_dossiers(tmp_path)
    out = tmp_path / "observed_meta_test.json"
    out.write_text(json.dumps(aggregate(tmp_path, fmt="gen9testfmt",
                                        names={"kingambit": "Kingambit"})), encoding="utf-8")
    from v_dance.parser.belief_state import BeliefState
    b = BeliefState(out)
    assert "Kingambit" in b.all_pokemon()
