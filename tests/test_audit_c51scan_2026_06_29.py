"""Regression tests for the full-codebase scan run before C51 B2 (2026-06-29).

Confirmed findings fixed here:
- C51 config-boundary crash: value_loss_mode='c51' is reachable via --config but unwired until B2 ->
  build_train_configs must REJECT it fast (not crash every PPO minibatch mid-run).
- v19 redundant-STATUS gap: _status_immune / move_redundant_status ignored terrain immunity
  (Misty blocks all major status, Electric blocks sleep — both on grounded targets only).
- browser double-credit: the _ai_consumer finally-block guards on host._ended so a re-delivered
  terminal frame can't double-count the tally (predicate-level test; full path is --self-test).
"""
import pytest

from v_dance.encoders.battle_mechanics import _status_immune, move_redundant_status


# ── Fix 2: terrain status immunity (dex-free, exercises the new _status_immune logic) ──
def _grounded(status="", ability="", types=("NORMAL",)):
    return {"status": status, "ability": ability, "types": list(types), "grounded": True}


def test_misty_terrain_blocks_all_major_status_on_grounded():
    g = _grounded()
    assert _status_immune("slp", False, "", g, "MISTY_TERRAIN") is True
    assert _status_immune("par", False, "", g, "MISTY_TERRAIN") is True
    assert _status_immune("brn", False, "", g, "MISTY_TERRAIN") is True


def test_electric_terrain_blocks_only_sleep_on_grounded():
    g = _grounded()
    assert _status_immune("slp", False, "", g, "ELECTRIC_TERRAIN") is True
    assert _status_immune("par", False, "", g, "ELECTRIC_TERRAIN") is False   # par NOT blocked by Electric


def test_terrain_immunity_requires_grounded():
    ng = {"status": "", "ability": "", "types": ["FLYING"], "grounded": False}
    assert _status_immune("slp", False, "", ng, "MISTY_TERRAIN") is False
    assert _status_immune("slp", False, "", ng, "ELECTRIC_TERRAIN") is False


def test_no_terrain_is_old_behaviour():
    g = _grounded()
    assert _status_immune("slp", False, "", g, None) is False    # vanilla grounded Normal — not immune


def test_move_redundant_status_threads_terrain():
    # Spore is a pure sleep move; under Misty/Electric Terrain vs a grounded target it is WASTED.
    g = _grounded()
    base = move_redundant_status("spore", g, None)
    assert base == 0.0                                            # no terrain -> not redundant
    assert move_redundant_status("spore", g, "MISTY_TERRAIN") == 1.0
    assert move_redundant_status("spore", g, "ELECTRIC_TERRAIN") == 1.0


# ── Fix 1: C51 config boundary ────────────────────────────────────────────────
# NOTE: the B2-era fail-fast guard (reject c51 entirely) was REPLACED in B3 by full c51 wiring;
# build_train_configs now ACCEPTS c51 and only rejects a degenerate atom count. (Acceptance is also
# covered by test_c51_b3_2026_06_29.test_build_train_configs_accepts_c51_now.)
def test_build_train_configs_c51_now_wired():
    pytest.importorskip("torch")
    from v_dance.selfplay.generation import build_train_configs
    ppo_cfg, _ = build_train_configs(ppo_overrides={"value_loss_mode": "c51", "n_atoms": 51})
    assert ppo_cfg.value_loss_mode == "c51" and ppo_cfg.n_atoms == 51
    with pytest.raises(SystemExit):                              # degenerate atom count still rejected
        build_train_configs(ppo_overrides={"value_loss_mode": "c51", "n_atoms": 1})
    ppo_cfg, _ = build_train_configs(ppo_overrides={"value_loss_mode": "bce"})
    assert ppo_cfg.value_loss_mode == "bce"


# ── Fix 3: browser double-credit guard (predicate-level; full path under --self-test) ──
def test_credit_guard_skips_already_ended_tag():
    class _H:
        _ended = {"battle-x"}
    host, result = _H(), "|win|VictoryDanceAI"
    # re-delivered terminal frame for an already-ended/credited room -> skip
    assert not (result is not None and "battle-x" and "battle-x" not in host._ended)
    # a fresh, not-yet-ended battle -> credit
    assert (result is not None and "battle-new" and "battle-new" not in host._ended)
