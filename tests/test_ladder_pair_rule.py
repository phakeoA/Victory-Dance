"""2026-10-03 THE PAIR RULE in the W3b ladder chain (memory 01: "keep team picker with the battle neural networks in
reinforcement learning and treat them like 1 neural network"): the candidate keeps the picker its ladder games were
played with, and is registered + deployed as that pair."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from v_dance.ladder import update as LU


class _Arm:
    def __init__(self, tp):
        self.tp_ckpt = tp

    def uses_no_tp(self):
        return str(self.tp_ckpt).strip().lower() == "none"

    def uses_default(self, which):
        return self.tp_ckpt is None or str(self.tp_ckpt).strip().lower() in ("", "default")


def test_env_default_tp_reads_only_the_picker_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("SECRET_PASSWORD=x\n# VD_TP_CKPT=old.pt\nVD_TP_CKPT=pickers/tp.pt\n", encoding="utf-8")
    assert LU.env_default_tp(env) == LU._REPO / "pickers" / "tp.pt"
    from v_dance.play.model_io import DEFAULT_TP_CHECKPOINT
    assert LU.env_default_tp(tmp_path / "missing.env") == Path(DEFAULT_TP_CHECKPOINT)


def test_pair_picker_prefers_the_sidecar_then_the_arms_and_refuses_mixed_data(tmp_path):
    from v_dance.selfplay import tp_learning as TPL
    base, side_tp = tmp_path / "base.pt", tmp_path / "paired_tp.pt"
    base.write_bytes(b"net")
    side_tp.write_bytes(b"picker")
    plain = tmp_path / "plain.pt"
    plain.write_bytes(b"net2")
    arms = {"learn": _Arm("default"), "twin": _Arm("default"), "own": _Arm(str(tmp_path / "own_tp.pt")),
            "first4": _Arm("none")}
    tp, how = LU.pair_picker(arms, plain, ["learn", "twin"], default_tp=tmp_path / "env_tp.pt")
    assert tp == tmp_path / "env_tp.pt" and "learn" in how                  # both played the .env default picker
    tp, _ = LU.pair_picker(arms, plain, ["own"], default_tp=None)
    assert Path(tp).name == "own_tp.pt"
    assert LU.pair_picker(arms, plain, ["first4"], default_tp=None)[0] is None   # the first-4 heuristic
    with pytest.raises(ValueError, match="DIFFERENT pickers"):
        LU.pair_picker(arms, plain, ["learn", "own"], default_tp=tmp_path / "env_tp.pt")
    with pytest.raises(ValueError, match="unknown"):
        LU.pair_picker(arms, plain, ["ghost"], default_tp=None)
    TPL.record_pair(base, side_tp)                                           # a verified pair beats the arms
    tp, how = LU.pair_picker(arms, base, ["learn", "own"], default_tp=None)
    assert Path(tp).resolve() == side_tp.resolve() and "sidecar" in how


def test_registration_and_the_env_deploy_carry_the_pair(tmp_path):
    from v_dance.ladder import ppo_update as PU
    cfg = tmp_path / "serve_bandit.json"
    cfg.write_text(json.dumps({"arms": [{"name": "era2", "battle_ckpt": "x.pt", "tp_ckpt": "default", "tau": 0.0,
                                         "incumbent": True}]}), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("VD_BATTLE_CKPT=old/battle_base.pt\nVD_TP_CKPT=old/tp.pt\n", encoding="utf-8")
    out = tmp_path / "cand"
    out.mkdir()
    ckpt = out / "battle_base.pt"
    ckpt.write_bytes(b"net")
    args = SimpleNamespace(run_gates=False, register=True, force_register=False, name=None, no_rotate=True,
                           twin=True, no_deploy_env=False, env_path=str(env), config=str(cfg))
    sel = SimpleNamespace(n_games=5, n_turn_steps=50, per_arm={"learn": 5}, tau=0.3)
    report = {"pair_picker": "pickers/paired_tp.pt", "kl_to_base_after": 0.01, "explained_variance": 0.5}
    rc = PU._gates_and_register(args, base=tmp_path / "base.pt", anchor=None, chain=False, arms_cfg={},
                                stamp="20261003", out_dir=out, ckpt=ckpt, report=report, sel=sel, ok=True, warns=[],
                                fails=[])
    assert rc == 0
    arms = {a["name"]: a for a in json.loads(cfg.read_text(encoding="utf-8"))["arms"]}
    assert arms["ppo_20261003"]["tp_ckpt"] == "pickers/paired_tp.pt"          # the learning / candidate arm
    twin = next(a for a in arms.values() if a.get("twin_of") == "ppo_20261003")
    assert twin["tp_ckpt"] == "pickers/paired_tp.pt"                          # and its argmax twin
    text = env.read_text(encoding="utf-8")
    assert "VD_TP_CKPT=pickers/paired_tp.pt" in text                           # the default stack is a pair too
    assert any(ln.startswith("VD_BATTLE_CKPT=") and "cand/battle_base.pt" in ln for ln in text.splitlines())


def test_a_first4_pair_never_touches_the_env(tmp_path):
    from v_dance.ladder import ppo_update as PU
    cfg = tmp_path / "serve_bandit.json"
    cfg.write_text(json.dumps({"arms": []}), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("VD_BATTLE_CKPT=old/battle_base.pt\n", encoding="utf-8")
    out = tmp_path / "cand"
    out.mkdir()
    ckpt = out / "battle_base.pt"
    ckpt.write_bytes(b"net")
    args = SimpleNamespace(run_gates=False, register=True, force_register=False, name=None, no_rotate=True,
                           twin=False, no_deploy_env=False, env_path=str(env), config=str(cfg))
    sel = SimpleNamespace(n_games=5, n_turn_steps=50, per_arm={"first4": 5}, tau=0.3)
    PU._gates_and_register(args, base=tmp_path / "b.pt", anchor=None, chain=False, arms_cfg={}, stamp="20261003",
                           out_dir=out, ckpt=ckpt, report={"pair_picker": "none", "kl_to_base_after": 0.0,
                                                           "explained_variance": 0.5},
                           sel=sel, ok=True, warns=[], fails=[])
    arms = json.loads(cfg.read_text(encoding="utf-8"))["arms"]
    assert arms[0]["tp_ckpt"] == "none"
    assert env.read_text(encoding="utf-8") == "VD_BATTLE_CKPT=old/battle_base.pt\n"   # .env untouched
