"""15b-train.1 — the TP trainer builds + trains the SBDA net (feature recipe + self-attn +
teammate-bias), stamps feature_schema/feat_dim/arch into the checkpoint, and the result
round-trips through model_io (load past the lockstep guard + serve via team_order) — the full
train->save->load->serve loop. The legacy default stays the 46-dim net with no schema.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("poke_env")

import v_dance.training.train_teampreview as T   # noqa: E402
import v_dance.play.model_io as M                # noqa: E402
from v_dance.training import tp_features as TPF   # noqa: E402

# real M-B species so the belief + teammate-affinity are non-trivial
_POOL = ["Incineroar", "Tyranitar", "Garchomp", "Amoonguss", "Sinistcha", "Staraptor",
         "Excadrill", "Charizard", "Rillaboom", "Landorus", "Ogerpon", "Whimsicott"]


def _write_synthetic_replays(folder: Path, n: int = 4):
    folder.mkdir(parents=True, exist_ok=True)
    for r in range(n):
        our = _POOL[r:r + 6]
        opp = _POOL[r + 3:r + 9] if r + 9 <= len(_POOL) else _POOL[-6:]
        line = {"players": {"p1": {"roster": our, "brought": our[:4]},
                            "p2": {"roster": opp, "brought": opp[:4]}},
                "replay_id": f"r{r}"}
        (folder / f"r{r}.jsonl").write_text(json.dumps(line) + "\n", encoding="utf-8")


def _run(tmp_path: Path, extra):
    data = tmp_path / "data"
    _write_synthetic_replays(data)
    out = tmp_path / "ckpt"
    argv = ["--data", str(data), "--out", str(out), "--epochs", "1", "--batch-size", "4",
            "--hidden", "16", "--emb-dim", "8", "--attn-heads", "4", "--device", "cpu",
            "--val-frac", "0.25", "--seed", "0"] + extra
    T.train(T.parse_args(argv))
    return next(out.glob("teampreview_*.pt"))   # mode-dependent name: teampreview_sbda.pt | teampreview_base.pt


def test_train_sbda_checkpoint_roundtrips_through_serve(tmp_path):
    ckpt = _run(tmp_path, ["--features", "sbda", "--self-attn", "--teammate-bias",
                           "--format", "gen9championsvgc2026regmb"])
    assert ckpt.exists()
    model, vocab, cfg = M.load_team_chooser(ckpt)          # MUST pass the load-time lockstep guard
    assert M.uses_tp_features(cfg) is True
    assert cfg["feat_dim"] == TPF.FEAT_DIM
    assert cfg["feature_schema"] == TPF.FEATURE_SCHEMA_VERSION
    assert cfg["use_self_attn"] and cfg["use_teammate_bias"]
    assert model.use_self_attn and model.use_teammate_bias
    # serve it: team_order drives without crashing, with the active (regmb) belief
    from v_dance.parser.belief_state import BeliefState
    order = M.team_order(model, vocab, cfg, _POOL[:6], _POOL[6:12], n=4, belief=BeliefState())
    assert len(order) == 4 and len(set(order)) == 4 and all(0 <= i < 6 for i in order)


def test_train_teammate_bias_implies_self_attn(tmp_path):
    # --teammate-bias without --self-attn must auto-enable self-attn (the bias acts inside it)
    ckpt = _run(tmp_path, ["--features", "sbda", "--teammate-bias",
                           "--format", "gen9championsvgc2026regmb"])
    _model, _vocab, cfg = M.load_team_chooser(ckpt)
    assert cfg["use_self_attn"] is True and cfg["use_teammate_bias"] is True


def test_train_legacy_checkpoint_is_46dim_no_schema(tmp_path):
    ckpt = _run(tmp_path, [])                               # default --features legacy
    model, vocab, cfg = M.load_team_chooser(ckpt)
    assert M.uses_tp_features(cfg) is False
    assert cfg["feat_dim"] == 46 and "feature_schema" not in cfg
    order = M.team_order(model, vocab, cfg, _POOL[:6], _POOL[6:12], n=4)
    assert len(order) == 4 and len(set(order)) == 4
