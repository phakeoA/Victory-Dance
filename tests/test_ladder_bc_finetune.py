"""W3b B4 (2026-09-04, USER: learning first) — the ladder-wins BC fine-tune script + its Mission Control card.

The plan must be the era-2 lineage recipe (docs/era3_kickoff_design.md Arm A) with the base as warm-start AND
advantage value head; the export is the STANDING two-pass Type-C pipeline; a same-day second run gets its own
stamp; the card's defaults build the verified command; the CLI answers --help and --dry-run (read-only) cleanly.
The real training smoke (`--smoke`, CPU, minutes) is opt-in: VD_BCFT_SMOKE=1."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scratch" / "ladder_bc_finetune.py"
_NO_CKPTS = {"battle": [], "tp": []}


def _mod():
    spec = importlib.util.spec_from_file_location("ladder_bc_finetune", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_plan_is_the_era2_lineage_recipe_off_the_base(tmp_path):
    m = _mod()
    base = tmp_path / "head.pt"
    argv = m.train_command(base, tmp_path / "out", typec=m.typec_folder("gen9championsvgc2026regmb"),
                           epochs=8, lr=1e-3, patience=3, device="cuda")
    assert argv[0] == sys.executable and argv[1:6] == ["-X", "utf8", "-u", "-m", "v_dance.training.train_bc"]
    data = argv[argv.index("--data") + 1: argv.index("--val-data")]
    assert len(data) == 5 and data[-1].endswith("Regulation_MB\\Jsonl_TypeC") or data[-1].endswith("Regulation_MB/Jsonl_TypeC")
    assert all("Jsonl_HF_OTS" in d for d in data[:4])
    val = argv[argv.index("--val-data") + 1: argv.index("--warm-start")]
    assert len(val) == 4 and all(("Jsonl_TypeA" in v or "Jsonl_TypeB" in v) for v in val)
    assert argv[argv.index("--warm-start") + 1] == str(base) == argv[argv.index("--adv-value-ckpt") + 1]
    for flag in ("--adv-weight", "--rating-weight", "--aux-opp-head", "--augment-move-order", "--mmap-cache"):
        assert flag in argv
    assert argv[argv.index("--adv-beta") + 1] == "1.6" and argv[argv.index("--d-model") + 1] == "256"
    assert argv[argv.index("--epochs") + 1] == "8" and argv[argv.index("--lr") + 1] == "0.001"
    assert argv[argv.index("--device") + 1] == "cuda" and argv[argv.index("--loader-workers") + 1] == "4"
    assert "--limit-files" not in argv
    # the smoke variant: file limit, no cache-backed loader flags
    smoke = m.train_command(base, tmp_path / "out", typec=tmp_path, epochs=1, lr=1e-3, patience=3, device="cpu",
                            limit_files=40)
    assert smoke[smoke.index("--limit-files") + 1] == "40" and "--mmap-cache" not in smoke and "--loader-workers" not in smoke


def test_export_is_the_two_pass_type_c_pipeline_and_the_folder_follows_the_format(tmp_path):
    m = _mod()
    cmds = m.export_commands(tmp_path / "Jsonl_TypeC")
    main, appr = cmds["main"], cmds["approved"]
    for c in (main, appr):
        assert c[4:6] == ["-m", "v_dance.datatools.bulk_parse_replays"] and "--winner-only" in c
        assert c[c.index("--type") + 1] == "C" and c[c.index("--output") + 1] == str(tmp_path / "Jsonl_TypeC")
    assert main[main.index("--input") + 1].endswith("Type_C") and "--rated-only" in main and "--overwrite" not in main
    assert appr[appr.index("--input") + 1].endswith("approved") and "--overwrite" in appr and "--rated-only" not in appr
    lim = m.export_commands(tmp_path, limit=5)
    assert lim["main"][-2:] == ["--limit", "5"]
    assert m.reg_folder("gen9championsvgc2026regmb") == "Regulation_MB"
    assert m.reg_folder("gen9championsvgc2026regmc") == "Regulation_MC"
    assert m.typec_folder("gen9championsvgc2026regmb").name == "Jsonl_TypeC"
    assert m.free_stamp("20260904", ["bcft_20260904", "era2"], ckpt_root=tmp_path) == "20260904b"
    (tmp_path / "checkpoints_attn_bcft_20260905").mkdir()
    assert m.free_stamp("20260905", [], ckpt_root=tmp_path) == "20260905b"
    assert m.free_stamp("20260906", [], ckpt_root=tmp_path) == "20260906"


def test_mission_control_card_defaults_and_progress_spec():
    from v_dance.datatools import mission_control as mc
    e = mc._REG_BY_ID["bc_finetune"]
    assert e["heavy"] is True and not e.get("bot_down") and (_REPO / e["script"]).is_file()
    argv = mc._build_argv(e, {}, [], _NO_CKPTS)
    assert argv[4].endswith("ladder_bc_finetune.py")
    assert argv[5:] == ["--base", "learning", "--epochs", "8", "--lr", "0.001", "--patience", "3", "--share", "0.15",
                        "--run-gates", "--register"]
    argv = mc._build_argv(e, {"base": "incumbent", "no-export": True, "dry-run": True, "register": False}, [], _NO_CKPTS)
    assert argv[5:7] == ["--base", "incumbent"] and "--no-export" in argv and "--dry-run" in argv and "--register" not in argv
    spec = mc._PROGRESS["bc_finetune"]
    assert spec["total_arg"] == "--epochs" and any(lbl == "val top1" for lbl, _ in spec["metrics"])


def test_cli_help_and_dry_run_are_clean():
    r = subprocess.run([sys.executable, str(_SCRIPT), "--help"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(_REPO), timeout=300)
    assert r.returncode == 0 and "--smoke" in r.stdout and "--run-gates" in r.stdout, r.stderr[-800:]
    if not (_REPO / "config" / "serve_bandit.json").is_file():
        pytest.skip("config/serve_bandit.json is local (gitignored)")
    r = subprocess.run([sys.executable, str(_SCRIPT), "--dry-run"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(_REPO), timeout=600)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-800:]
    assert "[bcft] plan (dry run" in r.stdout and "export main:" in r.stdout and "train:" in r.stdout
    assert "--warm-start" in r.stdout and "bcft_" in r.stdout


@pytest.mark.skipif(os.environ.get("VD_BCFT_SMOKE") != "1", reason="opt-in: VD_BCFT_SMOKE=1 trains 1 CPU epoch on 40 files")
def test_smoke_trains_one_cpu_epoch_and_the_candidate_loads():
    r = subprocess.run([sys.executable, str(_SCRIPT), "--smoke"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(_REPO), timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1500:]
    assert "[bcft] SMOKE OK" in r.stdout and "[bcft] candidate ->" in r.stdout
