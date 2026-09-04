"""2026-09-04 (USER: "make a UI version of this command in Mission Control ... on execution have the config
swap out the ppo model for the latest one in .env"): the W3b chain update as a Mission Control card.

The card's defaults must build the exact verified command; the launch is refused while the online bot is up
(and the bot launch is refused while the chain job runs); the Jobs tab parses the script's report lines into
a phase + numbers; the status carries the learning arm next to .env VD_BATTLE_CKPT; the page ships the hooks.
Torch-free (Mission Control is stdlib-only by design)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_NO_CKPTS = {"battle": [], "tp": []}


def _mc():
    from v_dance.datatools import mission_control as mc
    return mc


def test_card_defaults_build_the_verified_command_and_the_choices_map_to_flags():
    mc = _mc()
    e = mc._REG_BY_ID["ladder_ppo"]
    assert e["cat"] == "train" and e["bot_down"] is True and not e.get("heavy")   # launchable, one click
    assert e["script"] == "scratch/ladder_ppo_update.py" and (_REPO / e["script"]).is_file()
    argv = mc._build_argv(e, {}, [], _NO_CKPTS)
    assert argv[0].endswith("python.exe") and argv[1:4] == ["-X", "utf8", "-u"]
    assert argv[4].endswith("ladder_ppo_update.py")
    # == .venv/Scripts/python.exe scratch/ladder_ppo_update.py --run-gates --register --base learning
    #    --days 1 --actor-lr 1e-3 --epochs 4   (the USER's 2026-09-04 command; --min-steps 200 = the default)
    #    + --twin (2026-09-04 "go build it": the argmax twin ppo_<date>_t0 registered with every step)
    assert argv[5:] == ["--base", "learning", "--days", "1.0", "--actor-lr", "0.001", "--epochs", "4",
                        "--min-steps", "200", "--run-gates", "--register", "--twin"]
    # 'incumbent' = the script's own default -> the flag is OMITTED; the optional flags appear when ticked
    argv = mc._build_argv(e, {"base": "incumbent", "register": False, "twin": False, "dry-run": True,
                              "no-deploy-env": True, "actor-lr": "0.0005", "days": "2"}, [], _NO_CKPTS)
    tail = argv[5:]
    assert "--base" not in tail and "--register" not in tail and "--twin" not in tail
    assert tail[:4] == ["--days", "2.0", "--actor-lr", "0.0005"]
    assert "--dry-run" in tail and "--no-deploy-env" in tail and "--run-gates" in tail
    with pytest.raises(ValueError, match="invalid choice"):
        mc._build_argv(e, {"base": "era2"}, [], _NO_CKPTS)
    pub = next(c for c in mc._registry_public() if c["id"] == "ladder_ppo")
    assert pub["bot_down"] is True and pub["opts"][0]["omit"] == ["incumbent"]    # the page mirrors both


def test_the_script_accepts_the_deploy_switch():
    """The CLI deploys a registered arm to .env by default; --no-deploy-env is the opt-out (hidden --env-path
    makes it testable)."""
    r = subprocess.run([sys.executable, str(_REPO / "scratch" / "ladder_ppo_update.py"), "--help"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(_REPO),
                       timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert "--no-deploy-env" in r.stdout and "--env-path" not in r.stdout     # hidden (tests only)
    assert "--base" in r.stdout and "--actor-lr" in r.stdout and "--epochs" in r.stdout


class _FakeJob:
    made: list = []

    def __init__(self, jid, entry_id, argv, env_extra, heavy=False):
        self.jid, self.entry_id, self.argv, self.alive, self.heavy = jid, entry_id, argv, True, heavy
        _FakeJob.made.append(self)

    def info(self):
        return {"job": self.jid, "id": self.entry_id, "alive": self.alive}


@pytest.fixture
def jobs(monkeypatch):
    mc = _mc()
    _FakeJob.made = []
    monkeypatch.setattr(mc, "_Job", _FakeJob)                     # never Popen anything from a test
    monkeypatch.setattr(mc, "_panel_status", lambda: {"up": False})
    return mc._Jobs()


def test_chain_launch_is_refused_while_the_online_bot_is_up(jobs, monkeypatch):
    mc = _mc()
    chain, bot = mc._REG_BY_ID["ladder_ppo"], mc._REG_BY_ID["play_online"]
    # (a) a live bot PANEL (launched from a terminal, not through MC) -> refused, nothing spawned
    monkeypatch.setattr(mc, "_panel_status", lambda: {"up": True, "port": 8777, "run": {}, "tally": {}})
    with pytest.raises(ValueError, match="needs the online bot DOWN.*8777"):
        jobs.start(chain, {}, {})
    assert _FakeJob.made == []
    # (b) a live bot JOB of this MC session (its panel not up yet) -> refused too
    monkeypatch.setattr(mc, "_panel_status", lambda: {"up": False})
    live = _FakeJob("j1", "play_online", ["x"], {})
    jobs._jobs["j1"] = live
    with pytest.raises(ValueError, match="needs the online bot DOWN.*job j1"):
        jobs.start(chain, {}, {})
    # (c) the bot down -> the chain job starts with the verified argv
    live.alive = False
    info = jobs.start(chain, {}, {})
    assert info["id"] == "ladder_ppo" and _FakeJob.made[-1].argv[5:7] == ["--base", "learning"]
    # (d) and while the chain job runs, launching the bot is refused (it rotates config + .env at the end)
    with pytest.raises(ValueError, match="still running"):
        jobs.start(bot, {}, {})
    _FakeJob.made[-1].alive = False
    assert jobs.start(bot, {}, {})["id"] == "play_online"


_LOG_OK = """[ladder-ppo] chain mode: base = the learning arm ppo_20260903b's checkpoint
[ladder-ppo] base E:\\repo\\ai_train_scripts\\BC_model\\checkpoints_attn_ladder_ppo_20260903b\\battle_base.pt
[ladder-ppo] selection
  files            : 3
  tau              : 0.3   candidates (tau: turn steps) 0.3: 1386
  games / steps    : 221 / 1782  (turn 1386, replacement 396, dedup-inexact 0, empty-slot->PASS repaired 0)
[ladder-ppo] recipe  {"tau": 0.3}
[ladder-ppo] update  steps 1782  epochs 3  loss -0.0123  ratio 1.000  approx_kl 0.0208  KL-to-base 0.0157  EV 0.845  pair_flips 0.01  halted True (approx_kl 0.0277 > 0.02 after epoch 3)  17.5 s
[ladder-ppo] candidate -> E:\\repo\\ai_train_scripts\\BC_model\\checkpoints_attn_ladder_ppo_20260904\\battle_base.pt
[ladder-ppo] warning: early stop: approx_kl 0.0277 > 0.02 after epoch 3
[ladder-ppo] ruler delta -0.02 pp (floor -0.5) -> ok;  type-eff True -> ok
[ladder-ppo] ruler vs anchor ai_train_scripts/BC_model/checkpoints_attn_era2/battle_base.pt: 0.1 pp (floor -1.0) -> ok
[ladder-ppo] suite: .venv/Scripts/python.exe -m pytest tests -q
[ladder-ppo] arm registered: {"name": "ppo_20260904", "battle_ckpt": "ai_train_scripts/BC_model/checkpoints_attn_ladder_ppo_20260904/battle_base.pt", "tau": 0.3}
[ladder-ppo] chain: previous learning arm(s) benched: ['ppo_20260903b']
[ladder-ppo] twin registered: {"name": "ppo_20260904_t0", "battle_ckpt": "ai_train_scripts/BC_model/checkpoints_attn_ladder_ppo_20260904/battle_base.pt", "tau": 0.0, "twin_of": "ppo_20260904"}
[ladder-ppo] twin: previous twin(s) benched: ['ppo_20260903b_t0']
[ladder-ppo] .env VD_BATTLE_CKPT -> ai_train_scripts/BC_model/checkpoints_attn_ladder_ppo_20260904/battle_base.pt (was ai_train_scripts/BC_model/checkpoints_attn_era4_2b/battle_base.pt)
[ladder-ppo] restart the bot (lanes) — the ladder decides (retire at ~40 g, promote at >= 200 g)
"""


def test_jobs_tab_progress_parses_the_scripts_report_lines(tmp_path):
    mc = _mc()
    log = tmp_path / "mc_ladder_ppo_x.log"
    log.write_text(_LOG_OK, encoding="utf-8")
    p = mc._job_progress("ladder_ppo", ["python", "scratch/ladder_ppo_update.py"], log)
    assert p["label"] == "phase" and p["value"] == "deployed" and p["pct"] == 100
    m = p["metrics"]
    assert m["games"] == "221" and m["turn steps"] == "1386"
    assert m["KL-to-base"] == "0.0157" and m["EV"] == "0.845"
    assert m["ruler pp"] == "-0.02 ok" and m["vs anchor pp"] == "0.1 ok" and m["type-eff"] == "ok"
    assert m["arm"] == "ppo_20260904" and m[".env VD_BATTLE_CKPT"] == "checkpoints_attn_ladder_ppo_20260904"
    assert m["twin"] == "ppo_20260904_t0"
    # mid-run: the furthest phase reached, a partial pct
    log.write_text(_LOG_OK.split("[ladder-ppo] candidate")[0], encoding="utf-8")
    p = mc._job_progress("ladder_ppo", [], log)
    assert p["value"] == "trained" and 0 < p["pct"] < 100 and "KL-to-base" in p["metrics"]
    # the three ways it stops short are named as the phase
    log.write_text("[ladder-ppo] selection\n  games / steps    : 3 / 20  (turn 12, x)\n"
                   "[ladder-ppo] REFUSED — 12 turn steps < --min-steps 200\n", encoding="utf-8")
    assert mc._job_progress("ladder_ppo", [], log)["value"].startswith("REFUSED")
    log.write_text(_LOG_OK.split("[ladder-ppo] arm registered")[0] +
                   "[ladder-ppo] GATE FAILED: ruler\n[ladder-ppo] NOT registered — a gate failed\n", encoding="utf-8")
    assert mc._job_progress("ladder_ppo", [], log)["value"].startswith("GATE FAILED")
    log.write_text("[ladder-ppo] selection\n[ladder-ppo] dry run — nothing trained\n", encoding="utf-8")
    assert mc._job_progress("ladder_ppo", [], log)["value"] == "dry run done"
    assert mc._job_progress("ladder_ppo", [], tmp_path / "missing.log") is None


def test_status_carries_the_learning_arm_next_to_the_env_default(tmp_path, monkeypatch):
    mc = _mc()
    cfg = tmp_path / "serve_bandit.json"
    cfg.write_text(json.dumps({"arms": [
        {"name": "era2", "battle_ckpt": "a.pt", "tau": 0, "incumbent": True},
        {"name": "ppo_20260904", "battle_ckpt": "ai_train_scripts/BC_model/checkpoints_attn_ladder_ppo_20260904/battle_base.pt",
         "tau": 0.3, "learning": True}]}), encoding="utf-8")
    la = mc._bandit_learning_arm(cfg)
    assert la == {"name": "ppo_20260904", "tau": 0.3,
                  "battle_ckpt": "ai_train_scripts/BC_model/checkpoints_attn_ladder_ppo_20260904/battle_base.pt"}
    cfg.write_text(json.dumps({"arms": [{"name": "era2", "battle_ckpt": "a.pt"}]}), encoding="utf-8")
    assert mc._bandit_learning_arm(cfg) is None                       # no learning arm
    assert mc._bandit_learning_arm(tmp_path / "absent.json") is None  # unreadable -> None, never a 500
    monkeypatch.setattr(mc, "_port_open", lambda port: False)
    monkeypatch.setattr(mc, "_panel_status", lambda: {"up": False})
    st = mc._status()
    assert "learning_arm" in st and st["bot_up"] is False
    if st["learning_arm"]:                                            # the live config: the chain head
        assert st["learning_arm"]["name"].startswith("ppo_") and st["learning_arm"]["tau"] > 0


def test_page_ships_the_card_hooks():
    mc = _mc()
    html = mc._HTML_PATH.read_text(encoding="utf-8")
    assert 'id="chain-status"' in html and "function renderChainStatus" in html
    assert "o.omit && o.omit.includes(v)" in html                    # Copy command omits 'incumbent' too
    assert "bot must be DOWN" in html and "W3b chain head (learning arm)" in html
    assert 'if (curTab === "train") renderChainStatus();' in html   # kept live by the 3 s poll
    assert "poke-env @a8810da" in html                              # the Deploy card's env pins line
