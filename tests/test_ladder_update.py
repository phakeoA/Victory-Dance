"""W3b-2 (2026-09-02) — the nightly ladder PPO update (docs/w3b_ladder_ppo_design.md §5).

Drives the whole path on a tiny pair_cond checkpoint: games recorded the way the online bot records
them (the real serve sampler under the pair decode → behaviour log-prob + effective masks + decode
order), the selection rules (only clean, valid, on-base τ-arm games; one τ per update), the recipe
configs with the W3b-1b parity switches, one leashed update through the real trainer, the gates,
the saved candidate (loadable by the BC loader the bandit uses) and the arm registration. Also the
trainer's recipe options (AdamW, backbone LR scale, per-epoch approx-KL stop)."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import numpy as np
import torch

from v_dance.encoders.state_encoder import get_action_dim, get_state_dim
from v_dance.models.bc_model_attn import AttnBCPolicy
from v_dance.play import model_io as M
from v_dance.selfplay import ladder_update as LU
from v_dance.selfplay.actor_critic import ActorCritic, AttnCritic
from v_dance.selfplay.collector import TrajectoryCollector
from v_dance.selfplay.reward import place_terminal_reward
from v_dance.selfplay.schema import PASS_ACTION
from v_dance.selfplay.store import write_trajectories
from v_dance.selfplay.trainer import PPOTrainer, TrainConfig
from v_dance.selfplay.ppo import PPOConfig

A, S = get_action_dim(), get_state_dim()
HEADS = ("our_a", "our_b", "opp_a", "opp_b")
FMT = "gen9championsvgc2026regmb"
NOW = 1_800_000_000.0


def _base_ckpt(tmp_path: Path, seed: int = 3) -> Path:
    torch.manual_seed(seed)
    pol = AttnBCPolicy(d_model=32, n_heads=4, n_layers=1, dropout=0.0, heads=HEADS,
                       gimmick_heads=("our_a", "our_b"), pair_cond=True).eval()
    ac = ActorCritic(pol, AttnCritic(copy.deepcopy(pol)).eval(), pol.head_names, pol.gimmick_head_names, True)
    p = tmp_path / "base" / "battle_base.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    ac.save(p)
    return p


def _masks(rng):
    m = [False] * A
    for i in rng.choice(A, size=int(rng.integers(4, 12)), replace=False):
        m[int(i)] = True
    return m


def _game(policy, *, arm: str, tau: float, n: int, seed: int, won: bool, valid: bool = True,
          terminal: str = None, recorded_at: float = NOW, **over):
    """One ladder game as the recorder seals it — the real sampler under the pair decode."""
    rng = np.random.default_rng(seed)
    c = TrajectoryCollector(f"battle-{FMT}-{seed}", "p1")
    for turn in range(1, n + 1):
        x = rng.standard_normal(S).astype(np.float32)
        m0, m1 = _masks(rng), _masks(rng)
        M.LAST_DECODE.clear()
        a0, a1 = M.bc_action_indices(policy, policy.head_names, x, m0, m1, temperature=tau, rng=rng)
        rec = M.decode_record()
        lp = sum(float(t) for t in rec["logp"] if t is not None)
        c.add_step(state=x, action_s0=(PASS_ACTION if a0 is None else a0),
                   action_s1=(PASS_ACTION if a1 is None else a1), logprob=lp,
                   value=float(rng.uniform(-0.5, 0.5)), decision_type="turn", turn=turn,
                   mask_s0=list(rec["masks"][0]), mask_s1=list(rec["masks"][1]),
                   pair_first=rec["first"])
    sampling = {"source": "ladder", "session": "s", "fmt": FMT, "arm": arm, "pinned": False,
                "tau": tau, "top_p": 1.0, "pair_decode": True, "adapt_rules": False,
                "logprob_valid": valid, "logprob_reason": (None if valid else "placeholder"),
                "logprob_source": ("sampler" if valid else "placeholder"), "logprob_inexact_steps": 0,
                "gimmick_sampled": False, "replacement_sampled": False,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(recorded_at))}
    sampling.update(over)
    tr = c.finish(own_team=["A"] * 6, opp_team=["B"] * 6, tp_bring=[0, 1, 2, 3], tp_leads=[0, 1],
                  won=won, terminal_type=(terminal or ("win" if won else "loss")), n_turns=n,
                  sampling=sampling)
    if tr.meta.is_trainable:
        place_terminal_reward(tr)
    return tr


def _config(tmp_path: Path, base: Path, other: Path) -> Path:
    cfg = {"prior_games": 5, "prior_sd": 25, "min_games": 8, "retire_min_games": 40, "retire_margin": 0.1,
           "arms": [{"name": "inc", "incumbent": True, "battle_ckpt": str(base), "tau": 0},
                    {"name": "tau03", "battle_ckpt": str(base), "tau": 0.3, "top_p": 1.0, "adapt_rules": False},
                    {"name": "tau05", "battle_ckpt": str(base), "tau": 0.5, "top_p": 1.0, "adapt_rules": False},
                    {"name": "other03", "battle_ckpt": str(other), "tau": 0.3, "top_p": 1.0}]}
    p = tmp_path / "serve_bandit.json"
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ladder_update")
    base = _base_ckpt(tmp)
    other = tmp / "other" / "battle_base.pt"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"not a real checkpoint")
    cfg = _config(tmp, base, other)
    policy, _ = M.load_bc_policy(base)
    games = [_game(policy, arm="tau03", tau=0.3, n=5, seed=i, won=(i % 2 == 0)) for i in range(6)]
    games += [_game(policy, arm="tau05", tau=0.5, n=4, seed=100 + i, won=True) for i in range(2)]
    games += [_game(policy, arm="tau03", tau=0.3, n=3, seed=200, won=True, valid=False),     # pre-1a data
              _game(policy, arm="other03", tau=0.3, n=3, seed=201, won=True),               # off-base arm
              _game(policy, arm="inc", tau=0.0, n=3, seed=202, won=True, valid=False),      # argmax arm
              _game(policy, arm="tau03", tau=0.3, n=3, seed=203, won=True, terminal="fallback"),
              _game(policy, arm="tau03", tau=0.3, n=3, seed=204, won=True, recorded_at=NOW - 3 * 86400)]
    d = tmp / "ladder_rl" / FMT
    d.mkdir(parents=True)
    write_trajectories(d / "s1.jsonl", games)
    return SimpleNamespace(tmp=tmp, base=base, other=other, cfg=cfg, dir=d, policy=policy, games=games)


# ── selection ─────────────────────────────────────────────────────────────────
def test_selection_keeps_only_clean_on_base_valid_games_and_picks_one_tau(world):
    arms = LU.arm_table(world.cfg)
    assert set(arms) == {"inc", "tau03", "tau05", "other03"}
    assert LU.exploration_arms(arms, world.base) == ["tau03", "tau05"]     # τ>0 AND on the base
    files = LU.trajectory_files(world.tmp / "ladder_rl", FMT, days=1.0)
    assert len(files) == 1
    sel = LU.select_trajectories(files, base=world.base, arms=["tau03", "tau05"], arm_table=arms,
                                 days=1.0, now=NOW + 60, tau="auto", expected_state_dim=S)
    assert sel.tau == 0.3 and sel.n_games == 6 and sel.per_arm == {"tau03": 6}
    assert sel.n_turn_steps == 30 and sel.n_replacement_steps == 0
    assert sel.tau_candidates == {0.3: 30, 0.5: 8}
    sk = sel.skipped
    assert sk["tau 0.5 != chosen 0.3"] == 2
    assert sk["logprob invalid: placeholder"] == 1                        # the pre-W3b-1a game
    assert sk["arm not selected (other03)"] == 1 and sk["arm not selected (inc)"] == 1
    assert sk["terminal FALLBACK"] == 1 and sk["outside the window"] == 1
    assert sel.pair_decode is True and sel.gimmick_sampled is False and sel.replacement_sampled is False
    # asking for the other τ, or for an off-base arm, is refused for the right reason
    sel5 = LU.select_trajectories(files, base=world.base, arms=["tau03", "tau05"], arm_table=arms,
                                  now=NOW + 60, tau=0.5, expected_state_dim=S)
    assert sel5.tau == 0.5 and sel5.n_games == 2 and sel5.skipped["tau 0.3 != chosen 0.5"] == 7
    sel_off = LU.select_trajectories(files, base=world.base, arms=["other03"], arm_table=arms,
                                     now=NOW + 60, expected_state_dim=S)
    assert sel_off.n_games == 0 and sel_off.skipped["arm other03 plays another checkpoint"] == 1
    assert "skipped" in LU.format_selection(sel) and "tau03: 6" in LU.format_selection(sel)


def test_build_configs_applies_the_recipe_and_the_parity_switches(world):
    arms = LU.arm_table(world.cfg)
    sel = LU.select_trajectories([world.dir / "s1.jsonl"], base=world.base, arms=["tau03"], arm_table=arms,
                                 now=NOW + 60, expected_state_dim=S)
    ppo, train = LU.build_configs(sel, epochs=1, minibatch=0)
    assert ppo.tau == 0.3 and ppo.clip_eps == 0.1 and ppo.entropy_coef == 0.02 and ppo.kl_coef == 0.5
    assert ppo.pair_decode is True and ppo.gimmick_terms is False and ppo.replacement_policy is False
    assert train.ppo_epochs == 1 and train.minibatch_size == 0 and train.target_kl_from_bc == 0.15
    assert train.approx_kl_stop == 0.02 and train.actor_weight_decay == 1e-2 and train.backbone_lr_scale == 0.1
    assert train.gamma == 0.997 and train.lam == 0.95
    with pytest.raises(ValueError):                                       # an argmax arm cannot drive PPO
        LU.build_configs(LU.Selection(trajectories=[], tau=0.0, pair_decode=True))
    with pytest.raises(ValueError):                                       # mixed decodes
        LU.build_configs(LU.Selection(trajectories=[], tau=0.3, pair_decode=None))


# ── the update, the gates, the artefacts ──────────────────────────────────────
def test_run_update_gates_saves_a_loadable_candidate_and_registers_the_arm(world, tmp_path):
    arms = LU.arm_table(world.cfg)
    sel = LU.select_trajectories([world.dir / "s1.jsonl"], base=world.base, arms=["tau03"], arm_table=arms,
                                 days=1.0, now=NOW + 60, expected_state_dim=S)
    ppo, train = LU.build_configs(sel, epochs=2, minibatch=0, approx_kl_stop=None)
    ac, report = LU.run_update(world.base, sel, ppo, train, seed=1)
    assert report["n_games"] == 6 and report["n_steps"] == 30 and report["tau"] == 0.3
    assert abs(report["kl_to_base_init"]) < 1e-6                          # exactly the base before the step
    assert np.isfinite(report["kl_to_base_after"]) and report["update"]["epochs_run"] == 2
    assert report["pair_flips"] is not None and report["ppo_config"]["pair_decode"] is True
    ok, fails, warns = LU.gate(report, max_kl=10.0, min_ev=-10.0)
    assert ok and fails == []
    ok2, fails2, _ = LU.gate(dict(report, kl_to_base_after=1.0), max_kl=0.15, min_ev=-10.0)
    assert not ok2 and "KL to base" in fails2[0]
    _, _, w3 = LU.gate(dict(report, halted=True, halt_reason="approx_kl 0.03 > 0.02 after epoch 1"),
                       max_kl=10.0, min_ev=-10.0)
    assert any("early stop" in w for w in w3)
    out = tmp_path / "checkpoints_attn_ladder_ppo_test"
    ckpt = LU.save_candidate(ac, out, report)
    assert ckpt.is_file() and (out / "ladder_ppo_meta.json").is_file()
    cand, heads = M.load_bc_policy(ckpt)                                  # what the bandit will load
    assert getattr(cand, "pair_cond", False) and tuple(heads)[:2] == ("our_a", "our_b")
    meta = json.loads((out / "ladder_ppo_meta.json").read_text(encoding="utf-8"))
    assert meta["n_games"] == 6
    # the weights moved (a real update), the base did not
    base_pol, _ = M.load_bc_policy(world.base)
    diff = sum(float((p - q).abs().sum()) for p, q in zip(cand.parameters(), base_pol.parameters()))
    assert diff > 0.0
    # registration: same τ, clean knobs, repo-relative path when inside the repo, duplicate refused
    entry = LU.register_arm(world.cfg, name="ppo_test", battle_ckpt=ckpt, tau=sel.tau, note="n")
    assert entry["tau"] == 0.3 and entry["top_p"] == 1.0 and entry["adapt_rules"] is False
    assert [a["name"] for a in json.loads(world.cfg.read_text(encoding="utf-8"))["arms"]][-1] == "ppo_test"
    with pytest.raises(ValueError):
        LU.register_arm(world.cfg, name="ppo_test", battle_ckpt=ckpt, tau=0.3, note="dup")


def test_external_gate_parsing_and_runner(tmp_path):
    ruler = ("  [BASE] a\n  [CAND1] b\n"
             "                    BASE     CAND1 (delta)\n"
             "  top1           0.6155   0.6121 (-0.0034)   (n=1000)\n")
    assert LU.parse_ruler_delta_pp(ruler) == pytest.approx(-0.34)
    assert LU.parse_ruler_delta_pp("no rows") is None
    assert LU.parse_type_eff(">>> VERDICT: RESPECTS type-eff (graded): x") is True
    assert LU.parse_type_eff(">>> VERDICT: VIOLATES type-eff") is False and LU.parse_type_eff("") is None
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return SimpleNamespace(stdout=(ruler if "bc_val_report" in cmd else ">>> VERDICT: RESPECTS"),
                               stderr="", returncode=0)
    out = LU.run_external_gates("base.pt", "cand.pt", tmp_path, run=fake_run, ruler_floor_pp=-0.5)
    assert out["ruler_delta_pp"] == pytest.approx(-0.34) and out["ruler_ok"] and out["type_eff_ok"]
    assert (tmp_path / "gates" / "ruler.txt").is_file() and len(calls) == 2
    out2 = LU.run_external_gates("base.pt", "cand.pt", tmp_path, run=fake_run, ruler_floor_pp=-0.1)
    assert out2["ruler_ok"] is False                                      # −0.34 pp below a −0.1 floor
    cmds = LU.external_gate_commands("b.pt", "c.pt", tmp_path)
    assert "bc_val_report" in cmds["ruler"] and "type_eff_probe" in cmds["type_eff"] and "pytest" in cmds["suite"]
    # 2026-09-03: the gates EXECUTE with this interpreter's absolute path (cmd.exe cannot resolve
    # `.venv/Scripts/python.exe` — the first --run-gates run failed both gates with None), while the
    # recorded commands keep the pasteable form; an unparsable gate says why
    import sys
    assert all(c.startswith(f'"{sys.executable}"') for c in calls)
    assert out["commands"]["ruler"].startswith(".venv/Scripts/python.exe")
    dead = LU.run_external_gates("base.pt", "cand.pt", tmp_path, ruler_floor_pp=-0.5,
                                 run=lambda cmd: SimpleNamespace(stdout="", stderr="'.venv' is not recognized",
                                                                 returncode=1))
    assert dead["ruler_delta_pp"] is None and dead["ruler_ok"] is False and dead["type_eff_ok"] is False
    assert "not recognized" in dead["ruler_note"] and "rc 1" in dead["type_eff_note"]


# ── the trainer's recipe options ──────────────────────────────────────────────
def test_trainer_recipe_options_param_groups_adamw_and_approx_kl_stop(world):
    ac = ActorCritic.from_bc_checkpoint(world.base)
    t = PPOTrainer(ac, PPOConfig(), TrainConfig(actor_lr=1e-3, backbone_lr_scale=0.1, actor_weight_decay=1e-2))
    assert isinstance(t.actor_opt, torch.optim.AdamW) and len(t.actor_opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in t.actor_opt.param_groups)
    assert lrs == pytest.approx([1e-4, 1e-3])
    n_params = sum(len(g["params"]) for g in t.actor_opt.param_groups)
    assert n_params == len(ac.actor_parameters())                         # every actor param, once
    t.reset_optimizers()
    assert isinstance(t.actor_opt, torch.optim.AdamW) and len(t.actor_opt.param_groups) == 2
    t0 = PPOTrainer(ac, PPOConfig(), TrainConfig())
    assert type(t0.actor_opt) is torch.optim.Adam and len(t0.actor_opt.param_groups) == 1   # byte-identical default
    # approx-KL stop: a huge LR moves the policy → the epoch's approx KL trips the stop after epoch 1
    arms = LU.arm_table(world.cfg)
    sel = LU.select_trajectories([world.dir / "s1.jsonl"], base=world.base, arms=["tau03"], arm_table=arms,
                                 now=NOW + 60, expected_state_dim=S)
    ppo = PPOConfig(tau=0.3, pair_decode=True, gimmick_terms=False, replacement_policy=False, kl_coef=0.0)
    tr = PPOTrainer(ActorCritic.from_bc_checkpoint(world.base), ppo,
                    TrainConfig(actor_lr=5e-2, ppo_epochs=4, minibatch_size=0, approx_kl_stop=1e-6,
                                assert_value_space=False), seed=0)
    st = tr.ppo_update(sel.trajectories)
    assert st["halted"] and str(st["halt_reason"]).startswith("approx_kl") and st["epochs_run"] < 4


def test_selection_repairs_the_recorders_empty_slot_zero_into_pass(world):
    """2026-09-03: games recorded BEFORE the recorder fix carry action 0 under an all-zero mask for an
    empty / fainted slot (the first W3b update died in ``assert_actions_legal`` on transition 23,
    ~10 % of turn steps, every arm). The selector repairs them to PASS_ACTION, counts + reports them,
    and the legality guard then passes; an action illegal under a NON-EMPTY mask stays the alarm."""
    from v_dance.selfplay.policy_eval import assert_actions_legal
    g = _game(world.policy, arm="tau03", tau=0.3, n=4, seed=300, won=True)
    t = g.transitions[1]
    t.action_s1, t.mask_s1 = 0, [0] * A                                # what the old recorder wrote
    t2 = g.transitions[2]
    t2.action_s0, t2.mask_s0 = 0, None                                 # absent mask + action -> PASS too
    d = world.tmp / "ladder_rl_repair" / FMT
    d.mkdir(parents=True)
    write_trajectories(d / "s2.jsonl", [g])
    arms = LU.arm_table(world.cfg)
    with pytest.raises(AssertionError, match="illegal under"):        # the raw file is what tripped the run
        assert_actions_legal(g.transitions)
    sel = LU.select_trajectories([d / "s2.jsonl"], base=world.base, arms=["tau03"], arm_table=arms,
                                 now=NOW + 60, expected_state_dim=S)
    assert sel.n_games == 1 and sel.pass_repaired == 2
    tr = sel.trajectories[0]
    assert tr.transitions[1].action_s1 == PASS_ACTION and tr.transitions[1].action_s0 != PASS_ACTION
    assert tr.transitions[2].action_s0 == PASS_ACTION and tr.transitions[0].action_s0 != PASS_ACTION
    assert "empty-slot->PASS repaired 2" in LU.format_selection(sel)
    assert sel.summary()["pass_repaired"] == 2
    assert_actions_legal(tr.transitions)                               # the guard passes after the repair
    # a genuinely illegal action (NON-empty mask) is NOT repaired — still the corruption alarm
    bad = _game(world.policy, arm="tau03", tau=0.3, n=2, seed=301, won=True)
    bt = bad.transitions[0]
    m = [0] * A
    m[5] = 1
    bt.action_s0, bt.mask_s0 = 3, m
    write_trajectories(d / "s3.jsonl", [bad])
    sel_bad = LU.select_trajectories([d / "s3.jsonl"], base=world.base, arms=["tau03"], arm_table=arms,
                                     now=NOW + 60, expected_state_dim=S)
    assert sel_bad.n_games == 1 and sel_bad.pass_repaired == 0
    with pytest.raises(AssertionError, match="illegal under"):
        assert_actions_legal(sel_bad.trajectories[0].transitions)


def test_same_day_second_run_gets_a_suffixed_stamp(tmp_path):
    """2026-09-03: the evening chain update on the day ppo_20260903 was registered must not reuse its arm
    name (register_arm refuses AFTER training + gates) or overwrite its served checkpoint folder."""
    root = tmp_path / "ckpts"
    root.mkdir()
    assert LU.next_run_stamp("20260903", ["era2"], root) == "20260903"
    assert LU.next_run_stamp("20260903", ["era2", "ppo_20260903"], root) == "20260903b"
    (root / "checkpoints_attn_ladder_ppo_20260903b").mkdir()                 # folder taken, name free
    assert LU.next_run_stamp("20260903", ["era2", "ppo_20260903"], root) == "20260903c"
    assert LU.next_run_stamp("20260904", ["ppo_20260903", "ppo_20260903b"], root) == "20260904"


def test_chain_mode_learning_base_anchor_gate_and_rotation(world, tmp_path):
    """2026-09-03 L3: ``--base learning`` trains from the ONE learning arm's checkpoint; the ruler runs
    against the ANCHOR too (absolute floor); registering flags the new arm learning and benches the old."""
    import shutil
    cfg_p = tmp_path / "chain.json"
    shutil.copy(world.cfg, cfg_p)
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="exactly ONE learning arm"):
        LU.learning_base(LU.arm_table(cfg_p))                       # none flagged yet
    for a in cfg["arms"]:
        if a["name"] == "tau03":
            a["learning"] = True
    cfg_p.write_text(json.dumps(cfg), encoding="utf-8")
    arms = LU.arm_table(cfg_p)
    assert LU.learning_arms(arms) == ["tau03"]
    assert LU.learning_base(arms).resolve() == world.base.resolve()
    # the anchor gate: commands + parsing + the combined verdict
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        row = "  top1           0.6155   0.6121 (-0.0034)   (n=1000)\n" if "anchor.pt" not in cmd else \
              "  top1           0.6300   0.6100 (-0.0200)   (n=1000)\n"
        return SimpleNamespace(stdout=(row if "bc_val_report" in cmd else ">>> VERDICT: RESPECTS"),
                               stderr="", returncode=0)
    ext = LU.run_external_gates("base.pt", "cand.pt", tmp_path, run=fake_run, anchor="anchor.pt",
                                ruler_abs_floor_pp=-1.0)
    assert len(calls) == 3 and "anchor.pt" in ext["commands"]["ruler_anchor"]
    assert ext["ruler_anchor_delta_pp"] == pytest.approx(-2.0) and ext["ruler_anchor_ok"] is False
    assert ext["ruler_ok"] and ext["type_eff_ok"] and LU.external_ok(ext) is False   # the anchor floor bites
    ext2 = LU.run_external_gates("base.pt", "cand.pt", tmp_path, run=fake_run, anchor="anchor.pt",
                                 ruler_abs_floor_pp=-3.0)
    assert ext2["ruler_anchor_ok"] and LU.external_ok(ext2) is True
    assert LU.external_ok({"ruler_ok": True, "type_eff_ok": True}) is True   # no anchor = the old verdict
    # registration in chain mode: the new arm is the learning arm, the old one is benched + unflagged
    entry = LU.register_arm(cfg_p, name="ppo_x", battle_ckpt=world.base, tau=0.3, note="chain", learning=True)
    assert entry["learning"] is True
    assert LU.rotate_learning_arm(cfg_p, keep="ppo_x", stamp="20260904") == ["tau03"]
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    names = [a["name"] for a in cfg["arms"]]                        # (world.cfg may carry an arm an earlier test registered)
    assert "tau03" not in names and names[-1] == "ppo_x" and {"inc", "tau05", "other03"} <= set(names)
    old = next(a for a in cfg["benched"] if a["name"] == "tau03")
    assert old["learning"] is False and "superseded as the LEARNING arm by ppo_x" in old["benched_20260904"]
    assert LU.learning_arms(LU.arm_table(cfg_p)) == ["ppo_x"]
    assert LU.rotate_learning_arm(cfg_p, keep="ppo_x", stamp="20260905") == []   # idempotent


def test_twin_arm_registration_and_rotation(world, tmp_path):
    """2026-09-04 (design B3, USER "go build it"): ``--twin`` registers ``ppo_<stamp>_t0`` = the new head's checkpoint
    at τ 0 with the INCUMBENT's knobs (no adapt_rules key, not learning, retire rule applies), marked ``twin_of``;
    one twin at a time — the previous twin is benched with a dated note; the live loader accepts the entry."""
    import shutil
    from v_dance.play import serve_bandit as SB
    cfg_p = tmp_path / "twin.json"
    shutil.copy(world.cfg, cfg_p)
    head = LU.register_arm(cfg_p, name="ppo_20260904b", battle_ckpt=world.base, tau=0.3, note="head", learning=True)
    tw = LU.register_twin_arm(cfg_p, head=head["name"], battle_ckpt=world.base, stamp="20260904b")
    assert tw["name"] == "ppo_20260904b_t0" and tw["tau"] == 0.0 and tw["twin_of"] == "ppo_20260904b"
    assert tw["battle_ckpt"] == head["battle_ckpt"] and tw["top_p"] == 1.0 and tw["tp_tie_eps"] == 1.0
    assert "adapt_rules" not in tw and "learning" not in tw          # era2's knobs: launch default, retire rule applies
    with pytest.raises(ValueError, match="already exists"):
        LU.register_twin_arm(cfg_p, head="ppo_20260904b", battle_ckpt=world.base, stamp="20260904b")
    # the LIVE loader (what the bot runs at launch) accepts the extra key and reads the twin as a plain argmax arm
    loaded = {a.name: a for a in SB.load_arms(cfg_p, exists=lambda _p: True)}
    t = loaded["ppo_20260904b_t0"]
    assert t.tau == 0.0 and t.adapt_rules is None and t.learning is False and t.incumbent is False
    assert t.adapt_rules_for(True) is True and loaded["ppo_20260904b"].adapt_rules_for(True) is False
    assert LU.rotate_twin_arm(cfg_p, keep="ppo_20260904b_t0", stamp="20260904b") == []   # nothing to bench yet
    # the next step: a new head + twin; the old twin is benched (record kept by name), the new one stays
    LU.register_arm(cfg_p, name="ppo_20260905", battle_ckpt=world.base, tau=0.3, note="head 2", learning=True)
    assert LU.rotate_learning_arm(cfg_p, keep="ppo_20260905", stamp="20260905") == ["ppo_20260904b"]
    tw2 = LU.register_twin_arm(cfg_p, head="ppo_20260905", battle_ckpt=world.base, stamp="20260905")
    assert LU.rotate_twin_arm(cfg_p, keep=tw2["name"], stamp="20260905") == ["ppo_20260904b_t0"]
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    names = [a["name"] for a in cfg["arms"]]
    assert "ppo_20260905_t0" in names and "ppo_20260904b_t0" not in names and "ppo_20260904b" not in names
    old = next(a for a in cfg["benched"] if a["name"] == "ppo_20260904b_t0")
    assert "superseded as the argmax TWIN by ppo_20260905_t0" in old["benched_20260905"] and old["twin_of"] == "ppo_20260904b"
    assert LU.rotate_twin_arm(cfg_p, keep=tw2["name"], stamp="20260906") == []            # idempotent
    assert LU.learning_arms(LU.arm_table(cfg_p)) == ["ppo_20260905"]                    # the twin is never 'learning'


def test_deploy_env_battle_ckpt_swaps_the_key_and_keeps_one_rollback_line(tmp_path):
    """2026-09-04 (USER: "on execution have the config swap out the ppo model for the latest one in .env"):
    a registered chain head becomes .env VD_BATTLE_CKPT in place; the old value survives as ONE commented
    rollback line; nothing else moves; a current value is left alone; a missing key is appended."""
    env = tmp_path / ".env"
    env.write_text("# creds\nPS_USERNAME=bot\n# deploy defaults\n"
                   "VD_BATTLE_CKPT=ai_train_scripts/BC_model/checkpoints_attn_era4_2b/battle_base.pt\n"
                   "VD_TP_CKPT=tp.pt\nVD_SERVE_TAU=0\n", encoding="utf-8")
    ck1 = LU._REPO / "ai_train_scripts" / "BC_model" / "checkpoints_attn_ladder_ppo_20260904" / "battle_base.pt"
    r = LU.deploy_env_battle_ckpt(env, ck1, stamp="20260904", arm="ppo_20260904")
    assert r["changed"] and r["old"].endswith("checkpoints_attn_era4_2b/battle_base.pt")
    assert r["new"] == "ai_train_scripts/BC_model/checkpoints_attn_ladder_ppo_20260904/battle_base.pt"
    lines = env.read_text(encoding="utf-8").splitlines()
    assert [ln for ln in lines if ln.startswith("VD_BATTLE_CKPT=")] == ["VD_BATTLE_CKPT=" + r["new"]]
    assert lines[:3] == ["# creds", "PS_USERNAME=bot", "# deploy defaults"]
    assert lines[-2:] == ["VD_TP_CKPT=tp.pt", "VD_SERVE_TAU=0"]
    assert lines.index("VD_BATTLE_CKPT=" + r["new"]) == 5          # in place: two tag lines above it
    assert lines[4] == "# [ladder-ppo] VD_BATTLE_CKPT=" + r["old"]
    assert lines[3].startswith("# [ladder-ppo] 20") and "ppo_20260904" in lines[3] and "ROLLBACK" in lines[3]
    # the next night: ONE rollback line (last night's head); the older tag lines are gone
    ck2 = LU._REPO / "ai_train_scripts" / "BC_model" / "checkpoints_attn_ladder_ppo_20260905" / "battle_base.pt"
    r2 = LU.deploy_env_battle_ckpt(env, ck2, stamp="20260905", arm="ppo_20260905")
    lines = env.read_text(encoding="utf-8").splitlines()
    tags = [ln for ln in lines if ln.startswith("# [ladder-ppo]")]
    assert r2["old"] == r["new"] and len(tags) == 2 and tags[1] == "# [ladder-ppo] VD_BATTLE_CKPT=" + r["new"]
    assert [ln for ln in lines if ln.startswith("VD_BATTLE_CKPT=")] == ["VD_BATTLE_CKPT=" + r2["new"]]
    assert len(lines) == 8
    # already current: nothing written
    before = env.read_text(encoding="utf-8")
    assert LU.deploy_env_battle_ckpt(env, ck2)["changed"] is False
    assert env.read_text(encoding="utf-8") == before
    # no key at all: appended (a tag line + the key), nothing to roll back
    env2 = tmp_path / "bare.env"
    env2.write_text("PS_USERNAME=bot\n", encoding="utf-8")
    r3 = LU.deploy_env_battle_ckpt(env2, ck1)
    assert r3["old"] is None and env2.read_text(encoding="utf-8").splitlines()[-1] == "VD_BATTLE_CKPT=" + r["new"]
    assert not any("ROLLBACK" in ln for ln in env2.read_text(encoding="utf-8").splitlines())
    # the chain head reader (Mission Control shows it next to the .env default)
    cfg = {"arms": [{"name": "era2", "battle_ckpt": "a.pt", "tau": 0, "incumbent": True},
                    {"name": "ppo_20260904", "battle_ckpt": "b.pt", "tau": 0.3, "learning": True}]}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    assert LU.learning_head(LU.arm_table(p)) == {"name": "ppo_20260904", "battle_ckpt": "b.pt", "tau": 0.3}
    cfg["arms"][1]["learning"] = False
    p.write_text(json.dumps(cfg), encoding="utf-8")
    assert LU.learning_head(LU.arm_table(p)) is None
