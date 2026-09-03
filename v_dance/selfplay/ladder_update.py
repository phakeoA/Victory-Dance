"""Ladder PPO update — W3b-2 (2026-09-02, docs/w3b_ladder_ppo_design.md §5).

The learning half of the USER's live-training idea, as a NIGHTLY step the USER launches (box quiet):

    ladder (lanes, clean τ arms explore)  →  artifacts/ladder_rl/<fmt>/<session>.jsonl   (W3b-0 recorder)
        →  select: last N days · τ arms whose checkpoint IS the base · logprob_valid=true only
        →  ActorCritic.from_bc_checkpoint(base) · critic warm-up · rebase · ONE leashed PPO update
           (pair-mode evaluator = the served 2b decode, W3b-1b; KL-to-base 0.5; clip 0.1; ent 0.02)
        →  gates: KL-to-base ≤ 0.15 · critic EV not collapsed · (external) ruler within −0.5 pp ·
           type-eff RESPECTS · suite
        →  checkpoints_attn_ladder_ppo_<date>/battle_base.pt  +  bandit arm ppo_<date> (same τ) → the
           ladder decides (retire at ~40 g, promote = human at ≥200 g)

Everything the script decides is reported: per-arm / per-reason counts, the τ picked, the parity
switches, the loss stats, the gates. It REFUSES (exit 2) rather than trains on doubtful data: no
sampler log-prob (`logprob_valid=false`), an arm that played a different checkpoint, mixed pair
decodes, fewer than ``--min-steps`` turn steps. The CLI is ``scratch/ladder_ppo_update.py``.
"""
from __future__ import annotations

import calendar
import json
import logging
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
LADDER_RL_DIR = _REPO / "artifacts" / "ladder_rl"
DEFAULT_BANDIT_CONFIG = _REPO / "config" / "serve_bandit.json"
CKPT_ROOT = _REPO / "ai_train_scripts" / "BC_model"

# The recipe (design §5, revised from the ps-ppo review 2026-09-02): a SMALL leashed step.
RECIPE = {
    "clip": 0.1, "entropy": 0.02, "kl_coef": 0.5, "target_kl": 0.15, "approx_kl_stop": 0.02,
    "actor_lr": 3e-5, "critic_lr": 1e-3, "backbone_lr_scale": 0.1, "weight_decay": 1e-2,
    "epochs": 2, "minibatch": 256, "max_grad_norm": 0.5, "gamma": 0.997, "lam": 0.95,
    "warmup_updates": 1,
}
GATES = {"max_kl": 0.15, "min_ev": 0.0, "ruler_floor_pp": -0.5, "max_pair_flips": 0.05}


# ── selection ─────────────────────────────────────────────────────────────────
@dataclass
class Selection:
    trajectories: list
    tau: Optional[float] = None
    n_games: int = 0
    n_steps: int = 0
    n_turn_steps: int = 0
    n_replacement_steps: int = 0
    inexact_steps: int = 0
    per_arm: Dict[str, int] = field(default_factory=dict)        # games per selected arm
    skipped: Counter = field(default_factory=Counter)            # reason -> games
    files: List[str] = field(default_factory=list)
    tau_candidates: Dict[float, int] = field(default_factory=dict)   # τ -> turn steps seen (valid games)
    pair_decode: Optional[bool] = None                           # None = mixed / unknown
    gimmick_sampled: bool = False
    replacement_sampled: bool = False
    pass_repaired: int = 0                                       # empty-slot action 0 -> PASS (2026-09-03)

    def summary(self) -> dict:
        return {
            "files": list(self.files), "tau": self.tau, "games": self.n_games, "steps": self.n_steps,
            "turn_steps": self.n_turn_steps, "replacement_steps": self.n_replacement_steps,
            "inexact_steps": self.inexact_steps, "pass_repaired": self.pass_repaired,
            "per_arm": dict(self.per_arm),
            "skipped": dict(self.skipped), "tau_candidates": {str(k): v for k, v in self.tau_candidates.items()},
            "pair_decode": self.pair_decode, "gimmick_sampled": self.gimmick_sampled,
            "replacement_sampled": self.replacement_sampled,
        }


def _same_file(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a) == str(b)


def _recorded_at(sampling: dict) -> Optional[float]:
    s = (sampling or {}).get("recorded_at")
    if not s:
        return None
    try:
        return float(calendar.timegm(time.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ")))
    except Exception:
        return None


def arm_table(config_path=DEFAULT_BANDIT_CONFIG) -> dict:
    """``name -> Arm`` from the bandit config. Arms with a missing checkpoint are KEPT here (a
    dropped arm never played, so it cannot appear in the data; keeping it makes the report honest)."""
    from v_dance.play import serve_bandit as SB
    p = Path(config_path)
    if not p.is_file():
        return {}
    return {a.name: a for a in SB.load_arms(p, exists=lambda _p: True)}


def arm_plays_base(arm, base, *, default_battle=None) -> bool:
    """Whether ``arm``'s battle checkpoint IS ``base`` (one-step off-policy at most, design §5.1)."""
    from v_dance.play.serve_bandit import _resolve
    if arm.uses_default("battle"):
        return default_battle is not None and _same_file(default_battle, base)
    return _same_file(_resolve(arm.battle_ckpt), base)


def exploration_arms(arms: dict, base, *, default_battle=None) -> List[str]:
    """The τ > 0 arms that play ``base`` — the default ``--arms`` (design §1: only they produce
    on-policy, non-degenerate data)."""
    return [n for n, a in arms.items() if a.tau > 0.0 and arm_plays_base(a, base, default_battle=default_battle)]


def trajectory_files(root=LADDER_RL_DIR, fmt: Optional[str] = None, *, days: float = 1.0,
                     now: Optional[float] = None) -> List[Path]:
    """Session files touched inside the window (file mtime — a session file keeps growing while
    the bot runs; the per-game ``recorded_at`` window is applied in ``select_trajectories``)."""
    now = time.time() if now is None else float(now)
    d = Path(root) / fmt if fmt else Path(root)
    if not d.is_dir():
        return []
    cutoff = now - float(days) * 86400.0
    return sorted(p for p in d.glob("*.jsonl") if p.stat().st_mtime >= cutoff)


def repair_pass_slots(trajectories) -> int:
    """Recorder bug, 2026-09-03 (the first W3b update tripped ``assert_actions_legal``): before the
    fix the ladder recorder wrote the live player's EMPTY-slot encoding — action 0 with an all-zero
    effective mask (``_select_actions``: None -> 0, the slot passes) — where the self-play schema
    means ``PASS_ACTION`` (``game_runner.resolve_action``). Repair in place: a non-PASS action under
    an ABSENT or ALL-ZERO mask becomes ``PASS_ACTION`` (the evaluator then skips the slot, exactly
    as the sampler did — its term was never in the recorded joint log-prob, ``masked_sample_logp``
    returns None with no legal action). Returns the number of slots repaired. A non-PASS action
    that is illegal under a NON-EMPTY mask is NOT touched — that stays the corruption alarm."""
    from v_dance.selfplay.schema import PASS_ACTION
    n = 0
    for tr in trajectories:
        for t in tr.transitions:
            for a_attr, m_attr in (("action_s0", "mask_s0"), ("action_s1", "mask_s1")):
                a = getattr(t, a_attr)
                m = getattr(t, m_attr)
                if a != PASS_ACTION and (m is None or not any(m)):
                    setattr(t, a_attr, PASS_ACTION)
                    n += 1
    return n


def select_trajectories(files: Sequence, *, base, arms: Sequence[str], arm_table: dict,
                        days: Optional[float] = None, now: Optional[float] = None, tau="auto",
                        default_battle=None, expected_state_dim: Optional[int] = None) -> Selection:
    """Keep only what PPO may learn from (design §5.1): ladder games with a terminal outcome, played
    by one of ``arms``, whose checkpoint is ``base``, sealed ``logprob_valid=true``, inside the
    window; then ONE τ (``tau="auto"`` = the τ with the most turn steps) — the evaluator runs one
    temperature per update. Every rejection is counted by reason."""
    from v_dance.selfplay.store import iter_trajectories
    sel = Selection(trajectories=[])
    wanted = set(arms)
    cutoff = None if days is None else ((time.time() if now is None else float(now)) - float(days) * 86400.0)
    by_tau: Dict[float, list] = defaultdict(list)
    for f in files:
        sel.files.append(str(f))
        for tr in iter_trajectories(f, expected_state_dim=expected_state_dim):
            s = tr.meta.sampling or {}
            arm = s.get("arm")
            reason = None
            if not s:
                reason = "no sampling meta"
            elif s.get("source") not in (None, "ladder"):
                reason = f"source {s.get('source')}"
            elif not tr.meta.is_trainable:
                reason = "terminal FALLBACK"
            elif not tr.transitions:
                reason = "no steps"
            elif arm not in wanted:
                reason = f"arm not selected ({arm})"
            elif arm not in arm_table:
                reason = f"arm not in config ({arm})"
            elif not arm_plays_base(arm_table[arm], base, default_battle=default_battle):
                reason = f"arm {arm} plays another checkpoint"
            elif not s.get("logprob_valid"):
                reason = "logprob invalid: " + str(s.get("logprob_reason") or "placeholder (recorded before W3b-1a)")
            elif cutoff is not None and (_recorded_at(s) or 0.0) and _recorded_at(s) < cutoff:
                reason = "outside the window"
            if reason:
                sel.skipped[reason] += 1
                continue
            by_tau[round(float(s.get("tau", 0.0) or 0.0), 6)].append(tr)
    for t, trs in by_tau.items():
        sel.tau_candidates[t] = sum(1 for tr in trs for x in tr.transitions if x.decision_type == "turn")
    if not by_tau:
        return sel
    if tau == "auto":
        chosen = max(sel.tau_candidates.items(), key=lambda kv: (kv[1], kv[0]))[0]
    else:
        chosen = round(float(tau), 6)
        if chosen not in by_tau:
            sel.skipped[f"tau {chosen:g} has no valid games"] += 0
            for t, trs in by_tau.items():
                sel.skipped[f"tau {t:g} != requested {chosen:g}"] += len(trs)
            return sel
    for t, trs in sorted(by_tau.items()):
        if t != chosen:
            sel.skipped[f"tau {t:g} != chosen {chosen:g}"] += len(trs)
            continue
        sel.trajectories.extend(trs)
    sel.tau = chosen
    pair_values = set()
    for tr in sel.trajectories:
        s = tr.meta.sampling or {}
        sel.n_games += 1
        sel.per_arm[s.get("arm")] = sel.per_arm.get(s.get("arm"), 0) + 1
        sel.n_steps += len(tr.transitions)
        sel.n_turn_steps += sum(1 for x in tr.transitions if x.decision_type == "turn")
        sel.n_replacement_steps += sum(1 for x in tr.transitions if x.decision_type == "replacement")
        sel.inexact_steps += int(s.get("logprob_inexact_steps", 0) or 0)
        pair_values.add(bool(s.get("pair_decode", False)))
        sel.gimmick_sampled |= bool(s.get("gimmick_sampled", False))
        sel.replacement_sampled |= bool(s.get("replacement_sampled", False))
    sel.pair_decode = (pair_values.pop() if len(pair_values) == 1 else None)
    sel.pass_repaired = repair_pass_slots(sel.trajectories)    # 2026-09-03 recorder bug (see the helper)
    return sel


# ── configs ───────────────────────────────────────────────────────────────────
def build_configs(sel: Selection, **over):
    """``(PPOConfig, TrainConfig)`` for this batch: the recipe defaults (``RECIPE``) with keyword
    overrides, τ = the selection's, and the W3b-1b parity switches derived from the data (pair
    decode on when the games were served under it; gimmick / replacement terms only when those
    heads were SAMPLED — in serve they are argmax)."""
    from v_dance.selfplay.ppo import PPOConfig
    from v_dance.selfplay.trainer import TrainConfig
    if sel.tau is None or sel.tau <= 0.0:
        raise ValueError(f"no usable τ (selection tau={sel.tau}) — an argmax arm cannot drive PPO")
    if sel.pair_decode is None:
        raise ValueError("selected games mix pair-decode and independent serves — split them by arm")
    r = dict(RECIPE)
    r.update({k: v for k, v in over.items() if v is not None})
    ppo = PPOConfig(tau=float(sel.tau), clip_eps=float(r["clip"]), entropy_coef=float(r["entropy"]),
                    kl_coef=float(r["kl_coef"]), pair_decode=bool(sel.pair_decode),
                    gimmick_terms=bool(sel.gimmick_sampled),
                    replacement_policy=bool(sel.replacement_sampled))
    train = TrainConfig(actor_lr=float(r["actor_lr"]), critic_lr=float(r["critic_lr"]),
                        ppo_epochs=int(r["epochs"]), minibatch_size=int(r["minibatch"]),
                        max_grad_norm=float(r["max_grad_norm"]), gamma=float(r["gamma"]),
                        lam=float(r["lam"]), target_kl_from_bc=float(r["target_kl"]),
                        approx_kl_stop=(None if r["approx_kl_stop"] is None else float(r["approx_kl_stop"])),
                        actor_weight_decay=float(r["weight_decay"]),
                        backbone_lr_scale=float(r["backbone_lr_scale"]))
    return ppo, train


# ── the update ────────────────────────────────────────────────────────────────
def run_update(base, sel: Selection, ppo_cfg, train_cfg, *, device: str = "cpu", seed: int = 0,
               warmup_updates: int = 1):
    """Warm-start from ``base``, warm the critic, rebase the stored values, ONE leashed PPO update.
    Returns ``(actor_critic, report)``; the report carries every number the gates read."""
    from v_dance.selfplay.actor_critic import ActorCritic
    from v_dance.selfplay.policy_eval import assert_actions_legal
    from v_dance.selfplay.trainer import PPOTrainer
    ac = ActorCritic.from_bc_checkpoint(base, device=device)
    if ppo_cfg.pair_decode and not bool(getattr(ac.policy, "pair_cond", False)):
        raise ValueError("the games were served under the pair decode but the base is not a pair_cond checkpoint")
    trajs = sel.trajectories
    txns = [t for tr in trajs for t in tr.transitions]
    assert_actions_legal(txns)
    trainer = PPOTrainer(ac, ppo_cfg, train_cfg, device=device, seed=seed)
    kl_init = trainer._kl_from_bc(txns) if txns else float("nan")
    warm = trainer.warmup_critic(trajs, n_updates=int(warmup_updates))
    rebased = trainer.rebase_values(trajs)
    stats = trainer.ppo_update(trajs)
    kl_after = trainer._kl_from_bc(txns) if txns else float("nan")
    report = {
        "base": str(base), "device": device, "seed": int(seed),
        "n_games": sel.n_games, "n_steps": len(txns), "n_turn_steps": sel.n_turn_steps,
        "tau": sel.tau, "per_arm": dict(sel.per_arm),
        "ppo_config": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                       for k, v in vars(ppo_cfg).items()},
        "train_config": {k: v for k, v in vars(train_cfg).items()},
        "warmup": warm, "rebased": rebased,
        "kl_to_base_init": kl_init, "kl_to_base_after": kl_after,
        "update": stats, "explained_variance": stats.get("explained_variance"),
        "halted": bool(stats.get("halted")), "halt_reason": stats.get("halt_reason"),
        "pair_flips": stats.get("pair_flips"),
    }
    return ac, report


def gate(report: dict, *, max_kl: float = GATES["max_kl"], min_ev: float = GATES["min_ev"],
         max_pair_flips: float = GATES["max_pair_flips"]) -> Tuple[bool, List[str], List[str]]:
    """The in-process gates (design §5.3): ``(ok, failures, warnings)``. A benign approx-KL early
    stop is a warning; a KL-to-base overshoot, a collapsed critic or a non-finite update fail."""
    fails, warns = [], []
    kl = report.get("kl_to_base_after")
    if kl is None or not np.isfinite(kl):
        fails.append("KL to base is not finite")
    elif kl > max_kl:
        fails.append(f"KL to base {kl:.4f} > {max_kl}")
    ev = report.get("explained_variance")
    if ev is None or not np.isfinite(ev):
        fails.append("critic explained variance is not finite")
    elif ev < min_ev:
        fails.append(f"critic explained variance {ev:.3f} < {min_ev} (collapsed)")
    reason = str(report.get("halt_reason") or "")
    if report.get("halted"):
        if reason.startswith("approx_kl"):
            warns.append("early stop: " + reason)
        else:
            fails.append("update halted: " + reason)
    if int((report.get("update") or {}).get("nonfinite_skips", 0) or 0):
        fails.append("non-finite minibatches were skipped")
    pf = report.get("pair_flips")
    if pf is not None and pf > max_pair_flips:
        warns.append(f"pair-decode order flipped on {pf:.1%} of two-pick steps (parity drift)")
    return (not fails), fails, warns


# ── artefacts ─────────────────────────────────────────────────────────────────
def save_candidate(ac, out_dir, meta: dict) -> Path:
    """``<out_dir>/battle_base.pt`` (verified to reload through the BC loader — the same layout the
    bandit serves) + ``ladder_ppo_meta.json`` with the whole report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "battle_base.pt"
    ac.save(ckpt, verify=True)
    (out / "ladder_ppo_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return ckpt


def repo_relative(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(_REPO.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def register_arm(config_path, *, name: str, battle_ckpt, tau: float, note: str,
                 tp_ckpt: str = "default") -> dict:
    """Append arm ``name`` to the bandit config (same τ as the data, top-p 1.0, adapt-rules OFF —
    the next night's data from it stays clean). Refuses a duplicate name."""
    p = Path(config_path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    arms = cfg.setdefault("arms", [])
    if any(a.get("name") == name for a in arms):
        raise ValueError(f"arm {name!r} already exists in {p}")
    entry = {"name": name, "battle_ckpt": repo_relative(battle_ckpt), "tp_ckpt": tp_ckpt,
             "tau": float(tau), "top_p": 1.0, "tp_tie_eps": 1.0, "adapt_rules": False, "note": note}
    arms.append(entry)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return entry


# ── external gates (the ruler / type-eff probe / suite) ───────────────────────
def external_gate_commands(base, candidate, out_dir, *, py: str = ".venv/Scripts/python.exe") -> Dict[str, str]:
    """The three commands the design's gate list needs beyond the in-process numbers."""
    return {
        "ruler": f'{py} -m v_dance.eval.bc_val_report --ckpt "{base}" --ckpt "{candidate}"',
        "type_eff": f'{py} scratch/type_eff_probe.py --ckpt "{candidate}" --out "{Path(out_dir) / "type_eff_probe.json"}"',
        "suite": f"{py} -m pytest tests -q",
    }


_TOP1_ROW = re.compile(r"^\s*top1\s+([0-9.]+)\s+([0-9.]+)\s*\(([+-][0-9.]+)\)", re.M)


def parse_ruler_delta_pp(text: str) -> Optional[float]:
    """The candidate's pooled top-1 delta vs the base in percentage points, from the ruler's
    ``top1`` row (``BASE  CAND (delta)``); None when the row is absent."""
    m = _TOP1_ROW.search(text or "")
    return None if m is None else float(m.group(3)) * 100.0


def parse_type_eff(text: str) -> Optional[bool]:
    m = re.search(r"VERDICT:\s*(\S+)", text or "")
    return None if m is None else m.group(1).upper().startswith("RESPECTS")


def _first_line(text: str) -> str:
    for ln in (text or "").splitlines():
        if ln.strip():
            return ln.strip()[:160]
    return ""


def run_external_gates(base, candidate, out_dir, *, py: Optional[str] = None,
                       ruler_floor_pp: float = GATES["ruler_floor_pp"], run=None,
                       which: Sequence[str] = ("ruler", "type_eff")) -> dict:
    """Run the external gates (minutes each), keep their output under ``<out_dir>/gates/``, parse
    the verdicts. ``run`` is injectable (tests). Returns a dict with per-gate ok flags.

    2026-09-03 (the first ``--run-gates`` run): the commands go through ``cmd.exe`` on Windows, which
    cannot resolve the forward-slash ``.venv/Scripts/python.exe`` form ("'.venv' is not recognized as an
    internal or external command") — both gates returned None and the candidate was refused for the
    wrong reason. They now EXECUTE with this interpreter's absolute path (quoted); ``commands`` keeps the
    friendly form the user can paste. An unparsable gate reports its first output line as ``<name>_note``."""
    cmds = external_gate_commands(base, candidate, out_dir)              # the printed / recorded form
    exec_cmds = external_gate_commands(base, candidate, out_dir, py=(py or f'"{sys.executable}"'))
    gdir = Path(out_dir) / "gates"
    gdir.mkdir(parents=True, exist_ok=True)
    run = run or (lambda cmd: subprocess.run(cmd, shell=True, cwd=str(_REPO), capture_output=True,
                                             text=True, encoding="utf-8", errors="replace"))
    out: dict = {"commands": cmds}
    for name in which:
        r = run(exec_cmds[name])
        text = (getattr(r, "stdout", "") or "") + "\n" + (getattr(r, "stderr", "") or "")
        (gdir / f"{name}.txt").write_text(text, encoding="utf-8")
        parsed = None
        if name == "ruler":
            d = parse_ruler_delta_pp(text)
            out["ruler_delta_pp"] = parsed = d
            out["ruler_ok"] = (d is not None and d >= ruler_floor_pp)
        elif name == "type_eff":
            v = parse_type_eff(text)
            out["type_eff_verdict"] = parsed = v
            out["type_eff_ok"] = bool(v)
        out[f"{name}_returncode"] = int(getattr(r, "returncode", 0) or 0)
        if parsed is None:                       # the gate did not run / did not print its row
            out[f"{name}_note"] = f"gate output unparsable (rc {out[f'{name}_returncode']}): {_first_line(text)}"
    return out


# ── report text ───────────────────────────────────────────────────────────────
def format_selection(sel: Selection) -> str:
    # ASCII only: the USER's console (and this harness) is cp1252 — a Greek tau would crash the print.
    lines = [f"  files            : {len(sel.files)}",
             f"  tau              : {sel.tau if sel.tau is not None else '-'}   candidates (tau: turn steps) "
             + (", ".join(f"{k:g}: {v}" for k, v in sorted(sel.tau_candidates.items())) or "-"),
             f"  games / steps    : {sel.n_games} / {sel.n_steps}  (turn {sel.n_turn_steps}, replacement "
             f"{sel.n_replacement_steps}, dedup-inexact {sel.inexact_steps}, empty-slot->PASS repaired "
             f"{sel.pass_repaired})",
             f"  per arm          : " + (", ".join(f"{k}: {v}" for k, v in sorted(sel.per_arm.items())) or "-"),
             f"  parity switches  : pair_decode={sel.pair_decode} gimmick_terms={sel.gimmick_sampled} "
             f"replacement_policy={sel.replacement_sampled}"]
    if sel.skipped:
        lines.append("  skipped          :")
        for reason, n in sorted(sel.skipped.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {n:5d}  {reason}")
    return "\n".join(lines)
