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
    # 2026-09-04 B2: Elo per DOUBLING of a game's advantage weight, centred on the batch's mean opponent
    # rating (see opp_rating_weights); 0 = every game weighs 1.
    "opp_weight_scale": 400.0,
}
GATES = {"max_kl": 0.15, "min_ev": 0.0, "ruler_floor_pp": -0.5, "max_pair_flips": 0.05,
         # 2026-09-03 L3 (chain mode): the ruler vs the ANCHOR (the incumbent) — an absolute floor so a
         # nightly chain of −0.4 pp steps cannot drift away from era2 unnoticed
         "ruler_abs_floor_pp": -1.0}


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


def learning_arms(arms: dict) -> List[str]:
    """The arm(s) flagged ``learning`` (the adaptive τ arm the nightly update trains from)."""
    return [n for n, a in arms.items() if getattr(a, "learning", False)]


def learning_base(arms: dict, *, default_battle=None) -> Path:
    """2026-09-03 L3 (chain mode, ``--base learning``): the checkpoint the ONE learning arm plays — the
    nightly update trains FROM it on ITS games, so the policy compounds night over night instead of
    restarting from the incumbent. Exactly one learning arm is required (the chain has one head)."""
    from v_dance.play.serve_bandit import _resolve
    names = learning_arms(arms)
    if len(names) != 1:
        raise ValueError("chain mode needs exactly ONE learning arm in the bandit config, found "
                         f"{names or 'none'} — flag one with \"learning\": true")
    a = arms[names[0]]
    if a.uses_default("battle"):
        if default_battle is None:
            raise ValueError(f"learning arm {a.name!r} plays the deployed default checkpoint — pass --base explicitly")
        return Path(default_battle)
    return _resolve(a.battle_ckpt)


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


# ── B2: opponent-rating weights (2026-09-04, USER: learning first) ───────────
def opp_rating_weights(sel: Selection, *, scale: float = RECIPE["opp_weight_scale"], lo: float = 0.5,
                       hi: float = 2.0):
    """One advantage weight per selected game from the OPPONENT's pre-game rating (the recorder's
    ``sampling.opp_rating_before``): ``2 ** ((opp - mean_opp) / scale)`` clipped to ``[lo, hi]``, centred on
    the batch mean so the weights average about 1 (no change to the effective learning rate). Why: with the
    learning arm at 70 % of the games the rating sits in a low band, and a policy trained on beating 1300s is
    not trained to beat 1700s — games against stronger opponents should count more. Unknown ratings weigh 1;
    ``scale`` <= 0 = off (all ones). Returns ``(weights, info)``."""
    ratings = []
    for tr in sel.trajectories:
        r = (tr.meta.sampling or {}).get("opp_rating_before")
        try:
            ratings.append(None if r is None else float(r))
        except (TypeError, ValueError):
            ratings.append(None)
    known = [r for r in ratings if r is not None]
    info = {"scale": float(scale or 0.0), "lo": lo, "hi": hi, "n_known": len(known),
            "n_unknown": len(ratings) - len(known),
            "mean_opp_rating": (round(float(np.mean(known)), 1) if known else None)}
    if not scale or float(scale) <= 0.0 or not known:
        w = [1.0] * len(ratings)
    else:
        m = float(np.mean(known))
        w = [1.0 if r is None else float(min(hi, max(lo, 2.0 ** ((r - m) / float(scale))))) for r in ratings]
    if w:
        info.update({"w_mean": round(float(np.mean(w)), 3), "w_min": round(float(min(w)), 3),
                     "w_max": round(float(max(w)), 3)})
    return w, info


# ── the update ────────────────────────────────────────────────────────────────
def run_update(base, sel: Selection, ppo_cfg, train_cfg, *, device: str = "cpu", seed: int = 0,
               warmup_updates: int = 1, traj_weights=None):
    """Warm-start from ``base``, warm the critic, rebase the stored values, ONE leashed PPO update.
    Returns ``(actor_critic, report)``; the report carries every number the gates read.
    ``traj_weights`` (B2): one advantage weight per selected trajectory, see :func:`opp_rating_weights`."""
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
    stats = trainer.ppo_update(trajs, traj_weights=traj_weights)
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


def next_run_stamp(stamp: str, arm_names, ckpt_root=CKPT_ROOT) -> str:
    """2026-09-03: a SECOND update on the same date must not reuse the day's arm name (``register_arm``
    refuses a duplicate — after the training and the gates) nor its candidate folder (``save_candidate``
    would overwrite the checkpoint the served arm is playing). Returns ``stamp`` when both
    ``ppo_<stamp>`` and ``checkpoints_attn_ladder_ppo_<stamp>`` are free, else the first free suffixed
    stamp: ``<stamp>b``, ``<stamp>c``, …"""
    names = set(arm_names or ())
    root = Path(ckpt_root)
    for suffix in [""] + [chr(c) for c in range(ord("b"), ord("z") + 1)]:
        cand = f"{stamp}{suffix}"
        if f"ppo_{cand}" not in names and not (root / f"checkpoints_attn_ladder_ppo_{cand}").exists():
            return cand
    raise ValueError(f"no free run stamp for {stamp} (26 same-day runs?)")


_WARM_NOTE = (" Bandit prior warm-started from {src} (min(its games, warm_start_games) pseudo-games at its mean, "
              "Thompson only; each real game displaces one, so the prior is gone at warm_start_games games; "
              "the record itself starts at 0).")


def register_arm(config_path, *, name: str, battle_ckpt, tau: float, note: str,
                 tp_ckpt: str = "default", learning: bool = False, prior_from: Optional[str] = None,
                 adapt_rules: Optional[bool] = False, share: Optional[float] = None) -> dict:
    """Append arm ``name`` to the bandit config (same τ as the data, top-p 1.0, adapt-rules OFF —
    the next night's data from it stays clean). Refuses a duplicate name. ``learning=True`` (L3 chain
    mode) flags it as the learning arm: retire-exempt, the L2 floor share, tomorrow's ``--base learning``.
    ``prior_from`` (2026-09-04) names the predecessor whose ladder record seeds this arm's Thompson prior
    (``serve_bandit.Arm.prior_from``) — in chain mode the head it was trained from."""
    p = Path(config_path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    arms = cfg.setdefault("arms", [])
    if any(a.get("name") == name for a in arms):
        raise ValueError(f"arm {name!r} already exists in {p}")
    entry = {"name": name, "battle_ckpt": repo_relative(battle_ckpt), "tp_ckpt": tp_ckpt,
             "tau": float(tau), "top_p": 1.0, "tp_tie_eps": 1.0, "note": note}
    if adapt_rules is not None:                 # None (B4 argmax candidates) = the launch default, like era2
        entry["adapt_rules"] = bool(adapt_rules)
    if share is not None:                       # B1 fixed share (a B4 candidate needs one to get games at all)
        entry["share"] = float(share)
    if learning:
        entry["learning"] = True
    if prior_from:
        entry["prior_from"] = str(prior_from)
        entry["note"] = note + _WARM_NOTE.format(src=prior_from)
    arms.append(entry)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return entry


def rotate_learning_arm(config_path, *, keep: str, stamp: str) -> List[str]:
    """L3 chain mode: after ``keep`` is registered as the new learning arm, every OTHER learning arm is
    un-flagged and BENCHED (moved to ``benched`` with a dated note) — the chain has one head, and a
    superseded τ 0.3 snapshot kept in the rotation would only bleed rating (USER ruling 2026-09-03:
    bench superseded snapshots of one lineage). Its ladder record stays in the bandit state file.
    Returns the benched names."""
    p = Path(config_path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    arms = cfg.setdefault("arms", [])
    bench = cfg.setdefault("benched", [])
    out: List[str] = []
    for a in list(arms):
        if a.get("learning") and a.get("name") != keep:
            a["learning"] = False
            a[f"benched_{stamp}"] = (f"superseded as the LEARNING arm by {keep} (W3b chain: the newest PPO "
                                    f"checkpoint at tau {a.get('tau', 0)} is the learning arm)")
            arms.remove(a)
            bench.append(a)
            out.append(str(a.get("name")))
    if out:
        p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


# ── the argmax TWIN (2026-09-04, design B3; USER "go build it") ───────────────────────────────
def twin_name(head: str) -> str:
    return f"{head}_t0"


def register_twin_arm(config_path, *, head: str, battle_ckpt, stamp: str, tp_ckpt: str = "default",
                      prior_from: Optional[str] = "auto", share="auto") -> dict:
    """Register ``ppo_<stamp>_t0`` = the chain head's checkpoint at τ 0 (argmax), served NEXT to the incumbent.
    The learning arm's own ladder record always carries the τ-0.3 sampling cost (≈ −3 Elo/g), so it can never
    say whether the WEIGHTS beat era2 — the twin can. Same knobs as the incumbent (top-p 1.0, tie-eps 1.0, TP
    default, adapt-rules = the launch default — no key, exactly like era2's entry) so the comparison is apples
    to apples; NOT learning; promotion stays human.
    ``twin_of`` marks it for :func:`rotate_twin_arm` (``load_arms`` ignores unknown keys — verified 2026-09-04).
    ``prior_from`` "auto" (default) = the twin currently served (the one :func:`rotate_twin_arm` will bench) seeds the
    new twin's Thompson prior; None = no warm-start; a name = that arm.
    ``share`` (2026-09-05) "auto" (default) = the FIXED share of the twin being replaced, else the config's
    ``twin_share``; a number sets it; None = no fixed share. The 09-05 twin was registered without one while the
    other shares summed to 1 (Thompson's share 0 %), so it played its 8 warm-up games and never again — the 200-g
    read needs the share, which also makes the twin retire-EXEMPT (B1: a human benches a bad instrument)."""
    p = Path(config_path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    arms = cfg.setdefault("arms", [])
    name = twin_name(f"ppo_{stamp}") if head == f"ppo_{stamp}" else twin_name(head)
    if any(a.get("name") == name for a in arms):
        raise ValueError(f"arm {name!r} already exists in {p}")
    prev_twins = [a for a in arms if a.get("twin_of") and a.get("name") != name]
    if prior_from == "auto":
        prior_from = prev_twins[-1].get("name") if prev_twins else None
    if share == "auto":
        inherited = prev_twins[-1].get("share") if prev_twins else None
        share = inherited if inherited is not None else cfg.get("twin_share")
    entry = {"name": name, "battle_ckpt": repo_relative(battle_ckpt), "tp_ckpt": tp_ckpt, "tau": 0.0,
             "top_p": 1.0, "tp_tie_eps": 1.0,
             "note": (f"argmax TWIN of {head} (W3b B3, {stamp}): the same checkpoint at tau 0 next to the incumbent "
                      "- judges the chain's WEIGHTS on the ladder (the learning arm's record carries the tau-0.3 "
                      "sampling cost). Promotion = human (swap 'incumbent')."),
             "twin_of": head}
    if share is not None:
        entry["share"] = float(share)
        entry["note"] += (f" Fixed share {float(share):.0%} (retire-exempt, B1) so the 200-g read happens even when "
                          "the other shares sum to 1; a human benches a bad twin.")
    else:
        entry["note"] += " Retire rule applies."
    if prior_from:
        entry["prior_from"] = str(prior_from)
        entry["note"] += _WARM_NOTE.format(src=prior_from)
    arms.append(entry)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return entry


def rotate_twin_arm(config_path, *, keep: str, stamp: str) -> List[str]:
    """One twin at a time: every OTHER arm carrying ``twin_of`` is BENCHED with a dated note (its ladder record
    stays in the bandit state under its name, so the series of twins reads as the chain's trend). Returns the
    benched names; [] when nothing moved."""
    p = Path(config_path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    arms = cfg.setdefault("arms", [])
    bench = cfg.setdefault("benched", [])
    out: List[str] = []
    for a in list(arms):
        if a.get("twin_of") and a.get("name") != keep:
            a[f"benched_{stamp}"] = (f"superseded as the argmax TWIN by {keep} (one twin at a time; the record "
                                    f"of this one — head {a.get('twin_of')} — stays in artifacts/bandit)")
            arms.remove(a)
            bench.append(a)
            out.append(str(a.get("name")))
    if out:
        p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


# ── .env deploy (2026-09-04, USER: "on execution have the config swap out the ppo model for the
# latest one in .env") ─────────────────────────────────────────────────────────────────────────
ENV_BATTLE_KEY = "VD_BATTLE_CKPT"
_ENV_DEPLOY_TAG = "# [ladder-ppo]"          # every line this helper writes starts with this tag


def deploy_env_battle_ckpt(env_path, ckpt, *, key: str = ENV_BATTLE_KEY, stamp: Optional[str] = None,
                           arm: Optional[str] = None) -> dict:
    """Point ``.env`` ``key`` (the deployed battle net every harness loads at launch) at ``ckpt`` — the
    chain head that ``--register`` just made the learning arm — so the newest PPO checkpoint is ALSO the
    ``.env`` default (bandit-off serving, the websocket ladder, the launch echo, Mission Control's
    Deployed-models card). Bandit arms carry explicit checkpoints, so per-game serving is unchanged.

    Keeps the file's own convention: the previous value survives as ONE commented rollback line right
    above the key (older ``[ladder-ppo]`` lines are dropped, so nightly runs do not pile up). Atomic
    tmp-replace like ``mission_control._env_write``. Returns ``{key, old, new, changed}``; a value
    that is already current is left alone (``changed`` False, nothing written)."""
    env_path = Path(env_path)
    new = repo_relative(ckpt)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    old: Optional[str] = None
    idx: Optional[int] = None
    for i, ln in enumerate(lines):
        if ln.split("=", 1)[0].strip() == key and not ln.lstrip().startswith("#"):
            old = ln.split("=", 1)[1].strip()
            idx = i
            break
    if old == new:
        return {"key": key, "old": old, "new": new, "changed": False}
    kept = [ln for ln in lines if not ln.startswith(_ENV_DEPLOY_TAG)]      # one rollback line, not N
    if idx is not None:
        idx = next(i for i, ln in enumerate(kept)
                   if ln.split("=", 1)[0].strip() == key and not ln.lstrip().startswith("#"))
    when = time.strftime("%Y-%m-%d %H:%M")
    head = (f"{_ENV_DEPLOY_TAG} {when}: {key} -> {arm or Path(new).parent.name}"
            + (f" (run {stamp})" if stamp else "")
            + (f"; ROLLBACK = the commented line below (was {old})" if old else ""))
    block = [head] + ([f"{_ENV_DEPLOY_TAG} {key}={old}"] if old else []) + [f"{key}={new}"]
    if idx is None:
        kept += block
    else:
        kept[idx:idx + 1] = block
    tmp = env_path.with_suffix(env_path.suffix + ".tmp")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    tmp.replace(env_path)
    return {"key": key, "old": old, "new": new, "changed": True}


def learning_head(arms: dict) -> Optional[dict]:
    """``{name, battle_ckpt, tau}`` of the chain head (the ``learning: true`` arm), or None."""
    names = learning_arms(arms)
    if not names:
        return None
    a = arms[names[0]]
    return {"name": names[0], "battle_ckpt": getattr(a, "battle_ckpt", None), "tau": float(getattr(a, "tau", 0.0) or 0.0)}


# ── external gates (the ruler / type-eff probe / suite) ───────────────────────
def external_gate_commands(base, candidate, out_dir, *, py: str = ".venv/Scripts/python.exe",
                           anchor=None) -> Dict[str, str]:
    """The three commands the design's gate list needs beyond the in-process numbers (+ the ruler vs
    the ANCHOR checkpoint in chain mode — the absolute floor, 2026-09-03 L3)."""
    out = {
        "ruler": f'{py} -m v_dance.eval.bc_val_report --ckpt "{base}" --ckpt "{candidate}"',
        "type_eff": f'{py} scratch/type_eff_probe.py --ckpt "{candidate}" --out "{Path(out_dir) / "type_eff_probe.json"}"',
        "suite": f"{py} -m pytest tests -q",
    }
    if anchor is not None:
        out["ruler_anchor"] = f'{py} -m v_dance.eval.bc_val_report --ckpt "{anchor}" --ckpt "{candidate}"'
    return out


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


def external_ok(ext: dict) -> bool:
    """Every external gate that ran passed (the anchor gate only counts when it ran)."""
    ok = bool(ext.get("ruler_ok")) and bool(ext.get("type_eff_ok"))
    if "ruler_anchor_ok" in ext:
        ok = ok and bool(ext.get("ruler_anchor_ok"))
    return ok


_OOM_SIGNS = ("Unable to allocate", "_ArrayMemoryError", "MemoryError", "CUDA out of memory", "out of memory")
GATE_OOM_RETRY_WAIT_S = 20.0


def looks_like_oom(text: str) -> bool:
    """2026-09-05: the era2 anchor gate's val-report died on a 199 MiB numpy allocation (the box at its commit limit
    while the training process still held its CUDA context) — a crash, not a verdict."""
    t = text or ""
    return any(s in t for s in _OOM_SIGNS)


def run_external_gates(base, candidate, out_dir, *, py: Optional[str] = None,
                       ruler_floor_pp: float = GATES["ruler_floor_pp"], run=None,
                       which: Optional[Sequence[str]] = None, anchor=None,
                       ruler_abs_floor_pp: float = GATES["ruler_abs_floor_pp"],
                       oom_retries: int = 1, sleep=None, log=print) -> dict:
    """Run the external gates (minutes each), keep their output under ``<out_dir>/gates/``, parse
    the verdicts. ``run`` is injectable (tests). Returns a dict with per-gate ok flags.

    2026-09-03 (the first ``--run-gates`` run): the commands go through ``cmd.exe`` on Windows, which
    cannot resolve the forward-slash ``.venv/Scripts/python.exe`` form ("'.venv' is not recognized as an
    internal or external command") — both gates returned None and the candidate was refused for the
    wrong reason. They now EXECUTE with this interpreter's absolute path (quoted); ``commands`` keeps the
    friendly form the user can paste. An unparsable gate reports its first output line as ``<name>_note``."""
    cmds = external_gate_commands(base, candidate, out_dir, anchor=anchor)   # the printed / recorded form
    exec_cmds = external_gate_commands(base, candidate, out_dir, py=(py or f'"{sys.executable}"'), anchor=anchor)
    if which is None:
        which = ("ruler", "type_eff") + (("ruler_anchor",) if anchor is not None else ())
    gdir = Path(out_dir) / "gates"
    gdir.mkdir(parents=True, exist_ok=True)
    run = run or (lambda cmd: subprocess.run(cmd, shell=True, cwd=str(_REPO), capture_output=True,
                                             text=True, encoding="utf-8", errors="replace"))
    out: dict = {"commands": cmds}
    sleep = sleep or time.sleep
    for name in which:
        r = run(exec_cmds[name])
        text = (getattr(r, "stdout", "") or "") + "\n" + (getattr(r, "stderr", "") or "")
        # 2026-09-05: a gate that CRASHED out of memory is re-run after a pause (the previous gate's process and the
        # trainer's tensors release their commit charge slowly on Windows) — never a gate that ran and printed a verdict
        tries = 0
        while tries < oom_retries and int(getattr(r, "returncode", 0) or 0) != 0 and looks_like_oom(text):
            tries += 1
            log(f"[ladder-ppo] gate {name}: out of memory (rc {getattr(r, 'returncode', '?')}) — retry {tries}/{oom_retries} "
                f"in {GATE_OOM_RETRY_WAIT_S:.0f} s")
            sleep(GATE_OOM_RETRY_WAIT_S)
            r = run(exec_cmds[name])
            text = (getattr(r, "stdout", "") or "") + "\n" + (getattr(r, "stderr", "") or "")
        if tries:
            out[f"{name}_retries"] = tries
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
        elif name == "ruler_anchor":                  # L3: the absolute floor vs the incumbent
            d = parse_ruler_delta_pp(text)
            out["ruler_anchor_delta_pp"] = parsed = d
            out["ruler_anchor_ok"] = (d is not None and d >= ruler_abs_floor_pp)
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
