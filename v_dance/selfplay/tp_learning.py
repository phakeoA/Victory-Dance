"""The team picker INSIDE self-play (2026-10-03, USER: "I want it to learn that though for both team picker and battle
neural network" → "just do all stages").

Why: both collection paths built the self-play LEARNER without a team picker, so every self-play checkpoint trained on
the first-4 roster heuristic while eval and the ladder use the picker — g50 on Baltimore vs era2 won 77.5 % with
first-4 but 62.7 % with the served picker (memory 14). The picker never got a learning signal from self-play at all.

STAGE 1 (collection, ``mp_collect`` + ``model_io.team_order(explore=...)``): the learner (and the 'latest' opponent,
the same live actor) pick their four with ``--learner-tp`` and EXPLORE — a bring set sampled from
``(1 - eps) * softmax(set scores / tau) + eps * uniform``, then a lead pair inside it the same way — and record the
decision (``EpisodeMeta.tp_learn``: both rosters' packed picker inputs, the subsets / lead pairs, the chosen indices,
the picker's own probabilities ``p_*`` and the behaviour probabilities ``mu_*``). The battle net trains on those games
as usual, so it practises what the picker brings.

STAGE 2 (this module, ``--tp-learn``): after each generation's collection the picker takes a PPO-clip step on its set
head + lead logits from the games' results.
  · reward = +1 won / -1 lost (draws, fallbacks and unrecorded previews are skipped); 2026-10-04 under --reward-v2 a
    loss is -1 + the loss margin (the battle net's own terminal value, ``reward.terminal_reward``; no field credit);
  · baseline = the LEAVE-ONE-OUT mean reward of the other games of the SAME pairing (own team, opponent team) — it
    never depends on this game's pick, so the advantage stays unbiased (a turn-1 critic value would already contain
    the pick and cancel the signal);
  · ratio = pi_new(a) / p_old(a) (p_old = the picker's own probability when it sampled) with an importance weight
    p_old(a) / mu(a) for the eps-uniform part of the behaviour policy (clipped at ``max_iw``);
  · + ``kl_coef`` * KL(pi_new || pi_anchor) — the anchor is the picker the run STARTED from (human imitation), so the
    learned picker drifts only as far as the results justify; − ``ent_coef`` * entropy keeps it exploring.
The new weights are saved as ``<archive>/tp/tp_gen{N}.pt`` with the start checkpoint's config + vocab (so
``model_io.load_team_chooser`` and the bandit load it unchanged), and the run's eval / panel play the candidate with
ITS picker — every generation is a (battle net, picker) PAIR.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class TPLearnConfig:
    tau: float = 0.5            # sampling temperature over set / lead-pair scores (collection AND the update's pi);
                                # 10-03 measured on Baltimore vs the 68 pool teams (v9 picker, eps 0.15): tau 1.0 → the
                                # top set gets 37 % (63 % of picks off it), tau 0.5 → 58 % (42 % off) — 0.5 keeps the
                                # battle net training mostly on the picker's preferred sets while still exploring
    eps: float = 0.15           # eps-uniform exploration mixed into the behaviour policy (collection)
    lr: float = 3e-5
    epochs: int = 2
    batch: int = 128
    clip: float = 0.2
    kl_coef: float = 0.1
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    max_iw: float = 5.0         # cap on the eps importance weight p_old / mu
    min_records: int = 8
    adv_norm: bool = True
    seed: int = 0

    def __post_init__(self):
        # review C (10-03): tau <= 0 divides the set scores by zero in the update → NaN weights, written to disk
        if not float(self.tau) > 0:
            raise ValueError(f"--learner-tp-tau must be > 0 (got {self.tau})")
        if not 0.0 <= float(self.eps) <= 1.0:
            raise ValueError(f"--learner-tp-eps must be in [0, 1] (got {self.eps})")
        if float(self.lr) < 0:
            raise ValueError(f"--tp-lr must be >= 0 (got {self.lr})")


# ── records ───────────────────────────────────────────────────────────────────────────────────────────────
def _record_reward(m) -> float:
    """The picker's reward: the SAME terminal value the battle net gets (2026-10-04 REWARD v2 — a loss earns back the
    margin; no field credit: the picker is never shaped), ±1 for a v1 trajectory (byte-identical)."""
    try:
        from v_dance.rl.reward import terminal_reward
        r = terminal_reward(m)
    except Exception:
        r = None
    return float(r) if r is not None else (1.0 if m.won else -1.0)


def records_from_trajectories(trajectories) -> list:
    """One training record per trajectory whose preview an EXPLORING picker drove and whose result is a real win /
    loss. ``key`` = (own roster, sorted opponent roster) — the pairing the leave-one-out baseline groups by."""
    out = []
    for t in trajectories or ():
        m = getattr(t, "meta", None)
        rec = getattr(m, "tp_learn", None) if m is not None else None
        if not rec or m.won is None or not getattr(m, "is_trainable", True):
            continue
        key = (tuple(str(s) for s in (m.own_team or ())), tuple(sorted(str(s) for s in (m.opp_team or ()))))
        out.append({**rec, "won": bool(m.won), "reward": _record_reward(m), "key": key,
                    "battle_id": str(getattr(m, "battle_id", "") or ""),
                    "own_team": [str(s) for s in (m.own_team or ())],
                    "opp_team": [str(s) for s in (m.opp_team or ())]})
    return out


def training_records(trajectories) -> list:
    """The picker's training records of one generation: FIRST drop every fallback (backstop-forfeit) game AND its
    zero-sum twin by battle_id (``reward.drop_fallback_pairs`` — the twin of a forfeited self-copy game is a
    mislabeled +1 win; PPO drops it too, but only after collect_fn, where the picker update runs — review A,
    10-03), then one record per remaining exploring preview."""
    from v_dance.rl.reward import drop_fallback_pairs
    return records_from_trajectories(drop_fallback_pairs(list(trajectories or ())))


def loo_baselines(rewards: Sequence[float], keys: Sequence, games: Optional[Sequence] = None) -> List[float]:
    """Mean reward of the OTHER GAMES with the same key. Every record of THIS game is left out — its own perspective
    and, in a two-perspective self-copy game, its zero-sum twin (review B, 10-03: a twin's -r made the baseline
    depend on this game's outcome) — so the baseline never depends on this game. A key with no other game falls
    back to the mean of all OTHER games (0.0 when there are none). ``games`` = one id per record (battle_id);
    None = every record is its own game."""
    games = list(games) if games is not None else list(range(len(rewards)))
    tot, cnt = defaultdict(float), defaultdict(int)
    ktot, kcnt = defaultdict(float), defaultdict(int)     # per (key, game)
    gtot, gcnt = defaultdict(float), defaultdict(int)     # per game, all keys
    for r, k, g in zip(rewards, keys, games):
        tot[k] += float(r)
        cnt[k] += 1
        ktot[(k, g)] += float(r)
        kcnt[(k, g)] += 1
        gtot[g] += float(r)
        gcnt[g] += 1
    n, all_sum = len(rewards), float(sum(rewards))
    out = []
    for r, k, g in zip(rewards, keys, games):
        c = cnt[k] - kcnt[(k, g)]
        if c > 0:
            out.append((tot[k] - ktot[(k, g)]) / c)
        else:
            c2 = n - gcnt[g]
            out.append((all_sum - gtot[g]) / c2 if c2 > 0 else 0.0)
    return out


def _group_key(rec) -> tuple:
    return (tuple(tuple(int(i) for i in s) for s in rec["subsets"]), len(rec["pairs"]),
            rec.get("aff") is not None, bool(rec.get("ctx")))


# ── the learner ───────────────────────────────────────────────────────────────────────────────────────────
class TPLearner:
    """Holds the trainable picker (+ a frozen anchor) in the MAIN process; ``update(records)`` → stats,
    ``save(path, generation)`` → a checkpoint the workers / the eval / the bandit load with ``load_team_chooser``."""

    def __init__(self, start_path, *, anchor_path=None, cfg: Optional[TPLearnConfig] = None, device: str = "cpu",
                 opt_state_path=None):
        import torch
        from v_dance.play import model_io as M
        self.cfg = cfg or TPLearnConfig()
        self.device = device
        self.start_path = str(start_path)
        self.model, self.vocab, self.mcfg = M.load_team_chooser(self.start_path, device)
        if not getattr(self.model, "use_set_head", False):
            raise ValueError(f"--tp-learn needs a SET-HEAD picker (config use_set_head) — {start_path} has none")
        if int(self.mcfg.get("bring_k", 4)) != 4 or int(self.mcfg.get("lead_k", 2)) != 2:
            raise ValueError(f"--tp-learn assumes bring 4 / lead 2 (got {self.mcfg.get('bring_k')} / "
                             f"{self.mcfg.get('lead_k')}) — {start_path}")
        self.anchor_path = str(anchor_path or start_path)
        self.anchor, _, _ = M.load_team_chooser(self.anchor_path, device)
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.model.eval()                       # eval mode throughout: dropout OFF → pi matches the sampling pass
        self.anchor.eval()
        self.opt = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.cfg.lr)
        self._template = torch.load(self.start_path, map_location="cpu", weights_only=False)
        self._rng = np.random.default_rng(int(self.cfg.seed))
        # 2026-10-03: a resume restores the Adam moments + the minibatch-shuffle RNG saved beside tp_gen{N}.pt
        # (opt_path_for); the CURRENT --tp-lr wins over the saved one. Missing / unreadable → a fresh Adam.
        self.opt_restored = False
        if opt_state_path and Path(opt_state_path).exists():
            try:
                st = torch.load(str(opt_state_path), map_location="cpu", weights_only=False)
                self.opt.load_state_dict(st["opt_state"])
                for g in self.opt.param_groups:
                    g["lr"] = float(self.cfg.lr)
                if st.get("rng_state") is not None:
                    self._rng.bit_generator.state = st["rng_state"]
                self.opt_restored = True
            except Exception as exc:
                print(f"[tp] optimizer state {opt_state_path} not restored ({exc!r}) — fresh Adam")

    # -- tensors ----------------------------------------------------------------------------------------------------
    def _tensors(self, recs):
        import torch
        dev = self.device
        oi = torch.as_tensor(np.asarray([r["oi"] for r in recs], dtype=np.int64), device=dev)
        pi = torch.as_tensor(np.asarray([r["pi"] for r in recs], dtype=np.int64), device=dev)
        of = torch.as_tensor(np.stack([np.asarray(r["of"], dtype=np.float32) for r in recs]), device=dev)
        pf = torch.as_tensor(np.stack([np.asarray(r["pf"], dtype=np.float32) for r in recs]), device=dev)
        aff = (torch.as_tensor(np.stack([np.asarray(r["aff"], dtype=np.float32) for r in recs]), device=dev)
               if recs[0].get("aff") is not None else None)
        ctx = ({k: torch.as_tensor(np.stack([np.asarray(r["ctx"][k], dtype=np.float32) for r in recs]), device=dev)
                for k in recs[0]["ctx"]} if recs[0].get("ctx") else {})
        pa = torch.as_tensor(np.asarray([[q[0] for q in r["pairs"]] for r in recs], dtype=np.int64), device=dev)
        pb = torch.as_tensor(np.asarray([[q[1] for q in r["pairs"]] for r in recs], dtype=np.int64), device=dev)
        tau = torch.as_tensor(np.asarray([float(r["tau"]) for r in recs], dtype=np.float32), device=dev)
        return oi, pi, of, pf, aff, ctx, pa, pb, tau

    @staticmethod
    def _logps(model, oi, pi, of, pf, aff, ctx, subsets, pa, pb, tau):
        """(log-softmax over the subsets, log-softmax over the chosen subset's lead pairs) — the picker's pi, the
        same formula as model_io.explore_probs with eps = 0."""
        import torch
        scores, _bl, ll = model.score_subsets(oi, pi, of, pf, aff, subsets=subsets, **ctx)
        lps = torch.log_softmax(scores / tau[:, None], dim=-1)
        pair = ll.gather(1, pa) + ll.gather(1, pb)
        lpp = torch.log_softmax(pair / tau[:, None], dim=-1)
        return lps, lpp

    # -- the update -------------------------------------------------------------------------------------------------
    def update(self, records: list) -> dict:
        import torch
        cfg = self.cfg
        n = len(records)
        if n < int(cfg.min_records):
            return {"n": n, "skipped": f"fewer than {cfg.min_records} records"}
        rew = np.asarray([float(r.get("reward", 1.0 if r["won"] else -1.0)) for r in records], dtype=np.float32)
        base = np.asarray(loo_baselines(list(rew), [r["key"] for r in records],
                                        [r.get("battle_id") or f"#{i}" for i, r in enumerate(records)]),
                          dtype=np.float32)
        adv = rew - base
        adv_raw_abs = float(np.mean(np.abs(adv)))
        if cfg.adv_norm and float(adv.std()) > 1e-6:
            adv = adv / float(adv.std())
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[_group_key(r)].append(i)

        before = self._argmax_sets(records)
        clip_hits = steps = 0
        ratio_sum = 0.0
        for _ep in range(max(1, int(cfg.epochs))):
            for gk, idxs in groups.items():
                subsets = [list(s) for s in gk[0]]
                perm = self._rng.permutation(np.asarray(idxs))
                for s0 in range(0, len(perm), max(1, int(cfg.batch))):
                    bi = [int(i) for i in perm[s0:s0 + int(cfg.batch)]]
                    recs = [records[i] for i in bi]
                    oi, pi, of, pf, aff, ctx, pa, pb, tau = self._tensors(recs)
                    lps, lpp = self._logps(self.model, oi, pi, of, pf, aff, ctx, subsets, pa, pb, tau)
                    with torch.no_grad():
                        alps, alpp = self._logps(self.anchor, oi, pi, of, pf, aff, ctx, subsets, pa, pb, tau)
                    si = torch.as_tensor([int(r["set_idx"]) for r in recs], device=self.device)
                    pj = torch.as_tensor([int(r["pair_idx"]) for r in recs], device=self.device)
                    logp_new = lps.gather(1, si[:, None])[:, 0] + lpp.gather(1, pj[:, None])[:, 0]
                    p_old = torch.as_tensor([float(r["p_set"]) * float(r["p_pair"]) for r in recs],
                                            device=self.device).clamp_min(1e-12)
                    mu = torch.as_tensor([float(r["mu_set"]) * float(r["mu_pair"]) for r in recs],
                                         device=self.device).clamp_min(1e-12)
                    iw = (p_old / mu).clamp(max=float(cfg.max_iw))
                    ratio = torch.exp(logp_new - torch.log(p_old))
                    a = torch.as_tensor(adv[bi], device=self.device)
                    surr = torch.minimum(ratio * a, ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * a)
                    ps, pp = lps.exp(), lpp.exp()
                    ent = -(ps * lps).sum(-1) - (pp * lpp).sum(-1)
                    kl = (ps * (lps - alps)).sum(-1) + (pp * (lpp - alpp)).sum(-1)
                    loss = -(iw * surr).mean() - cfg.ent_coef * ent.mean() + cfg.kl_coef * kl.mean()
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.max_grad_norm))
                    self.opt.step()
                    steps += 1
                    with torch.no_grad():
                        clip_hits += int(((ratio < 1.0 - cfg.clip) | (ratio > 1.0 + cfg.clip)).sum())
                        ratio_sum += float(ratio.sum())
        after = self._argmax_sets(records)
        anchor_sets = self._argmax_sets(records, model=self.anchor)
        kl_end, ent_end = self._kl_ent(records)
        sampled_dev = float(np.mean([int(r["set_idx"]) != before[i] for i, r in enumerate(records)]))
        return {
            "n": n, "win": float(np.mean(rew > 0)), "adv_abs": adv_raw_abs, "steps": steps,
            "ratio_mean": ratio_sum / max(1, n * max(1, int(cfg.epochs))),
            "clip_frac": clip_hits / max(1, n * max(1, int(cfg.epochs))),
            "kl_anchor": kl_end, "entropy": ent_end,
            "explored": sampled_dev,                                   # share of picks off the picker's argmax
            "argmax_moved": float(np.mean([a != b for a, b in zip(after, before)])),   # this update
            "argmax_vs_start": float(np.mean([a != b for a, b in zip(after, anchor_sets)])),
            "bring_change": self._bring_change(records, after, anchor_sets),
        }

    def _argmax_sets(self, records, model=None) -> list:
        import torch
        model = model or self.model
        out = [0] * len(records)
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[_group_key(r)].append(i)
        with torch.no_grad():
            for gk, idxs in groups.items():
                subsets = [list(s) for s in gk[0]]
                for s0 in range(0, len(idxs), 256):
                    bi = idxs[s0:s0 + 256]
                    recs = [records[i] for i in bi]
                    oi, pi, of, pf, aff, ctx, _pa, _pb, _tau = self._tensors(recs)
                    scores, _bl, _ll = model.score_subsets(oi, pi, of, pf, aff, subsets=subsets, **ctx)
                    for j, i in enumerate(bi):
                        out[i] = int(torch.argmax(scores[j]).item())
        return out

    def _kl_ent(self, records):
        import torch
        kls, ents = [], []
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[_group_key(r)].append(i)
        with torch.no_grad():
            for gk, idxs in groups.items():
                subsets = [list(s) for s in gk[0]]
                for s0 in range(0, len(idxs), 256):
                    recs = [records[i] for i in idxs[s0:s0 + 256]]
                    oi, pi, of, pf, aff, ctx, pa, pb, tau = self._tensors(recs)
                    lps, lpp = self._logps(self.model, oi, pi, of, pf, aff, ctx, subsets, pa, pb, tau)
                    alps, alpp = self._logps(self.anchor, oi, pi, of, pf, aff, ctx, subsets, pa, pb, tau)
                    ps, pp = lps.exp(), lpp.exp()
                    kls += ((ps * (lps - alps)).sum(-1) + (pp * (lpp - alpp)).sum(-1)).tolist()
                    ents += (-(ps * lps).sum(-1)).tolist()
        return float(np.mean(kls)) if kls else 0.0, float(np.mean(ents)) if ents else 0.0

    @staticmethod
    def _bring_change(records, after, anchor_sets, top: int = 4) -> dict:
        """For the most common OWN roster: the picker's argmax bring rate per species, learned vs start (pp)."""
        teams = Counter(tuple(r["own_team"]) for r in records)
        if not teams:
            return {}
        team, _ = teams.most_common(1)[0]
        idx = [i for i, r in enumerate(records) if tuple(r["own_team"]) == team]
        if not idx:
            return {}

        def rates(sets):
            c = Counter()
            for i in idx:
                for m in records[i]["subsets"][sets[i]]:
                    c[team[int(m)]] += 1
            return {sp: c[sp] / len(idx) for sp in team}
        new, old = rates(after), rates(anchor_sets)
        diffs = sorted(((sp, new[sp] - old[sp]) for sp in team), key=lambda kv: -abs(kv[1]))
        return {"team_games": len(idx), "top": [(sp, round(old[sp] * 100, 1), round(new[sp] * 100, 1))
                                                for sp, _d in diffs[:top]]}

    # -- persistence ------------------------------------------------------------------------------------------------
    def save(self, path, generation: int) -> str:
        import torch
        from v_dance.play import model_io as M
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = dict(self._template)
        obj["model_state"] = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        obj["tp_learn"] = {"generation": int(generation), "start": self.start_path, "anchor": self.anchor_path,
                           "cfg": asdict(self.cfg)}
        tmp = path.with_name(path.name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)
        M.load_team_chooser(str(path))           # verify the round trip LOUDLY (lockstep guard + strict load)
        # the optimizer sidecar (a resume continues the SAME Adam moments) — kept out of the served picker file
        op = opt_path_for(path)
        otmp = op.with_name(op.name + ".tmp")
        torch.save({"generation": int(generation), "opt_state": self.opt.state_dict(),
                    "rng_state": self._rng.bit_generator.state}, otmp)
        os.replace(otmp, op)
        return str(path)


def anchor_of(path) -> Optional[str]:
    """The anchor a saved tp_genN.pt was trained against (None for a plain picker checkpoint)."""
    try:
        import torch
        meta = (torch.load(str(path), map_location="cpu", weights_only=False) or {}).get("tp_learn") or {}
        return meta.get("anchor")
    except Exception:
        return None


def opt_path_for(tp_path) -> Path:
    """``tp/tp_genN.pt`` → ``tp/tp_genN_opt.pt`` (the picker's Adam state + shuffle RNG for a resume)."""
    p = Path(str(tp_path))
    return p.with_name(p.stem + "_opt" + p.suffix)


def tp_path_for(archive, generation: int) -> Path:
    return Path(archive) / "tp" / f"tp_gen{int(generation)}.pt"


def pair_sidecar(ckpt) -> Path:
    return Path(str(ckpt) + ".tp.json")


def record_pair(ckpt, tp_path) -> Optional[str]:
    """Write ``<ckpt>.tp.json``: the picker this battle checkpoint plays with + the checkpoint file's exact size and
    mtime_ns. Written by a picker-in-the-loop run right after it saves / copies the checkpoint (save_fn, keep_fn),
    for Stage 2 (tp_genN) AND Stage 1 (the fixed --learner-tp). Returns the sidecar path (None if either file is
    missing)."""
    try:
        c, t = Path(str(ckpt)), Path(str(tp_path))
        if not (c.exists() and t.exists()):
            return None
        try:
            rel = os.path.relpath(t, c.parent).replace("\\", "/")
        except ValueError:                          # another drive on Windows
            rel = str(t.resolve())
        st = c.stat()
        sp = pair_sidecar(c)
        tmp = sp.with_name(sp.name + ".tmp")
        tmp.write_text(json.dumps({"tp": rel, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}),
                       encoding="utf-8")
        os.replace(tmp, sp)
        return str(sp)
    except Exception:
        return None


def paired_tp_for(ckpt) -> Optional[str]:
    """The picker a RUN checkpoint was co-trained / evaluated with, so a past self (a league snapshot, prev_best,
    the champion mirror, a HoF suspect, a registered arm) plays with ITS picker. Read from the ``<ckpt>.tp.json``
    sidecar (record_pair) and accepted ONLY if the checkpoint is still the exact file it was written for (size +
    mtime_ns) — review D, 10-03: a later run reusing the archive overwrites genN.pt, so a stale pairing can never
    attach an older run's picker. None (no / stale sidecar, every external checkpoint, every run without
    --learner-tp) → the caller keeps the shared picker, byte-identical."""
    try:
        c = Path(str(ckpt))
        sp = pair_sidecar(c)
        if not (sp.exists() and c.exists()):
            return None
        d = json.loads(sp.read_text(encoding="utf-8"))
        st = c.stat()
        if int(d.get("size", -1)) != int(st.st_size) or int(d.get("mtime_ns", -1)) != int(st.st_mtime_ns):
            return None
        t = Path(d["tp"])
        t = t if t.is_absolute() else Path(os.path.normpath(c.parent / t))
        return str(t) if t.exists() else None
    except Exception:
        return None


def picker_after_restore(restored_ckpt, *, base_ckpt, learner_tp) -> Optional[str]:
    """Review F (10-03): after a collapse REVERT the battle net is the champion again, so the picker must be the
    champion's pair too — its recorded pair, or the start picker when the champion IS the base checkpoint.
    None = no known pair (the caller keeps the current picker and says so)."""
    p = paired_tp_for(restored_ckpt)
    if p:
        return p
    try:
        if learner_tp and Path(str(restored_ckpt)).resolve() == Path(str(base_ckpt)).resolve():
            return str(learner_tp)
    except Exception:
        pass
    return None


def summarize(stats: dict) -> str:
    if stats.get("skipped"):
        return f"TP update SKIPPED ({stats['skipped']}; n={stats.get('n', 0)})"
    bc = stats.get("bring_change") or {}
    tops = " · ".join(f"{sp} {a:.0f}→{b:.0f}%" for sp, a, b in bc.get("top", []))
    return (f"TP update: n={stats['n']} win {stats['win'] * 100:.0f}% · explored {stats['explored'] * 100:.0f}% · "
            f"KL-to-start {stats['kl_anchor']:.3f} · entropy {stats['entropy']:.2f} · ratio {stats['ratio_mean']:.2f} "
            f"clip {stats['clip_frac'] * 100:.0f}% · argmax moved {stats['argmax_moved'] * 100:.0f}% "
            f"(vs start {stats['argmax_vs_start'] * 100:.0f}%)"
            + (f" | bring vs start ({bc.get('team_games', 0)} g): {tops}" if tops else ""))


def append_stats(archive, generation: int, stats: dict) -> None:
    p = Path(archive) / "tp" / "tp_stats.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"generation": int(generation), **stats}, default=str) + "\n")
