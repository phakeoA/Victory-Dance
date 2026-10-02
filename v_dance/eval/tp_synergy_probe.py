"""15b-feat.0 — TP SYNERGY-SENSITIVITY probe (go/no-go BEFORE the SBDA retrain).

Question: does the CURRENT 46-dim TP net up-weight a weather ABUSER's bring logit
specifically because its weather SETTER is on the team? If the mean-pool + no-ability
architecture truly washes synergy out, this is ~0 — and that ~0 is the BASELINE the
SBDA redesign (mechanic tags + self-attention + Pikalytics teammate prior) must beat.

Method (matched control, isolates synergy from generic context shift):
  roster = [abuser@0, VARIED@1, filler, filler, filler, filler]; opp fixed.
  synergy_delta = bring_logit[abuser] with the SETTER@1  MINUS  the mean bring_logit
  [abuser] over many random NON-setter controls@1. Averaged over many (filler, opp)
  contexts. Compared against (a) the teammate-swap noise band and (b) the opponent-
  swap |delta| (matchup reference). Positive AND > noise => the net encodes synergy.

We use BASE-FORME abusers (Excadrill/Barraskewda/Lilligant): a Sand-Force MEGA Steelix
is invisible at preview (the net sees base "Steelix"), so only base-forme abilities
are observable to the current net.

Pure CPU, offline, reads only the TP checkpoint + pinned pikalytics — safe during training.
Importable (synergy_delta / make_score_fn are unit-tested); `python -m v_dance.eval.tp_synergy_probe` runs it (moved from scratch/ in the 2026-09-10 refactor, Phase 1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_REPO = Path(__file__).resolve().parents[2]          # v_dance/eval/ -> repo root
assert (_REPO / "pyproject.toml").is_file(), _REPO

from v_dance.play.model_io import load_team_chooser, _pack_side, DEFAULT_TP_CHECKPOINT  # noqa: E402
from v_dance.dex.pokedex import norm_species             # noqa: E402

# Default to the SERVED SBDA TP net (model_io single source of truth, post 2026-06-25 rename). The probe
# auto-detects the legacy-vs-SBDA recipe (make_score_fn), so pass --ckpt
# ai_train_scripts/teamPreview_model/checkpoints_pre_sbda/teampreview_base.pt for the legacy ~0 baseline.
TP_PATH = DEFAULT_TP_CHECKPOINT

# Weather abilities, classified by BASE-forme ability so the PREVIEW species the net sees
# actually carries the mechanic (a Sand-Force MEGA Steelix is invisible at preview).
SETTER_ABIL = {"Sand Stream": "sand", "Drizzle": "rain", "Drought": "sun", "Snow Warning": "snow"}
ABUSER_ABIL = {"Sand Rush": "sand", "Sand Force": "sand", "Swift Swim": "rain",
               "Chlorophyll": "sun", "Solar Power": "sun", "Slush Rush": "snow"}


def discover_pairs(pika: dict, in_vocab, max_pairs: int = 12):
    """Auto-find (weather, setter, abuser) triples where BOTH base-forme species are in the
    TP vocab and carry the weather setter/abuser ability per Pikalytics. Returns the triples,
    the set of all weather mons (to exclude from controls), sorted by combined usage."""
    setters, abusers, weather_mons, usage = {}, {}, set(), {}
    for name, e in pika.items():
        if "Mega" in name or not in_vocab(name):     # classify by BASE forme only
            continue
        abil = {a["name"] for a in (e.get("abilities") or [])}
        usage[name] = float(e.get("usage_pct", 0.0))
        for ab in abil & set(SETTER_ABIL):
            setters.setdefault(SETTER_ABIL[ab], []).append(name); weather_mons.add(name)
        for ab in abil & set(ABUSER_ABIL):
            abusers.setdefault(ABUSER_ABIL[ab], []).append(name); weather_mons.add(name)
    triples = []
    for w in sorted(set(setters) & set(abusers)):
        for s in setters[w]:
            for a in abusers[w]:
                if s != a:
                    triples.append((w, s, a, usage.get(s, 0) + usage.get(a, 0)))
    triples.sort(key=lambda t: -t[3])
    return [(f"{w}:{s}+{a}", s, a) for w, s, a, _ in triples[:max_pairs]], weather_mons


def make_score_fn(model, vocab, cfg, device: str = "cpu", belief=None):
    """Return score(own_species, opp_species) -> bring_logits[6] using the production packing.

    Auto-detects the recipe: legacy 46-dim dex (no belief) vs SBDA tp_features (needs the belief,
    use_tp_features=True) — so the SAME probe measures synergy on the legacy baseline AND the SBDA net."""
    from v_dance.play.model_io import uses_tp_features
    fd = int(cfg.get("feat_dim", 46))
    use_tp = uses_tp_features(cfg)
    # Include the teammate-bias channel (the Pikalytics co-occurrence prior) when the SBDA net carries
    # it — it's a core synergy mechanism; omitting it under-measures the net.
    use_bias = bool(getattr(model, "use_teammate_bias", False)) and belief is not None
    if use_bias:
        from v_dance.training.tp_features import teammate_affinity_matrix

    def score(own, opp):
        oi, of = _pack_side(own, vocab, fd, belief=belief, use_tp_features=use_tp,
                            tp_schema=cfg.get("feature_schema"))      # 2026-10-02: v8 view of the v9 vector
        pi, pf = _pack_side(opp, vocab, fd, belief=belief, use_tp_features=use_tp,
                            tp_schema=cfg.get("feature_schema"))
        aff = None
        if use_bias:
            aff = torch.as_tensor(teammate_affinity_matrix(own, belief, n=6)[None], device=device)
        with torch.no_grad():
            bl, _ = model(
                torch.as_tensor([oi], device=device), torch.as_tensor([pi], device=device),
                torch.as_tensor(of[None], device=device), torch.as_tensor(pf[None], device=device),
                aff,
            )
        return np.asarray(bl[0]).ravel()

    return score


def synergy_delta(score_fn, setter, abuser, fillers, opp, controls):
    """abuser@slot0, varied mon@slot1, fillers@2..5. Returns (l_setter, l_control_mean,
    l_control_std): the abuser's bring logit with the SETTER vs the mean over non-setter controls."""
    l_setter = float(score_fn([abuser, setter] + list(fillers), opp)[0])
    ls = [float(score_fn([abuser, c] + list(fillers), opp)[0]) for c in controls]
    lc = np.asarray(ls, dtype=float)
    return l_setter, float(lc.mean()), float(lc.std())


def run(n_ctx: int = 10, n_controls: int = 14, seed: int = 0,
        ckpt=TP_PATH, belief_path=None):
    model, vocab, cfg = load_team_chooser(ckpt, "cpu")
    from v_dance.play.model_io import uses_tp_features
    belief = None
    if uses_tp_features(cfg):                       # SBDA net needs the Pikalytics belief
        from v_dance.parser.belief_state import BeliefState
        belief = BeliefState(belief_path) if belief_path else BeliefState()
    score = make_score_fn(model, vocab, cfg, belief=belief)
    pika = json.load(open(_REPO / "data" / "pikalytics_regma.json", encoding="utf-8"))["pokemon"]
    in_vocab = lambda s: norm_species(s) in vocab
    pool = [n for n in pika if in_vocab(n)]
    pairs, weather_mons = discover_pairs(pika, in_vocab)
    controls_pool = [n for n in pool if n not in weather_mons]   # controls = weather-neutral mons
    rng = np.random.RandomState(seed)
    print(f"TP ckpt : {ckpt}")
    print(f"vocab   : {len(vocab)} | in-vocab pool: {len(pool)} | control pool: {len(controls_pool)}")
    print(f"auto-discovered in-vocab setter+abuser pairs: {len(pairs)}")
    rows = []
    for label, setter, abuser in pairs:
        deltas, noise, opp_ref = [], [], []
        for _ in range(n_ctx):
            avail = [p for p in controls_pool if p not in (setter, abuser)]
            fillers = list(rng.choice(avail, 4, replace=False))
            opp = list(rng.choice(pool, 6, replace=False))
            ctrls = list(rng.choice([p for p in avail if p not in fillers], n_controls, replace=False))
            l_set, l_mean, l_std = synergy_delta(score, setter, abuser, fillers, opp, ctrls)
            deltas.append(l_set - l_mean)
            noise.append(l_std)
            # opponent-swap reference: hold own fixed, vary the opponent
            own = [abuser, setter] + fillers
            base = float(score(own, opp)[0])
            opp_ref.append(np.mean([abs(float(score(own, list(rng.choice(pool, 6, replace=False)))[0]) - base)
                                    for _ in range(n_controls)]))
        d = np.asarray(deltas)
        nb = float(np.mean(noise))
        verdict = "SIGNAL" if d.mean() > 2 * nb else "~0 (synergy washed out)"
        print(f"\n[{label}] setter={setter}  abuser={abuser}")
        print(f"   synergy_delta (abuser logit: SETTER vs avg non-setter): mean={d.mean():+.4f}  std={d.std():.4f}")
        print(f"   teammate-swap noise band (std of control logits)      : ~{nb:.4f}")
        print(f"   opponent-swap |delta| (matchup reference)             : ~{float(np.mean(opp_ref)):.4f}")
        print(f"   => {verdict}")
        rows.append({"pair": label, "setter": setter, "abuser": abuser,
                     "synergy_delta_mean": float(d.mean()), "synergy_delta_std": float(d.std()),
                     "noise_band": nb, "opp_swap_ref": float(np.mean(opp_ref)), "verdict": verdict})
    overall = {}
    if rows:
        md = float(np.mean([r["synergy_delta_mean"] for r in rows]))
        mn = float(np.mean([r["noise_band"] for r in rows]))
        n_signal = sum(r["verdict"] == "SIGNAL" for r in rows)
        overall = {"mean_synergy_delta": md, "mean_noise_band": mn,
                   "pairs_with_signal": n_signal, "n_pairs": len(rows)}
        print("\n================= OVERALL =================")
        print(f"  pairs tested: {len(rows)} | pairs showing SIGNAL: {n_signal}")
        print(f"  mean synergy_delta across pairs: {md:+.4f}  vs mean noise band ~{mn:.4f}")
        print(f"  => {'SYNERGY PRESENT' if md > mn else 'NO SYNERGY (washed out) — redesign justified'}")
    out = _REPO / "artifacts" / "logs" / "tp_synergy_report.json"
    out.write_text(json.dumps({"tp": str(ckpt), "n_ctx": n_ctx, "n_controls": n_controls,
                               "overall": overall, "pairs": rows}, indent=2), encoding="utf-8")
    print(f"\nThis is the BASELINE the SBDA redesign must beat. report -> {out.relative_to(_REPO).as_posix()}")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="15b-feat.0 TP synergy-sensitivity probe")
    ap.add_argument("--ckpt", default=str(TP_PATH),
                    help="TP checkpoint (default: served SBDA net; pass the legacy base for the ~0 baseline)")
    ap.add_argument("--belief", default=None, help="Pikalytics belief json (SBDA net only)")
    _a = ap.parse_args()
    run(ckpt=_a.ckpt, belief_path=_a.belief)
