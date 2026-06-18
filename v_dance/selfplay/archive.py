"""Generation archive (data) — task 3c.5.

Writes ``manifest.json`` (in the self-play area): the per-generation index (gen, checkpoint,
scripted win-rate, Elo, verdict, promoted) + best pointer + league pool. This is the DATA the
metrics/logging dashboard (3c.6, an html/js/css UI) reads and re-renders over time. No static
graphs are produced here.

(Per-game showcase replays used to also be written here as ``Type_D`` Showdown-replay HTML; that
is now superseded by the ``--save-replays`` flag, which saves real playable Showdown ``.html``
replays via poke-env's native template under ``<archive>/live/.../{replays,eval}/``.)

Pure (no poke-env / torch) — built from a GenerationHistory.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ── manifest (the dashboard's data source) ────────────────────────────────────
def _numeric_stats(d: dict) -> dict:
    """Keep only the plottable scalars from a PPO ``update_stats`` dict (loss,
    kl_to_bc, clip_fraction, explained_variance, entropy, value, policy, …);
    booleans like ``halted`` collapse to 0.0/1.0 so the dashboard can chart them."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def build_manifest(history, league=None) -> dict:
    """The dashboard's single data source. Per generation: the raw result plus the
    PPO/critic health (``update_stats``) and *improvement* signals (Elo / win-rate
    deltas vs the prior gen, and whether this gen is the best win-rate so far). The
    top level carries a summary (best gen/Elo/win-rate, promotion count)."""
    recs = history.records
    # THE CHAMPION is the latest-PROMOTED gen (the gate's accepted best), NOT the argmax-scripted
    # gen — those diverge once scripted saturates and the v2 head-to-head ladder takes over, so the
    # dashboard must star the real champion (red-team observability fix, sec 16).
    champ_gen = history.champion_generation() if hasattr(history, "champion_generation") else None
    gens = []
    prev_elo = prev_wr = best_wr_so_far = None
    for r in recs:
        wr = (r.scripted_wins / r.scripted_games) if r.scripted_games else None
        elo_delta = (r.model_elo - prev_elo) if (r.model_elo is not None and prev_elo is not None) else None
        wr_delta = (wr - prev_wr) if (wr is not None and prev_wr is not None) else None
        best_wr_so_far = wr if (wr is not None and best_wr_so_far is None) else best_wr_so_far
        gens.append({
            "generation": r.generation,
            "checkpoint": f"checkpoints/gen{r.generation}.pt",
            "scripted_win_rate": wr,
            "model_elo": r.model_elo,
            "champion_elo": getattr(r, "champion_elo", None),     # non-saturating lineage Elo curve
            "verdict": r.verdict, "promoted": r.promoted,
            "n_trajectories": r.n_trajectories,
            "update_stats": _numeric_stats(r.update_stats),
            "elo_delta": elo_delta,
            "win_rate_delta": wr_delta,
            "is_best": wr is not None and (best_wr_so_far is None or wr >= best_wr_so_far),
            "is_champion": (champ_gen is not None and r.generation == champ_gen),
            "hof": getattr(r, "hof", None),     # Phase-2 breadth-veto result (None = HoF didn't run)
        })
        if r.model_elo is not None:
            prev_elo = r.model_elo
        if wr is not None:
            prev_wr = wr
            best_wr_so_far = wr if best_wr_so_far is None else max(best_wr_so_far, wr)

    wr_pts = [(g["generation"], g["scripted_win_rate"]) for g in gens if g["scripted_win_rate"] is not None]
    best_gen, best_wr = max(wr_pts, key=lambda t: t[1]) if wr_pts else (None, None)
    best_elo = max((g["model_elo"] for g in gens if g["model_elo"] is not None), default=None)
    return {
        "n_generations": history.generation,
        # the real CHAMPION (gate-accepted = latest promoted), first-class:
        "champion_path": history.best_path,
        "champion_generation": champ_gen,
        "champion_elo": getattr(history, "champion_elo", None),   # non-saturating progress metric
        "best_path": history.best_path,
        # the best SCRIPTED-win-rate gen (a secondary stat; saturates — NOT the champion):
        "best_scripted_generation": best_gen,
        "best_scripted_win_rate": best_wr,
        # back-compat summary (dashboard reads these): star follows the CHAMPION, not argmax-scripted.
        "best_generation": champ_gen if champ_gen is not None else best_gen,
        "best_win_rate": best_wr,
        "best_elo": best_elo,
        "n_promotions": sum(1 for r in recs if r.promoted),
        "league": [s.snapshot_id for s in league.snapshots] if league is not None else [],
        "generations": gens,
    }


def write_manifest(archive_dir, history, league=None) -> Path:
    d = Path(archive_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "manifest.json"
    p.write_text(json.dumps(build_manifest(history, league), indent=2), encoding="utf-8")
    return p


def write_generation_artifacts(archive_dir, history, league=None):
    """Each generation: write the self-play ``manifest.json`` (the dashboard's data source).

    (Per-game replays are no longer written here — see ``--save-replays``, which saves real
    playable Showdown ``.html`` replays via poke-env's native template.)"""
    return {"manifest": str(write_manifest(archive_dir, history, league))}


# ── offline demo (no server): write a sample manifest ─────────────────────────
def _demo(out_dir=str(_REPO_ROOT / "artifacts" / "self_play_archive")) -> None:
    """`python -m v_dance.selfplay.archive` -> a sample manifest.json the dashboard can read."""
    from v_dance.selfplay.generation import GenerationHistory, GenerationRecord

    h = GenerationHistory()
    for g, (wr, elo, verd, prom) in enumerate([
            (0.48, 1040, "promote", True), (0.56, 1080, "promote", True),
            (0.63, 1135, "hold", False)]):
        h.add(GenerationRecord(g, 200, int(wr * 100), 100, float(elo), verd, prom))
    h.best_path = "gen1.pt"
    out = write_generation_artifacts(out_dir, h)
    print("wrote sample archive artifacts:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _demo()
