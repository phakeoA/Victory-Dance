"""Generation archive (data) + Type_D Showdown replays — task 3c.5.

Two things, per docs/ppo_reward_design.md sec 18:

  * **Archive data** (in the self-play area): ``manifest.json`` — the per-generation index
    (gen, checkpoint, scripted win-rate, Elo, verdict, promoted) + best pointer + league
    pool. This is the DATA the metrics/logging dashboard (3c.6, an html/js/css UI) reads
    and re-renders over time. No static graphs are produced here.

  * **Type_D replays** (in ``data/vods/Type_D/``, alongside Type_A/B/C): standard Showdown
    replay HTML — the SAME format Showdown's own ``test/common.js saveReplay`` and the
    in-client "Download replay" button produce: the raw ``|``-protocol log embedded in a
    ``<script class="battle-log-data">`` tag, rendered by ``replay-embed.js``. That is
    EXACTLY the format the VOD parser already ingests (Type_B IS this format), so every
    Type_D replay doubles as future training data and re-renders in any browser.

Pure (no poke-env / torch) — built from a GenerationHistory + a captured ``|``-log.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
TYPE_D_DIR = _REPO_ROOT / "data" / "vods" / "Type_D"

# The Showdown replay-log CSS, copied verbatim from a downloaded replay (data/vods/Type_B)
# so a Type_D file renders identically to a real Showdown replay even before JS loads.
REPLAY_CSS = (
    "html,body {font-family:Verdana, sans-serif;font-size:10pt;margin:0;padding:0;}"
    "body{padding:12px 0;} .battle-log {font-family:Verdana, sans-serif;font-size:10pt;} "
    ".battle-log-inline {border:1px solid #AAAAAA;background:#EEF2F5;color:black;"
    "max-width:640px;margin:0 auto 80px;padding-bottom:5px;} .battle-log .inner "
    "{padding:4px 8px 0px 8px;} .battle-log .inner-preempt {padding:0 8px 4px 8px;} "
    ".battle-log .inner-after {margin-top:0.5em;} .battle-log h2 {margin:0.5em -8px;"
    "padding:4px 8px;border:1px solid #AAAAAA;background:#E0E7EA;border-left:0;"
    "border-right:0;font-family:Verdana, sans-serif;font-size:13pt;} .battle-log .chat "
    "{vertical-align:middle;padding:3px 0 3px 0;font-size:8pt;} .battle-log .chat strong "
    "{color:#40576A;} .battle-log .chat em {padding:1px 4px 1px 3px;color:#000000;"
    "font-style:normal;} .chat.mine {background:rgba(0,0,0,0.05);margin-left:-8px;"
    "margin-right:-8px;padding-left:8px;padding-right:8px;} .spoiler {color:#BBBBBB;"
    "background:#BBBBBB;padding:0px 3px;} .spoiler:hover, .spoiler:active, .spoiler-shown "
    "{color:#000000;background:#E2E2E2;padding:0px 3px;} .spoiler a {color:#BBBBBB;} "
    ".spoiler:hover a, .spoiler:active a, .spoiler-shown a {color:#2288CC;} .chat code, "
    ".chat .spoiler:hover code, .chat .spoiler:active code, .chat .spoiler-shown code "
    "{border:1px solid #C0C0C0;background:#EEEEEE;color:black;padding:0 2px;} .chat "
    ".spoiler code {border:1px solid #CCCCCC;background:#CCCCCC;color:#CCCCCC;} "
    ".battle-log .rated {padding:3px 4px;} .battle-log .rated strong {color:white;"
    "background:#89A;padding:1px 4px;border-radius:4px;} .spacer {margin-top:0.5em;} "
    ".message-announce {background:#6688AA;color:white;padding:1px 4px 2px;} .broadcast-green "
    "{background-color:#559955;color:white;padding:2px 4px;} .broadcast-blue "
    "{background-color:#6688AA;color:white;padding:2px 4px;} .infobox {border:1px solid "
    "#6688AA;padding:2px 4px;} .broadcast-red {background-color:#AA5544;color:white;"
    "padding:2px 4px;} .message-effect-weak {font-weight:bold;color:#CC2222;} "
    ".message-effect-resist {font-weight:bold;color:#6688AA;} .message-effect-immune "
    "{font-weight:bold;color:#666666;} .subtle {color:#3A4A66;}"
)


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
            "checkpoint": f"gen{r.generation}.pt",
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


# ── Type_D: standard Showdown replay HTML ─────────────────────────────────────
def _parse_replay_meta(log_lines: List[str]) -> dict:
    """Pull format / player names from the protocol log for the page title/header."""
    fmt, names = "Self-play battle", {}
    for ln in log_lines:
        p = ln.split("|")
        if len(p) >= 3 and p[1] == "tier":
            fmt = p[2]
        elif len(p) >= 4 and p[1] == "player" and p[3]:
            names[p[2]] = p[3]
    return {"format": fmt, "p1": names.get("p1", "p1"), "p2": names.get("p2", "p2")}


def render_replay_html(log_lines: List[str], *, replayid: str = "selfplay",
                       title: Optional[str] = None) -> str:
    """Standard Showdown replay HTML (the ``data/vods/Type_B`` / saveReplay format): the
    raw ``|``-log embedded in ``<script class="battle-log-data">``, rendered by
    ``replay-embed.js``. Doubles as VOD-parser-ingestible training data."""
    meta = _parse_replay_meta(log_lines)
    page_title = title or f"{meta['format']} replay: {meta['p1']} vs. {meta['p2']}"
    # `</` would close the inline <script> tag early — neutralise it (standard precaution).
    battle_log = "\n".join(log_lines).replace("</", "<\\/")
    import html as _html
    pt = _html.escape(page_title)
    return (
        "<!DOCTYPE html>\n<meta charset=\"utf-8\" />\n<!-- version 1 -->\n"
        f"<title>{pt}</title>\n<style>\n{REPLAY_CSS}\n</style>\n"
        "<div class=\"wrapper replay-wrapper\" style=\"max-width:1180px;margin:0 auto\">\n"
        f"<input type=\"hidden\" name=\"replayid\" value=\"{_html.escape(replayid)}\" />\n"
        "<div class=\"battle\"></div><div class=\"battle-log\"></div>"
        "<div class=\"replay-controls\"></div><div class=\"replay-controls-2\"></div>\n"
        "<h1 style=\"font-weight:normal;text-align:center\"><strong>"
        f"{_html.escape(meta['format'])}</strong><br />{_html.escape(meta['p1'])} vs. "
        f"{_html.escape(meta['p2'])}</h1>\n"
        f"<script type=\"text/plain\" class=\"battle-log-data\">{battle_log}</script>\n"
        "</div>\n<script>\nlet daily = Math.floor(Date.now()/1000/60/60/24);"
        "document.write('<script src=\"https://play.pokemonshowdown.com/js/"
        "replay-embed.js?version'+daily+'\"></'+'script>');\n</script>\n"
    )


def write_type_d(out_dir, generation: int, tag: str, log_lines: List[str]) -> Path:
    """Write ``gen_<N>_<tag>.html`` (Showdown replay format) to ``out_dir``
    (default ``data/vods/Type_D``). The HTML embeds the raw log, so it is itself the
    parser-ingestible training artifact — no separate ``.log`` file."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(c if (c.isalnum() or c in "-_") else "_" for c in tag)[:48]
    p = d / f"gen_{generation}_{safe_tag}.html"
    p.write_text(render_replay_html(log_lines, replayid=f"selfplay-gen{generation}-{safe_tag}"),
                 encoding="utf-8")
    return p


def write_generation_artifacts(archive_dir, history, league=None, *,
                               type_d_dir=None, showcase_log=None, tag: str = "showcase"):
    """Each generation: write the manifest (self-play area) and, if a showcase ``|``-log is
    given, a Type_D Showdown replay to ``data/vods/Type_D`` (overridable)."""
    out = {"manifest": str(write_manifest(archive_dir, history, league))}
    if showcase_log and history.records:
        gen = history.records[-1].generation
        td = Path(type_d_dir) if type_d_dir is not None else TYPE_D_DIR
        out["type_d_html"] = str(write_type_d(td, gen, tag, showcase_log))
    return out


# ── offline demo (no server): write a sample Type_D replay + manifest ─────────
def _demo(out_dir=str(_REPO_ROOT / "artifacts" / "self_play_archive")) -> None:
    """`python local_battle/self_play/archive.py` -> a sample manifest + a Type_D Showdown
    replay (data/vods/Type_D/gen_2_demo.html) you can open in a browser (renders via
    replay-embed.js when online)."""
    import sys as _sys
    from v_dance.selfplay.generation import GenerationHistory, GenerationRecord

    h = GenerationHistory()
    for g, (wr, elo, verd, prom) in enumerate([
            (0.48, 1040, "promote", True), (0.56, 1080, "promote", True),
            (0.63, 1135, "hold", False)]):
        h.add(GenerationRecord(g, 200, int(wr * 100), 100, float(elo), verd, prom))
    h.best_path = "gen1.pt"
    log = [
        "|gametype|doubles", "|player|p1|Gen2-Bot|101|1200", "|player|p2|Anchor|rosa|1300",
        "|gen|9", "|tier|[Gen 9 Champions] VGC 2026 Reg M-A", "|teamsize|p1|4",
        "|teamsize|p2|4", "|start",
        "|switch|p1a: Incineroar|Incineroar, L50, F|100/100",
        "|switch|p2a: Flutter Mane|Flutter Mane, L50|100/100", "|turn|1",
        "|move|p1a: Incineroar|Fake Out|p2a: Flutter Mane", "|-damage|p2a: Flutter Mane|82/100",
        "|faint|p2a: Flutter Mane", "|win|Gen2-Bot",
    ]
    out = write_generation_artifacts(out_dir, h, showcase_log=log, tag="demo")
    print("wrote sample archive artifacts:")
    for k, v in out.items():
        print(f"  {k}: {v}")
    print("Open the Type_D .html in a browser (rendered by replay-embed.js).")


if __name__ == "__main__":
    _demo()
