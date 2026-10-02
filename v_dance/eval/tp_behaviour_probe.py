"""Team-preview BEHAVIOUR probe (2026-10-01): what each TP checkpoint would BRING on the bot's REAL ladder matchups.

The val ruler (``tp_val_report``) scores imitation of humans; this answers the behavioural question — e.g. "does the
new picker bring Salamence more?" It replays every saved ladder game (``data/vods/Type_C/*.html``) where the bot
fielded ``--team``: our six = the team file (crisp own builds, as the player passes them), theirs = the replay's
preview. Each checkpoint is decoded through the SERVE path (``model_io.team_order``, closed sheets) — argmax, and with
the served near-tie sampling (``--eps``, the live VD_TP_TIE_EPS) averaged over ``--samples`` draws. Validation: the
served checkpoint's sampled bring rates should land near what the ladder actually did (printed alongside).

    .venv/Scripts/python.exe -X utf8 -m v_dance.eval.tp_behaviour_probe --team Baltimore_Sand_Psy \\
        --ckpt old=<served tp> --ckpt new=<candidate tp>
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import glob
import io
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
BOT = "victoriousdancing"


def _id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def ladder_matchups(team_path: Path, replay_glob: str, since: float):
    """[(opp_species[6], actually_brought_set)] for every replay after ``since`` where the bot fielded this team."""
    ours = None
    out = []
    from v_dance.online.accounts import our_userids
    bots = our_userids() | {BOT}                       # 2026-10-02: every .env ladder account (EncoreFN too)
    for f in glob.glob(replay_glob):
        if os.path.getmtime(f) <= since:
            continue
        lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
        side = next((m.group(1) for ln in lines for m in [re.match(r"\|player\|(p[12])\|([^|]*)\|", ln)]
                     if m and _id(m.group(2)) in bots), None)
        if not side:
            continue
        other = "p2" if side == "p1" else "p1"
        mine = [re.sub(r",.*", "", ln.split("|")[3]) for ln in lines if ln.startswith(f"|poke|{side}|")]
        opp = [re.sub(r",.*", "", ln.split("|")[3]) for ln in lines if ln.startswith(f"|poke|{other}|")]
        if ours is None:
            ours = {_id(s.split("-")[0]) for s in mine}
        if {_id(s.split("-")[0]) for s in mine} != ours or len(opp) != 6:
            continue
        brought = {_id(re.sub(r",.*", "", ln.split("|")[3]).split("-")[0]) for ln in lines
                   if ln.startswith(("|switch|", "|drag|")) and ln.split("|")[2].startswith(side)}
        out.append((opp, brought))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True, help="the bot's team (name or path)")
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=TP_CKPT")
    ap.add_argument("--replays", default=str(_REPO / "data" / "vods" / "Type_C" / "*.html"))
    ap.add_argument("--eps", type=float, default=1.0, help="served near-tie eps (VD_TP_TIE_EPS)")
    ap.add_argument("--samples", type=int, default=5)
    a = ap.parse_args(argv)

    import numpy as np
    import v_dance.play.run_local_battle as R
    from v_dance.dex.pokedex import norm_species
    from v_dance.dex.team_sheet import parse_showdown_team
    from v_dance.formats import default_format, pikalytics_path_for
    from v_dance.parser.belief_state import BeliefState
    from v_dance.play import model_io as M

    team_path = Path(R.resolve_team_path(a.team))
    sheet = parse_showdown_team(team_path.read_text(encoding="utf-8"))
    our_species = [m["species"] for m in sheet]
    own_build = {norm_species(m["species"]): {"ability": m.get("ability"), "item": m.get("item"),
                                              "moves": list(m.get("moves") or [])} for m in sheet}
    belief = BeliefState(pikalytics_path_for(default_format()))
    games = ladder_matchups(team_path, a.replays, since=os.path.getmtime(team_path))
    print(f"[tp-probe] {len(games)} ladder games with {team_path.name} (since its last edit); "
          f"our six: {', '.join(our_species)}")
    real = collections.Counter(s for _o, b in games for s in b)
    rows = {"ladder (actual)": {_id(s): real[_id(s)] / max(1, len(games)) for s in our_species}}
    for item in a.ckpt:
        name, _, path = item.partition("=")
        model, vocab, cfg = M.load_team_chooser(path, device="cpu")
        for mode, eps, reps in (("argmax", 0.0, 1), (f"eps {a.eps:g}", a.eps, a.samples)):
            os.environ["VD_TP_TIE_EPS"] = str(eps)
            np.random.seed(0)
            cnt = collections.Counter()
            for opp, _b in games:
                for _ in range(reps):
                    with contextlib.redirect_stdout(io.StringIO()):     # the per-deviation "[tp] near-tie" prints
                        order = M.team_order(model, vocab, cfg, our_species, opp, 4, "cpu",
                                             belief=belief, own_build=own_build)
                    for i in order[:4]:
                        cnt[_id(our_species[i])] += 1
            rows[f"{name} {mode}"] = {_id(s): cnt[_id(s)] / max(1, len(games) * reps) for s in our_species}
    w = max(len(k) for k in rows)
    print("\n  brought %".ljust(w + 4) + "".join(f"{s[:10]:>11s}" for s in our_species))
    for k, r in rows.items():
        print(f"  {k:<{w}}  " + "".join(f"{r[_id(s)] * 100:10.0f}%" for s in our_species))
    return 0


if __name__ == "__main__":
    sys.exit(main())
