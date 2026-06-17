import sys
sys.path.insert(0, 'data/scripts')
sys.path.insert(0, 'data/scripts/tests')
import numpy as np
from pathlib import Path
import _parity_harness as H
from vod_parser.replay_parser import extract_log_from_html, ShowdownReplayParser
from live_state_encoder import LiveStateEncoder
from state_encoder import STATE_DIM, POKEMON_FEATURES

FIXTURES = {
    'dual_zoroark_ditto': 'Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html',
    'clean': 'Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html',
}

def base_sp(m):
    return (getattr(m, 'base_species', None) or getattr(m, 'species', '') or '')

def is_seen(m):
    return bool(getattr(m, 'revealed', False)) and bool(getattr(m, 'max_hp', 0))

for fixname, fname in FIXTURES.items():
    log = extract_log_from_html(next(Path('data/vods').rglob(fname)).read_text(encoding='utf-8'))
    names = H.player_usernames(log)
    print(f"\n========== {fixname}  players={names} ==========")
    for persp in ('p1', 'p2'):
        user = names[persp]
        print(f"\n### perspective {persp} (user={user})")
        # ---- OFFLINE raw ----
        parser = ShowdownReplayParser(log, our_player=persp)
        res = parser.parse()
        # ---- LIVE raw: drive battle, sample at each turn ----
        from poke_env.battle import DoubleBattle
        import logging
        lg = logging.getLogger("probe"); lg.disabled = True
        battle = DoubleBattle("parity-tag", user, lg, gen=9)
        live_snaps = {}
        for ln in log.split("\n"):
            if not ln.startswith("|"):
                continue
            parts = ln.split("|")
            if len(parts) < 2 or parts[1] in {"t:", "uhtmlchange", "win", "tie"}:
                continue
            try:
                battle.parse_message(parts)
            except Exception:
                pass
            if parts[1] == "turn":
                t = int(parts[2])
                # snapshot opp roster state
                opp_active = battle.opponent_active_pokemon
                opp_active_set = {p for p in opp_active if p is not None}
                opp_active_bases = {base_sp(p) for p in opp_active_set}
                opp_team = list(getattr(battle, 'opponent_team', {}).values())
                tp_opp = list(getattr(battle, 'teampreview_opponent_team', None) or [])
                live_snaps[t] = {
                    'active': [(base_sp(p), getattr(p,'fainted',None)) for p in opp_active if p],
                    'active_bases': opp_active_bases,
                    'team': [(base_sp(p), getattr(p,'revealed',None), getattr(p,'max_hp',None), getattr(p,'fainted',None)) for p in opp_team],
                    'tp_count': len(tp_opp),
                    'tp_species': [base_sp(p) for p in tp_opp],
                }
        for turn in res['turns']:
            t = turn['turn']
            snap = turn['state_before_actions'][persp]
            opp_a = snap.get('opp_active') or {}
            opp_bench = snap.get('opp_bench') or []
            off_active = [(opp_a.get(k,{}) or {}).get('base_species') or (opp_a.get(k,{}) or {}).get('species') for k in ('opp_a','opp_b') if opp_a.get(k)]
            off_bench_view = [(m.get('base_species') or m.get('species'), m.get('seen', True), m.get('is_fainted'), m.get('hp_pct')) for m in opp_bench]
            off_seen_alive = sum(1 for m in opp_bench if m.get('seen', True) and not m.get('is_fainted'))
            off_seen_fnt   = sum(1 for m in opp_bench if m.get('seen', True) and m.get('is_fainted'))
            ls = live_snaps.get(t, {})
            live_opp_active_bases = ls.get('active_bases', set())
            # live computed counts
            team = ls.get('team', [])
            seen_nonactive = [m for m in team if m[1] and m[2] and m[0] not in live_opp_active_bases]
            live_opp_live = sum(1 for m in seen_nonactive if not m[3])
            live_opp_fnt  = sum(1 for m in seen_nonactive if m[3])
            print(f" turn {t}:")
            print(f"   OFF active={off_active} | OFF bench(sp,seen,fnt,hp)={off_bench_view}")
            print(f"        OFF opp_seen_alive={off_seen_alive} opp_seen_fnt={off_seen_fnt}")
            print(f"   LIVE active={ls.get('active')} tp_count={ls.get('tp_count')} tp={ls.get('tp_species')}")
            print(f"        LIVE team(sp,rev,maxhp,fnt)={team}")
            print(f"        LIVE opp_live={live_opp_live} opp_fnt={live_opp_fnt}")
