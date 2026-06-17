import sys
sys.path.insert(0, 'data/scripts')
sys.path.insert(0, 'data/scripts/tests')

from pathlib import Path
import _parity_harness as H
from vod_parser.replay_parser import extract_log_from_html, ShowdownReplayParser
from live_state_encoder import LiveStateEncoder

FNAME = 'Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html'
PERSP = 'p2'


def find_fix(name):
    return next(Path('data/vods').rglob(name))


def dump_live_team_at_turn(log, user, target_turns):
    """Replay protocol, snapshot battle.team at each |turn| boundary."""
    from poke_env.battle import DoubleBattle
    import logging
    lg = logging.getLogger("probe"); lg.disabled = True
    enc = LiveStateEncoder()
    battle = DoubleBattle("probe", user, lg, gen=9)
    snaps = {}
    for ln in log.split("\n"):
        if not ln.startswith("|"):
            continue
        parts = ln.split("|")
        if len(parts) < 2 or parts[1] in H._parity_harness_skip if False else (len(parts) < 2):
            continue
        if parts[1] in {"t:", "uhtmlchange", "win", "tie"}:
            continue
        try:
            battle.parse_message(parts)
        except Exception:
            pass
        if parts[1] == "turn":
            t = int(parts[2])
            if t in target_turns:
                # snapshot of own team
                try:
                    own_active = battle.active_pokemon
                except ValueError:
                    own_active = [None, None]
                own_active_set = {p for p in own_active if p is not None}
                rows = []
                for key, p in battle.team.items():
                    rows.append({
                        "key": key,
                        "species": p.species,
                        "base_species": getattr(p, "base_species", None),
                        "fainted": p.fainted,
                        "revealed": getattr(p, "revealed", None),
                        "max_hp": getattr(p, "max_hp", None),
                        "active": p in own_active_set,
                    })
                snaps[t] = rows
    return snaps


def dump_offline_snap(log, persp, target_turns):
    parser = ShowdownReplayParser(log, our_player=persp)
    res = parser.parse()
    out = {}
    for turn in res["turns"]:
        t = turn["turn"]
        if t not in target_turns:
            continue
        snap = turn["state_before_actions"][persp]
        out[t] = {
            "our_active": snap.get("our_active"),
            "our_bench": snap.get("our_bench"),
        }
    return out


if __name__ == "__main__":
    log = extract_log_from_html(find_fix(FNAME).read_text(encoding='utf-8'))
    user = H.player_usernames(log)[PERSP]
    targets = {1, 2, 3, 4}

    print("##### LIVE battle.team snapshots (own side =", user, ") #####")
    snaps = dump_live_team_at_turn(log, user, targets)
    for t in sorted(snaps):
        print(f"\n--- turn {t} (start) --- own team count={len(snaps[t])}")
        for r in snaps[t]:
            print(f"  {r['species']:<18} base={str(r['base_species']):<14} "
                  f"fnt={int(bool(r['fainted']))} rev={int(bool(r['revealed']))} "
                  f"max_hp={r['max_hp']} active={int(r['active'])}")
        # recompute live own_live the way encode does
        own_active_set = {r['key'] for r in snaps[t] if r['active']}
        bench = [r for r in snaps[t] if (not r['active']) and (not r['fainted'])]
        own_live = sum(1 for r in snaps[t] if (not r['active']) and (not r['fainted']))
        own_fnt = sum(1 for r in snaps[t] if r['fainted'])
        print(f"  >> live encode: own_live={own_live} own_fnt={own_fnt}  "
              f"bench-selected={[r['species'] for r in bench][:4]}")

    print("\n\n##### OFFLINE snapshot (our_bench, our_active) #####")
    osnaps = dump_offline_snap(log, PERSP, targets)
    for t in sorted(osnaps):
        ob = osnaps[t]["our_bench"] or []
        oa = osnaps[t]["our_active"] or {}
        print(f"\n--- turn {t} (state_before) ---")
        print("  our_active:")
        for k, m in oa.items():
            if m:
                print(f"    {k}: {m.get('species')} fnt={int(bool(m.get('is_fainted')))} "
                      f"seen={m.get('seen', True)} hp={m.get('hp_pct')}")
        print(f"  our_bench (n={len(ob)}):")
        for m in ob:
            print(f"    {m.get('species'):<18} base={m.get('base_species')} "
                  f"fnt={int(bool(m.get('is_fainted')))} seen={m.get('seen', True)} "
                  f"hp={m.get('hp_pct')}")
        live_b = [m for m in ob if not m.get('is_fainted')]
        fnt_b = [m for m in ob if m.get('is_fainted')]
        print(f"  >> offline encode: own_live={len(live_b)} own_fnt={len(fnt_b)}  "
              f"bench-selected={[m.get('species') for m in live_b][:4]}")
