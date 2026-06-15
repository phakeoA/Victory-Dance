"""Drill into the GAP #4 divergences: dump raw own/opp team + active mons on the
LIVE side and our_active/our_bench/opp_active/opp_bench on the OFFLINE side at the
diverging turns (ZOROARK_DITTO p2 turns 2,3)."""
import sys
sys.path.insert(0, 'data/scripts')
sys.path.insert(0, 'data/scripts/tests')

from pathlib import Path
import _parity_harness as H
from vod_parser.replay_parser import extract_log_from_html, ShowdownReplayParser
from live_state_encoder import LiveStateEncoder
from poke_env.battle import DoubleBattle
import logging

FIX = "Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html"
PERSP = "p2"
TARGET_TURNS = {2, 3}

path = next(Path('data/vods').rglob(FIX))
log = path.read_text(encoding='utf-8')
log = extract_log_from_html(log)
names = H.player_usernames(log)
user = names[PERSP]
print(f"user={user} role={PERSP}")

_LOG = logging.getLogger("drill"); _LOG.disabled = True
_SKIP = {"t:", "uhtmlchange", "win", "tie"}


def dump_live(battle, turn):
    print(f"\n===== LIVE @ start of turn {turn} =====")
    try:
        own_active = battle.active_pokemon
    except ValueError:
        own_active = [None, None]
    try:
        opp_active = battle.opponent_active_pokemon
    except ValueError:
        opp_active = [None, None]
    own_active_set = {p for p in own_active if p is not None}
    print("player_role:", battle.player_role)
    print("own active:", [(p.species if p else None) for p in own_active])
    print("opp active:", [(p.species if p else None) for p in opp_active])
    print("\n  battle.team (OWN) entries:")
    for k, p in battle.team.items():
        print(f"    {k}: species={p.species} base={p.base_species} "
              f"fainted={p.fainted} revealed={p.revealed} max_hp={p.max_hp} "
              f"hpfrac={p.current_hp_fraction} active={p.active} "
              f"in_active_set={p in own_active_set}")
    # replicate the encoder bench + count logic
    bench = [p for p in battle.team.values()
             if p not in own_active_set and not p.fainted][:4]
    own_team = list(battle.team.values())
    own_live = sum(1 for p in own_team if p not in own_active_set and not p.fainted)
    own_fnt = sum(1 for p in own_team if p.fainted)
    print(f"\n  -> live bench (slots) = {[p.species for p in bench]}")
    print(f"  -> live own_live={own_live} own_fnt={own_fnt}")


def dump_offline(snap, turn):
    print(f"\n===== OFFLINE @ start of turn {turn} (state_before_actions[{PERSP}]) =====")
    oa = snap.get("our_active") or {}
    print("our_active:", {k: (v.get("species"), v.get("is_fainted")) for k, v in oa.items()})
    opa = snap.get("opp_active") or {}
    print("opp_active:", {k: (v.get("species"), v.get("is_fainted")) for k, v in opa.items()})
    print("\n  our_bench:")
    for m in (snap.get("our_bench") or []):
        print(f"    species={m.get('species')} base={m.get('base_species')} "
              f"is_fainted={m.get('is_fainted')} seen={m.get('seen')} "
              f"hp_pct={m.get('hp_pct')} transformed={m.get('is_transformed')}")
    print("\n  opp_bench:")
    for m in (snap.get("opp_bench") or []):
        print(f"    species={m.get('species')} base={m.get('base_species')} "
              f"is_fainted={m.get('is_fainted')} seen={m.get('seen')}")
    our_bench_all = snap.get("our_bench") or []
    own_live = sum(1 for m in our_bench_all if not m.get("is_fainted"))
    own_fnt = sum(1 for m in our_bench_all if m.get("is_fainted"))
    print(f"\n  -> offline own_live={own_live} own_fnt={own_fnt}")


# LIVE: re-drive and snapshot at target turns
battle = DoubleBattle("parity-tag", user, _LOG, gen=9)
for ln in log.split("\n"):
    if not ln.startswith("|"):
        continue
    parts = ln.split("|")
    if len(parts) < 2 or parts[1] in _SKIP:
        continue
    try:
        battle.parse_message(parts)
    except Exception:
        pass
    if parts[1] == "turn" and int(parts[2]) in TARGET_TURNS:
        dump_live(battle, int(parts[2]))

# OFFLINE
parser = ShowdownReplayParser(log, our_player=PERSP)
res = parser.parse()
for turn in res["turns"]:
    if turn["turn"] in TARGET_TURNS:
        dump_offline(turn["state_before_actions"][PERSP], turn["turn"])
