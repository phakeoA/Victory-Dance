"""Final live verification of BOTH fixes (bench + belief)."""
from __future__ import annotations
import sys, asyncio, logging
from pathlib import Path
from collections import Counter
_HERE=Path(__file__).resolve().parent;_REPO=_HERE.parent
for _p in (str(_REPO/"data"/"scripts"),str(_REPO),str(_HERE)):
    if _p in sys.path: sys.path.remove(_p)
    sys.path.insert(0,_p)
import numpy as np
from poke_env import AccountConfiguration
from player import VGCPlayer
from state_encoder import (ACTIVE_SLOTS,POKEMON_FEATURES,_move_target_kind,_ALLY_KINDS,NUM_MOVES)
from vgc_base import build_legal_action_mask,random_legal_action
import model_io as _M
logging.basicConfig(level=logging.WARNING)

ERRORS=[];SLOT=Counter();MODEL_ALLY=Counter();OPP_EST_NONZERO=Counter()
# opp_a est-stat byte offset: slot2 start + (hp1 + types40 + base6) = 47..52
_OPP_EST_LO=(ACTIVE_SLOTS-2)*POKEMON_FEATURES + 1+20+20+6
_OPP_EST_HI=_OPP_EST_LO+6

class V(VGCPlayer):
    async def _handle_battle_message(self,sm):
        try:
            for m in sm:
                if len(m)>=2 and m[1]=="error": ERRORS.append("|".join(m))
        except Exception: pass
        return await super()._handle_battle_message(sm)
    def _select_actions(self,battle,state_vec):
        # belief wired?
        OPP_EST_NONZERO["calls"]+=1
        if float(np.abs(state_vec[_OPP_EST_LO:_OPP_EST_HI]).sum())>0: OPP_EST_NONZERO["nonzero"]+=1
        a0=a1=None;src="random"
        if self._model is not None:
            try:
                m0=build_legal_action_mask(battle,0);m1=build_legal_action_mask(battle,1)
                a0,a1=_M.bc_action_indices(self._model,self._model_heads,state_vec,m0,m1,self._device)
                # per-slot model-drive among slots with >=1 legal action
                for sl,mk,a in ((0,m0,a0),(1,m1,a1)):
                    if sum(mk)>0:
                        SLOT[f"slot_legal"]+=1
                        if a is not None:
                            SLOT["slot_model"]+=1
                            # model ally-attack?
                            if a<12 and a%3==2:
                                mon=battle.active_pokemon[sl]
                                mv=list(mon.moves.values())[:NUM_MOVES]
                                if a//3<len(mv) and _move_target_kind(mv[a//3].id) not in _ALLY_KINDS:
                                    MODEL_ALLY["ally"]+=1
            except Exception as e:
                SLOT["exc"]+=1
        fa0=a0 if a0 is not None else random_legal_action(battle,0)
        fa1=a1 if a1 is not None else random_legal_action(battle,1)
        if self._model is None: return fa0,fa1,"random"
        return fa0,fa1,("random" if (a0 is None or a1 is None) else "model")

def make(u,team):
    return V(model_path=_REPO/"ai_train_scripts/BC_model/checkpoints/bc_best.pt",
        team_chooser_path=_REPO/"ai_train_scripts/teamPreview_model/checkpoints/teampreview_best.pt",
        replay_path=_REPO/"replay_buffer"/f"_fv_{u}.jsonl",device="cpu",
        account_configuration=AccountConfiguration(u,None),
        battle_format="gen9championsvgc2026regma",team=team,max_concurrent_battles=1,log_level=logging.WARNING)

async def main(n):
    team=(_REPO/"teams"/"M-A"/"team1").read_text(encoding="utf-8").strip()
    p1=make("FvRed",team);p2=make("FvBlue",team)
    print("encoder.belief loaded:", p1._encoder.belief is not None)
    await p1.battle_against(p2,n_battles=n)
    await p1.ps_client.stop_listening();await p2.ps_client.stop_listening();p1.close();p2.close()
    print("\n"+"="*60+"\nFINAL VERIFY (both fixes)\n"+"="*60)
    inv=[e for e in ERRORS if "Invalid choice" in e or "Unavailable" in e]
    print(f"Showdown order errors (rejections) : {len(inv)}")
    for e in inv[:5]: print("   ",e[:90])
    sl=SLOT["slot_legal"] or 1
    print(f"slots with >=1 legal action        : {SLOT['slot_legal']}")
    print(f"  driven by MODEL                  : {SLOT['slot_model']}  ({SLOT['slot_model']/sl*100:.1f}%)")
    print(f"  fell back to random              : {SLOT['slot_legal']-SLOT['slot_model']}")
    print(f"MODEL ally-attack picks            : {MODEL_ALLY['ally']}")
    print(f"opp est-stat bytes non-zero        : {OPP_EST_NONZERO['nonzero']}/{OPP_EST_NONZERO['calls']} encodes (belief live)")

if __name__=="__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv)>1 else 4))
