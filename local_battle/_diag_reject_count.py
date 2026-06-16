"""Count Showdown order rejections by cause across N battles (production player,
team1 vs WolfeGlick — the setup that reproduced disabled-move rejections)."""
from __future__ import annotations
import sys, asyncio, logging, re
from pathlib import Path
from collections import Counter
_HERE=Path(__file__).resolve().parent;_REPO=_HERE.parent
for _p in (str(_REPO/"data"/"scripts"),str(_REPO),str(_HERE)):
    if _p in sys.path: sys.path.remove(_p)
    sys.path.insert(0,_p)
from poke_env import AccountConfiguration
from player import VGCPlayer
logging.basicConfig(level=logging.WARNING)
ERR=[]
class C(VGCPlayer):
    async def _handle_battle_message(self,sm):
        try:
            for m in sm:
                if len(m)>=2 and m[1]=="error": ERR.append("|".join(m))
        except Exception: pass
        return await super()._handle_battle_message(sm)
def make(u,team):
    return C(model_path=_REPO/"ai_train_scripts/BC_model/checkpoints/bc_best.pt",
        team_chooser_path=_REPO/"ai_train_scripts/teamPreview_model/checkpoints/teampreview_best.pt",
        replay_path=_REPO/"replay_buffer"/f"_rc_{u}.jsonl",device="cpu",
        account_configuration=AccountConfiguration(u,None),
        battle_format="gen9championsvgc2026regma",team=team,max_concurrent_battles=1,log_level=logging.WARNING)
async def main(n):
    t1=(_REPO/"teams"/"M-A"/"team1").read_text(encoding="utf-8").strip()
    t2=(_REPO/"teams"/"M-A"/"WolfeGlick").read_text(encoding="utf-8").strip()
    p1=make("RcRed",t1);p2=make("RcBlue",t2)
    await p1.battle_against(p2,n_battles=n)
    await p1.ps_client.stop_listening();await p2.ps_client.stop_listening();p1.close();p2.close()
    causes=Counter()
    for e in ERR:
        if "disabled" in e: causes["move_disabled"]+=1
        elif "switch" in e: causes["switch"]+=1
        elif "Unavailable" in e or "Invalid" in e: causes["other_order"]+=1
        else: causes["nonorder"]+=1
    print(f"\n{'='*50}\nREJECTIONS over {n} battles (team1 vs WolfeGlick)\n{'='*50}")
    print(f"total |error| lines : {len(ERR)}")
    print(f"by cause            : {dict(causes)}")
    for e in ERR[:6]: print("   ",e[:95])
if __name__=="__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv)>1 else 8))
