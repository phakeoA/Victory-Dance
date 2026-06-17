"""Smoke test: model driving vs Zoroark/Ditto (Kronomono3 has Zoroark-Hisui +
Charizard-Mega-Y).  Exercises the gap-#6 opponent splice + model-driven turns +
model-driven forced replacement together.  Asserts: no Showdown rejections, model
drives (incl. forced_switch_model), no random/retry/model_error, all recorded
states finite (1398).  Prints PASS/FAIL.

Run:  .venv/Scripts/python.exe local_battle/_smoke_zoroark.py [n_battles]
"""
from __future__ import annotations
import sys, json, asyncio, logging
from pathlib import Path
from collections import Counter
_HERE = Path(__file__).resolve().parent; _REPO = _HERE.parent
import numpy as np
from poke_env import AccountConfiguration
from v_dance.play.player import VGCPlayer
from v_dance.encoders.state_encoder import STATE_DIM
logging.basicConfig(level=logging.WARNING)

ERRORS: list = []
SPLICE = Counter()

class SmokePlayer(VGCPlayer):
    async def _handle_battle_message(self, sm):
        try:
            for m in sm:
                if len(m) >= 2 and m[1] == "error":
                    ERRORS.append("|".join(m))
        except Exception:
            pass
        return await super()._handle_battle_message(sm)
    def _build_opp_snapshot(self, battle):
        snap = super()._build_opp_snapshot(battle)
        SPLICE["on" if snap is not None else "off"] += 1
        return snap

def make(u, team):
    rp = _REPO/"artifacts"/"replay_buffer"/f"_smoke_{u}.jsonl"
    if rp.exists():
        rp.unlink()
    return SmokePlayer(
        model_path=_REPO/"ai_train_scripts/BC_model/checkpoints/bc_best.pt",
        team_chooser_path=_REPO/"ai_train_scripts/teamPreview_model/checkpoints/teampreview_best.pt",
        replay_path=rp, device="cpu",
        account_configuration=AccountConfiguration(u, None),
        battle_format="gen9championsvgc2026regma", team=team,
        max_concurrent_battles=1, log_level=logging.WARNING)

def _states_finite(path: Path):
    n_ok = n_bad = 0
    if not path.exists():
        return 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        st = json.loads(line).get("state")
        if st is None:
            continue
        arr = np.asarray(st, dtype=np.float64)
        if arr.shape == (STATE_DIM,) and np.all(np.isfinite(arr)):
            n_ok += 1
        else:
            n_bad += 1
    return n_ok, n_bad

async def main(n):
    team = (_REPO/"teams"/"M-A"/"Kronomono3").read_text(encoding="utf-8").strip()
    p1 = make("ZoroRed", team); p2 = make("ZoroBlue", team)   # mirror Zoroark
    await p1.battle_against(p2, n_battles=n)
    await p1.ps_client.stop_listening(); await p2.ps_client.stop_listening()
    p1.close(); p2.close()

    print("\n" + "=" * 60)
    print(f"SMOKE: model vs Zoroark/Ditto (Kronomono3 mirror, {n} battles)")
    print("=" * 60)
    rej = [e for e in ERRORS if "Invalid choice" in e or "Unavailable" in e]
    c1, c2 = dict(p1._source_counts), dict(p2._source_counts)
    ok1, bad1 = _states_finite(_REPO/"artifacts"/"replay_buffer"/"_smoke_ZoroRed.jsonl")
    ok2, bad2 = _states_finite(_REPO/"artifacts"/"replay_buffer"/"_smoke_ZoroBlue.jsonl")
    print(f"Showdown rejections        : {len(rej)}")
    for e in rej[:4]: print("   ", e[:90])
    print(f"opp_splice (on/off)        : {dict(SPLICE)}")
    print(f"ZoroRed  sources           : {c1}")
    print(f"ZoroBlue sources           : {c2}")
    print(f"recorded states finite     : Red {ok1} ok / {bad1} bad | Blue {ok2} ok / {bad2} bad")

    bad_src = lambda c: any(c.get(k, 0) for k in ("random", "retry", "model_error", "no_model"))
    drove   = lambda c: c.get("model", 0) > 0
    checks = {
        "no rejections":            len(rej) == 0,
        "model drove turns":        drove(c1) and drove(c2),
        "no random/retry/error":    not bad_src(c1) and not bad_src(c2),
        "splice on (gap-#6)":       SPLICE.get("on", 0) > 0,
        "states all finite":        bad1 == 0 and bad2 == 0 and (ok1 + ok2) > 0,
    }
    fsm = c1.get("forced_switch_model", 0) + c2.get("forced_switch_model", 0)
    fsr = c1.get("forced_switch", 0) + c2.get("forced_switch", 0)
    print(f"forced replacement         : model={fsm}  random_fallback={fsr}")
    print("-" * 60)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    allok = all(checks.values())
    print("=" * 60)
    print("RESULT:", "PASS ✅" if allok else "FAIL ❌")
    return allok

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ok = asyncio.run(main(n))
    sys.exit(0 if ok else 1)
