"""Diagnostic: capture the EXACT Showdown rejection reason behind the retries in
the Kronomono3 mirror, to characterise what's being rejected (illusion-target vs
disabled/trapped vs phantom).  poke-env logs the raw "|error|..." at level 25,
which the players suppress at WARNING — so we lower their loggers here.

    python local_battle/_diag_rejections.py [n]
"""
from __future__ import annotations

import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
import v_dance.play.run_local_battle as R  # noqa: E402

CK = _REPO / "ai_train_scripts" / "BC_model" / "checkpoints" / "bc_best.pt"
TC = _REPO / "ai_train_scripts" / "teamPreview_model" / "checkpoints" / "teampreview_best.pt"

_REASONS = Counter()


class _Capture(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "Error message received" in msg and "|error|" in msg:
            reason = msg.split("|error|", 1)[1].strip()
            # collapse to the leading bracketed tag + a few words
            _REASONS[reason[:80]] += 1
            print("  REJECT:", reason[:110])


async def main(n: int) -> None:
    proc = R.start_showdown()
    team = R.load_team(R.resolve_team_path("Kronomono3"))
    p1 = R.make_player("DgRed", team, model_path=CK, team_chooser_path=TC)
    p2 = R.make_player("DgBlue", team, model_path=CK, team_chooser_path=TC)
    cap = _Capture()
    for p in (p1, p2):
        p.logger.setLevel(24)          # below 25 so "Error message received" passes
        p.logger.addHandler(cap)
    try:
        await p1.battle_against(p2, n_battles=n)
    finally:
        await p1.ps_client.stop_listening()
        await p2.ps_client.stop_listening()
        p1.close()
        p2.close()
        R.stop_showdown(proc)
    print("\n" + "=" * 56)
    print(f"  REJECTION REASONS over {p1.n_finished_battles} battles")
    print("=" * 56)
    for reason, k in _REASONS.most_common():
        print(f"  {k:3d} ×  {reason}")
    if not _REASONS:
        print("  (none captured)")
    print("=" * 56)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(main(n))
