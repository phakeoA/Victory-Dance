"""Corpus probe (diagnostic): runs the retrained BC policy over real transitions to
measure (#12) ally mis-targets and (#13) move-order sensitivity.

  .venv/Scripts/python.exe data/scripts/tests/_probe_policy.py [n_files]
"""
import sys, json, glob
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "data" / "scripts", ROOT / "ai_train_scripts" / "BC_model",
          ROOT / "local_battle"):
    sys.path.insert(0, str(p))

import numpy as np, torch
from state_encoder import VodStateEncoder, SWITCH_OFFSET
from policy_analysis import (
    is_ally_mistarget, permute_move_slots, permute_mask_row, unpermute_action,
    flatten_move_known, HEAD_SLOT,
)
import model_io as M

HEADS = ("our_a", "our_b")
# A few non-identity permutations of the 4 move slots to probe order-sensitivity.
PERMS = [(3, 2, 1, 0), (1, 0, 3, 2), (1, 2, 3, 0)]


def masked_argmax(logits, mask):
    bi, bv = None, -1e30
    for i, ok in enumerate(mask):
        if ok and logits[i] > bv:
            bv, bi = logits[i], i
    return bi


def main(n_files, ckpt=None):
    ckpt = ckpt or (ROOT / "ai_train_scripts/BC_model/checkpoints/bc_best.pt")
    print(f"[probe] checkpoint: {ckpt}")
    model, heads = M.load_bc_policy(ckpt)
    enc = VodStateEncoder()
    files = sorted(glob.glob(str(ROOT / "data/vods/Prepared_training_data/Regulation_MA/Jsonl_TypeB/**/*.jsonl"), recursive=True))
    import random as _r; _r.seed(0); _r.shuffle(files)
    files = files[:n_files]

    c = Counter()
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                t = json.loads(line)
                snap = t.get("state_before_actions") or {}
                mask_all = t.get("action_mask") or {}
                first = {}
                for a in (t.get("our_actions") or []):
                    sl = a.get("slot")
                    if sl and sl not in first:
                        first[sl] = a
                # need an active mon dict per head for is_ally_mistarget + a mask
                oa = snap.get("our_active") or {}
                vec = None
                for head in HEADS:
                    row = mask_all.get(head)
                    mon = oa.get(head)
                    if not row or sum(row) == 0 or not mon:
                        continue
                    if vec is None:
                        vec = enc.encode_snapshot(snap, turn=t.get("turn") or 0)
                    with torch.no_grad():
                        out = model(torch.as_tensor(vec))[0]   # (actions, gimmicks) → actions
                    logit = np.asarray(out[head].detach()).ravel()
                    am = masked_argmax(logit, row)
                    c["decisions"] += 1

                    # #12 — model ally mis-target + human ally mis-target (data)
                    if is_ally_mistarget(mon, am):
                        c["model_ally_mistarget"] += 1
                    ha = (first.get(head) or {}).get("action_index")
                    if is_ally_mistarget(mon, ha):
                        c["human_ally_mistarget"] += 1

                    # #13 — move-order sensitivity (does a permuted order change the
                    # PHYSICAL move the policy picks?).  Measured two ways:
                    #   raw  = on the training board (Type-B is_known gradient present)
                    #   flat = with own is_known flattened to 1.0 (the SERVE board:
                    #          all 4 own moves are known) → the residual a live bot sees
                    slot_index = HEAD_SLOT[head]

                    def move_flip(base_vec, base_am):
                        for perm in PERMS:
                            pvec = permute_move_slots(base_vec, slot_index, perm)
                            prow = permute_mask_row(row, perm)
                            with torch.no_grad():
                                pout = model(torch.as_tensor(pvec))[0]
                            pam = masked_argmax(np.asarray(pout[head].detach()).ravel(), prow)
                            mapped = unpermute_action(pam, perm)
                            a_move = base_am // 3 if (base_am is not None and base_am < SWITCH_OFFSET) else base_am
                            m_move = mapped // 3 if (mapped is not None and mapped < SWITCH_OFFSET) else mapped
                            if a_move != m_move:
                                return True
                        return False

                    if move_flip(vec, am):
                        c["order_move_flip_raw"] += 1

                    fvec = flatten_move_known(vec, slot_index)
                    with torch.no_grad():
                        fam = masked_argmax(np.asarray(model(torch.as_tensor(fvec))[0][head].detach()).ravel(), row)
                    if move_flip(fvec, fam):
                        c["order_move_flip_flat"] += 1

    d = c["decisions"] or 1
    print("=" * 60)
    print(f"POLICY PROBE — {c['decisions']} slot-decisions over {len(files)} files")
    print("=" * 60)
    print(f"#12 model ally mis-targets : {c['model_ally_mistarget']} "
          f"({c['model_ally_mistarget']/d*100:.3f}% of decisions)")
    print(f"    human ally mis-targets : {c['human_ally_mistarget']} "
          f"({c['human_ally_mistarget']/d*100:.3f}% — the data's own rate)")
    print(f"#13 MOVE-order flip (raw)  : {c['order_move_flip_raw']} "
          f"({c['order_move_flip_raw']/d*100:.2f}% — Type-B board w/ is_known gradient)")
    print(f"    MOVE-order flip (flat) : {c['order_move_flip_flat']} "
          f"({c['order_move_flip_flat']/d*100:.2f}% — SERVE board, own is_known=1.0)")
    print(f"    (flip = any of {len(PERMS)} move-slot permutations changed the chosen PHYSICAL move)")
    print("=" * 60)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    main(n, ckpt)
