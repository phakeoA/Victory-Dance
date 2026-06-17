"""For transitions whose our_actions has a mega=True move, dump the acting mon's
state_before_actions dict so we can see which fields (item / is_mega / species)
are reliably available to build_gimmick_mask at decision time."""
import json
from pathlib import Path

HERE = Path(__file__).resolve()
PREP = HERE.parents[2] / "vods" / "Prepared_training_data" / "Regulation_MA"

dirs = [PREP / "Jsonl_TypeB"] + [PREP / "Jsonl_TypeA" / f"Kronomono{n}" for n in (1, 2, 3)]

shown = 0
item_present = item_absent = ismega_false = ismega_true = ismega_missing = 0
sample_keys = None
for d in dirs:
    for fp in sorted(d.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if '"mega": true' not in line:
                continue
            t = json.loads(line)
            snap = t.get("state_before_actions") or {}
            actives = snap.get("our_active") or {}
            for act in t.get("our_actions") or []:
                if not act.get("mega"):
                    continue
                mon = actives.get(act.get("slot"))
                if mon is None:
                    continue
                if sample_keys is None:
                    sample_keys = sorted(mon.keys())
                itm = mon.get("item", mon.get("known_item"))
                if itm:
                    item_present += 1
                else:
                    item_absent += 1
                im = mon.get("is_mega")
                if im is True:
                    ismega_true += 1
                elif im is False:
                    ismega_false += 1
                else:
                    ismega_missing += 1
                if shown < 6:
                    shown += 1
                    print(f"[{fp.parent.name}] turn {t.get('turn')} slot {act.get('slot')} "
                          f"persp={t.get('perspective')}")
                    print("   species=", mon.get("species"), "| base=", mon.get("base_species"),
                          "| item=", mon.get("item"), "| known_item=", mon.get("known_item"),
                          "| is_mega=", mon.get("is_mega"))

print("\n--- sample mon keys ---")
print(sample_keys)
print("\n--- totals over all mega-move acting mons ---")
print(f"item present : {item_present}")
print(f"item absent  : {item_absent}")
print(f"is_mega False: {ismega_false}   True: {ismega_true}   missing: {ismega_missing}")
