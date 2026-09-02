"""Thought feed (USER 2026-09-02): the "what the nets are thinking" narration for the Online tab —
pure text builders over the decode stashes (no torch, no poke-env), the per-battle ring buffer, the
live-object adapter on duck-typed fakes, and the no-op contract without an installed feed."""
from __future__ import annotations

import pytest

from v_dance.play import thought_feed as TF
from v_dance.play.thought_feed import ThoughtFeed, action_label, replacement_text, tp_text, turn_text

TAG = "battle-gen9championsvgc2026regmb-123"


@pytest.fixture(autouse=True)
def _plain_names(monkeypatch):
    monkeypatch.setattr(TF, "_display", lambda kind, raw: TF._title(raw))


MOVES = [{"name": "Heat Wave", "target": "spread"}, {"name": "Protect", "target": "field"},
         {"name": "Air Slash", "target": "normal"}, {"name": "Helping Hand", "target": "ally"}]
TARGETS = ["Incineroar", "Garchomp", "Raichu"]
BENCH = ["Swampert", "Sneasler"]


def test_action_label_words():
    assert action_label(0, MOVES, TARGETS, BENCH) == "Heat Wave→spread"
    assert action_label(3, MOVES, TARGETS, BENCH) == "Protect"
    assert action_label(6, MOVES, TARGETS, BENCH) == "Air Slash→Incineroar"
    assert action_label(7, MOVES, TARGETS, BENCH) == "Air Slash→Garchomp"
    assert action_label(8, MOVES, TARGETS, BENCH) == "Air Slash→partner (Raichu)"
    assert action_label(9, MOVES, TARGETS, BENCH) == "Helping Hand→partner (Raichu)"
    assert action_label(12, MOVES, TARGETS, BENCH) == "switch→Swampert"
    assert action_label(15, MOVES, TARGETS, BENCH) == "switch→bench 4 (?)"
    assert action_label(None, MOVES, TARGETS, BENCH) == "pass (no legal action)"
    assert action_label(6, [], ["", "", ""], []) == "move 3→opp slot 1"


def _probs(**kv):
    p = [0.0] * 16
    for k, v in kv.items():
        p[int(k[1:])] = v
    return p


def test_turn_text_pair_decode_narration():
    ctx = {"turn": 3, "bench": BENCH,
           "own": [{"species": "Charizard", "moves": MOVES},
                   {"species": "Raichu", "moves": [{"name": "Fake Out", "target": "normal"},
                                                   {"name": "Volt Switch", "target": "normal"},
                                                   {"name": "Protect", "target": "field"},
                                                   {"name": "Nuzzle", "target": "normal"}]}],
           "opp": [{"species": "Incineroar", "hp": 72, "item": "Leftovers", "item_src": "belief",
                    "item_pct": 60, "ability": "Intimidate", "ability_src": "belief"},
                   {"species": "Garchomp", "hp": 100, "item": "Life Orb", "item_src": "seen",
                    "ability": "Rough Skin", "ability_src": "seen"}]}
    mask = [True] * 16
    decode = {"pair": True, "first": 0, "conf": (0.61, 0.47), "cond_on": 0,
              "probs": (_probs(a0=0.61, a3=0.22, a12=0.09, a6=0.08),
                        _probs(a0=0.47, a4=0.30, a6=0.12, a13=0.11)),
              "masks": (mask, mask), "tau": 0.3, "top_p": 1.0, "dropped": [9]}
    txt = turn_text(ctx, decode, picks=(0, 0), wp=0.52, arm="era4_2b")
    lines = txt.split("\n")
    assert lines[0] == "turn 3 · win-prob 0.52 (even) · arm era4_2b"   # 0.55 / 0.45 = the player's own labels
    assert turn_text(ctx, decode, picks=(0, 0), wp=0.58).startswith("turn 3 · win-prob 0.58 (winning)")
    assert lines[1] == ("opp: Incineroar 72% (item ? → Leftovers 60% belief · Intimidate belief) · "
                        "Garchomp 100% (Life Orb revealed · Rough Skin revealed)")
    assert lines[2] == ("slot 1 Charizard [decoded first, conf 61%]: ✔ Heat Wave→spread 61% · "
                        "Protect 22% · switch→Swampert 9%")
    # a4 = (move 1, target 1) = Volt Switch→Garchomp; a6 = (move 2, target 0) = Protect (field move)
    assert lines[3] == ("slot 2 Raichu [given Heat Wave→spread]: ✔ Fake Out→Incineroar 47% · "
                        "Volt Switch→Garchomp 30% · Protect 12%")
    # slot 2's move index 3 (Nuzzle) with target 0 = action 9 was futility-dropped
    assert lines[4] == "sampled at τ=0.3 · futility dropped: Nuzzle→Incineroar"


def test_turn_text_independent_decode_and_offlist_pick():
    ctx = {"turn": 7, "bench": [], "own": [{"species": "Charizard", "moves": MOVES}, None],
           "opp": [None, {"species": "Kingambit", "item": None, "item_src": "none", "ability": None,
                          "ability_src": "none"}]}
    mask = [True] * 16
    decode = {"pair": False, "probs": (_probs(a0=0.5, a3=0.3, a6=0.15, a7=0.05), None),
              "masks": (mask, mask), "tau": 0.0, "top_p": 1.0, "dropped": []}
    txt = turn_text(ctx, decode, picks=(7, None), wp=None)
    lines = txt.split("\n")
    assert lines[0] == "turn 7"
    assert lines[1] == "opp: Kingambit (item ?)"
    assert lines[2] == ("slot 1 Charizard: Heat Wave→spread 50% · Protect 30% · Air Slash→opp slot 1 15%"
                        " · picked: Air Slash→Kingambit 5%")
    assert lines[3] == "slot 2: empty / fainted"
    assert len(lines) == 4                            # no τ line at argmax
    # an empty decode never breaks the narration
    assert "(no decode numbers)" in turn_text(ctx, {}, picks=(0, 0))


def test_tp_text_set_head_with_near_tie():
    our = ["Charizard", "Raichu", "Swampert", "Sneasler", "Farigiraf", "Mawile"]
    opp = ["Incineroar", "Garchomp", "Kingambit", "Farigiraf", "Pelipper", "Whimsicott"]
    stash = {"path": "set_head", "subsets": [[0, 1, 2, 3], [0, 1, 2, 4], [1, 2, 3, 4]],
             "scores": [2.31, 2.12, 1.5], "lead_logits": [1.0, 0.87, 0.8, 0.3, 0.2, 0.1],
             "set": [0, 1, 2, 4], "leads": [0, 2], "eps": 1.0, "set_dev": True, "lead_dev": True}
    txt = tp_text(our, opp, [0, 2, 1, 4], 2, stash, arm="2b_tp_eps05")
    lines = txt.split("\n")
    assert lines[0] == "TEAM PREVIEW vs Incineroar, Garchomp, Kingambit, Farigiraf, Pelipper, Whimsicott · arm 2b_tp_eps05"
    assert lines[1] == "NET: bring Charizard + Swampert + Raichu + Farigiraf — leads Charizard + Swampert"
    assert lines[2] == ("set head: Charizard + Raichu + Swampert + Farigiraf 2.12 (best was "
                        "Charizard + Raichu + Swampert + Sneasler 2.31) · runner-up "
                        "Charizard + Raichu + Swampert + Sneasler 2.31 (Δ+0.19)")
    assert lines[3] == "leads Charizard + Swampert 1.80 (argmax pair Charizard + Raichu 1.87, Δ-0.07)"
    assert lines[4] == "near-tie sampling (eps 1): SET DEVIATED from the argmax · LEADS DEVIATED from the argmax"
    # argmax path: best set, kept leads, no eps line
    stash2 = dict(stash, set=[0, 1, 2, 3], leads=[0, 1], eps=0.0, set_dev=False, lead_dev=False)
    lines2 = tp_text(our, opp, [0, 1, 2, 3], 2, stash2).split("\n")
    assert lines2[2].startswith("set head: Charizard + Raichu + Swampert + Sneasler 2.31 (best) · runner-up")
    assert lines2[3] == "leads Charizard + Raichu 1.87 (next pair Raichu + Swampert 1.67, Δ-0.20)"
    assert len(lines2) == 4
    # greedy + unknown paths and an empty stash are all words, never exceptions
    g = tp_text(our, opp, [0, 1, 2, 3], 2, {"path": "greedy", "bring_logits": [1.0, 0.9, 0.8, 0.7, 0.1, 0.0],
                                            "lead_logits": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]}, source="NET")
    assert "greedy: bring logits Charizard 1.00 · Raichu 0.90" in g and "lead logits Charizard 0.50" in g
    assert tp_text(our, opp, [0, 1, 2, 3], 2, {}, source="HEURISTIC").endswith("leads Charizard + Raichu")


def test_replacement_text():
    mask = [False] * 12 + [True, True, False, False]
    txt = replacement_text([{"slot": 1, "species": None, "probs": _probs(a12=0.62, a13=0.38),
                             "mask": mask, "pick": 12}], BENCH, arm="era2")
    assert txt == "forced replacement · arm era2: slot 2 (fainted) → ✔ switch→Swampert 62% · switch→Sneasler 38%"
    assert replacement_text([], BENCH) == "forced replacement: nothing to replace"


def test_feed_ring_buffer_order_and_pruning():
    clock = {"t": 1000.0}
    f = ThoughtFeed(maxlen=3, max_battles=2, now=lambda: clock["t"])
    f.add(TAG, "tp", "preview", turn=0, arm="era4_2b", opponent="Alice")
    for t in (1, 2, 3):
        clock["t"] += 1
        f.add(TAG + "-privsuffix", "turn", f"turn {t}", turn=t)
    f.amend(TAG, "gimmick: Mega Charizard NOW (margin +1.30)", turn=3)
    f.amend(TAG, "late note", turn=9)               # turn mismatch -> its own note entry
    s = f.summary(live_tags=[TAG])
    assert s["battles"] == [{"tag": TAG, "live": True, "finished": False, "arm": "era4_2b",
                             "opponent": "Alice", "turn": 9, "n": 3}]
    texts = [e["text"] for e in s["entries"]]
    assert texts == ["late note", "turn 3\ngimmick: Mega Charizard NOW (margin +1.30)", "turn 2"]  # maxlen 3, newest first
    assert all(e["tag"] == TAG for e in s["entries"])
    # a second battle, then a third evicts the FINISHED one first
    other = "battle-gen9championsvgc2026regmb-200"
    f.add(other, "turn", "x", turn=1)
    f.mark_finished(other)
    third = "battle-gen9championsvgc2026regmb-300"
    f.add(third, "turn", "y", turn=1)
    tags = {b["tag"] for b in f.summary(live_tags=[TAG, third])["battles"]}
    assert tags == {TAG, third}
    # live battles come first, then finished ones; the entry cap holds
    f.mark_finished(TAG)
    s2 = f.summary(live_tags=[third], limit=2)
    assert [b["tag"] for b in s2["battles"]] == [third, TAG]
    assert len(s2["entries"]) == 2 and s2["total"] == 7


def test_install_is_the_only_switch():
    class P:
        pass
    p = P()
    assert getattr(p, "_thoughts", None) is None
    f = ThoughtFeed().install(p)
    assert p._thoughts is f and "thought feed ACTIVE" in f.banner()


def test_battle_context_from_duck_typed_battle():
    class Move:
        def __init__(self, mid, target):
            self.id, self.target = mid, target

    class T:                                        # a poke-env Target-like enum member
        def __init__(self, name):
            self.name = name

    class Mon:
        def __init__(self, species, moves=(), item="unknown_item", ability=None, hp=1.0, fainted=False):
            self.species, self.item, self.ability = species, item, ability
            self.current_hp_fraction, self.fainted = hp, fainted
            self.moves = {m.id: m for m in moves}

    class Battle:
        turn = 4
        active_pokemon = [Mon("charizard", [Move("heatwave", T("ALL_ADJACENT_FOES")), Move("protect", T("SELF")),
                                            Move("airslash", T("NORMAL")), Move("helpinghand", T("ADJACENT_ALLY"))]),
                          None]
        opponent_active_pokemon = [Mon("incineroar", item="unknown_item", ability="intimidate", hp=0.72),
                                   Mon("garchomp", item="lifeorb", fainted=True)]

    class Belief:
        def top_item(self, sp):
            return "Leftovers" if sp == "incineroar" else None

        def item_distribution(self, sp, k=5):
            return [{"name": "Leftovers", "p": 0.6}]

        def top_ability(self, sp):
            return None

        def _resolve(self, sp):
            return {"incineroar": "Incineroar"}.get(sp)

    b = Battle()
    ctx = TF.battle_context(b, belief=Belief(),
                            move_list_fn=lambda battle, slot, mon: list(mon.moves.values()),
                            bench_fn=lambda battle: [Mon("swampert")])
    assert ctx["turn"] == 4 and ctx["bench"] == ["Swampert"]
    assert ctx["own"][0]["species"] == "Charizard" and ctx["own"][1] is None
    assert [m["target"] for m in ctx["own"][0]["moves"]] == ["spread", "field", "normal", "ally"]
    assert ctx["own"][0]["moves"][0]["name"] == "Heatwave"          # title-case fallback in tests
    assert ctx["opp"][0] == {"species": "Incineroar", "hp": 72, "item": "Leftovers", "item_src": "belief",
                             "item_pct": 60, "ability": "Intimidate", "ability_src": "seen"}
    assert ctx["opp"][1] is None                                     # fainted foe hidden
    # the narration composes end-to-end from that context
    txt = turn_text(ctx, {"pair": False, "probs": (_probs(a0=1.0), None), "masks": ([True] * 16, [])},
                    picks=(0, None))
    assert "slot 1 Charizard: ✔ Heatwave→spread 100%" in txt and "slot 2: empty / fainted" in txt
    # USER bug 09-02: a mega'd foe (poke-env: base species, mega ability, NO stone) → the stone is
    # the revealed item and the species says (Mega)
    from v_dance.play import matchup_book as MB
    if MB._dex() is not None:
        view = TF._opp_view(Mon("gardevoir", item="unknown_item", ability="pixilate", hp=0.5), Belief())
        assert view["species"] == "Gardevoir (Mega)" or view["species"].endswith("(Mega)")
        assert (view["item"], view["item_src"], view["ability"], view["ability_src"]) == \
            ("Gardevoirite", "seen", "Pixilate", "seen")
        assert "Gardevoirite revealed · Pixilate revealed" in TF.opp_line([view])
