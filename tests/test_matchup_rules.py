"""2026-10-01: matchup rules — the Rillaboom → Mega Salamence rule (v_dance/play/matchup_rules.py), its team-preview
constraint (model_io.team_order ``require``) and its per-arm opt-in (serve_bandit ``matchup_rules``)."""
import json
from pathlib import Path

import pytest

from v_dance.play.matchup_rules import active_rule, mega_override

OURS = ["Salamence", "Sneasler", "Indeedee", "Tyranitar", "Corviknight", "Excadrill"]


def test_the_rule_fires_only_on_its_matchup():
    r = active_rule(["rillaboom_salamence"], OURS, ["Rillaboom", "Incineroar", "Gholdengo", "Kingambit", "Pelipper", "Basculegion"])
    assert r and r["name"] == "rillaboom_salamence" and r["bring_index"] == 0
    assert active_rule(["rillaboom_salamence"], OURS, ["Rillaboom", "Archaludon", "A", "B", "C", "D"]) is None
    assert active_rule(["rillaboom_salamence"], OURS, ["Incineroar", "A", "B", "C", "D", "E"]) is None
    assert active_rule([], OURS, ["Rillaboom"]) is None                         # no arm opt-in, no rule
    assert active_rule(["rillaboom_salamence"], OURS[1:], ["Rillaboom"]) is None  # team without Salamence


def test_mega_goes_to_salamence_and_is_held_from_tyranitar():
    rule = active_rule(["rillaboom_salamence"], OURS, ["Rillaboom"])
    alive = ["salamence", "tyranitar", "excadrill", "indeedee"]
    kw = dict(mega=1, none=0)
    # Tyranitar wants to mega, Salamence doesn't: swapped
    assert mega_override(rule, alive, ["tyranitar", "salamence"], [True, True], [True, True], [1, 0], **kw) == [0, 1]
    # Salamence switching out / unable to mega this turn: Tyranitar is still held
    assert mega_override(rule, alive, ["tyranitar", "salamence"], [True, False], [True, True], [1, 0], **kw) == [0, 0]
    assert mega_override(rule, alive, ["tyranitar", "salamence"], [True, True], [True, False], [1, 0], **kw) == [0, 0]
    # spent: Salamence already mega'd (its live species id changes) or fainted -> picks pass through
    spent = ["salamencemega", "tyranitar"]
    assert mega_override(rule, spent, ["tyranitar", "salamencemega"], [True, True], [True, False], [1, 0], **kw) == [1, 0]
    assert mega_override(rule, ["tyranitar"], ["tyranitar", None], [True, False], [True, False], [1, 0], **kw) == [1, 0]
    assert mega_override(None, alive, ["tyranitar", "salamence"], [True, True], [True, True], [1, 0], **kw) == [1, 0]


def test_arms_opt_in_and_the_bundle_carries_the_rules(tmp_path):
    from v_dance.play.serve_bandit import apply_bundle, load_arms, load_bundle
    cfg = tmp_path / "b.json"
    cfg.write_text(json.dumps({"arms": [{"name": "a", "battle_ckpt": "default"},
                                        {"name": "a_rule", "battle_ckpt": "default",
                                         "matchup_rules": ["rillaboom_salamence"]}]}), encoding="utf-8")
    arms = {a.name: a for a in load_arms(cfg)}
    assert arms["a"].matchup_rules == () and arms["a_rule"].matchup_rules == ("rillaboom_salamence",)
    b = load_bundle(arms["a_rule"], {}, default_battle="x.pt", default_tp="y.pt",
                    loader=lambda kind, p: (object(), {}) if kind == "battle" else (object(), {}, {}))
    class P: pass
    p = P()
    apply_bundle(p, b)
    assert p._matchup_rules == ("rillaboom_salamence",)


_TP = Path(__file__).resolve().parents[1] / "ai_train_scripts" / "teamPreview_model" / "checkpoints_set_ots_ctx" / "teampreview_sbda.pt"


@pytest.mark.skipif(not _TP.is_file(), reason="served TP checkpoint not on disk")
def test_the_real_picker_brings_salamence_under_the_constraint():
    from v_dance.formats import default_format, pikalytics_path_for
    from v_dance.parser.belief_state import BeliefState
    from v_dance.play import model_io as M
    model, vocab, cfg = M.load_team_chooser(str(_TP), device="cpu")
    belief = BeliefState(pikalytics_path_for(default_format()))
    opp = ["Rillaboom", "Incineroar", "Kingambit", "Garchomp", "Pelipper", "Sinistcha"]
    free = M.team_order(model, vocab, cfg, OURS, opp, 4, "cpu", belief=belief)
    ruled = M.team_order(model, vocab, cfg, OURS, opp, 4, "cpu", belief=belief, require=[0], require_lead=[0])
    assert len(ruled) == 4 and len(set(ruled)) == 4 and 0 in ruled[:2]      # brought AND leading (order = leads first)
    assert set(free) != set(ruled) or 0 in free               # the constraint only bites when the net left it home


def test_required_lead_keeps_the_best_partner():
    from v_dance.play.model_io import _leads_with
    ll = [0.1, 0.9, 0.5, 0.7, 0.2, 0.3]
    assert _leads_with([0, 1, 3, 5], ll, 2) == [1, 3]                 # no rule: the two best lead logits
    assert _leads_with([0, 1, 3, 5], ll, 2, [0]) == [0, 1]            # rule: the required mon + the best partner
    assert _leads_with([1, 3, 5, 2], ll, 2, [0]) == [1, 3]            # required mon not brought -> unchanged
