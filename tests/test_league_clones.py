"""League P1 (2026-09-30): the 'clone' opponent kind — behaviour-cloned human opponents in the self-play league."""
import numpy as np

from v_dance.play.parallel_battles import collect_account_names
from v_dance.selfplay.league import LeagueConfig, OpponentLeague, make_league_opponent
from v_dance.selfplay.mp_collect import _spec_from_sample


def test_no_clones_keeps_the_old_mixture():
    lg = OpponentLeague(latest_path="latest.pt")
    lg.admit("s1", "s1.pt", 1)
    mix = lg.mixture()
    s = lg.scripted_fraction()
    assert mix["clone"] == 0.0
    assert abs(mix["latest"] - (1 - s) * 0.5 / 0.8) < 1e-12 and abs(mix["past"] - (1 - s) * 0.3 / 0.8) < 1e-12


def test_clones_take_their_share_and_are_sampled():
    lg = OpponentLeague(latest_path="latest.pt", clones=("nemesis.pt", "arch03.pt"),
                        cfg=LeagueConfig(clone_frac=0.3))
    lg.admit("s1", "s1.pt", 1)
    mix = lg.mixture()
    assert abs(sum(mix.values()) - 1.0) < 1e-12 and mix["clone"] == 0.3
    rng = np.random.default_rng(0)
    kinds = [lg.sample(rng)[0] for _ in range(20000)]
    assert abs(kinds.count("clone") / len(kinds) - 0.3) < 0.02
    paths = {val for kind, val in (lg.sample(rng) for _ in range(2000)) if kind == "clone"}
    assert {"nemesis.pt", "arch03.pt"} <= paths


def test_clone_state_round_trips_and_old_states_still_load():
    lg = OpponentLeague(latest_path="l.pt", clones=("c.pt",), cfg=LeagueConfig(clone_frac=0.25))
    back = OpponentLeague.from_obj(lg.to_obj())
    assert back.clones == ("c.pt",) and back.cfg.clone_frac == 0.25
    old = {"latest_path": "l.pt", "scripted": ["random"], "snapshots": [], "cfg": {"latest_frac": 0.5}}
    assert OpponentLeague.from_obj(old).clones == ()


def test_clone_spec_builds_a_model_player_everywhere():
    made = {}
    make_league_opponent(("clone", "arch03.pt"), username="u", team="t",
                         make_model=lambda u, t, model_path: made.setdefault("path", model_path),
                         make_scripted=lambda *a: None)
    assert made["path"] == "arch03.pt"
    spec = _spec_from_sample(("clone", "arch03.pt"), "ta", "tb", 5, 7, 2)
    assert spec.kind == "clone" and spec.opp_ref == "arch03.pt" and spec.snapshot_id is None
    assert collect_account_names("clone", 7)[1].startswith("LGkx")


def test_archetype_clone_plays_its_own_archetype_teams():
    from v_dance.selfplay.build_clone import clone_archetype, clone_team_map
    from v_dance.selfplay.mp_collect import build_chunk_specs
    a3 = "BC/checkpoints_attn_clone_arch03_k10_20260930/battle_base.pt"
    nem = "BC/checkpoints_attn_clone_nemesis_regmb_regmc_20260930/battle_base.pt"
    assert (clone_archetype(a3), clone_archetype(nem)) == (3, None)
    z = {"t_rain": 3, "t_sun": 6, "t_bal": 3}
    assert clone_team_map([a3, nem], list(z), team_archetype=z.get) == {a3: ["t_rain", "t_bal"]}
    lg = OpponentLeague(latest_path="l.pt", clones=(a3, nem), cfg=LeagueConfig(clone_frac=0.3),
                        clone_teams={a3: ("t_rain", "t_bal")})
    lg.cover_kinds = True
    specs = build_chunk_specs(lg, ["p1", "p2", "p3"], 12, chunk_size=1)
    by_kind = {(s.kind, s.opp_ref): s.team_b for s in specs}
    assert by_kind[("clone", a3)] in ("t_rain", "t_bal") and by_kind[("clone", nem)] in ("p1", "p2", "p3")
    assert OpponentLeague.from_obj(lg.to_obj()).clone_teams == {a3: ("t_rain", "t_bal")}


def test_resume_drops_league_members_whose_checkpoint_is_gone():
    lg = OpponentLeague(latest_path="l.pt")
    for g in (13, 15, 16):
        lg.admit(f"gen{g}", f"ck/gen{g}.pt", g)
    gone = lg.drop_missing(exists=lambda p: not p.endswith("gen15.pt"))
    assert [s.snapshot_id for s in gone] == ["gen15"]
    assert [s.snapshot_id for s in lg.snapshots] == ["gen13", "gen16"]
    assert lg.drop_missing(exists=lambda p: True) == []
