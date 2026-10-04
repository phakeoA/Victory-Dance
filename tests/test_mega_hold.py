"""2026-10-03 MEGA-HOLD — the probe (v_dance/eval/mega_hold_probe.py), the self-play exploration
(v_dance/selfplay/mega_hold.py + its player / recorder / PPO plumbing) and the ``megatime`` drill."""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("poke_env")

from v_dance.selfplay import mega_hold as MH  # noqa: E402


class _Mon:
    def __init__(self, species, item=None, ability=None, fainted=False):
        self.species = species
        self.base_species = species
        self.item, self.ability = item, ability
        self.fainted = fainted


ALL = lambda mon: True                                                   # noqa: E731 — relevance not under test


class _Battle:
    def __init__(self, active, team, opp, turn=1, weather=None, tag="battle-x-1"):
        self.active_pokemon = active
        self.team = {m.species: m for m in team}
        self.teampreview_opponent_team = opp
        self.opponent_team = {}
        self.turn = turn
        self.weather = weather or {}
        self.battle_tag = tag


class _W:
    def __init__(self, name):
        self.name = name


def _cfg(**kw):
    c = MH.MegaHoldConfig(**kw).to_spec()
    c["rng"] = np.random.default_rng(0)
    return c


OURS = [_Mon("tyranitar", "tyranitarite", "sandstream"), _Mon("indeedee", "psychicseed", "psychicsurge"),
        _Mon("salamence", "salamencite", "intimidate"), _Mon("sneasler"), _Mon("excadrill"), _Mon("corviknight")]
RAIN = [_Mon("pelipper"), _Mon("archaludon"), _Mon("incineroar"), _Mon("gholdengo")]
DRY = [_Mon("incineroar"), _Mon("gholdengo"), _Mon("rillaboom"), _Mon("kingambit")]


# ── config + weather setters ──────────────────────────────────────────────────────────────────────────────
def test_config_validates_and_reports_on():
    for bad in ({"p": 1.5}, {"p_weather": -0.1}, {"kmin": 1}, {"kmin": 5, "kmax": 4}, {"w_min": 2.0}):
        with pytest.raises(ValueError):
            MH.MegaHoldConfig(**bad)
    assert not MH.MegaHoldConfig().on and MH.MegaHoldConfig(p_weather=0.5).on
    spec = MH.MegaHoldConfig(p=0.1, p_weather=0.5, kmin=2, kmax=4, w_min=0.2).to_spec()
    assert spec == {"p": 0.1, "p_weather": 0.5, "kmin": 2, "kmax": 4, "end_on_weather": True, "w_min": 0.2}


def test_weather_setters_come_from_the_dex_incl_megas():
    assert MH.species_weather_kinds("pelipper") == {"rain"} and MH.species_weather_kinds("tyranitar") == {"sand"}
    assert MH.species_weather_kinds("charizard") == {"sun"}              # via its MEGA (Charizardite Y: Drought)
    assert MH.species_weather_kinds("archaludon") == frozenset()
    assert MH.weather_conflict(["tyranitar", "indeedee"], ["pelipper", "archaludon"])
    assert not MH.weather_conflict(["indeedee"], ["pelipper"])          # we set no weather → nothing to fight for
    assert not MH.weather_conflict(["tyranitar"], ["hippowdon"])        # same kind
    assert MH.current_weather_kinds(_Battle([], OURS, RAIN, weather={_W("RAINDANCE"): 1})) == {"rain"}


def test_decide_uses_the_weather_rate_and_draws_k():
    b = _Battle([], OURS, RAIN)
    st = MH.decide(b, _cfg(p=0.0, p_weather=1.0, kmin=3, kmax=5))
    assert st["hold"] and st["conflict"] and 3 <= st["k"] <= 5 and st["ours"] == {"sand"} and not st["ended"]
    st = MH.decide(_Battle([], OURS, DRY), _cfg(p=0.0, p_weather=1.0))
    assert not st["hold"] and not st["conflict"] and st["ended"]       # no weather fight → the base rate (0)


def test_p_none_matches_the_ppo_definition():
    from v_dance.rl.policy_eval import masked_log_softmax
    rng = np.random.default_rng(3)
    for tau in (0.7, 1.0, 1.6):
        logits = rng.normal(size=3) * 3
        mask = [True, True, False]
        ref = masked_log_softmax(torch.tensor(logits, dtype=torch.float32), torch.tensor(mask), tau).exp()[0]
        assert MH.p_none(logits, mask, tau) == pytest.approx(float(ref), rel=1e-5)
    assert MH.p_none([0.0, 1.0, 0.0], [False, True, False], 1.0) == 0.0   # none illegal → 0, never a NaN


# ── the force + its importance weight ─────────────────────────────────────────────────────────────────────
def _player(**cfg):
    return types.SimpleNamespace(_mega_hold=_cfg(**cfg))


def _mask_for(capable):
    return lambda battle, slot: [True, bool(capable[slot]), False]


def test_apply_forces_none_on_mega_capable_move_slots_and_stashes_pi_none():
    p = _player(p=1.0, p_weather=1.0, kmin=4, kmax=4)
    b = _Battle([], OURS, RAIN, turn=1)
    glog = (np.array([0.0, 2.0, 0.0]), np.array([0.0, 2.0, 0.0]))
    out = [1, 0]                                                       # slot 0 sampled MEGA
    w = MH.apply(p, b, glog, out, (0, 3), none=0, mega=1, switch_offset=12, tau=1.0,
                 build_mask=_mask_for((True, False)), relevant=ALL)
    assert out == [0, 0] and w == pytest.approx(MH.p_none(glog[0], [True, True, False], 1.0))
    assert p._mega_hold_w[(b.battle_tag, "turn")] == (1, w) and p._mega_hold_state[b.battle_tag]["forced_steps"] == 1
    # a SWITCH slot never sampled a gimmick → not forced, not weighted
    out = [0, 0]
    assert MH.apply(p, b, glog, out, (12, 3), none=0, mega=1, switch_offset=12, tau=1.0,
                    build_mask=_mask_for((True, False)), relevant=ALL) is None
    # two forced slots → the product
    out = [1, 1]
    w2 = MH.apply(p, b, glog, out, (0, 1), none=0, mega=1, switch_offset=12, tau=1.0,
                  build_mask=_mask_for((True, True)), relevant=ALL)
    assert w2 == pytest.approx(MH.p_none(glog[0], [True, True, False], 1.0) ** 2) and out == [0, 0]


def test_the_hold_ends_at_turn_k_or_when_their_weather_is_up():
    p = _player(p=1.0, p_weather=1.0, kmin=3, kmax=3)
    b = _Battle([], OURS, RAIN, turn=2)
    glog = (np.array([0.0, 2.0, 0.0]), np.zeros(3))
    kw = dict(none=0, mega=1, switch_offset=12, tau=1.0, build_mask=_mask_for((True, False)), relevant=ALL)
    assert MH.apply(p, b, glog, [1, 0], (0, 0), **kw) is not None       # turn 2 < k 3 → held
    b.turn = 3
    out = [1, 0]
    assert MH.apply(p, b, glog, out, (0, 0), **kw) is None and out == [1, 0]   # turn k → free (the policy megas)
    assert p._mega_hold_state[b.battle_tag]["end"] == "turn"
    b.turn = 1
    assert MH.apply(p, b, glog, [1, 0], (0, 0), **kw) is None           # ended for good
    p2 = _player(p=1.0, p_weather=1.0, kmin=6, kmax=6)
    b2 = _Battle([], OURS, RAIN, turn=1, tag="battle-x-2")
    assert MH.apply(p2, b2, glog, [1, 0], (0, 0), **kw) is not None
    b2.turn, b2.weather = 2, {_W("RAINDANCE"): 2}                      # THEIR rain is up → mega free now
    out = [1, 0]
    assert MH.apply(p2, b2, glog, out, (0, 0), **kw) is None and out == [1, 0]
    assert p2._mega_hold_state[b2.battle_tag]["end"] == "weather"
    p3 = _player(p=1.0, p_weather=1.0, kmin=6, kmax=6, end_on_weather=False)
    b3 = _Battle([], OURS, RAIN, turn=2, weather={_W("RAINDANCE"): 2}, tag="battle-x-3")
    assert MH.apply(p3, b3, glog, [1, 0], (0, 0), **kw) is not None     # keep-on-weather: still held
    b4 = _Battle([], OURS, RAIN, turn=1, weather={_W("SANDSTORM"): 1}, tag="battle-x-4")
    assert MH.apply(_player(p=1.0, p_weather=1.0, kmin=6, kmax=6), b4, glog, [1, 0], (0, 0), **kw) is not None


def test_w_min_floors_and_tau_zero_never_explores():
    p = _player(p=1.0, p_weather=1.0, kmin=6, kmax=6, w_min=0.3)
    b = _Battle([], OURS, RAIN)
    glog = (np.array([0.0, 9.0, 0.0]), np.zeros(3))                      # π(none) ~ 1e-4
    w = MH.apply(p, b, glog, [1, 0], (0, 0), none=0, mega=1, switch_offset=12, tau=1.0,
                 build_mask=_mask_for((True, False)), relevant=ALL)
    assert w == pytest.approx(0.3)
    assert MH.apply(_player(p=1.0, p_weather=1.0), _Battle([], OURS, RAIN, tag="b9"), glog, [1, 0], (0, 0), none=0,
                    mega=1, switch_offset=12, tau=0.0, build_mask=_mask_for((True, False)), relevant=ALL) is None


def test_game_meta_pops_the_record_and_stale_weights():
    p = _player(p=1.0, p_weather=1.0, kmin=6, kmax=6)
    b = _Battle([], OURS, RAIN)
    MH.apply(p, b, (np.array([0.0, 2.0, 0.0]), np.zeros(3)), [1, 0], (0, 0), none=0, mega=1, switch_offset=12,
             tau=1.0, build_mask=_mask_for((True, False)), relevant=ALL)
    meta = MH.game_meta(p, b.battle_tag)
    assert meta == {"held": True, "k": 6, "conflict": True, "end": None, "forced_steps": 1, "reasons": {"any": 1}}
    assert b.battle_tag not in p._mega_hold_state and not p._mega_hold_w
    assert MH.game_meta(p, "never-seen") is None


def test_the_player_hook_forces_the_learner_only():
    """VGCPlayer._select_gimmicks (unbound, fake self — tests/test_gimmick_player.py): a mega-favouring head on a
    mega-capable Tyranitar megas; with _mega_hold in a held game it plays none and stashes w; the serve path
    (no _collect_sample) never explores."""
    from tests.test_gimmick_player import _mega_favoring_model
    from v_dance.encoders.state_encoder import GIMMICK_MEGA, GIMMICK_NONE, STATE_DIM
    from v_dance.play.player import VGCPlayer
    sv = np.zeros(STATE_DIM, np.float32)
    battle = _Battle([_Mon("tyranitar", "tyranitarite", "sandstream"), _Mon("indeedee")], OURS, RAIN, turn=1)
    serve = types.SimpleNamespace(_model=_mega_favoring_model(), _model_heads=("our_a", "our_b"), _device="cpu",
                                  _mega_hold=_cfg(p=1.0, p_weather=1.0, kmin=6, kmax=6))
    assert VGCPlayer._select_gimmicks(serve, battle, sv, 0, 0)[0] == GIMMICK_MEGA   # serve: argmax, no exploration
    learner = types.SimpleNamespace(_model=_mega_favoring_model(), _model_heads=("our_a", "our_b"), _device="cpu",
                                    _collect_sample=True, _temperature=1.0, _rng=np.random.default_rng(1),
                                    _mega_hold=_cfg(p=1.0, p_weather=1.0, kmin=6, kmax=6))
    g = VGCPlayer._select_gimmicks(learner, battle, sv, 0, 0)
    assert g[0] == GIMMICK_NONE                                          # Tyranitar's mega HELD
    assert g[1] != GIMMICK_MEGA                                          # Indeedee cannot mega — not forced
    turn, w = learner._mega_hold_w[(battle.battle_tag, "turn")]
    assert turn == 1 and 0.0 < w < 1e-3                                  # π(none) of a head biased [0, 10, 0]
    off = types.SimpleNamespace(_model=_mega_favoring_model(), _model_heads=("our_a", "our_b"), _device="cpu",
                                _collect_sample=True, _temperature=1.0, _rng=np.random.default_rng(1))
    assert VGCPlayer._select_gimmicks(off, battle, sv, 0, 0)[0] == GIMMICK_MEGA     # no _mega_hold → untouched


# ── recording + schema + PPO ──────────────────────────────────────────────────────────────────────────────
def test_the_weight_is_recorded_and_round_trips():
    from v_dance.rl.schema import Transition
    t = Transition(state=np.zeros(4, np.float32), action_s0=0, action_s1=1, is_weight=0.25)
    d = t.to_obj()
    assert d["is_weight"] == 0.25 and Transition.from_obj(d).is_weight == 0.25
    plain = Transition(state=np.zeros(4, np.float32), action_s0=0, action_s1=1).to_obj()
    assert "is_weight" not in plain and Transition.from_obj(plain).is_weight == 1.0   # legacy / ordinary step


def test_the_self_play_recorder_pops_the_stash(monkeypatch):
    import v_dance.selfplay.game_runner as GR
    seen = {}
    monkeypatch.setattr(GR, "record_decision", lambda c, ac, **kw: seen.update(kw))
    monkeypatch.setattr(GR, "build_legal_action_mask", lambda b, s: [1] * 4)
    monkeypatch.setattr(GR, "build_gimmick_legal_mask", lambda b, s: [1, 1, 0])
    tag = "battle-x-7"
    fake = types.SimpleNamespace(_sampling_masks={}, _collectors={}, _ac=None, _tau=1.0,
                                 _source_counts={"rejected_resample": 0}, _mega_hold_w={(tag, "turn"): (1, 0.125)})
    fake._collector_for = lambda b: types.SimpleNamespace(last_step=lambda: None)
    b = types.SimpleNamespace(battle_tag=tag, turn=1)
    GR.SelfPlayVGCPlayer._record_rl_decision(fake, b, np.zeros(3), 0, 1, 0, 0, "model", "turn")
    assert seen["is_weight"] == 0.125 and not fake._mega_hold_w        # popped
    GR.SelfPlayVGCPlayer._record_rl_decision(fake, b, np.zeros(3), 0, 1, 0, 0, "model", "turn")
    assert seen["is_weight"] == 1.0                                      # the policy's own step
    # review: a stash left by a DISCARDED step of turn 1 must never weight turn 2's (unforced) step
    fake._mega_hold_w[(tag, "turn")] = (1, 0.125)
    b.turn = 2
    GR.SelfPlayVGCPlayer._record_rl_decision(fake, b, np.zeros(3), 0, 1, 0, 0, "model", "turn")
    assert seen["is_weight"] == 1.0 and not fake._mega_hold_w


def test_ppo_weights_scale_the_surrogate_and_none_is_byte_identical():
    from v_dance.rl.ppo import PPOConfig, ppo_losses
    cfg = PPOConfig(standardize_adv=False, entropy_coef=0.0, value_coef=0.0)
    new = torch.tensor([-0.5, -1.0], requires_grad=True)
    kw = dict(old_logprob=torch.tensor([-0.5, -1.0]), advantages=torch.tensor([1.0, 2.0]),
              value_pm=torch.zeros(2), old_value_pm=torch.zeros(2), returns=torch.zeros(2), entropy=torch.zeros(2),
              cfg=cfg)
    base, st = ppo_losses(new_logprob=new, **kw)
    ones, _ = ppo_losses(new_logprob=new, weights=torch.ones(2), **kw)
    assert float(base) == float(ones) and st["is_weight_mean"] == 1.0
    half, st2 = ppo_losses(new_logprob=new, weights=torch.tensor([1.0, 0.0]), **kw)
    assert float(half) == pytest.approx(-(1.0 * 1.0) / 2) and st2["is_weight_mean"] == 0.5   # step 2 drops out
    half.backward()
    assert float(new.grad[1]) == 0.0 and float(new.grad[0]) != 0.0


def test_ppo_loss_from_batch_reads_the_step_weights(monkeypatch):
    from v_dance.rl import ppo as PPO
    from v_dance.rl.schema import Transition
    got = {}

    def fake_losses(**kw):
        got["w"] = kw.get("weights")
        return torch.tensor(0.0), {}
    monkeypatch.setattr(PPO, "ppo_losses", fake_losses)
    ev = types.SimpleNamespace(logprob=torch.zeros(2), value_pm=torch.zeros(2), entropy=torch.zeros(2),
                               kl_to_ref=None, opp_ce=None, atoms_logits=None, pair_flips=None)
    monkeypatch.setattr(PPO.policy_eval, "ppo_forward", lambda *a, **k: ev)
    ac = types.SimpleNamespace(critic=None)
    ts = [Transition(state=np.zeros(2, np.float32), action_s0=0, action_s1=0),
          Transition(state=np.zeros(2, np.float32), action_s0=0, action_s1=0, is_weight=0.2)]
    PPO.ppo_loss_from_batch(ac, ts, [0.0, 0.0], [0.0, 0.0])
    assert torch.allclose(got["w"], torch.tensor([1.0, 0.2]))
    PPO.ppo_loss_from_batch(ac, ts[:1], [0.0], [0.0])
    assert got["w"] is None                                              # no forced step → unweighted path


# ── worker plumbing, the run's fail-fast, the readout, the preflight ─────────────────────────────────────────
def test_chunk_specs_carry_the_config_and_recording_players_get_a_seeded_rng(monkeypatch):
    from v_dance.selfplay import mp_collect as MP
    import v_dance.selfplay.game_runner as GR
    import v_dance.play.run_local_battle as R

    class FakeSP:
        def __init__(self, ac, **kw):
            self.kw = kw

    monkeypatch.setattr(GR, "SelfPlayVGCPlayer", FakeSP)
    monkeypatch.setattr(R, "load_team", lambda p: "TEAM")
    monkeypatch.setattr(R, "resolve_team_path", lambda t: t)
    spec = MP.ChunkSpec("A", "B", 2, "latest", None, None, 1, 3)
    spec.mega_hold = MH.MegaHoldConfig(p=0.1, p_weather=0.5).to_spec()
    our, opp = MP._build_players_real("AC", spec, 1.0, 0, "tp.pt")
    for p in (our, opp):
        assert p._mega_hold["p_weather"] == 0.5 and p._mega_hold["rng"] is not None
    plain = MP.ChunkSpec("A", "B", 2, "latest", None, None, 2, 3)
    our2, _ = MP._build_players_real("AC", plain, 1.0, 0, "tp.pt")
    assert not hasattr(our2, "_mega_hold")

    class League:
        def sample(self, rng):
            return ("latest",)
    specs = MP.build_chunk_specs(League(), ["A", "B"], 6, chunk_size=2,
                                 mega_hold=MH.MegaHoldConfig(p=0.2).to_spec())
    assert specs and all(s.mega_hold and s.mega_hold["p"] == 0.2 for s in specs)
    assert all(s.mega_hold is None for s in MP.build_chunk_specs(League(), ["A", "B"], 4, chunk_size=2))


def test_mega_hold_needs_multiprocess(tmp_path, capsys):
    from v_dance.selfplay.generation import run_live_generations
    with pytest.raises(SystemExit) as ei:
        run_live_generations("no_such.pt", team_pool=["A", "B"], team_chooser=None, archive_dir=tmp_path / "a",
                             collect_procs=1, mega_hold=MH.MegaHoldConfig(p_weather=0.5))
    assert ei.value.code == 2 and "need --collect-procs >= 2" in capsys.readouterr().err


def test_cli_knobs_build_the_config():
    from v_dance.selfplay.generation import _mega_hold_cfg_from_args
    ns = types.SimpleNamespace
    assert _mega_hold_cfg_from_args(ns(mega_hold_p=0.0, mega_hold_p_weather=None)) is None
    c = _mega_hold_cfg_from_args(ns(mega_hold_p=0.1, mega_hold_p_weather=None, mega_hold_kmin=2, mega_hold_kmax=5,
                                    mega_hold_keep_on_weather=False, mega_hold_w_min=0.2))
    assert (c.p, c.p_weather, c.kmax, c.end_on_weather, c.w_min) == (0.1, 0.1, 5, True, 0.2)
    c = _mega_hold_cfg_from_args(ns(mega_hold_p=0.0, mega_hold_p_weather=0.5, mega_hold_kmin=2, mega_hold_kmax=6,
                                    mega_hold_keep_on_weather=True, mega_hold_w_min=0.0))
    assert c.on and c.p == 0.0 and not c.end_on_weather


def test_summarize_trajectories_counts_games_steps_and_weights():
    def traj(mh, ws):
        meta = types.SimpleNamespace(sampling={"mega_hold": mh} if mh else {})
        return types.SimpleNamespace(meta=meta, transitions=[types.SimpleNamespace(is_weight=w) for w in ws])
    s = MH.summarize_trajectories([
        traj({"held": True, "conflict": True, "forced_steps": 2}, [0.1, 0.3, 1.0]),
        traj({"held": False, "conflict": True, "forced_steps": 0}, [1.0]),
        traj({"held": True, "conflict": False, "forced_steps": 1}, [0.5]), traj(None, [1.0])])
    assert (s["decided"], s["held"], s["conflict"], s["held_conflict"], s["forced_steps"], s["weighted_steps"]) == \
        (3, 2, 2, 1, 3, 3)
    assert s["w_mean"] == pytest.approx(0.3) and s["w_min"] == pytest.approx(0.1)
    assert "held 2/3 games (1/2 with a field fight)" in MH.format_stats(s)


def test_preflight_requires_the_exploration_to_fire():
    from v_dance.selfplay import preflight as PF

    class Rec:
        def __init__(self, g):
            self.generation, self.n_trajectories, self.update_stats = g, 5, {"loss": 0.1}
            self.promoted, self.panel, self.hof = g == 1, {}, None

    class H:
        def __init__(self, n):
            self.records = [Rec(g) for g in range(n)]

    good = [{"generation": 0, "decided": 9, "held": 3, "conflict": 4, "weighted_steps": 2}]
    ok, warns, _ = PF.check({"history": H(3), "mega_hold": good}, {"history": H(4), "mega_hold": good},
                            mega_hold_on=True)
    assert not [p for p in ok if "mega-hold" in p or "HOLD" in p or "importance" in p]
    none_held = [{"generation": 0, "decided": 9, "held": 0, "conflict": 4, "weighted_steps": 0}]
    bad, _, _ = PF.check({"history": H(3), "mega_hold": none_held}, {"history": H(4), "mega_hold": good},
                         mega_hold_on=True)
    assert any("no learner game was a HOLD game" in p for p in bad)
    unforced = [{"generation": 0, "decided": 9, "held": 3, "conflict": 0, "weighted_steps": 0}]
    bad, warns, _ = PF.check({"history": H(3), "mega_hold": unforced}, {"history": H(4), "mega_hold": good},
                             mega_hold_on=True)
    assert any("no step carried an importance weight" in p for p in bad)
    assert any("weather rate" in w for w in warns)
    bad, _, _ = PF.check({"history": H(3)}, {"history": H(4)}, mega_hold_on=True)
    assert any("reported nothing" in p for p in bad)


# ── the megatime drill ────────────────────────────────────────────────────────────────────────────────────
_PASTE = "{}\nAbility: X\n- Protect\n\n"


def _team(tmp_path, name, species):
    p = tmp_path / name
    p.write_text("".join(f"{s} @ Leftovers\n" + _PASTE.format("") for s in species), encoding="utf-8")
    return str(p)


def test_megatime_pool_needs_the_threat_and_a_setter(tmp_path):
    from v_dance.selfplay.drills import get_drill
    own = tmp_path / "Baltimore"
    own.write_text("Tyranitar @ Tyranitarite\nAbility: Sand Stream\n- Rock Slide\n\nIndeedee-F @ Psychic Seed\n"
                   "Ability: Psychic Surge\n- Follow Me\n", encoding="utf-8")
    hit = _team(tmp_path, "ArchRain", ["Pelipper", "Archaludon", "Incineroar", "Gholdengo"])
    rain_only = _team(tmp_path, "RainOnly", ["Politoed", "Basculegion", "Incineroar", "Gholdengo"])
    arch_only = _team(tmp_path, "ArchOnly", ["Archaludon", "Garchomp", "Incineroar", "Gholdengo"])
    d = get_drill("megatime:share=0.5,cap=1")
    dp = d.build_pool([hit, rain_only, arch_only], str(own))
    assert [t.path for t in dp.teams] == [hit]                           # AND, not OR
    assert dp.weights[hit] == pytest.approx(0.5) and dp.weights[rain_only] == pytest.approx(0.25)
    assert d.pressure == "field" and get_drill("megatime:pressure=off").pressure is None
    with pytest.raises(ValueError):
        get_drill("megatime:share=0").build_pool([hit], str(own))
    with pytest.raises(ValueError):
        get_drill("megatime:threat=kyogre").build_pool([hit, rain_only], str(own))


_LOG = """|player|p1|OurBot|1|
|player|p2|Rival|2|
|poke|p2|Pelipper, L50, M|
|poke|p2|Archaludon, L50, M|
|switch|p1a: Tyranitar|Tyranitar, L50, M|100/100
|switch|p2a: Archaludon|Archaludon, L50, M|100/100
|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar
|turn|1
|move|p2a: Archaludon|Electro Shot|p1a: Tyranitar
|-damage|p1a: Tyranitar|50/100
|switch|p2a: Pelipper|Pelipper, L50, M|100/100
|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Pelipper
|turn|2
|detailschange|p1a: Tyranitar|Tyranitar-Mega, L50, M
|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar
|move|p1a: Tyranitar|Rock Slide|p2a: Pelipper
|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Pelipper
|turn|3
|move|p2a: Archaludon|Electro Shot|p1b: Indeedee
|-damage|p1b: Indeedee|0 fnt
|faint|p1b: Indeedee
|win|OurBot
"""


def test_megatime_score_reads_the_mega_after_their_rain_and_the_reset():
    from v_dance.selfplay.drills import get_drill
    d = get_drill("megatime")
    g = d.score_game(_LOG, {"ourbot"}, d.ctx)
    assert g["won"] and g["hit"] and g["their_weather"] and g["mega"] and not g["mega_t1"]
    assert g["mega_after_weather"] and g["mega_reset"]
    assert g["es_ko_rain"] == 1 and g["es_ko_other"] == 0               # turn 3: rain back up → the Electro Shot KO
    early = _LOG.replace("|turn|2\n|detailschange|p1a: Tyranitar|Tyranitar-Mega, L50, M\n"
                         "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar\n", "|turn|2\n")
    early = early.replace("|turn|1\n", "|turn|1\n|detailschange|p1a: Tyranitar|Tyranitar-Mega, L50, M\n"
                                       "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar\n", 1)
    e = d.score_game(early, {"ourbot"}, d.ctx)
    assert e["mega_t1"] and not e["mega_after_weather"] and not e["mega_reset"]
    agg = d.aggregate([g, e])
    assert agg["hit_games"] == 2 and agg["mega_after_weather"] == 0.5 and agg["mega_reset"] == 0.5


def test_megatime_counts_an_electro_shot_ko_under_rain():
    from v_dance.selfplay.drills import get_drill
    d = get_drill("megatime")
    no_reset = _LOG.replace("|turn|2\n|detailschange|p1a: Tyranitar|Tyranitar-Mega, L50, M\n"
                            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar\n", "|turn|2\n")
    g = d.score_game(no_reset, {"ourbot"}, d.ctx)
    assert g["es_ko_rain"] == 1 and not g["mega"]


# ── the probe ─────────────────────────────────────────────────────────────────────────────────────────────
def test_probe_snapshots_label_the_situations():
    from v_dance.eval import mega_hold_probe as P
    snaps = P.turn_snapshots(_LOG, {"ourbot"})
    t1, t2, t3 = snaps["turns"][1], snaps["turns"][2], snaps["turns"][3]
    assert t1["active"][0] == "tyranitar" and t1["weather"] == "sand" and t1["opp_active"] == {"archaludon"}
    setters = {"pelipper", "politoed", "kyogre"}
    assert P.situation(t1, snaps["opp_six"], setters, "rain") == "hold:unseen"     # Pelipper in the back
    assert P.situation(t2, snaps["opp_six"], setters, "rain") == "reset"           # their rain is up
    assert P.situation(t3, snaps["opp_six"], setters, "rain") == "reset"
    assert P.situation(dict(t1, opp_fainted={"pelipper"}), snaps["opp_six"], setters, "rain") == "free"


def test_probe_mega_probs_and_summary():
    from tests.test_gimmick_player import _mega_favoring_model
    from v_dance.encoders.state_encoder import STATE_DIM
    from v_dance.eval import mega_hold_probe as P
    rows = [{"state": np.zeros(STATE_DIM, np.float32), "slot": s, "gmask": [1, 1, 0], "recorded": 1,
             "label": lab, "arm": "a"} for s, lab in ((0, "hold:unseen"), (1, "reset"))]
    pm, am = P.mega_probs(_mega_favoring_model(), ("our_a", "our_b"), rows)
    assert np.all(pm > 0.99) and am.all()
    summ = {r["label"]: r for r in P.summarize(rows, {"m": (pm, am)})}
    assert summ["hold"]["n"] == 1 and summ["hold"]["m:argmax_mega"] == 1.0 and summ["free"]["n"] == 0
    assert len(P.usable_rows(rows + [{"state": np.zeros(5, np.float32)}], STATE_DIM)) == 2


# ── 2026-10-03 (USER): only DELAY-RELEVANT megas are held; Salamence & co never — and guarded ─────────────────────
@pytest.mark.parametrize("species,item,ability,reason", [
    ("tyranitar", "tyranitarite", "sandstream", "field"),     # the mega re-sets sand
    ("charizard", "charizarditey", "blaze", "field"),         # the mega sets sun
    ("raichu", "raichunitex", "static", "field"),             # Electric Surge on mega
    ("metagross", "metagrossite", "clearbody", "ability"),    # the base form blocks the drop the mega would take
    ("raichu", "raichunitey", "lightningrod", "ability"),     # USER: Lightning Rod eats Electro Shot, the mega can't
    ("charizard", "charizarditex", "blaze", "type"),          # Fire/Flying → Fire/Dragon: the Ground immunity is gone
    ("golisopod", "golisopite", "emergencyexit", "type"),     # Bug/Water → Bug/Steel
    ("metagross", "metagrossite", "lightmetal", None),        # same types, nothing guarding → never held
    ("raichu", "raichunitey", "static", None),
    ("salamence", "salamencite", "intimidate", None),         # USER: "mega salamence doesn't really need to delay"
    ("tyranitar", "leftovers", "sandstream", None),           # no stone → no mega → nothing to hold
])
def test_delay_reason_reads_the_stone_the_base_ability_and_the_types(species, item, ability, reason):
    assert MH.delay_reason(_Mon(species, item, ability)) == reason
    assert MH.delay_relevant(_Mon(species, item, ability)) is (reason is not None)
    assert MH.delay_reason(None) is None and MH.delay_relevant(None) is False


def test_the_hold_ends_only_in_the_domain_our_field_mega_fights_and_counts_reasons():
    class _F:
        def __init__(self, name):
            self.name = name
    raichu_team = [_Mon("raichu", "raichunitex", "static"), _Mon("garchomp"), _Mon("incineroar")]
    rain_and_grass = [_Mon("pelipper"), _Mon("rillaboom"), _Mon("archaludon")]
    st = MH.decide(_Battle([], raichu_team, rain_and_grass), _cfg(p=0.0, p_weather=1.0))
    assert st["domains"] == {"terrain"} and st["conflict"] and st["hold"]       # Grassy fights Electric Surge
    st2 = MH.decide(_Battle([], OURS, [_Mon("rillaboom"), _Mon("incineroar")]), _cfg(p=0.0, p_weather=1.0))
    assert st2["domains"] == {"weather"} and not st2["conflict"]                # Grassy is no fight for Tyranitar
    p = _player(p=1.0, p_weather=1.0, kmin=6, kmax=6)
    b = _Battle([_Mon("raichu", "raichunitex", "static"), _Mon("garchomp")], raichu_team, rain_and_grass,
                weather={_W("RAINDANCE"): 1})
    glog = (np.array([0.0, 2.0, 0.0]), np.zeros(3))
    kw = dict(none=0, mega=1, switch_offset=12, tau=1.0, build_mask=_mask_for((True, False)))
    assert MH.apply(p, b, glog, [1, 0], (0, 0), **kw) is not None              # THEIR RAIN does not end a terrain hold
    b.turn, b.fields = 2, {_F("GRASSY_TERRAIN"): 2}
    assert MH.apply(p, b, glog, [1, 0], (0, 0), **kw) is None                  # their terrain does
    assert p._mega_hold_state[b.battle_tag]["end"] == "terrain"
    meta = MH.game_meta(p, b.battle_tag)
    assert meta["reasons"] == {"field": 1}
    s = MH.summarize_trajectories([types.SimpleNamespace(meta=types.SimpleNamespace(sampling={"mega_hold": meta}),
                                                         transitions=[])])
    assert s["forced_by_reason"] == {"field": 1} and "by reason: field 1" in MH.format_stats(s)


def test_megatime_guard_auto_is_our_megas_without_a_delay_reason(tmp_path):
    from v_dance.selfplay.drills import get_drill
    own = tmp_path / "Ours"
    own.write_text("Tyranitar @ Tyranitarite\nAbility: Sand Stream\n- Rock Slide\n\n"
                   "Salamence @ Salamencite\nAbility: Intimidate\n- Hyper Voice\n\n"
                   "Charizard @ Charizardite X\nAbility: Blaze\n- Flare Blitz\n", encoding="utf-8")
    hit = _team(tmp_path, "ArchRain", ["Pelipper", "Archaludon", "Incineroar", "Gholdengo"])
    d = get_drill("megatime")
    d.build_pool([hit], str(own))
    assert d.ctx["guard"] == ["salamence"]                    # Tyranitar = field, Charizard X = type → not guards
    g = d.score_game(_GUARD_LOG, {"ourbot"}, d.ctx)
    assert g["guard_chance"] and g["guard_first"]


def test_a_hold_game_never_forces_salamence():
    p = _player(p=1.0, p_weather=1.0, kmin=6, kmax=6)
    b = _Battle([_Mon("tyranitar", "tyranitarite", "sandstream"), _Mon("salamence", "salamencite", "intimidate")],
                OURS, RAIN, turn=1)
    glog = (np.array([0.0, 2.0, 0.0]), np.array([0.0, 3.0, 0.0]))
    out = [1, 1]                                                         # both sampled mega
    w = MH.apply(p, b, glog, out, (0, 0), none=0, mega=1, switch_offset=12, tau=1.0,
                 build_mask=_mask_for((True, True)))                     # the REAL relevance rule
    assert out == [0, 1]                                                 # Tyranitar held, Salamence untouched
    assert w == pytest.approx(MH.p_none(glog[0], [True, True, False], 1.0))   # the weight is Tyranitar's alone
    b2 = _Battle([_Mon("salamence", "salamencite", "intimidate"), _Mon("indeedee")], OURS, RAIN, tag="battle-x-9")
    out = [1, 0]
    assert MH.apply(p, b2, glog, out, (0, 0), none=0, mega=1, switch_offset=12, tau=1.0,
                    build_mask=_mask_for((True, False))) is None and out == [1, 0]


_GUARD_LOG = """|player|p1|OurBot|1|
|player|p2|Rival|2|
|poke|p2|Incineroar, L50, M|
|switch|p1a: Salamence|Salamence, L50, M|100/100
|switch|p1b: Indeedee|Indeedee, L50, F|100/100
|switch|p2a: Incineroar|Incineroar, L50, M|100/100
|turn|1
|detailschange|p1a: Salamence|Salamence-Mega, L50, M
|move|p1a: Salamence|Hyper Voice|p2a: Incineroar
|turn|2
|win|OurBot
"""


def test_megatime_guard_counts_salamence_first_chance_megas():
    from v_dance.selfplay.drills import get_drill
    d = get_drill("megatime:guard=salamence")
    now = d.score_game(_GUARD_LOG, {"ourbot"}, d.ctx)
    late_log = _GUARD_LOG.replace("|turn|1\n|detailschange|p1a: Salamence|Salamence-Mega, L50, M\n", "|turn|1\n")
    late_log = late_log.replace("|turn|2\n", "|turn|2\n|detailschange|p1a: Salamence|Salamence-Mega, L50, M\n")
    late = d.score_game(late_log, {"ourbot"}, d.ctx)
    assert now["guard_chance"] and now["guard_first"] and late["guard_chance"] and not late["guard_first"]
    agg = d.aggregate([now, late])
    assert agg["guard_games"] == 2 and agg["guard_first_mega"] == 0.5
    assert not get_drill("megatime:guard=").score_game(_GUARD_LOG, {"ourbot"}, {})["guard_chance"]


def test_probe_collects_guard_rows(tmp_path):
    import json
    from v_dance.eval import mega_hold_probe as P
    rl = tmp_path / "rl"
    rl.mkdir()
    tag = "battle-gen9championsvgc2026regmc-123"
    tr = {"state": [0.0] * 4, "action_s0": 0, "action_s1": 0, "gimmick_s0": 1, "gimmick_s1": 1, "turn": 1,
          "decision_type": "turn", "gmask_s0": [1, 1, 0], "gmask_s1": [1, 1, 0]}
    meta = {"battle_id": tag, "own_team": ["tyranitar", "salamence"], "won": True, "sampling": {"arm": "a"}}
    (rl / "s.jsonl").write_text(json.dumps({"meta": meta, "transitions": [tr]}) + "\n", encoding="utf-8")
    rep = tmp_path / "rep"
    rep.mkdir()
    log = _LOG.replace("|switch|p2a: Archaludon",
                       "|switch|p1b: Salamence|Salamence, L50, M|100/100\n|switch|p2a: Archaludon")
    (rep / f"x_{tag}.html").write_text(log, encoding="utf-8")
    rows, counts = P.collect_decisions(rl, rep, {"ourbot"}, guards=["salamence"])
    assert sorted(r["label"] for r in rows) == ["guard:salamence", "hold:unseen"] and counts["games_used"] == 1
    rows_no_guard, _ = P.collect_decisions(rl, rep, {"ourbot"})
    assert [r["label"] for r in rows_no_guard] == ["hold:unseen"]


# ── 2026-10-03 USER RULE: the picker + the battle net are ONE network ─────────────────────────────────────────
def _pair(tmp_path):
    from v_dance.selfplay import tp_learning as TPL
    ck, tp = tmp_path / "battle_base.pt", tmp_path / "tp.pt"
    ck.write_bytes(b"net")
    tp.write_bytes(b"picker")
    TPL.record_pair(ck, tp)
    return str(ck), str(tp)


def test_a_live_run_trains_the_net_with_its_paired_picker(tmp_path, capsys):
    from v_dance.selfplay.generation import _resolve_picker_pair
    ck, tp = _pair(tmp_path)
    ns = types.SimpleNamespace
    a = ns(ckpt=ck, learner_tp=None, tp_learn=False, tp_fixed=False, no_picker_in_loop=False)
    _resolve_picker_pair(a)
    assert Path(a.learner_tp).resolve() == Path(tp).resolve() and a.tp_learn is True
    b = ns(ckpt=ck, learner_tp=None, tp_learn=False, tp_fixed=True, no_picker_in_loop=False)
    _resolve_picker_pair(b)
    assert b.learner_tp and b.tp_learn is False                          # --tp-fixed: paired but frozen
    lone = tmp_path / "lone.pt"
    lone.write_bytes(b"net")
    c = ns(ckpt=str(lone), learner_tp=None, tp_learn=False, tp_fixed=False, no_picker_in_loop=False)
    with pytest.raises(SystemExit) as ei:
        _resolve_picker_pair(c)
    assert ei.value.code == 2 and "the PAIR rule" in capsys.readouterr().err
    d = ns(ckpt=str(lone), learner_tp="v9.pt", tp_learn=False, tp_fixed=False, no_picker_in_loop=False)
    _resolve_picker_pair(d)
    assert d.learner_tp == "v9.pt" and d.tp_learn is True                # an explicit picker trains with the net
    e = ns(ckpt=str(lone), learner_tp=None, tp_learn=False, tp_fixed=False, no_picker_in_loop=True)
    _resolve_picker_pair(e)
    assert e.learner_tp is None and e.tp_learn is False                  # the explicit opt-out
    f = ns(ckpt=ck, learner_tp="x.pt", tp_learn=False, tp_fixed=False, no_picker_in_loop=True)
    with pytest.raises(SystemExit):
        _resolve_picker_pair(f)


def test_panel_eval_plays_a_pair_as_a_pair():
    from v_dance.eval.panel_eval import resolve_candidate_tp
    paired = {"a.pt": "tp_a.pt", "b.pt": "tp_b.pt", "c.pt": None}.get
    assert resolve_candidate_tp({"A": "a.pt"}, None, paired)[0] == "tp_a.pt"          # auto: its picker
    assert resolve_candidate_tp({"C": "c.pt"}, None, paired) == (None, "")             # no pair: eval default
    assert resolve_candidate_tp({"A": "a.pt"}, "none", paired)[0] == "none"            # explicit wins
    assert resolve_candidate_tp({"A": "a.pt"}, "default", paired)[0] is None           # explicit eval default
    with pytest.raises(ValueError, match="PAIR"):
        resolve_candidate_tp({"A": "a.pt", "B": "b.pt"}, None, paired)
    with pytest.raises(ValueError, match="PAIR"):
        resolve_candidate_tp({"A": "a.pt", "C": "c.pt"}, None, paired)
