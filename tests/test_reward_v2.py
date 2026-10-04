"""2026-10-04 REWARD v2 (v_dance/rl/reward.py + v_dance/selfplay/field_shaping.py; memory 14 'REWARD v2: DESIGN
DECIDED'): the loss margin, the potential-based weather / terrain credit, the 1/1.2 scale, the fade, the picker's
reward, the recording plumbing, the run's fail-fast + readout + preflight — and OFF = byte-identical."""
from __future__ import annotations

import itertools
import pickle
import types
from pathlib import Path

import numpy as np
import pytest

from v_dance.rl import reward as RW
from v_dance.rl.gae import compute_batch_gae, compute_gae
from v_dance.rl.schema import EpisodeMeta, Trajectory, Transition

REPO = Path(__file__).resolve().parents[1]
GAMMA = 0.997
BALTIMORE = (REPO / "teams" / "Champions" / "M-C" / "Baltimore_Sand_Psy").read_text(encoding="utf-8")
WOLFEY = (REPO / "teams" / "Champions" / "M-C" / "Wolfey_01_Golisopod_Pelipper").read_text(encoding="utf-8")


def _meta(won, *, v2=True, opp_fainted=None, tt=None, battle_id="b1", role="p1", tp_learn=None, guard=None):
    tt = tt or ("win" if won else "loss" if won is False else "draw")
    return EpisodeMeta(battle_id=battle_id, own_role=role, own_team=["a"] * 6, opp_team=["b"] * 6,
                       tp_bring=[0, 1, 2, 3], tp_leads=[0, 1], won=won, terminal_type=tt, n_turns=5,
                       reward_mode=("v2" if v2 else None), opp_fainted=opp_fainted, tp_learn=tp_learn, guard=guard)


def _traj(won, phis, *, v2=True, opp_fainted=None, values=None, tt=None, **kw):
    ts = [Transition(state=np.zeros(3, np.float32), action_s0=0, action_s1=0, turn=i + 1,
                     value=(0.0 if values is None else float(values[i])), phi=p) for i, p in enumerate(phis)]
    ts[-1].done = True
    t = Trajectory(meta=_meta(won, v2=v2, opp_fainted=opp_fainted, tt=tt, **kw), transitions=ts)
    RW.place_terminal_reward(t)
    return t


def _old_gae(traj, gamma=GAMMA, lam=0.95):
    """The pre-v2 compute_gae, verbatim (a real terminal) — the byte-identity reference."""
    r = np.array([t.reward for t in traj.transitions], dtype=np.float64)
    v = np.array([t.value for t in traj.transitions], dtype=np.float64)
    n = len(r)
    adv = np.zeros(n, dtype=np.float64)
    g = 0.0
    for t in range(n - 1, -1, -1):
        nv, nt = (0.0, 0.0) if t == n - 1 else (v[t + 1], 1.0)
        delta = r[t] + gamma * nv * nt - v[t]
        g = delta + gamma * lam * nt * g
        adv[t] = g
    return adv.astype(np.float32), (adv + v).astype(np.float32)


def _mc_return0(traj, kappa=1.0):
    """The Monte-Carlo return of the first step (λ = 1, values 0) under GAE's own reward path."""
    _a, ret = compute_gae(traj, GAMMA, 1.0, field_kappa=kappa)
    return float(ret[0])


# ── OFF = v1, byte-identical ─────────────────────────────────────────────────────────────────────────────────
def test_v1_rewards_and_gae_are_byte_identical():
    rng = np.random.default_rng(0)
    for won in (True, False, None):
        for k in (None, 0, 3):
            t = _traj(won, [0.1, -0.2, None, 0.0], v2=False, opp_fainted=k, values=rng.uniform(-1, 1, 4))
            assert t.transitions[-1].reward == {True: 1.0, False: -1.0, None: 0.0}[won]
            assert np.array_equal(RW.shaped_rewards(t, GAMMA, 0.3), [x.reward for x in t.transitions])
            for kappa in (1.0, 0.0, 0.5):
                a, r = compute_gae(t, field_kappa=kappa)
                a0, r0 = _old_gae(t)
                assert np.array_equal(a, a0) and np.array_equal(r, r0)
    t1 = _traj(True, [None] * 3, v2=False)
    a, r = compute_batch_gae([t1, _traj(False, [None] * 5, v2=False)], field_kappa=0.0)
    a0, r0 = compute_batch_gae([t1, _traj(False, [None] * 5, v2=False)])
    assert np.array_equal(a, a0) and np.array_equal(r, r0)


def test_a_v1_record_writes_no_new_keys_and_v2_round_trips():
    import json
    v1 = _traj(False, [None, None], v2=False).to_obj()
    assert not {"reward_mode", "opp_fainted"} & set(v1["meta"]) and all("phi" not in s for s in v1["transitions"])
    v2 = _traj(False, [0.1, -0.1], opp_fainted=2, guard={"protect": 3})
    back = Trajectory.from_obj(json.loads(json.dumps(v2.to_obj())))
    assert back.meta.reward_mode == "v2" and back.meta.opp_fainted == 2 and back.meta.guard is None   # guard: memory only
    assert [s.phi for s in back.transitions] == [0.1, -0.1]
    assert back.transitions[-1].reward == pytest.approx(-0.875)
    legacy = Trajectory.from_obj(v1)
    assert legacy.meta.reward_mode is None and legacy.meta.opp_fainted is None and legacy.transitions[0].phi is None


def test_the_store_guards_and_the_zero_sum_check_still_hold_under_v2():
    from v_dance.rl.collector import assert_zero_sum
    from v_dance.rl.store import assert_terminal_rewards_clean
    us = _traj(True, [0.1, 0.2], opp_fainted=3, battle_id="g", role="p1")
    them = _traj(False, [-0.1, -0.2], opp_fainted=1, battle_id="g", role="p2")
    assert_terminal_rewards_clean([us, them])                  # still sparse, terminal in [-1, 1]
    assert_zero_sum(us, them)                                  # the OUTCOMES are mirrored (the margin is not zero-sum)


# ── the loss margin ──────────────────────────────────────────────────────────────────────────────────────────
def test_a_win_is_plus_one_however_it_was_won_and_beats_every_loss():
    wins = [RW.terminal_reward(_meta(True, opp_fainted=k)) for k in (None, 0, 1, 2, 3, 4)]
    assert wins == [1.0] * 6
    losses = {k: RW.terminal_reward(_meta(False, opp_fainted=k)) for k in (None, 0, 1, 2, 3, 4, 7, -2)}
    assert losses == {None: -1.0, 0: -1.0, 1: -0.9375, 2: -0.875, 3: -0.8125, 4: -0.75, 7: -0.75, -2: -1.0}
    assert min(wins) - max(losses.values()) >= 1.75
    assert RW.terminal_reward(_meta(False, opp_fainted=2, tt="adjudicated")) == pytest.approx(-0.875)   # a timer loss
    assert RW.terminal_reward(_meta(None, opp_fainted=3)) == 0.0                                      # a draw
    assert RW.terminal_reward(_meta(False, opp_fainted=3, v2=False)) == -1.0                          # v1 ignores it


def test_a_sacrifice_win_scores_exactly_like_any_win():
    """Our own faints are never counted: two wins with the same field timeline score the same, whatever we lost."""
    sac = _traj(True, [0.1, 0.0, -0.1, 0.1], opp_fainted=4, guard={"own_fainted": 3})
    clean = _traj(True, [0.1, 0.0, -0.1, 0.1], opp_fainted=4, guard={"own_fainted": 0})
    for kappa in (1.0, 0.4):
        assert np.array_equal(compute_gae(sac, field_kappa=kappa)[1], compute_gae(clean, field_kappa=kappa)[1])


# ── the field potential's credit ─────────────────────────────────────────────────────────────────────────────
def test_the_field_credit_telescopes():
    rng = np.random.default_rng(1)
    levels = [-0.2, -0.1, 0.0, 0.1, 0.2]
    for n in (1, 2, 7, 40):
        for won, k in ((True, None), (False, 2), (None, None)):
            for kappa in (1.0, 0.25):
                phis = [float(rng.choice(levels)) for _ in range(n)]
                t = _traj(won, phis, opp_fainted=k)
                f = RW.shaped_rewards(t, GAMMA, kappa)
                disc = float(sum(GAMMA ** i * f[i] for i in range(n)))
                r_t = t.transitions[-1].reward
                assert disc == pytest.approx(RW.V2_SCALE * (GAMMA ** (n - 1) * r_t - kappa * phis[0]), abs=1e-12)
                assert _mc_return0(t, kappa) == pytest.approx(disc, abs=1e-6)


def test_weather_cycling_earns_nothing():
    """Setting the same weather again and again (or trading it back and forth) pays nothing net: from a neutral start
    the field credit sums to exactly 0 however the weather moved."""
    flat = _traj(True, [0.0] * 7)
    for phis in ([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0], [0.0, 0.1, -0.1, 0.1, -0.1, 0.2, 0.2], [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]):
        assert _mc_return0(_traj(True, phis)) == pytest.approx(_mc_return0(flat), abs=1e-6)


def test_the_field_credit_never_outweighs_a_result():
    """From the same starting state, EVERY win beats EVERY loss — whatever the field did in between (the worst win
    loses the field all game, the best loss takes it and three KOs) and however long either game ran: the credit sums
    to −κ·Φ(s₀) for both, so only the result separates them. And the credit itself is bounded by 0.2."""
    rng = np.random.default_rng(3)
    levels = [-0.2, -0.1, 0.0, 0.1, 0.2]
    for nw, nl in itertools.product((1, 5, 30, 120, 300), repeat=2):
        for phi0 in levels:
            worst_win = _mc_return0(_traj(True, [phi0] + [-0.2] * (nw - 1)))
            best_loss = _mc_return0(_traj(False, [phi0] + [0.2] * (nl - 1), opp_fainted=3))
            assert worst_win > best_loss, (nw, nl, phi0, worst_win, best_loss)
    for _ in range(100):
        n = int(rng.integers(1, 80))
        t = _traj(True, [float(rng.choice(levels)) for _ in range(n)])
        credit = float(sum(GAMMA ** i * f for i, f in enumerate(RW.shaped_rewards(t, GAMMA, 1.0)))) \
            - RW.V2_SCALE * GAMMA ** (n - 1)
        assert abs(credit) <= 0.2 * RW.V2_SCALE + 1e-12


def test_every_monte_carlo_return_stays_in_the_critic_range():
    rng = np.random.default_rng(2)
    levels = [-0.2, -0.1, 0.0, 0.1, 0.2]
    for _ in range(300):
        n = int(rng.integers(1, 60))
        won = [True, False, None][int(rng.integers(3))]
        t = _traj(won, [float(rng.choice(levels)) for _ in range(n)], opp_fainted=int(rng.integers(0, 4)))
        _a, ret = compute_gae(t, GAMMA, 1.0, field_kappa=float(rng.uniform(0, 1)))
        assert np.all(np.abs(ret) <= 1.0 + 1e-6), ret
    assert RW.V2_SCALE == pytest.approx(1 / 1.2)


def test_kappa_scales_the_credit_and_zero_leaves_the_scaled_result():
    t = _traj(False, [0.1, -0.1, 0.2], opp_fainted=1)
    r = t.transitions[-1].reward
    assert np.allclose(RW.shaped_rewards(t, GAMMA, 0.0), RW.V2_SCALE * np.array([0.0, 0.0, r]))
    f1, fh = RW.shaped_rewards(t, GAMMA, 1.0), RW.shaped_rewards(t, GAMMA, 0.5)
    base = RW.V2_SCALE * np.array([0.0, 0.0, r])
    assert np.allclose(fh - base, 0.5 * (f1 - base))
    assert not np.array_equal(compute_batch_gae([t], field_kappa=1.0)[1], compute_batch_gae([t], field_kappa=0.0)[1])


def test_the_trainer_passes_its_kappa_to_gae():
    pytest.importorskip("torch")
    from v_dance.rl.trainer import PPOTrainer, TrainConfig
    t = _traj(True, [0.2, 0.0, -0.2])
    fake = types.SimpleNamespace(tcfg=TrainConfig(assert_value_space=False), field_kappa=0.0)
    _tx, _adv, ret0 = PPOTrainer._flatten(fake, [t])
    fake.field_kappa = 1.0
    _tx, _adv, ret1 = PPOTrainer._flatten(fake, [t])
    assert np.array_equal(ret0, compute_batch_gae([t], standardize_adv=False, field_kappa=0.0)[1])
    assert not np.array_equal(ret0, ret1)


def test_the_fade():
    k = [RW.field_fade(g, 40) for g in range(45)]
    assert k[:27] == [1.0] * 27 and k[27] == pytest.approx(12 / 13) and k[39] == 0.0 and k[44] == 0.0
    assert all(a >= b for a, b in zip(k, k[1:]))
    assert RW.field_fade(5, None) == 1.0 and RW.field_fade(5, 0) == 1.0          # open-ended: never fades
    assert [RW.field_fade(g, 1) for g in range(2)] == [1.0, 1.0]
    assert [RW.field_fade(g, 3) for g in range(3)] == [1.0, 1.0, 0.0]             # the preflight's 3 gens


# ── ownership + Φ ────────────────────────────────────────────────────────────────────────────────────────────
from v_dance.selfplay import field_shaping as FS  # noqa: E402


class _E:
    def __init__(self, name):
        self.name = name


def test_ownership_on_the_real_pool_sheets():
    assert FS.sheet_kinds(BALTIMORE) == {"weather": frozenset({"sand"}), "terrain": frozenset({"psychic"})}
    assert FS.sheet_kinds(WOLFEY) == {"weather": frozenset({"rain"}), "terrain": frozenset({"psychic"})}
    own = FS.owners(BALTIMORE, WOLFEY)
    assert own == {"weather": {"sand": 1, "rain": -1}, "terrain": {}}       # Psychic: both set it → neutral
    assert FS.owners(WOLFEY, BALTIMORE) == FS.flip(own)
    assert FS.describe(own) == "ours: sand | theirs: rain"


def test_a_mirror_is_neutral_everywhere():
    own = FS.owners(BALTIMORE, BALTIMORE)
    assert own == {"weather": {}, "terrain": {}}
    for w, t in itertools.product(({"sand"}, {"rain"}, ()), ({"psychic"}, {"grassy"}, ())):
        assert FS.phi_from_kinds(w, t, own) == 0.0


def test_moves_and_mega_stones_set_kinds_and_both_or_neither_is_neutral():
    a = ("Charizard @ Charizardite Y\nAbility: Blaze\n- Heat Wave\n- Protect\n\n"
         "Tornadus @ Sitrus Berry\nAbility: Prankster\n- Rain Dance\n- Tailwind\n\n"
         "Indeedee-F @ Psychic Seed\nAbility: Psychic Surge\n- Follow Me\n- Grassy Terrain\n")
    b = ("Rillaboom @ Miracle Seed\nAbility: Grassy Surge\n- Fake Out\n\n"
         "Pelipper @ Focus Sash\nAbility: Drizzle\n- Hurricane\n")
    assert FS.sheet_kinds(a) == {"weather": frozenset({"sun", "rain"}), "terrain": frozenset({"psychic", "grassy"})}
    own = FS.owners(a, b)
    assert own == {"weather": {"sun": 1}, "terrain": {"psychic": 1}}        # rain + grassy: both can → neutral
    assert FS.phi_from_kinds({"sun"}, {"psychic"}, own) == pytest.approx(0.2)
    assert FS.phi_from_kinds({"rain"}, {"grassy"}, own) == 0.0
    assert FS.owners(a, "") == {"weather": {"sun": 1, "rain": 1}, "terrain": {"psychic": 1, "grassy": 1}}


def test_phi_reads_the_live_battle():
    own = FS.owners(BALTIMORE, WOLFEY)
    b = types.SimpleNamespace(weather={_E("SANDSTORM"): 1}, fields={_E("PSYCHIC_TERRAIN"): 1, _E("TRICK_ROOM"): 2})
    assert FS.phi(b, own) == pytest.approx(0.1)                              # sand ours, Psychic neutral, TR ignored
    b.weather = {_E("RAINDANCE"): 3}
    assert FS.phi(b, own) == pytest.approx(-0.1)
    b.weather, b.fields = {}, {}
    assert FS.phi(b, own) == 0.0


def test_the_game_guard_counts_our_moves_and_both_sides_faints():
    mon = lambda f: types.SimpleNamespace(fainted=f)                        # noqa: E731
    b = types.SimpleNamespace(
        player_role="p1", team={"a": mon(True), "b": mon(False)},
        opponent_team={"x": mon(True), "y": mon(True), "z": mon(False)},
        _replay_data=[["", "move", "p1a: Indeedee", "Trick Room", "p1a: Indeedee"],
                      ["", "move", "p1b: Tyranitar", "Protect", ""], ["", "move", "p2a: X", "Protect", ""],
                      ["", "move", "p1a: Indeedee", "Detect", ""], ["", "move", "p2b: Y", "Trick Room", ""]])
    assert FS.game_guard(b) == {"own_fainted": 1, "opp_fainted": 2, "protect": 2, "trick_room": 1}


# ── recording plumbing ───────────────────────────────────────────────────────────────────────────────────────
def _recorder(monkeypatch, **extra):
    import v_dance.selfplay.game_runner as GR
    seen = {}
    monkeypatch.setattr(GR, "record_decision", lambda c, ac, **kw: seen.update(kw))
    monkeypatch.setattr(GR, "build_legal_action_mask", lambda b, s: [1] * 4)
    monkeypatch.setattr(GR, "build_gimmick_legal_mask", lambda b, s: [1, 1, 0])
    fake = types.SimpleNamespace(_sampling_masks={}, _collectors={}, _ac=None, _tau=1.0, _mega_hold_w={},
                                 _source_counts={"rejected_resample": 0}, **extra)
    fake._collector_for = lambda b: types.SimpleNamespace(last_step=lambda: None)
    b = types.SimpleNamespace(battle_tag="battle-x-1", turn=1, weather={_E("SANDSTORM"): 1}, fields={})
    return GR, fake, b, seen


def test_the_recorder_records_phi_only_under_v2(monkeypatch):
    GR, fake, b, seen = _recorder(monkeypatch, _reward_v2={"owners": FS.owners(BALTIMORE, WOLFEY)})
    GR.SelfPlayVGCPlayer._record_rl_decision(fake, b, np.zeros(3), 0, 1, 0, 0, "model", "turn")
    assert seen["phi"] == pytest.approx(0.1)
    GR, fake, b, seen = _recorder(monkeypatch)                               # v1: no _reward_v2 at all
    GR.SelfPlayVGCPlayer._record_rl_decision(fake, b, np.zeros(3), 0, 1, 0, 0, "model", "turn")
    assert seen["phi"] is None
    GR, fake, b, seen = _recorder(monkeypatch, _reward_v2={"owners": None})  # a broken Φ never drops the step
    GR.SelfPlayVGCPlayer._record_rl_decision(fake, b, np.zeros(3), 0, 1, 0, 0, "model", "turn")
    assert "phi" in seen and seen["phi"] is None


def test_the_finish_callback_stamps_the_margin_only_under_v2(monkeypatch):
    pytest.importorskip("poke_env")
    import v_dance.selfplay.game_runner as GR
    from v_dance.rl.collector import TrajectoryCollector
    monkeypatch.setattr(GR.VGCPlayer, "_battle_finished_callback", lambda self, battle: None)
    mon = lambda f: types.SimpleNamespace(fainted=f, species="x")           # noqa: E731

    def run(rv2):
        p = object.__new__(GR.SelfPlayVGCPlayer)
        c = TrajectoryCollector("battle-x-9", "p1")
        c.add_step(state=np.zeros(3), action_s0=0, action_s1=0, turn=1, phi=(0.1 if rv2 else None))
        p._collectors, p._finished, p._tau, p._source_counts = {"battle-x-9": c}, {}, 1.0, {}
        p._reward_v2 = rv2
        b = types.SimpleNamespace(battle_tag="battle-x-9", won=False, lost=True, turn=7, player_role="p1",
                                  team={"a": mon(True)}, opponent_team={"x": mon(True), "y": mon(True)},
                                  _replay_data=[["", "move", "p1a: A", "Protect", ""]])
        GR.SelfPlayVGCPlayer._battle_finished_callback(p, b)
        return p._finished["battle-x-9"]

    t = run({"owners": {"weather": {}, "terrain": {}}})
    assert t.meta.reward_mode == "v2" and t.meta.opp_fainted == 2
    assert t.meta.guard == {"own_fainted": 1, "opp_fainted": 2, "protect": 1, "trick_room": 0}
    assert t.transitions[-1].reward == pytest.approx(-0.875)
    t1 = run(None)
    assert t1.meta.reward_mode is None and t1.meta.opp_fainted is None and t1.meta.guard is None
    assert t1.transitions[-1].reward == -1.0


def test_chunk_specs_carry_the_switch_and_each_recording_player_owns_from_its_side(monkeypatch):
    from v_dance.selfplay import mp_collect as MP
    import v_dance.selfplay.game_runner as GR
    import v_dance.play.run_local_battle as R

    class FakeSP:
        def __init__(self, ac, **kw):
            self.kw = kw

    monkeypatch.setattr(GR, "SelfPlayVGCPlayer", FakeSP)
    monkeypatch.setattr(R, "load_team", lambda p: {"A": BALTIMORE, "B": WOLFEY}[p])
    monkeypatch.setattr(R, "resolve_team_path", lambda t: t)
    spec = MP.ChunkSpec("A", "B", 2, "latest", None, None, 1, 3)
    spec.reward_v2 = True
    our, opp = MP._build_players_real("AC", spec, 1.0, 0, "tp.pt")
    assert our._reward_v2["owners"] == FS.owners(BALTIMORE, WOLFEY)
    assert opp._reward_v2["owners"] == FS.owners(WOLFEY, BALTIMORE)
    plain = MP.ChunkSpec("A", "B", 2, "latest", None, None, 2, 3)
    assert not plain.reward_v2 and pickle.loads(pickle.dumps(spec)).reward_v2
    our2, _ = MP._build_players_real("AC", plain, 1.0, 0, "tp.pt")
    assert not hasattr(our2, "_reward_v2")

    def boom(*a, **k):
        raise RuntimeError("dex down")
    monkeypatch.setattr(FS, "owners", boom)                                 # a failure → a neutral Φ, never a lost chunk
    our3, _ = MP._build_players_real("AC", spec, 1.0, 0, "tp.pt")
    assert our3._reward_v2["owners"] == {"weather": {}, "terrain": {}}

    class League:
        def sample(self, rng):
            return ("latest",)
    assert all(s.reward_v2 for s in MP.build_chunk_specs(League(), ["A", "B"], 6, chunk_size=2, reward_v2=True))
    assert not any(s.reward_v2 for s in MP.build_chunk_specs(League(), ["A", "B"], 4, chunk_size=2))


# ── the picker ───────────────────────────────────────────────────────────────────────────────────────────────
def test_the_picker_gets_the_same_terminal_value_without_field_credit():
    from v_dance.selfplay import tp_learning as TPL
    rec = {"subsets": [], "pairs": []}
    trajs = [_traj(False, [0.2, 0.2], opp_fainted=2, tp_learn=dict(rec), battle_id="g1"),
             _traj(True, [-0.2], opp_fainted=1, tp_learn=dict(rec), battle_id="g2"),
             _traj(False, [None], v2=False, opp_fainted=3, tp_learn=dict(rec), battle_id="g3"),
             _traj(True, [None], v2=False, tp_learn=dict(rec), battle_id="g4")]
    out = TPL.records_from_trajectories(trajs)
    assert [r["reward"] for r in out] == [pytest.approx(-0.875), 1.0, -1.0, 1.0]
    assert [r["won"] for r in out] == [False, True, False, True]


# ── the run: fail-fast, CLI, readout, preflight ──────────────────────────────────────────────────────────────
def test_reward_v2_needs_multiprocess(tmp_path, capsys):
    from v_dance.selfplay.generation import run_live_generations
    with pytest.raises(SystemExit) as ei:
        run_live_generations("no_such.pt", team_pool=["A", "B"], team_chooser=None, archive_dir=tmp_path / "a",
                             collect_procs=1, reward_v2=True)
    assert ei.value.code == 2 and "--reward-v2 needs --collect-procs >= 2" in capsys.readouterr().err


def test_the_cli_lists_reward_v2_off_by_default():
    import os
    import subprocess
    import sys
    env = {**os.environ, "PYTHONUTF8": "1"}
    hlp = subprocess.run([sys.executable, "-X", "utf8", "-m", "v_dance.selfplay.generation", "--help"],
                         capture_output=True, text=True, env=env, cwd=str(REPO))
    assert hlp.returncode == 0 and "--reward-v2" in hlp.stdout


def test_the_generation_readout_counts_margin_field_and_guards():
    win = _traj(True, [0.0, 0.1, 0.1, -0.1, 0.1], opp_fainted=4, guard={"own_fainted": 2, "protect": 3, "trick_room": 1})
    loss = _traj(False, [-0.1, 0.0, 0.0], opp_fainted=2, guard={"own_fainted": 4, "protect": 1, "trick_room": 0})
    s = FS.summarize_trajectories([win, loss, _traj(True, [None], v2=False)], 0.5)
    assert (s["trajectories"], s["v2"], s["steps"], s["steps_phi"]) == (3, 2, 8, 8)
    assert (s["ours"], s["theirs"], s["gains"], s["losses_field"]) == (3, 2, 3, 1)
    assert (s["games_won"], s["games_lost"], s["lost_margin_recorded"], s["opp_ko_on_loss"]) == (1, 1, 1, 2)
    assert s["margin_sum"] == pytest.approx(0.125) and s["own_faints_in_wins"] == 2 and s["guard_wins"] == 1
    line = FS.format_stats(s)
    assert line.startswith("reward v2: κ 0.50 · 2/3 trajectories") and "margin +0.125 avg (their KOs 2.0/4)" in line
    assert "Protect 2.0/game, Trick Room 0.50/game" in line


def test_the_preflight_requires_reward_v2_to_be_wired():
    from v_dance.selfplay import preflight as PF

    class Rec:
        def __init__(self, g):
            self.generation, self.n_trajectories, self.update_stats = g, 5, {"loss": 0.1}
            self.promoted, self.panel, self.hof = g == 1, {}, None

    class H:
        def __init__(self, n):
            self.records = [Rec(g) for g in range(n)]

    good = [{"generation": 0, "v2": 9, "steps": 40, "steps_phi": 40, "games_lost": 4, "lost_margin_recorded": 4,
             "ours": 5, "theirs": 3, "kappa": 1.0},
            {"generation": 2, "v2": 9, "steps": 40, "steps_phi": 40, "games_lost": 4, "lost_margin_recorded": 4,
             "ours": 5, "theirs": 3, "kappa": 0.0}]
    probs, warns, _ = PF.check({"history": H(3), "reward_v2": good}, {"history": H(4), "reward_v2": good},
                               reward_v2_on=True)
    assert not [p for p in probs if "reward" in p or "potential" in p or "faint" in p] and not warns
    bad = [dict(good[0], steps_phi=30), dict(good[1], lost_margin_recorded=3), dict(good[1], v2=0, generation=1)]
    probs, _, _ = PF.check({"history": H(3), "reward_v2": bad}, {"history": H(4), "reward_v2": good}, reward_v2_on=True)
    assert any("10 of 40 reward-v2 steps carry no field potential" in p for p in probs)
    assert any("carries no opponent faint count" in p for p in probs)
    assert any("no trajectory was collected under reward v2" in p for p in probs)
    probs, _, _ = PF.check({"history": H(3)}, {"history": H(4)}, reward_v2_on=True)
    assert any("reward v2 reported nothing" in p for p in probs)
    flat = [dict(g, ours=0, theirs=0, kappa=1.0) for g in good]
    _p, warns, _ = PF.check({"history": H(3), "reward_v2": flat}, {"history": H(4), "reward_v2": flat},
                            reward_v2_on=True)
    assert any("Φ ≡ 0" in w for w in warns) and any("fade never ran" in w for w in warns)
