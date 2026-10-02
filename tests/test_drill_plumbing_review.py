"""Drill league (2026-10-02) — plumbing / failure-mode review, offline.

Each test pins one way a run can believe it is drilling while it is not (or fail without a cause):
  * the drill's opponent WEIGHTS go through ``own_team_matchups``' deterministic largest-remainder split, so at
    the default ``--games 100`` the advertised neutral share never reaches collection;
  * a pool entry given as a bare team NAME (``--train-teams <names>``, resolvable everywhere else) silently gets
    no drill weight;
  * the preflight's pressure flag follows ``--drill-bias`` only, not whether the drill HAS a pressure;
  * a worker-side drill-row failure is logged at DEBUG, which a spawn worker never emits.
No server, no GPU, no network.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from v_dance.eval.gauntlet import _team_file, collection_pairings, team_key
from v_dance.selfplay import mp_collect as MP
from v_dance.selfplay import preflight as PF
from v_dance.selfplay.drills import setup_drill
from v_dance.selfplay.gate import GenerationHistory, GenerationRecord
from v_dance.selfplay.league import OpponentLeague
from v_dance.selfplay.mp_collect import ChunkSpec

_REPO_ROOT = Path(__file__).resolve().parents[1]

OURS = "Tyranitar @ Tyranitarite\nAbility: Sand Stream\n- Rock Slide\n"
RAIN = "Pelipper @ Damp Rock\nAbility: Drizzle\n- Hurricane\n"
PLAIN = "Garchomp @ Life Orb\nAbility: Rough Skin\n- Earthquake\n"


def _write(folder: Path, name: str, text: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text(text, encoding="utf-8")
    return str(p.resolve())


# ── 1. the advertised drill mix vs what collection actually plays ─────────────────────────────────────
@pytest.fixture
def rain_pool(tmp_path):
    d = tmp_path / "M-C"
    own = _write(d, "Ours_Sand", OURS)
    rain = [_write(d, f"Rain_{i:02d}", RAIN) for i in range(20)]
    plain = [_write(d, f"Plain_{i:03d}", PLAIN) for i in range(100)]
    return own, [own] + rain + plain, set(team_key(p) for p in plain)


def test_the_drill_advertises_a_20pct_neutral_share(rain_pool):
    own, pool, _ = rain_pool
    s = setup_drill("field", own_team_path=own, team_pool=pool)
    assert s.pool.shares["neutral"] == pytest.approx(0.2)
    assert sum(w for k, w in s.opp_weights.items() if team_key(k).startswith("plain_")) == pytest.approx(0.2)


def test_the_neutral_share_reaches_collection_at_the_default_games(rain_pool):
    own, pool, plain = rain_pool
    s = setup_drill("field", own_team_path=own, team_pool=pool)
    neutral = total = 0
    for gen in range(10):                                   # matchup_seed = gen in collect_fn
        for a, b, n in collection_pairings(pool, 100, seed=gen, own_team=own, own_mirror_frac=0.2,
                                           opp_weights=s.opp_weights, opp_draw="multinomial"):
            if team_key(a) == team_key(b):
                continue
            total += n
            neutral += n if team_key(b) in plain else 0
    assert total > 0
    assert neutral / total >= 0.1                           # advertised 0.2; currently 0.0


# ── 2. a bare-name pool entry (resolvable everywhere else) silently gets no drill weight ─────────────────
_BARE = "Balt27_01_Salamence_Tyranitar"                    # a real M-C team file (sets sand)


@pytest.mark.skipif(not (_REPO_ROOT / "teams" / "Champions" / "M-C" / _BARE).is_file(),
                    reason="needs the repo's M-C team files")
def test_a_bare_name_pool_entry_keeps_its_drill_weight(tmp_path):
    d = tmp_path / "M-C"
    own = _write(d, "Ours_Plain", PLAIN)                    # we set nothing: every setter team is a conflict
    other = _write(d, "Rain_Team", RAIN)
    assert _team_file(_BARE) is not None                    # the rest of the system resolves the bare name
    s = setup_drill("field", own_team_path=own, team_pool=[own, _BARE, other])
    assert other in s.opp_weights                           # (the path entry is fine)
    assert _BARE in s.opp_weights


# ── 3. the preflight's pressure flag ignores whether the drill has a pressure ─────────────────────────────
def _hist(n, drill):
    h = GenerationHistory()
    for g in range(n):
        h.add(GenerationRecord(generation=g, n_trajectories=10, scripted_wins=1, scripted_games=2,
                               model_elo=None, verdict="promote" if g == 1 else "hold",
                               promoted=(g == 1), update_stats={"loss": 0.1}, drill=dict(drill)))
    return h


def _launch(drill):
    def launch(a):
        n = 3 if a.resume_gen is None else 4
        lg = OpponentLeague(latest_path="l.pt")
        lg.note_played({k: 1 for k in lg.opponent_keys()})
        return {"history": _hist(n, drill), "league": lg}
    return launch


@pytest.mark.parametrize("spec", ["focus:opp=rillaboom,mon=salamence", "field:pressure=off"])
def test_preflight_pressure_flag_follows_the_drill(tmp_path, monkeypatch, spec):
    seen = {}
    real = PF.check

    def spy(fresh, resumed, **kw):
        seen.update(kw)
        return real(fresh, resumed, **kw)

    monkeypatch.setattr(PF, "PREFLIGHT_DIR", tmp_path)
    monkeypatch.setattr(PF, "check", spy)
    sb = {"games": 6, "pressure": {"games": 0}}             # gen_scoreboard of a pressure-less drill
    args = SimpleNamespace(generations=40, games=100, league_clones=None, clone_frac=0.0,
                           register_arms=False, hof=False, preflight="auto", preflight_only=False,
                           run_cfg_gate={}, drill=spec, drill_bias=2.5)
    PF.run_preflight(args, _launch(sb))
    assert seen["drill_on"] is True
    assert seen["pressure_on"] is False


# ── 4. a worker-side drill-row failure must be visible at WARNING (DEBUG never leaves a spawn worker) ─────
class _P:
    def __init__(self, username, trajs=None, battles=None):
        self.username = username
        self._trajs = trajs or {}
        self._source_counts = {"model": 1}
        self.battles = battles or {}
        self.n_won_battles = self.n_finished_battles = 0

        async def _stop():
            return None
        self.ps_client = SimpleNamespace(stop_listening=_stop)

    async def battle_against(self, opp, n_battles):
        self.n_finished_battles += n_battles

    def finished_trajectories(self):
        return dict(self._trajs)


def test_a_drill_row_failure_is_logged_at_warning(caplog):
    tag = "battle-1"

    def build(ac, spec, tau, seed, tc, live_dir=None, save_replays=False, port=None):
        t = SimpleNamespace(meta=SimpleNamespace(won=True, battle_id=tag, terminal_type="win"))
        our = _P("LG0x", trajs={tag: t}, battles={tag: SimpleNamespace(_replay_data=[["", 1, None]])})
        return our, _P("OP0y")

    MP._WARNED.discard("drill rows")                       # warn-once is per process
    caplog.set_level(logging.WARNING)
    res = asyncio.run(MP._collect_specs(None, [ChunkSpec("A", "B", 1, "clone", "c.pt", None, 1, score=True)],
                                        tau=1.0, seed=0, team_chooser=None, battle_timeout=None,
                                        async_workers=1, build_players=build))
    assert res.drill_rows == [] and res.n_games == 1         # non-fatal (as intended) ...
    assert any(r.levelno >= logging.WARNING and "drill" in r.getMessage().lower()
               for r in caplog.records)                     # ... but it must leave a trace
