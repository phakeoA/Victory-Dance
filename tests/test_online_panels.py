"""Online-tab panels (USER 2026-09-02) — the control panel's wiring for the matchup tables and the
"what the nets are thinking" feed, plus the model_io stashes the narration reads.

Controller half: status carries ``matchups`` / ``thoughts``; the GAME_DONE_HOOK tap feeds BOTH
tables (session + all-time for THIS team); the regulation table's team select (follow the pin /
all teams / a named team); start_control_ui chains the hook and installs the feed on the player.
Model half (torch): the 2b decode and team_order leave their real numbers in LAST_DECODE / LAST_TP.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import v_dance.play.play_vs_human_browser as _pvhb
from v_dance.play import bot_control_ui as bcu
from v_dance.play.bot_control_ui import BotController, start_control_ui

FMT = "gen9championsvgc2026regmb"
POOL = ["alpha_team", "beta_team"]


class FakePage:
    def __init__(self):
        self.sent = []

    async def evaluate(self, js, arg=None):
        self.sent.append((js, arg))
        return None


class _FakeTeam:
    def yield_team(self):
        return "PACKED"


class FakePlayer:
    def __init__(self):
        self._team = _FakeTeam()
        self._team_name = None

    def update_team(self, team):
        pass


class FakeHost:
    def __init__(self):
        self.player = FakePlayer()


class _Mon:
    def __init__(self, species, item="unknown_item", ability=None):
        self.species, self.item, self.ability = species, item, ability


class _DoneBattle:
    def __init__(self, tag, *mons):
        self.battle_tag = tag
        self.opponent_username = "Opp"
        self.opponent_team = {f"p2: {m.species}": m for m in mons}


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])
    monkeypatch.setenv("VD_SITE_POLL", "0")


def _ctrl(loop, **kw):
    c = BotController(page=FakePage(), host=FakeHost(), tally={"ai": 0, "you": 0, "draw": 0},
                      ai_pool=POOL, fmt=FMT, username="VictoriousDancing", loop=loop,
                      env_path=Path("unused.env"), **kw)
    c._load_scoped_team = lambda scoped: "LOADED"
    return c


def test_status_carries_matchups_and_the_game_done_tap_feeds_both_tables(tmp_path):
    loop = asyncio.new_event_loop()
    try:
        c = _ctrl(loop, session_id="S1", dossier_dir=tmp_path)      # empty dossier dir = no seed
        st = c.status()
        m = st["matchups"]
        assert (m["format"], m["reg"], m["mode"]) == (FMT, "M-B", "all")   # no pin → all teams
        assert m["session"]["footer"]["games"] == 0 and m["loaded"]["games"] == 0
        assert st["thoughts"] is None
        assert any("matchup book: 0 game(s)" in e for e in c.events)
        c.set_team("alpha_team")
        assert (c.matchup_status()["mode"], c.matchup_status()["team"]) == ("pin", "alpha_team")
        tag = f"battle-{FMT}-501"
        c.on_game_done(tag, _DoneBattle(tag, _Mon("incineroar", "leftovers", "intimidate"),
                                        _Mon("garchomp")),
                       {"battle_tag": tag, "ai_team": "alpha_team", "result": "ai",
                        "opponent": "Opp", "session_id": "S1"})
        m = c.matchup_status()
        assert m["session"]["footer"]["games"] == 1
        rows = {r["species"]: r for r in m["alltime"]["rows"]}
        assert rows["incineroar"]["wins"] == 1 and rows["incineroar"]["item_src"] == "seen"
        assert rows["garchomp"]["item_src"] == "none"                   # no belief wired here
        assert m["teams"] == [{"team": "alpha_team", "games": 1}]
        assert any("matchups: WON vs" in e and "team alpha_team" in e for e in c.events)
        # the team select: all teams / a named team / back to following the pin; unknown = error
        assert c.set_matchup_team("*") == "*" and c.matchup_status()["mode"] == "all"
        assert c.set_matchup_team("alpha_team") == "alpha_team"
        assert c.matchup_status()["mode"] == "select"
        with pytest.raises(ValueError):
            c.set_matchup_team("no_such_team")
        assert c.set_matchup_team("") == "" and c.matchup_status()["mode"] == "pin"
        # another team's game lands in ITS table; the session table spans both
        tag2 = f"battle-{FMT}-502"
        c.on_game_done(tag2, _DoneBattle(tag2, _Mon("incineroar")),
                       {"battle_tag": tag2, "ai_team": "beta_team", "result": "human",
                        "opponent": "Opp2", "session_id": "S1"})
        assert c.matchup_status()["alltime"]["footer"]["games"] == 1   # still alpha_team's table
        c.set_matchup_team("beta_team")
        assert c.matchup_status()["alltime"]["rows"][0]["losses"] == 1
        assert c.matchup_status()["session"]["footer"]["games"] == 2
        # a session id learned from the first row when the controller had none
        c2 = _ctrl(loop)
        c2.on_game_done(tag, _DoneBattle(tag, _Mon("kingambit")),
                        {"battle_tag": tag, "ai_team": "alpha_team", "result": "ai", "session_id": "S7"})
        assert c2.session_id == "S7" and c2.matchup_status()["session"]["footer"]["games"] == 1
        # matchups off → status carries None and the tap is harmless
        off = _ctrl(loop, matchups=False)
        assert off.status()["matchups"] is None
        off.on_game_done(tag, _DoneBattle(tag), {"battle_tag": tag})
        assert off.set_matchup_team("zzz") == "zzz"                     # nothing to validate against
    finally:
        loop.close()


def test_options_endpoint_accepts_matchup_team_and_http_status_has_the_new_fields(tmp_path):
    """One real-HTTP pass (the panel's own server): POST /api/options {matchup_team} and the
    /api/status payload — the same JSON Mission Control proxies."""
    import json
    import urllib.error
    import urllib.request
    prev = (_pvhb.RATING_HOOK, _pvhb.RATING_CHANGE_HOOK, getattr(_pvhb, "GAME_DONE_HOOK", None))
    calls = []
    _pvhb.GAME_DONE_HOOK = lambda tag, b, row: calls.append(("prev", tag))

    async def main():
        host = FakeHost()
        ctrl = start_control_ui(page=FakePage(), host=host, tally={"ai": 0, "you": 0, "draw": 0},
                                ai_pool=POOL, fmt=FMT, username="VictoriousDancing",
                                loop=asyncio.get_running_loop(), env_path=Path("unused.env"),
                                port=18879, open_browser=False, session_id="S2",
                                dossier_dir=tmp_path)
        try:
            # the feed is installed on the served player; the previous hook still runs
            assert ctrl.thoughts is not None and host.player._thoughts is ctrl.thoughts
            tag = f"battle-{FMT}-9"
            ctrl.thoughts.add(tag, "turn", "turn 1 · win-prob 0.40 (losing)", turn=1, arm="era2")
            _pvhb.GAME_DONE_HOOK(tag, _DoneBattle(tag, _Mon("kingambit")),      # ends the battle
                                 {"battle_tag": tag, "ai_team": "alpha_team", "result": "human",
                                  "session_id": "S2"})
            assert calls == [("prev", tag)]
            loop = asyncio.get_running_loop()

            def post(path, body):
                req = urllib.request.Request(ctrl.url.rstrip("/") + path, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                try:
                    r = urllib.request.urlopen(req, timeout=5)
                except urllib.error.HTTPError as e:      # the panel answers 400 with {"ok": False}
                    r = e
                return json.loads(r.read())

            js = await loop.run_in_executor(None, lambda: post("/api/options", {"matchup_team": "*"}))
            assert js["ok"] is True and ctrl.matchup_team == "*"
            js = await loop.run_in_executor(None, lambda: post("/api/options", {"matchup_team": "nope"}))
            assert js["ok"] is False and "unknown team" in js["error"]
            raw = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(ctrl.url + "api/status", timeout=5).read())
            st = json.loads(raw)
            assert st["matchups"]["mode"] == "all"
            assert st["matchups"]["session"]["rows"][0]["species"] == "kingambit"
            assert st["matchups"]["alltime"]["rows"][0]["losses"] == 1
            assert st["thoughts"]["entries"][0]["text"].startswith("turn 1")
            assert st["thoughts"]["battles"][0]["arm"] == "era2" and st["thoughts"]["battles"][0]["finished"]
            html = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(ctrl.url, timeout=5).read())
            assert b"What the nets are thinking" in html and b'id="muTeam"' in html
        finally:
            ctrl.stop()

    try:
        asyncio.run(main())
    finally:
        _pvhb.RATING_HOOK, _pvhb.RATING_CHANGE_HOOK, _pvhb.GAME_DONE_HOOK = prev


# ── model half: the stashes the narration reads (torch) ──────────────────────
def test_pair_decode_stashes_the_narration_numbers():
    torch = pytest.importorskip("torch")
    import v_dance.play.model_io as model_io
    from v_dance.encoders.state_encoder import get_action_dim, get_state_dim
    from v_dance.models.bc_model_attn import AttnBCPolicy
    A = get_action_dim()
    m = AttnBCPolicy(d_model=32, n_heads=4, n_layers=1, dropout=0.0,
                     heads=("our_a", "our_b", "opp_a", "opp_b"), gimmick_heads=("our_a", "our_b"),
                     pair_cond=True).eval()
    with torch.no_grad():                        # the test_pair_cond_2b rig: a decides 5, b|5 = 7
        for h in ("our_a", "our_b", "opp_a", "opp_b"):
            m.heads[h].weight.zero_()
            m.heads[h].bias.zero_()
        m.heads["our_a"].bias[5] = 10.0
        m.heads["our_b"].bias[2] = 1.0
        m.heads["our_b"].weight[7, -A + 5] = 100.0
    x = torch.rand(get_state_dim(), generator=torch.Generator().manual_seed(7)).numpy()
    mask = [True] * A
    m._pair_decode = True
    assert model_io.bc_action_indices(m, ("our_a", "our_b"), x, mask, mask) == (5, 7)
    d = model_io.LAST_DECODE
    assert d["pair"] is True and d["first"] == 0 and d["cond_on"] == 5 and d["picks"] == (5, 7)
    assert d["probs"][0][5] > 0.99 and d["probs"][1][7] > 0.99      # slot 1 = CONDITIONAL on 5
    assert d["conf"][0] > d["conf"][1] and d["tau"] == 0.0 and d["dropped"] == ()
    assert len(d["masks"][1]) == A and abs(sum(d["probs"][0]) - 1.0) < 1e-6
    # the futility hook's EFFECTIVE drops are reported (legal ones only)
    a0, a1 = model_io.bc_action_indices(m, ("our_a", "our_b"), x, mask, mask,
                                        pair_futility=lambda second, first, a: [7, 99])
    assert (a0, a1) == (5, 2) and model_io.LAST_DECODE["dropped"] == (7,)
    m._pair_decode = False
    assert model_io.bc_action_indices(m, ("our_a", "our_b"), x, mask, mask) == (5, 2)
    d = model_io.LAST_DECODE
    assert d["pair"] is False and d["picks"] == (5, 2)
    assert max(range(A), key=lambda i: d["probs"][1][i]) == 2      # zero-cond: bias[2] wins (low conf)
    # no legal action on slot 0 → slot 1 decodes FIRST (higher masked confidence), slot 0 stays None
    m._pair_decode = True
    assert model_io.bc_action_indices(m, ("our_a", "our_b"), x, [False] * A, mask) == (None, 2)
    d = model_io.LAST_DECODE
    assert d["picks"] == (None, 2) and d["first"] == 1 and d["cond_on"] == 2


def test_team_order_stashes_the_tp_narration(monkeypatch):
    torch = pytest.importorskip("torch")
    import v_dance.play.model_io as model_io
    from v_dance.models.teampreview_model import TeamPreviewModel
    from v_dance.training.teampreview_dataset import BRING_K, LEAD_K, MON_FEAT_DIM
    torch.manual_seed(2)
    model = TeamPreviewModel(vocab_size=16, feat_dim=MON_FEAT_DIM, emb_dim=8, hidden=16,
                             dropout=0.1, use_self_attn=True, attn_heads=2, use_set_head=True).eval()
    cfg = {"feat_dim": MON_FEAT_DIM, "bring_k": BRING_K, "lead_k": LEAD_K, "use_set_head": True}
    our = ["Pikachu", "Charizard", "Garchomp", "Amoonguss", "Incineroar", "Rillaboom"]
    opp = ["Torkoal", "Lilligant", "Flutter Mane", "Iron Hands", "Gholdengo", "Dragonite"]
    monkeypatch.setattr(model_io, "TP_SET_HEAD", True)
    monkeypatch.delenv("VD_TP_TIE_EPS", raising=False)
    order = model_io.team_order(model, {}, cfg, our, opp, n=4, device="cpu")
    s = model_io.LAST_TP
    assert s["path"] == "set_head" and len(s["subsets"]) == 15 and len(s["scores"]) == 15
    assert sorted(s["set"]) == sorted(order) and s["leads"] == order[:LEAD_K]
    assert len(s["lead_logits"]) == 6 and s["eps"] == 0.0 and not s["set_dev"] and not s["lead_dev"]
    monkeypatch.setattr(model_io, "TP_SET_HEAD", False)
    order2 = model_io.team_order(model, {}, cfg, our, opp, n=4, device="cpu")
    s = model_io.LAST_TP
    assert s["path"] == "greedy" and len(s["bring_logits"]) == 6 and sorted(s["set"]) == sorted(order2)
    # the narration composes from the real stash
    from v_dance.play.thought_feed import tp_text
    txt = tp_text(our, opp, order2, LEAD_K, dict(s))
    assert txt.startswith("TEAM PREVIEW vs Torkoal") and "greedy: bring logits" in txt
