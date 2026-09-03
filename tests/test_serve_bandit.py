"""Era-5 W0 — the serve-side bandit (the ladder as the arbiter) + the rating reflector.

Locks: the post-battle rating parse (the in-game line reads OLD → NEW; the panel used to show
OLD, one game behind the site), arm loading, warm-up then Thompson allocation, per-tag
attribution through private-suffixed room ids, the retire rule (incumbent never retired),
persistence, applying an arm to the served player, the site JSON parse, and the panel wiring.
No network, no torch, no Playwright."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("poke_env")

import v_dance.play.play_vs_human_browser as _pvhb
from v_dance.play import serve_bandit as SB
from v_dance.play.bot_control_ui import BotController, RatingBook

FMT = "gen9championsvgc2026regmb"


def _tag(n, suffix=""):
    return f"battle-{FMT}-{n}{suffix}"


# ── rating reflector: parse OLD → NEW ────────────────────────────────────────
def test_rating_change_parse_gives_old_and_new_while_legacy_parse_gives_old():
    frame = (f">{_tag(1)}\n"
             "|raw|VictoriousDancing's rating: 1120 &rarr; <strong>1141</strong><br />(+21 for winning)\n"
             "|raw|SomeOpp's rating: 1300 &rarr; <strong>1279</strong><br />(-21 for losing)")
    assert _pvhb._parse_rating_changes(frame) == [("VictoriousDancing", 1120, 1141),
                                                  ("SomeOpp", 1300, 1279)]
    assert _pvhb._parse_rating_lines(frame) == [("VictoriousDancing", 1120), ("SomeOpp", 1300)]
    assert _pvhb._parse_rating_changes(f">{_tag(1)}\n|turn|3") == []


# ── arms config ──────────────────────────────────────────────────────────────
def test_load_arms_drops_missing_checkpoints_and_keeps_defaults(tmp_path: Path):
    cfg = {"arms": [
        {"name": "inc", "battle_ckpt": "default", "tp_ckpt": "default", "incumbent": True},
        {"name": "real", "battle_ckpt": "ai_train_scripts/x/battle_base.pt", "tau": 0.2, "top_p": 0.9},
        {"name": "ghost", "battle_ckpt": "ai_train_scripts/missing/battle_base.pt"},
    ]}
    p = tmp_path / "arms.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    arms = SB.load_arms(p, exists=lambda q: "missing" not in str(q))
    assert [a.name for a in arms] == ["inc", "real"]
    assert arms[0].incumbent and arms[0].uses_default("battle") and arms[0].uses_default("tp")
    assert arms[1].tau == 0.2 and arms[1].top_p == 0.9 and not arms[1].uses_default("battle")


def test_load_arms_promotes_first_arm_to_incumbent_when_none_marked(tmp_path: Path):
    p = tmp_path / "arms.json"
    p.write_text(json.dumps({"arms": [{"name": "a"}, {"name": "b"}]}), encoding="utf-8")
    arms = SB.load_arms(p, exists=lambda q: True)
    assert arms[0].incumbent and not arms[1].incumbent


def test_repo_config_loads_and_names_real_checkpoints():
    # config/ is gitignored (local machine config) and CI has no checkpoints: skip when the file
    # is absent, and treat every checkpoint path as present so the parse itself is what is tested.
    if not SB.DEFAULT_CONFIG.is_file():
        pytest.skip("config/serve_bandit.json is a local (gitignored) file — not on this machine")
    arms = SB.load_arms(SB.DEFAULT_CONFIG, exists=lambda p: True)
    names = [a.name for a in arms]
    assert "era4_2b" in names and any(a.incumbent for a in arms)
    assert any(a.tau > 0 for a in arms)            # the W3a exploration arms are present


# ── allocation ───────────────────────────────────────────────────────────────
def _bandit(tmp_path, names=("inc", "a", "b"), **kw):
    arms = [SB.Arm(name=n, incumbent=(i == 0)) for i, n in enumerate(names)]
    kw.setdefault("min_games", 2)
    return SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / "state.json", seed=1,
                          now=lambda: 1000.0, **kw)


def test_warm_up_round_robins_every_arm_before_thompson(tmp_path: Path):
    b = _bandit(tmp_path)
    played = []
    for i in range(6):
        arm = b.choose()
        b.bind(_tag(i))
        b.observe(_tag(i), +5)
        played.append(arm.name)
    assert sorted(played) == sorted(["inc", "a", "b"] * 2)   # 2 games each before any Thompson


def test_choose_is_idempotent_until_the_battle_starts(tmp_path: Path):
    b = _bandit(tmp_path)
    first = b.choose().name
    assert b.choose().name == first and b.pending == first
    b.bind(_tag(1))
    assert b.pending is None and b.arm_for(_tag(1)) == first


def test_attribution_survives_the_private_room_suffix(tmp_path: Path):
    b = _bandit(tmp_path)
    arm = b.choose().name
    b.bind(_tag(7, "-6x833abcpw"))                 # the room's own frames carry the suffix
    assert b.observe(_tag(7), -12) == arm          # the rating line may come under the bare id
    assert b.stats[arm].n == 1 and b.stats[arm].losses == 1 and b.stats[arm].mean_delta() == -12


def test_thompson_prefers_the_arm_with_better_rating_deltas(tmp_path: Path):
    b = _bandit(tmp_path, names=("inc", "good", "bad"), min_games=1)
    for i, (name, d) in enumerate([("inc", 0), ("good", 0), ("bad", 0)]):
        b.pending = name
        b.bind(_tag(i))
        b.observe(_tag(i), d)
    for i in range(10):                            # good: +20 each, bad: −20 each, inc: 0
        for name, d in (("good", 20), ("bad", -20), ("inc", 0)):
            b.pending = name
            b.bind(_tag(100 + 3 * i + ("good", "bad", "inc").index(name)))
            b.observe(_tag(100 + 3 * i + ("good", "bad", "inc").index(name)), d)
    picks = []
    for i in range(60):
        b.pending = None
        picks.append(b.choose().name)
    assert picks.count("good") >= 40 and picks.count("bad") <= 6


def test_retire_rule_kills_a_clearly_worse_arm_but_never_the_incumbent_or_the_learning_arm(tmp_path: Path):
    """2026-09-03 rule: retire at >= retire_min_games when P(mean delta/game < the incumbent's) >= retire_prob.
    Never the incumbent, never the LEARNING (adaptive) arm — it bleeds rating by design."""
    arms = [SB.Arm(name="inc", incumbent=True), SB.Arm(name="a"), SB.Arm(name="b"),
            SB.Arm(name="learn", tau=0.3, learning=True)]
    b = SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / "state.json", seed=1, now=lambda: 1000.0,
                       min_games=2, retire_min_games=40, retire_prob=0.90)
    # incumbent 30W-10L (+5/g), 'a' 10W-30L (-5/g, P(worse) ~ 1), 'b' 28W-12L (+4/g, P ~ .70),
    # 'learn' = the same record as 'a' but flagged learning — only 'a' dies
    k = 0
    for name, wins, losses in (("inc", 30, 10), ("a", 10, 30), ("b", 28, 12), ("learn", 10, 30)):
        for d in [+10] * wins + [-10] * losses:
            b.pending = name
            b.bind(_tag(k))
            b.observe(_tag(k), d)
            k += 1
    assert b.stats["a"].retired and "P(worse than incumbent inc)" in b.stats["a"].retired_reason
    assert "after 40 games" in b.stats["a"].retired_reason
    assert not b.stats["inc"].retired and not b.stats["b"].retired and not b.stats["learn"].retired
    assert [a.name for a in b.active_arms()] == ["inc", "b", "learn"]
    assert b.p_worse("inc") is None and b.p_worse("a") > 0.99 and b.p_worse("learn") > 0.99
    assert 0.6 < b.p_worse("b") < 0.8
    # a retired arm is never chosen again
    for _ in range(30):
        b.pending = None
        assert b.choose().name != "a"
    # the UIs' fields: the learning label, P(worse) per arm, the rule summary, the banner
    rows = {r["name"]: r for r in b.summary()}
    assert rows["learn"]["learning"] is True and rows["a"]["learning"] is False
    assert rows["inc"]["p_worse"] is None and rows["a"]["p_worse"] > 0.99
    assert b.rule() == {"min_games": 2, "retire_min_games": 40, "retire_prob": 0.9, "retire_margin_elo": 0.0,
                        "incumbent": "inc", "learning": ["learn"], "learning_share": 0.0}
    assert "P(worse than the incumbent) ≥ 0.90" in b.banner() and "learning arm(s) exempt: learn" in b.banner()
    # a margin in Elo/game makes the rule stricter: 'a' is 10/g under the incumbent, so 12/g never retires it
    b2 = SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / "state2.json", seed=1, now=lambda: 1000.0,
                        min_games=2, retire_min_games=40, retire_prob=0.90, retire_margin_elo=12.0)
    k = 0
    for name, wins, losses in (("inc", 30, 10), ("a", 10, 30)):
        for d in [+10] * wins + [-10] * losses:
            b2.pending = name
            b2.bind(_tag(k))
            b2.observe(_tag(k), d)
            k += 1
    assert not b2.stats["a"].retired and b2.p_worse("a") < 0.5


def test_config_params_reads_the_new_retire_keys_and_ignores_the_old_margin(tmp_path: Path):
    p = tmp_path / "arms.json"
    p.write_text(json.dumps({"min_games": 8, "retire_min_games": 40, "retire_margin": 0.1, "retire_prob": 0.9,
                             "retire_margin_elo": 0.0,
                             "arms": [{"name": "inc", "incumbent": True},
                                      {"name": "tau03", "tau": 0.3, "learning": True}]}), encoding="utf-8")
    assert SB.config_params(p) == {"min_games": 8, "retire_min_games": 40, "retire_prob": 0.9,
                                   "retire_margin_elo": 0.0}
    arms = SB.load_arms(p, exists=lambda q: True)
    assert [a.learning for a in arms] == [False, True]


def test_state_persists_across_restarts(tmp_path: Path):
    b = _bandit(tmp_path)
    b.pending = "a"
    b.bind(_tag(1))
    b.observe(_tag(1), +17)
    b2 = _bandit(tmp_path)                         # same state_path → reloads
    assert b2.stats["a"].n == 1 and b2.stats["a"].sum_delta == 17
    assert b2.arm_for(_tag(1)) == "a"


def test_apply_pending_invokes_the_applier_only_when_the_arm_changes(tmp_path: Path):
    applied = []
    arms = [SB.Arm(name="inc", incumbent=True), SB.Arm(name="a")]
    b = SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / "s.json", seed=0,
                       applier=lambda arm: applied.append(arm.name), min_games=1)
    first = b.apply_pending()
    b.apply_pending()                              # same pending arm → no second apply
    assert applied == [first.name]
    b.bind(_tag(1))
    b.observe(_tag(1), 0)
    nxt = b.apply_pending()
    assert applied[-1] == nxt.name or nxt.name == first.name


def test_summary_and_banner_shape(tmp_path: Path):
    b = _bandit(tmp_path)
    s = b.summary()
    assert [d["name"] for d in s] == ["inc", "a", "b"] and s[0]["incumbent"]
    assert set(s[0]) >= {"n", "wins", "losses", "mean_delta", "win_rate", "retired", "current", "tau"}
    assert "serve BANDIT ACTIVE" in b.banner() and "incumbent inc" in b.banner()


# ── applying an arm to the served player ─────────────────────────────────────
class _FakePlayer:
    def __init__(self):
        self._model = self._model_heads = None
        self._team_chooser = self._tc_vocab = self._tc_cfg = None
        self._temperature, self._top_p, self._rng = 0.0, 1.0, None


class _FakeHost:
    def __init__(self):
        self.player = _FakePlayer()


def test_apply_arm_swaps_handles_knobs_and_caches_loads(monkeypatch):
    loads = []

    def loader(kind, path):
        loads.append((kind, str(path)))
        return (("M", path), ("H",)) if kind == "battle" else (("TC", path), {"v": 1}, {"cfg": 1})

    host, cache = _FakeHost(), {}
    monkeypatch.delenv("VD_TP_TIE_EPS", raising=False)
    a1 = SB.Arm(name="x", battle_ckpt="ai_train_scripts/x/battle_base.pt", tp_ckpt="default",
                tau=0.3, top_p=0.9, tp_tie_eps=0.5)
    SB.apply_arm(host, a1, cache, default_battle="D_B.pt", default_tp="D_TP.pt", loader=loader)
    p = host.player
    assert p._model == ("M", SB._resolve("ai_train_scripts/x/battle_base.pt")) and p._model_heads == ("H",)
    assert p._team_chooser == ("TC", Path("D_TP.pt")) and p._tc_vocab == {"v": 1}
    assert p._temperature == 0.3 and p._top_p == 0.9 and p._rng is not None
    assert __import__("os").environ["VD_TP_TIE_EPS"] == "0.5" and p._arm_name == "x"
    a2 = SB.Arm(name="inc", battle_ckpt="default", tp_ckpt="default", tau=0.0, incumbent=True)
    SB.apply_arm(host, a2, cache, default_battle="D_B.pt", default_tp="D_TP.pt", loader=loader)
    assert p._temperature == 0.0 and p._rng is None and p._arm_name == "inc"
    SB.apply_arm(host, a1, cache, default_battle="D_B.pt", default_tp="D_TP.pt", loader=loader)
    assert len(loads) == 3                         # x-battle, default-tp, default-battle: each ONCE


# ── the site's JSON ──────────────────────────────────────────────────────────
def test_fetch_official_ratings_parses_the_profile_json():
    body = json.dumps({"username": "VictoriousDancing", "ratings": {
        FMT: {"elo": 1141.0273, "gxe": 47.1, "rpr": 1478.36, "rprd": 25, "w": 441, "l": 450, "coil": None}}})
    seen = []
    out = SB.fetch_official_ratings("victoriousdancing", opener=lambda u: (seen.append(u), body)[1])
    assert seen == ["https://pokemonshowdown.com/users/victoriousdancing.json"]
    assert out[FMT] == {"elo": 1141, "gxe": 47.1, "glicko": 1478, "glicko_dev": 25, "w": 441, "l": 450}


def test_rating_book_prefers_the_post_battle_value(tmp_path: Path):
    p = tmp_path / "bench.jsonl"
    p.write_text(json.dumps({"type": "rating_update", "battle_tag": _tag(1), "rating": 1120,
                             "rating_after": 1141}) + "\n", encoding="utf-8")
    assert RatingBook(p).summary()[0]["all_time_peak"] == 1141


# ── panel wiring ─────────────────────────────────────────────────────────────
class _Page:
    async def evaluate(self, js, arg=None):
        return None


class _PanelHost:
    def __init__(self):
        class _T:
            def yield_team(self):
                return "PACKED"

        self.player = type("P", (), {"_team": _T(), "_team_name": None,
                                     "update_team": lambda self, t: None})()


def test_panel_applies_an_arm_before_search_and_binds_it_on_battle_start(monkeypatch, tmp_path):
    from v_dance.play import bot_control_ui as bcu
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])
    monkeypatch.setenv("VD_SITE_POLL", "0")
    applied = []
    arms = [SB.Arm(name="inc", incumbent=True), SB.Arm(name="a")]
    b = SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / "s.json", seed=0, min_games=1,
                       applier=lambda arm: applied.append(arm.name))

    async def main():
        c = BotController(page=_Page(), host=_PanelHost(), tally={"ai": 0, "you": 0, "draw": 0},
                          ai_pool=["alpha"], fmt=FMT, username="VictoriousDancing",
                          loop=asyncio.get_running_loop(), env_path=tmp_path / "u.env", bandit=b)
        c._load_scoped_team = lambda scoped: "LOADED"
        await c.start_ladder(3, "alpha")
        assert applied and b.pending == applied[-1]           # an arm was applied for the next game
        assert any(e.endswith("(search)") or "arm →" in e for e in c.events)
        c._battle_seen(_tag(5, "-privpw"))
        assert b.arm_for(_tag(5)) == applied[-1] and b.pending is None
        assert any("[arm " in e for e in c.events)
        # a second apply while the battle is LIVE is refused (no mid-game model swap)
        assert c.apply_next_arm("challenge") is None
        st = c.status()
        assert st["bandit"][0]["name"] == "inc" and "official" in st
        return c

    asyncio.run(main())


def test_fetch_official_ratings_default_opener_sends_a_real_user_agent(monkeypatch):
    """2026-09-02 live check: pokemonshowdown.com answers 403 to the default Python-urllib
    User-Agent and 200 to anything else — the default opener must send our own header."""
    import io as _io
    import urllib.request as _ur
    seen = {}

    class _Resp(_io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")
        seen["url"] = req.full_url
        return _Resp(b'{"ratings": {"gen9championsvgc2026regmb": {"elo": 1165.5, "gxe": 47.4, '
                     b'"rpr": 1480.0, "rprd": 25, "w": 443, "l": 451}}}')

    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)
    out = SB.fetch_official_ratings("victoriousdancing")
    assert seen["url"].endswith("/users/victoriousdancing.json")
    assert seen["ua"] and "python-urllib" not in seen["ua"].lower() and "Victory-Dance" in seen["ua"]
    assert out["gen9championsvgc2026regmb"]["elo"] == 1166 and out["gen9championsvgc2026regmb"]["w"] == 443


def test_learning_share_is_a_floor_for_the_learning_arm_after_warm_up(tmp_path: Path):
    """2026-09-03 L2: Thompson starved the learning arm (6 % of night 1); with learning_share the arm
    gets at least that share of games after the warm-up. Share 0 / no learning arm = pure Thompson."""
    def bandit(share, learning=True, seed=1):
        arms = [SB.Arm(name="inc", incumbent=True), SB.Arm(name="good"),
                SB.Arm(name="learn", tau=0.3, learning=learning)]
        b = SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / f"s{share}{learning}.json", seed=seed,
                           now=lambda: 1000.0, min_games=2, learning_share=share)
        k = 0
        for name, deltas in (("inc", [+8, +8]), ("good", [+20, +20]), ("learn", [-15, -15])):   # warm-up
            for d in deltas:
                b.pending = name
                b.bind(_tag(k))
                b.observe(_tag(k), d)
                k += 1
        picks = []
        for i in range(200):
            b.pending = None
            picks.append(b.choose().name)
            b.bind(_tag(1000 + i))
            b.observe(_tag(1000 + i), {"inc": 8, "good": 20, "learn": -15}[picks[-1]])
        return b, picks
    _, p0 = bandit(0.0)
    assert p0.count("learn") <= 30                                  # Thompson alone starves the -15/g arm
    b3, p3 = bandit(0.3)
    assert 40 <= p3.count("learn") <= 90                            # the floor: ~30 % of 200 (+ Thompson's few)
    assert p3.count("good") >= 80                                   # the rest still goes to Thompson's favourite
    assert p3.count("learn") > p0.count("learn") + 20               # the floor is what made the difference
    assert b3.rule()["learning_share"] == 0.3 and "floor share 30%" in b3.banner()
    _, p1 = bandit(1.0)
    assert p1.count("learn") == 200                                 # share 1.0 = every post-warm-up game
    _, pn = bandit(0.3, learning=False)
    assert pn.count("learn") <= 30                                  # no learning arm → the floor is inert
    assert SB.config_params(_cfg_with(tmp_path, {"learning_share": 0.3})) == {"learning_share": 0.3}


def _cfg_with(tmp_path: Path, extra: dict) -> Path:
    p = tmp_path / "share.json"
    p.write_text(json.dumps({**extra, "arms": [{"name": "inc"}]}), encoding="utf-8")
    return p
