"""B-L2 OpponentDossier capture tests: merge semantics, W-L bookkeeping, robustness."""
import json
from types import SimpleNamespace

import pytest

import v_dance.play.opponent_dossier as od


def _battle(opp="DeathTheUser", moves=("psychic", "trickroom"), item="safetygoggles"):
    mon = SimpleNamespace(species="Farigiraf", moves={m: None for m in moves},
                          item=item, ability="armortail")
    return SimpleNamespace(opponent_username=opp, opponent_team={"p2: Farigiraf": mon},
                           battle_tag="battle-x-1", turn=12)


def _redirect(monkeypatch, tmp_path):
    monkeypatch.setattr(od, "DOSSIER_DIR", tmp_path)


class TestDossier:
    def test_merge_and_wl(self, monkeypatch, tmp_path):
        _redirect(monkeypatch, tmp_path)
        p = od.update_from_battle(_battle(), "ai", our_team="maw_zard", note="t")
        assert p is not None and p.is_file() and p.stem == "deaththeuser"
        od.update_from_battle(_battle(moves=("psychic", "wideguard")), "human")
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["losses_vs_us"] == 1 and d["wins_vs_us"] == 1       # "ai"=our win=their loss
        rec = d["mons"]["farigiraf"]
        assert rec["times_seen"] == 2
        assert set(rec["moves"]) == {"psychic", "trickroom", "wideguard"}  # union across games
        assert rec["item"] == "safetygoggles"
        assert len(d["games"]) == 2 and d["games"][0]["our_team"] == "maw_zard"
        assert "2 game(s)" in od.summary("DeathTheUser")

    def test_never_raises(self, monkeypatch, tmp_path):
        _redirect(monkeypatch, tmp_path)
        assert od.update_from_battle(SimpleNamespace(opponent_username=None), "ai") is None
        assert od.update_from_battle(None, "ai") is None             # garbage in → None out

    def test_games_capped(self, monkeypatch, tmp_path):
        _redirect(monkeypatch, tmp_path)
        monkeypatch.setattr(od, "MAX_GAMES", 3)
        for _ in range(5):
            od.update_from_battle(_battle(), "draw")
        d = od.load("DeathTheUser")
        assert len(d["games"]) == 3 and d["draws_vs_us"] == 5        # history capped, tally full

    def test_mega_evolved_mon_stores_the_stone_and_keeps_the_base_ability_clean(self, monkeypatch, tmp_path):
        """USER 2026-09-02 ("fix at the source"): poke-env keeps the base species after a mega, swaps
        the ability to the mega forme's and never sets the stone — the dossier must not store
        Pixilate as a Gardevoir's base ability with no item."""
        from v_dance.play.matchup_book import _dex
        if _dex() is None:
            pytest.skip("data/pokedex.json not available")
        _redirect(monkeypatch, tmp_path)
        mega = SimpleNamespace(species="gardevoir", moves={"hypervoice": None}, item="unknown_item",
                               ability="pixilate")
        plain = SimpleNamespace(species="garchomp", moves={"earthquake": None}, item="unknown_item",
                                ability="roughskin")
        b = SimpleNamespace(opponent_username="MegaFan", battle_tag="battle-x-7", turn=9,
                            opponent_team={"p2: Gardevoir": mega, "p2: Garchomp": plain})
        od.update_from_battle(b, "human", our_team="The_Big_6")
        d = od.load("MegaFan")
        g = d["mons"]["gardevoir"]
        assert (g["item"], g["mega"], g["mega_ability"], g["ability"], g["mega_seen"]) == \
            ("gardevoirite", "Gardevoir-Mega", "pixilate", None, 1)
        assert d["mons"]["garchomp"]["ability"] == "roughskin" and d["mons"]["garchomp"]["item"] is None
        assert d["games"][-1]["megas"] == ["gardevoir"]
        assert d["games"][-1]["revealed"] == ["garchomp", "gardevoir"]
        # a later game without the mega: the stone and the mega keys stay, the base ability lands
        mega.ability = "trace"
        od.update_from_battle(b, "ai")
        d = od.load("MegaFan")
        g = d["mons"]["gardevoir"]
        assert (g["item"], g["ability"], g["mega_seen"], g["mega"]) == \
            ("gardevoirite", "trace", 1, "Gardevoir-Mega")
        assert d["games"][-1]["megas"] == [] and g["times_seen"] == 2
        # the backfill rewrites an OLD-shape record the same way (dry-run plan + rewrite)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dossier_mega_backfill", od._REPO / "scratch" / "dossier_mega_backfill.py")
        bf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bf)
        old = {"opponent": "Old", "games": [], "mons": {
            "gardevoir": {"species": "gardevoir", "moves": [], "item": None, "ability": "pixilate", "times_seen": 2},
            "latios": {"species": "latios", "moves": [], "item": None, "ability": "levitate", "times_seen": 1},
            "garchomp": {"species": "garchomp", "moves": [], "item": "lifeorb", "ability": "roughskin", "times_seen": 1}}}
        (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
        changes, scanned, bad = bf.plan(tmp_path)
        assert scanned == 2 and bad == 0 and [(p.name, [s for s, _ in todo]) for p, _d, todo in changes] == \
            [("old.json", ["gardevoir"])]                   # Latios Levitate is shared → untouched
        bf.rewrite(changes[0][1], changes[0][2])
        rec = changes[0][1]["mons"]["gardevoir"]
        assert (rec["item"], rec["mega"], rec["mega_ability"], rec["ability"], rec["mega_seen"]) == \
            ("gardevoirite", "Gardevoir-Mega", "pixilate", None, 1)
        assert changes[0][1]["mons"]["latios"]["ability"] == "levitate"
