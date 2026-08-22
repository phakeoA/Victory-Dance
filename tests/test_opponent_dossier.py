"""B-L2 OpponentDossier capture tests: merge semantics, W-L bookkeeping, robustness."""
import json
from types import SimpleNamespace

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
