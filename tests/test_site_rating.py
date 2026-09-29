"""2026-09-29: the official ladder numbers are the truth the panel's chained rating is checked against."""
import io
import json

from v_dance.online import site_rating as SR


def test_fetch_reads_the_format_block(monkeypatch):
    payload = {"userid": "victoriousdancing", "ratings": {
        "gen9championsvgc2026regmc": {"elo": 1231.64, "gxe": 50.5, "rpr": 1504, "rprd": 25, "w": 478, "l": 469}}}
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(SR.urllib.request, "urlopen", fake_urlopen)
    assert SR.fetch("VictoriousDancing", "gen9championsvgc2026regmc") == {"elo": 1231.6, "gxe": 50.5, "w": 478, "l": 469}
    assert seen["url"].endswith("/users/victoriousdancing.json")
    assert SR.fetch("VictoriousDancing", "gen9championsvgc2026regmb") is None


def test_drift_line_flags_a_desync():
    site = {"elo": 1067.0, "gxe": 40.0, "w": 478, "l": 469}
    assert "DESYNC" in SR.drift_line(site, 1233)
    assert "DESYNC" not in SR.drift_line({**site, "elo": 1231.6}, 1233)
    assert "panel" not in SR.drift_line(site, None)
