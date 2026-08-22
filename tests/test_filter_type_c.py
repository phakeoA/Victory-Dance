"""Type_C pre-ingest filter (USER, 2026-07-10): turn-1 forfeits / turn-1 timeouts / disconnect-
aborted games carry nothing to imitate — they must be classified junk and (with --apply) moved to
_excluded/ so bulk_parse_replays never ingests them. The manifest also audits every kept game's
pre-game |player| ratings — the field replay_parser lifts into players.pN.rating_before, which
train_bc --rating-weight / --rating-min consume (elo-valued Type_C training)."""
from __future__ import annotations

import json

from v_dance.datatools.filter_type_c import classify_log, run, scan

FMT = "gen9championsvgc2026regmb"


def _html(log_lines, tag=f"battle-{FMT}-123"):
    log = "\n".join(log_lines)
    return (f'<input type="hidden" name="replayid" value="{tag}" />\n'
            f'<script type="text/plain" class="battle-log-data">{log}</script>')


def _header(rated=True):
    r1, r2 = ("|1347", "|1333") if rated else ("", "")
    return [
        "|init|battle",
        f"|player|p1|Opponent|266{r1}",
        f"|player|p2|VictoriousDancing|cynthia{r2}",
        "|gametype|doubles",
        "|teampreview|4",
        "|start",
    ]


def _real_game():
    lines = _header()
    for t in range(1, 8):
        lines += [f"|turn|{t}",
                  "|move|p1a: Politoed|Weather Ball|p2a: Mawile",
                  "|move|p2a: Mawile|Play Rough|p1a: Politoed"]
    lines.append("|win|VictoriousDancing")
    return lines


# ── classification ────────────────────────────────────────────────────────────
def test_classify_real_game_kept_with_ratings():
    info = classify_log("\n".join(_real_game()))
    assert info["status"] == "ok" and info["turns"] == 7 and info["moves"] == 14
    assert info["ratings"] == {"Opponent": 1347, "VictoriousDancing": 1333}
    assert info["winner"] == "VictoriousDancing"


def test_classify_turn1_forfeit_and_timeout_are_junk():
    ff = _header() + ["|turn|1", "|-message|Opponent forfeited.", "|win|VictoriousDancing"]
    assert classify_log("\n".join(ff))["status"] == "turn1_forfeit"
    to = _header() + ["|turn|1", "|move|p2a: Mawile|Play Rough|p1a: Politoed",
                      "|-message|Opponent lost due to inactivity.", "|win|VictoriousDancing"]
    assert classify_log("\n".join(to))["status"] == "turn1_timeout"
    # a teampreview-stage forfeit never reaches |turn|1 → turn 0, still junk
    tp = _header() + ["|-message|Opponent forfeited.", "|win|VictoriousDancing"]
    assert classify_log("\n".join(tp))["status"] == "turn0_forfeit"


def test_classify_disconnect_without_result_is_junk():
    dc = _header() + ["|turn|1", "|move|p1a: A|Protect|p1a: A"]     # log just stops
    assert classify_log("\n".join(dc))["status"] == "no_result"


def test_classify_late_forfeit_is_kept_by_default_threshold():
    late = _header()
    for t in range(1, 6):
        late += [f"|turn|{t}", "|move|p1a: A|Surf|p2a: B"]
    late += ["|-message|Opponent forfeited.", "|win|VictoriousDancing"]
    assert classify_log("\n".join(late))["status"] == "ok"           # 5 real turns = learnable
    assert classify_log("\n".join(late), max_turn=5)["status"] == "turn5_forfeit"


def test_unrated_game_has_no_ratings_but_is_kept():
    info = classify_log("\n".join(_header(rated=False) + ["|turn|1", "|move|p1a: A|Surf|p2a: B",
                                                          "|turn|2", "|move|p1a: A|Surf|p2a: B",
                                                          "|win|Opponent"]))
    assert info["status"] == "ok" and info["ratings"] == {}


# ── end-to-end on a folder ────────────────────────────────────────────────────
def _seed(tmp_path):
    (tmp_path / "good.html").write_text(_html(_real_game()), encoding="utf-8")
    (tmp_path / "ff.html").write_text(
        _html(_header() + ["|turn|1", "|-message|X forfeited.", "|win|VictoriousDancing"],
              tag=f"battle-{FMT}-666"), encoding="utf-8")
    (tmp_path / "broken.html").write_text("<html>not a replay</html>", encoding="utf-8")


def test_run_dry_then_apply_deletes_junk_and_writes_manifest(tmp_path):
    _seed(tmp_path)

    m = run(tmp_path, apply=False)                                  # dry-run: nothing touched
    assert m["good.html"]["status"] == "ok" and not m["good.html"]["removed"]
    assert m["ff.html"]["status"] == "turn1_forfeit"
    assert m["broken.html"]["status"] == "unreadable"
    assert (tmp_path / "ff.html").exists()

    m = run(tmp_path, apply=True)                                   # apply: junk is DELETED (USER)
    assert not (tmp_path / "ff.html").exists()
    assert not (tmp_path / "broken.html").exists()
    assert not (tmp_path / "_excluded").exists()                    # no quarantine by default
    assert (tmp_path / "good.html").exists()                        # real game untouched
    saved = json.loads((tmp_path / "type_c_manifest.json").read_text(encoding="utf-8"))
    assert saved["ff.html"]["removed"] is True and saved["ff.html"]["action"] == "deleted"
    assert saved["good.html"]["removed"] is False and saved["good.html"]["action"] is None
    assert saved["good.html"]["ratings"]["Opponent"] == 1347        # the elo audit trail
    assert saved["good.html"]["battle_tag"] == f"battle-{FMT}-123"

    run(tmp_path, apply=True)                                       # rerun: idempotent
    assert set(scan(tmp_path)) == {"good.html"}


def test_run_quarantine_mode_moves_instead_of_deleting(tmp_path):
    _seed(tmp_path)
    m = run(tmp_path, apply=True, quarantine=True)
    assert (tmp_path / "_excluded" / "ff.html").exists()            # preserved, not deleted
    assert (tmp_path / "_excluded" / "broken.html").exists()
    assert (tmp_path / "good.html").exists()
    assert m["ff.html"]["action"] == "quarantined"
    assert set(scan(tmp_path)) == {"good.html"}                     # _excluded never rescanned


# ── the bulk_parse_replays pipeline gate (junk can never become corpus JSONL) ──
def test_bulk_parse_junk_gate(tmp_path):
    from v_dance.datatools.bulk_parse_replays import _junk_status, _purge_stale_exports

    assert _junk_status(_html(_real_game())) is None                # real game → parse normally
    ff = _html(_header() + ["|turn|1", "|move|p2a: Mawile|Play Rough|p1a: Politoed",
                            "|-message|X forfeited.", "|win|VictoriousDancing"])
    assert _junk_status(ff) == "turn1_forfeit"
    assert _junk_status(ff, max_turn=0) is None                     # threshold respected
    ff0 = _html(_header() + ["|turn|1", "|-message|X forfeited.", "|win|VictoriousDancing"])
    assert _junk_status(ff0, max_turn=0) == "no_moves"              # zero-move game = junk anyway
    assert _junk_status("<html>not a replay</html>") is None        # corruption → REAL error path

    out = tmp_path / "battle-x-1.jsonl"
    legacy = tmp_path / "stemname.jsonl"
    out.write_text("{}", encoding="utf-8")
    legacy.write_text("{}", encoding="utf-8")
    assert _purge_stale_exports(out, legacy) == 2                   # stale junk exports removed
    assert not out.exists() and not legacy.exists()
    assert _purge_stale_exports(out, out) == 0                      # absent / same-path safe


# ── --winner-only ingest (USER 2026-07-10: every Type_C game teaches from its WINNER) ──
def test_winner_side_resolution():
    from v_dance.datatools.bulk_parse_replays import _winner_side

    assert _winner_side(_html(_real_game())) == "p2"                # |win|VictoriousDancing
    p1win = _html(_header() + ["|turn|1", "|move|p1a: A|Surf|p2a: B", "|win|Opponent"])
    assert _winner_side(p1win) == "p1"
    # display-name normalisation: |win| name matches the |player| name case-insensitively
    mixed = _html(_header() + ["|turn|1", "|move|p1a: A|Surf|p2a: B", "|win|OPPONENT"])
    assert _winner_side(mixed) == "p1"
    tie = _html(_header() + ["|turn|1", "|move|p1a: A|Surf|p2a: B", "|tie|"])
    assert _winner_side(tie) is None                                # tie → no perspective
    nores = _html(_header() + ["|turn|1", "|move|p1a: A|Surf|p2a: B"])
    assert _winner_side(nores) is None                              # aborted → no perspective
    assert _winner_side("<html>not a replay</html>") is None
