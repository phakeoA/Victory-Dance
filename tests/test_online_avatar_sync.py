"""Avatar .env sync (play_online_browser) — browser avatar picks → .env PS_AVATAR.

v2 (2026-07-10): watches OUTGOING ``/avatar`` sends (`_sync_avatar_from_send` + `_write_avatar`) —
the server sends no updateuser on avatar changes, so v1's incoming-frame watch never fired.

The server confirms avatar changes with ``|updateuser|USER|NAMED|AVATAR|{settings}``; the sync
must apply ONLY named (post-login) updates for OUR account, rewrite just the PS_AVATAR line
(every other .env line byte-identical), and never write when nothing changed.
"""
import v_dance.play.play_online_browser as O


def _env_file(tmp_path, body):
    p = tmp_path / ".env"
    p.write_text(body, encoding="utf-8")
    return p


def _set_env(monkeypatch, **kv):
    for k, v in kv.items():
        monkeypatch.setitem(O._ENV, k, v)


BODY = "PS_USERNAME=VictoryDanceAI\nPS_PASSWORD=s3cret\nPS_AVATAR=169\nVD_DEFAULT_TEAM=maw_zard\n"


def test_outgoing_avatar_pick_rewrites_env(tmp_path, monkeypatch):
    # the client's UI pick sends a JSON-array-wrapped `|/avatar <x>` (global room = empty prefix)
    p = _env_file(tmp_path, BODY)
    _set_env(monkeypatch, PS_AVATAR="169")
    O._sync_avatar_from_send('["|/avatar lucas"]', env_path=p)
    txt = p.read_text(encoding="utf-8")
    assert "PS_AVATAR=lucas" in txt
    assert "PS_PASSWORD=s3cret" in txt and "VD_DEFAULT_TEAM=maw_zard" in txt  # others untouched
    assert O._ENV["PS_AVATAR"] == "lucas"                                     # in-memory view too


def test_bare_and_room_prefixed_sends_work(tmp_path, monkeypatch):
    p = _env_file(tmp_path, BODY)
    _set_env(monkeypatch, PS_AVATAR="169")
    O._sync_avatar_from_send("|/avatar blue", env_path=p)                     # bare frame
    assert "PS_AVATAR=blue" in p.read_text(encoding="utf-8")
    _set_env(monkeypatch, PS_AVATAR="blue")
    O._sync_avatar_from_send('["lobby|/avatar dawn"]', env_path=p)            # room-prefixed
    assert "PS_AVATAR=dawn" in p.read_text(encoding="utf-8")


def test_own_startup_send_is_a_noop(tmp_path, monkeypatch):
    # the harness's own startup `/avatar <env value>` re-sends the CURRENT value → no write
    p = _env_file(tmp_path, BODY)
    _set_env(monkeypatch, PS_AVATAR="169")
    O._sync_avatar_from_send('["|/avatar 169"]', env_path=p)
    assert p.read_text(encoding="utf-8") == BODY


def test_non_avatar_sends_ignored(tmp_path, monkeypatch):
    p = _env_file(tmp_path, BODY)
    _set_env(monkeypatch, PS_AVATAR="169")
    O._sync_avatar_from_send('["battle-x-1|/choose move 1 1, move 2 2"]', env_path=p)
    O._sync_avatar_from_send('["|/cmd userdetails avatarfan"]', env_path=p)
    assert p.read_text(encoding="utf-8") == BODY


def test_missing_avatar_line_is_appended(tmp_path, monkeypatch):
    p = _env_file(tmp_path, "PS_USERNAME=VictoryDanceAI\nPS_PASSWORD=s3cret\n")
    _set_env(monkeypatch, PS_AVATAR="")
    O._write_avatar("dawn", env_path=p)
    assert "PS_AVATAR=dawn" in p.read_text(encoding="utf-8")


def test_malformed_json_array_falls_back_to_bare_parse(tmp_path, monkeypatch):
    # a wrapper that fails json.loads degrades to the bare parse — the pick still lands
    p = _env_file(tmp_path, BODY)
    _set_env(monkeypatch, PS_AVATAR="169")
    O._sync_avatar_from_send("[not-json |/avatar erika", env_path=p)
    assert "PS_AVATAR=erika" in p.read_text(encoding="utf-8")


def test_sockjs_a_frame_unwraps_messages():
    """psim.us frames arrive SockJS-wrapped; the battle/challenge parsers need the bare text."""
    out = O._sockjs_unwrap('a["|updateuser| Bot|1|169|{}",">battle-x\\n|request|{}"]')
    assert out == ["|updateuser| Bot|1|169|{}", ">battle-x\n|request|{}"]


def test_sockjs_control_frames_dropped():
    assert O._sockjs_unwrap("o") == []
    assert O._sockjs_unwrap("h") == []
    assert O._sockjs_unwrap('c[3000,"Go away"]') == []


def test_bare_local_frames_pass_through():
    for raw in ("|pm| A| B|/challenge gen9x|", ">battle-y\n|move|p1a: X|Tackle|p2a: Y"):
        assert O._sockjs_unwrap(raw) == [raw]


def test_malformed_a_frame_passes_through():
    assert O._sockjs_unwrap("a[not-json") == ["a[not-json"]


def test_parse_challenge_ignores_our_own_outgoing_pm(monkeypatch):
    import v_dance.play.play_vs_human_browser as P
    monkeypatch.setattr(P, "_AI_NAME", "VictoriousDancing")
    theirs = P._parse_challenge("|pm| Kronomono| VictoriousDancing|/challenge gen9championsvgc2026regmb|")
    assert theirs == ("Kronomono", "gen9championsvgc2026regmb")
    ours = P._parse_challenge("|pm| VictoriousDancing| Kronomono|/challenge gen9championsvgc2026regmb|")
    assert ours is None


def test_consumer_stops_when_window_closed():
    """Window closed → the consumer notices on its idle tick (no frames arrive from a dead tab),
    sets ``stop`` and returns — the harness then shuts down exactly like a Ctrl-C."""
    import asyncio
    from v_dance.play.play_vs_human_browser import _ai_consumer

    class ClosedPage:
        def is_closed(self):
            return True

    stop = asyncio.Event()

    async def go():
        await asyncio.wait_for(
            _ai_consumer(ClosedPage(), None, asyncio.Queue(), ["team_a"],
                         {"ai": 0, "you": 0, "draw": 0}, stop),
            timeout=5.0)                          # must return on its own well before this

    asyncio.run(go())
    assert stop.is_set()
