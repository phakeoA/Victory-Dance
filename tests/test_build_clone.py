"""League P1 (2026-09-29): the nemesis clone picks exactly the bot's LOST games (winner = the human) and links them."""
import json

from v_dance.selfplay.build_clone import link_into, loss_files


def _write(path, winner):
    path.write_text(json.dumps({"winner": winner, "turn": 1}) + "\n" + json.dumps({"turn": 2}) + "\n", encoding="utf-8")


def test_loss_files_keeps_only_games_the_bot_lost(tmp_path):
    src = tmp_path / "Jsonl_TypeC"
    src.mkdir()
    _write(src / "a.jsonl", "VictoriousDancing")     # bot win (self-imitation) -> excluded
    _write(src / "b.jsonl", "thesean37")             # bot loss -> the human winner's side
    _write(src / "c.jsonl", "")                      # no result -> excluded
    (src / "d.jsonl").write_text("not json\n", encoding="utf-8")
    got = [f.name for f in loss_files(src, {"victoriousdancing"})]
    assert got == ["b.jsonl"]


def test_link_into_is_idempotent_and_keeps_contents(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write(src / "b.jsonl", "thesean37")
    dest = tmp_path / "Jsonl_TypeC_nemesis"
    assert link_into([src / "b.jsonl"], dest) == 1
    assert link_into([src / "b.jsonl"], dest) == 0
    assert (dest / "b.jsonl").read_text(encoding="utf-8") == (src / "b.jsonl").read_text(encoding="utf-8")


def test_split_by_archetype_writes_one_file_per_side(tmp_path):
    from v_dance.selfplay.build_clone import _is_val, split_by_archetype
    src = tmp_path / "src"
    src.mkdir()
    rows = [{"replay_id": "r1", "perspective": p, "turn": t} for p in ("p1", "p2") for t in (1, 2)]
    (src / "r1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    out = tmp_path / "Archetypes_k10"
    counts = split_by_archetype([src], {("r1", "p1"): 3, ("r1", "p2"): 7}, out)
    for arch, persp in ((3, "p1"), (7, "p2")):
        split = "val" if _is_val("r1", persp) else "train"
        f = out / f"a{arch:02d}" / split / f"r1_{persp}.jsonl"
        got = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]
        assert [r["perspective"] for r in got] == [persp, persp] and counts[(arch, split)] == 1
