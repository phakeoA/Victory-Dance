"""Generator for ``champions_move_overrides.py`` (v11 N1).

The custom **Champions** VGC format (`pokemon-showdown/data/mods/champions/moves.ts`) overrides the
basePower / type / accuracy / pp of ~40 moves vs standard Gen-9. Both encoders otherwise read STANDARD
Gen-9 move data (offline ``GenData(9)``, live poke-env ``Move``), so those moves get the wrong damage
band / type-eff / accuracy. This script parses the mod file and emits a FROZEN, committed Python module of
the overrides — static (not parsed at runtime) so a serve box without the cloned ``pokemon-showdown/`` repo
can't silently fall back to standard values (a parity hazard).

Regenerate after a Champions mod bump:  ``python -m v_dance.encoders.gen_champions_move_overrides``
A test (``tests/test_champions_move_overrides.py::test_generator_idempotent``) re-runs this and asserts the
committed module is byte-identical, so an un-regenerated bump fails loud.
"""
from __future__ import annotations

import re
from pathlib import Path

# v_dance/encoders/gen_…py → parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MOVES_TS = _REPO_ROOT / "pokemon-showdown" / "data" / "mods" / "champions" / "moves.ts"
_OUT = Path(__file__).resolve().parent / "champions_move_overrides.py"

# the gameplay fields the ENCODER consumes (pp emitted for completeness; see the module docstring)
_NUM_RE = {
    "basePower": re.compile(r"\n\t\tbasePower:\s*(\d+),"),
    "pp":        re.compile(r"\n\t\tpp:\s*(\d+),"),
}
_TYPE_RE = re.compile(r'\n\t\ttype:\s*"([^"]+)",')
_ACC_RE = re.compile(r"\n\t\taccuracy:\s*([^,\n]+),")


def parse_overrides(ts_path: Path = _MOVES_TS) -> dict[str, dict]:
    """{field: {move_id: value}} for basePower/type/accuracy/pp, from each top-level ``inherit:true``
    move block (only listed fields change). accuracy is a percent int or ``True`` (always-hit)."""
    txt = ts_path.read_text(encoding="utf-8")
    # top-level entries: `\tname: { ... \n\t},` (single-tab key, body until the closing `\n\t},`)
    entries = re.findall(r"\n\t(\w+):\s*\{(.*?)\n\t\},", txt, re.S)
    out: dict[str, dict] = {"basePower": {}, "type": {}, "accuracy": {}, "pp": {}}
    for name, body in entries:
        for field, rx in _NUM_RE.items():
            m = rx.search(body)
            if m:
                out[field][name] = int(m.group(1))
        mt = _TYPE_RE.search(body)
        if mt:
            out["type"][name] = mt.group(1)
        ma = _ACC_RE.search(body)
        if ma:
            raw = ma.group(1).strip()
            out["accuracy"][name] = True if raw == "true" else int(raw)
    return out


def _render_dict(name: str, d: dict, comment: str = "") -> str:
    head = f"{name} = {{" + (f"  # {comment}" if comment else "")
    lines = [head]
    for k in sorted(d):
        v = d[k]
        vs = repr(v) if isinstance(v, str) else ("True" if v is True else str(v))
        lines.append(f'    "{k}": {vs},')
    lines.append("}")
    return "\n".join(lines)


def render_module(overrides: dict[str, dict]) -> str:
    header = (
        '"""AUTO-GENERATED from pokemon-showdown/data/mods/champions/moves.ts — DO NOT EDIT BY HAND.\n'
        "\n"
        "Regenerate:  python -m v_dance.encoders.gen_champions_move_overrides\n"
        "\n"
        "v11 N1 — the custom Champions VGC format overrides these move fields; both encoders otherwise read\n"
        "standard Gen-9 data. Champions-ONLY by construction (the battle encoder is hard-scoped to Champions);\n"
        "revisit if a non-Champions battle format is ever encoded. Keys are move ids (norm_species(name) ==\n"
        'poke-env Move.id), so the overlay is byte-identical on the offline + live encoders (parity)."""\n'
    )
    bp = _render_dict("CHAMP_BP", overrides["basePower"], "basePower override (feeds band + Technician/-ate gates)")
    ty = _render_dict("CHAMP_TYPE", overrides["type"], "type override (STAB + type-eff + immunity + one-hot)")
    ac = _render_dict("CHAMP_ACC", overrides["accuracy"], "accuracy override (percent int or True=always-hit)")
    pp = _render_dict("CHAMP_PP", overrides["pp"],
                      "EMITTED, NOT consumed — pp is entangled with N5 (global PP cap) and low value; fold into N5")
    return header + "\n" + bp + "\n\n" + ty + "\n\n" + ac + "\n\n" + pp + "\n"


def main() -> None:
    text = render_module(parse_overrides())
    _OUT.write_text(text, encoding="utf-8", newline="\n")
    ov = parse_overrides()
    print(f"wrote {_OUT}  (BP {len(ov['basePower'])} / type {len(ov['type'])} / "
          f"acc {len(ov['accuracy'])} / pp {len(ov['pp'])})")


if __name__ == "__main__":
    main()
