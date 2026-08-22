"""Shared fixtures for the Victory-Dance test suite.

Run from the repo root (the `v_dance` package is pip install -e .):
    pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tests/ lives at the repo root; data/ is the sibling data tree.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_DATA = _REPO_ROOT / "data"
# The example VOD lives under data/vods/, which is organised into per-type
# subfolders (Type_A … Type_D) that may be reshuffled — search for it.
_VOD_NAME = "Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html"
VOD_PATH = next(
    (p for p in [_PROJECT_DATA / "vods" / "Type_B" / _VOD_NAME,
                 _PROJECT_DATA / "vods" / _VOD_NAME]
     if p.exists()),
    None,
) or next((_PROJECT_DATA / "vods").rglob(_VOD_NAME),
          _PROJECT_DATA / "vods" / _VOD_NAME)
POKEDEX_PATH = _PROJECT_DATA / "pokedex.json"


@pytest.fixture(autouse=True)
def _hermetic_serve_env(monkeypatch):
    """Strip serve-deploy knobs from the test environment. Importing play_online_browser
    (any test's import chain) exports the user's .env deploy values process-wide — the
    suite must not inherit live serve config (a VD_TP_TIE_EPS=0.5 deploy made the TP
    set-head dispatch test sample non-argmax, 2026-07-21). Tests that WANT a knob set it
    via monkeypatch.setenv after this fixture clears it."""
    for k in ("VD_TP_TIE_EPS", "VD_SERVE_TAU", "VD_SERVE_TOP_P", "VD_PAIR_DECODE",
              "VD_FUTILITY_MASK"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="session")
def vod_html() -> str:
    if not VOD_PATH.exists():   # vods are local-only (not in the published repo) — skip on CI
        pytest.skip(f"example VOD not present (local-only fixture): {VOD_PATH.name}")
    return VOD_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def vod_path() -> Path:
    if not VOD_PATH.exists():   # vods are local-only (not in the published repo) — skip on CI
        pytest.skip(f"example VOD not present (local-only fixture): {VOD_PATH.name}")
    return VOD_PATH


@pytest.fixture(scope="session")
def dex():
    from v_dance.parser.vod_parser.pokedex import Pokedex
    return Pokedex(POKEDEX_PATH)


def make_log(*lines: str) -> str:
    """Build a minimal Showdown protocol log from raw lines."""
    return "\n".join(lines)


# ── Attn battle-net test helpers (#27 attn-only refactor) ─────────────────────
# The flat BCPolicy was retired; tests build a TINY AttnBCPolicy + an attn checkpoint
# (model_type=="attn") that model_io.load_bc_policy / ActorCritic.from_bc_checkpoint accept.
def tiny_attn_policy(*, heads=("our_a", "our_b"), gimmick_heads=("our_a", "our_b"),
                     value_readout="mean", state_dim=None, action_dim=None, gimmick_dim=None):
    from v_dance.models.bc_model_attn import AttnBCPolicy
    from v_dance.encoders.state_encoder import get_state_dim, get_action_dim, get_gimmick_dim
    return AttnBCPolicy(
        state_dim=state_dim if state_dim is not None else get_state_dim(),
        action_dim=action_dim if action_dim is not None else get_action_dim(),
        gimmick_dim=gimmick_dim if gimmick_dim is not None else get_gimmick_dim(),
        d_model=32, n_heads=4, n_layers=1, value_readout=value_readout,
        heads=tuple(heads),
        gimmick_heads=tuple(gimmick_heads) if gimmick_heads is not None else None,
    )


def write_attn_ckpt(path, *, heads=("our_a", "our_b"), gimmick_heads=("our_a", "our_b"),
                    value_trained=True, gimmick_trained=True, value_readout="mean",
                    state_dim=None, action_dim=None, gimmick_dim=None, seed=4):
    """Write a minimal attn BC checkpoint loadable by model_io / ActorCritic. Pass
    ``gimmick_heads=()`` for a pre-gimmick checkpoint (no gimmick head weights)."""
    import torch
    from v_dance.encoders.state_encoder import (
        get_state_dim, get_action_dim, get_gimmick_dim, get_state_layout_version)
    torch.manual_seed(seed)
    sd = state_dim if state_dim is not None else get_state_dim()
    ad = action_dim if action_dim is not None else get_action_dim()
    gd = gimmick_dim if gimmick_dim is not None else get_gimmick_dim()
    m = tiny_attn_policy(heads=heads, gimmick_heads=gimmick_heads, value_readout=value_readout,
                         state_dim=sd, action_dim=ad, gimmick_dim=gd)
    cfg = {"model_type": "attn", "state_dim": sd, "action_dim": ad, "gimmick_dim": gd,
           "state_layout_version": get_state_layout_version(),
           "d_model": 32, "n_heads": 4, "n_layers": 1, "ff_mult": 2, "dropout": 0.0,
           "value_readout": value_readout, "heads": list(heads),
           "gimmick_heads": (list(gimmick_heads) if gimmick_heads is not None else None),
           "value_trained": value_trained, "gimmick_trained": gimmick_trained}
    torch.save({"model_state": m.state_dict(), "config": cfg}, path)
    return path


# ── Synthetic log building blocks (importable from tests) ────────────────
HEADER = [
    "|player|p1|alice|101|1500",
    "|player|p2|bob|102|1500",
    "|teamsize|p1|4",
    "|teamsize|p2|4",
    "|tier|[Gen 9 Champions] VGC 2026 Reg M-A",
    "|poke|p1|Meganium, L50, M|",
    "|poke|p1|Incineroar, L50, F|",
    "|poke|p2|Aerodactyl, L50, M|",
    "|poke|p2|Palafin, L50, M|",
]
