"""Layout v19 → v20 AUTO-UPGRADE for battle-net checkpoints, optimiser states and stored states (2026-10-02).

v20 (mega audit gap 2) inserts the MEGA PREVIEW block (MEGA_PREVIEW_FEATURES = 110 floats) into every mon row,
AFTER the tera block and BEFORE the 4 trailing slot flags (encoder_layout.MEGA_PREVIEW_REL = 409 = where v19's
flags started). The AttnBCPolicy reads a mon row through ONE Linear, ``mon_enc.0`` (input = the mon row, then the
104 identity-embedding columns), so a v19 net is upgraded by inserting 110 ZERO input columns into
``mon_enc.0.weight`` at column 409: the old columns keep their weights, the new block's inputs are multiplied by 0,
and the v20 net computes EXACTLY the v19 policy / value on every state until training gives the new channels
weight (gradients reach the zero columns the moment the preview is non-zero). Nothing else in the net has a
mon-row-width tensor. So no served checkpoint breaks and the era chain does NOT restart.

Applied at every load site that reads a battle-net state dict: ``model_io.load_bc_policy`` (serve, self-play
base, league / clone opponents, panel, gauntlet, PPO base, every eval), ``ActorCritic.restore_from`` (revert /
mp workers), ``train_bc --warm-start``, ``selfplay.resume.load_into`` (incl. the Adam moments of the two
mon_enc.0.weight params, matched by NAME), and ``rl.store`` (recorded v19 ladder states are zero-padded the same
way — the log-probs recorded under the v19 net stay exact for the upgraded net: review 10-02 measured ≤ 8.3e-6
on 600 real steps). ``train_bc --resume`` of a v19 run is REFUSED by its config fingerprint (state_dim / layout
are fingerprinted) — correct, the corpus re-encodes at v20: continue such a run with --warm-start.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from v_dance.encoders.encoder_layout import (
    ACTIVE_SLOTS, BENCH_SLOTS, GLOBAL_FEATURES, MEGA_PREVIEW_FEATURES, MEGA_PREVIEW_REL, OPP_BENCH_SLOTS,
    POKEMON_FEATURES, STATE_DIM, STATE_LAYOUT_VERSION, V19_POKEMON_FEATURES, V19_STATE_DIM,
)

N_MON_SLOTS = ACTIVE_SLOTS + BENCH_SLOTS + OPP_BENCH_SLOTS
EMB_EXTRA = 24 + 16 + 4 * 16            # AttnBCPolicy ability / item / 4×move identity embedding widths
_V19_IN = V19_POKEMON_FEATURES + EMB_EXTRA      # 517
_V20_IN = POKEMON_FEATURES + EMB_EXTRA          # 627
_MON_ENC = "mon_enc.0.weight"
assert STATE_LAYOUT_VERSION == 20 and POKEMON_FEATURES == V19_POKEMON_FEATURES + MEGA_PREVIEW_FEATURES


def _is_v19_mon_enc(key: str, t) -> bool:
    return key.endswith(_MON_ENC) and getattr(t, "dim", lambda: 0)() == 2 and int(t.shape[1]) == _V19_IN


def _widen(t):
    import torch
    z = torch.zeros(t.shape[0], MEGA_PREVIEW_FEATURES, dtype=t.dtype, device=t.device)
    return torch.cat([t[:, :MEGA_PREVIEW_REL], z, t[:, MEGA_PREVIEW_REL:]], dim=1)


def upgrade_state_dict(sd: Optional[dict]) -> int:
    """IN PLACE: widen every v19 ``…mon_enc.0.weight`` (any key prefix: '', 'net.', 'policy.', 'critic.net.').
    Returns how many tensors were widened (0 = nothing to do — a v20 or unrelated dict)."""
    if not isinstance(sd, dict):
        return 0
    n = 0
    for k in list(sd.keys()):
        if _is_v19_mon_enc(k, sd[k]):
            sd[k] = _widen(sd[k])
            n += 1
    return n


def is_v19_config(cfg: Optional[dict]) -> bool:
    cfg = cfg or {}
    return cfg.get("state_layout_version") == 19 or cfg.get("state_dim") == V19_STATE_DIM


def upgrade_checkpoint(ck) -> bool:
    """IN PLACE on a loaded checkpoint dict (``model_state`` [+ ``critic_state``] + ``config``). A v19 dict is
    widened and re-stamped as v20 (``config['layout_upgraded_from'] = 19``); anything else is left untouched.
    Returns True when it upgraded."""
    if not isinstance(ck, dict) or "model_state" not in ck:
        return False
    cfg = ck.get("config")
    if not isinstance(cfg, dict) or not is_v19_config(cfg):
        return False
    n = upgrade_state_dict(ck["model_state"])
    upgrade_state_dict(ck.get("critic_state"))
    if n == 0:
        raise ValueError("layout_upgrade: a v19-stamped checkpoint without a v19-shaped mon_enc.0.weight "
                         f"(expected input width {_V19_IN}) — refusing to guess")
    cfg["state_dim"] = STATE_DIM
    cfg["state_layout_version"] = STATE_LAYOUT_VERSION
    cfg["layout_upgraded_from"] = 19
    return True


def upgrade_optimizer_state(opt_sd: Optional[dict], param_names=None) -> int:
    """IN PLACE: widen the Adam moments (exp_avg / exp_avg_sq …) of a v19 mon_enc.0.weight with zeros, so a v19
    resume snapshot continues under v20 with the new columns' moments at 0. ``param_names`` = the optimizer's
    params in order (their names): only a ``…mon_enc.0.weight`` slot is widened; without it, by SHAPE (out, 517)."""
    if not isinstance(opt_sd, dict):
        return 0
    n = 0
    for idx, st in (opt_sd.get("state") or {}).items():
        if not isinstance(st, dict):
            continue
        if param_names is not None:
            try:
                nm = param_names[int(idx)]
            except (IndexError, ValueError, TypeError):
                continue
            if not str(nm).endswith(_MON_ENC):
                continue
        for k, v in list(st.items()):
            if getattr(v, "dim", lambda: 0)() == 2 and int(v.shape[1]) == _V19_IN:
                st[k] = _widen(v)
                n += 1
    return n


def pad_states(x: np.ndarray) -> np.ndarray:
    """v19 state rows (…, 5057) → v20 (…, 6377): a zero mega-preview block inserted into each of the 12 mon rows
    (the globals tail is unchanged). v20 rows pass through untouched."""
    x = np.asarray(x)
    if x.shape[-1] == STATE_DIM:
        return x
    if x.shape[-1] != V19_STATE_DIM:
        raise ValueError(f"pad_states: width {x.shape[-1]} is neither v19 ({V19_STATE_DIM}) nor v20 ({STATE_DIM})")
    lead = x.shape[:-1]
    mons = x[..., :N_MON_SLOTS * V19_POKEMON_FEATURES].reshape(*lead, N_MON_SLOTS, V19_POKEMON_FEATURES)
    z = np.zeros((*lead, N_MON_SLOTS, MEGA_PREVIEW_FEATURES), dtype=x.dtype)
    mons20 = np.concatenate([mons[..., :MEGA_PREVIEW_REL], z, mons[..., MEGA_PREVIEW_REL:]], axis=-1)
    out = np.concatenate([mons20.reshape(*lead, N_MON_SLOTS * POKEMON_FEATURES),
                          x[..., N_MON_SLOTS * V19_POKEMON_FEATURES:]], axis=-1)
    assert out.shape[-1] == STATE_DIM and GLOBAL_FEATURES == STATE_DIM - N_MON_SLOTS * POKEMON_FEATURES
    return out


_NOTED = [False]


def note(ck_path, upgraded: bool) -> None:
    """ONE visible line per process the first time a checkpoint is lifted (WARNING: the self-play / gauntlet
    loggers run at WARNING), DEBUG after that — so a run's log shows its checkpoints were lifted, not retrained."""
    if upgraded:
        import logging
        lg = logging.getLogger(__name__)
        msg = ("[layout] %s: v19 checkpoint auto-upgraded to v20 (zero mega-preview columns; identical policy "
               "until trained)")
        if not _NOTED[0]:
            _NOTED[0] = True
            lg.warning(msg, ck_path)
        else:
            lg.debug(msg, ck_path)
