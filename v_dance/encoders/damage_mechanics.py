"""v9 damage-correctness primitives — make the damage-band (B1.2) ability-type-aware and stat/BP-aware so
the model's damage estimate is correct for the non-standard-damage family the user called out:
  * ability-type change: Normalize / Pixilate / Refrigerate / Aerilate / Galvanize / Dragonize / Liquid Voice
  * variable base power: Low Kick/Grass Knot (target weight) · Heavy Slam/Heat Crash (weight ratio) ·
    Gyro Ball/Electro Ball (speed) · Eruption/Water Spout/Dragon Energy (HP) · Stored Power/Power Trip/
    Punishment (boosts) · Last Respects (faints) · Rage Fist (hits)
  * the stat-substitution moves (Body Press=def, Foul Play=target atk, Psyshock=hit def) are flagged by the
    move TAGS (dmg_uses_def / dmg_uses_target_atk / dmg_hits_def) in mechanic_tags — applied by the encoder.

Plus ``species_weight`` (kg, parsed from the dex) for the new per-mon weight feature AND the weight-based BP.
Standalone + tested here; wired into the encode path at the layout cutover. Types are UPPERCASE to match
the encoder's type convention.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Set

from v_dance.encoders.mechanic_vocab import _DEX, _norm

# -ate abilities: convert a NORMAL-type move to this type (+ a small power boost the band already folds via
# the damage_boost tag). Data set is cross-checked against mechanic_tags._DATA_MOVE_TYPECHANGE.
_ATE_TYPE = {"pixilate": "FAIRY", "refrigerate": "ICE", "aerilate": "FLYING",
             "galvanize": "ELECTRIC", "dragonize": "DRAGON"}


def _sound_moves() -> Set[str]:
    p = _DEX / "moves.ts"
    out: Set[str] = set()
    if not p.exists():
        return out
    text = p.read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in re.finditer(r"(?m)^\t(\w+): \{", text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]
        if re.search(r"flags:\s*\{[^}]*\bsound:\s*1", block):
            out.add(_norm(re.match(r"\t(\w+):", block).group(1)))
    return out


_SOUND_MOVES = _sound_moves()


def effective_move_type(move_id: Optional[str], base_type: Optional[str], ability_id: Optional[str]) -> str:
    """The move's EFFECTIVE type after a type-changing ability (UPPERCASE). Falls back to base_type."""
    bt = (base_type or "").upper()
    aid = _norm(ability_id)
    if aid == "normalize":
        return "NORMAL"
    if aid in _ATE_TYPE and bt == "NORMAL":
        return _ATE_TYPE[aid]
    if aid == "liquidvoice" and _norm(move_id) in _SOUND_MOVES:
        return "WATER"
    return bt


def _species_weights() -> Dict[str, float]:
    p = _DEX / "pokedex.ts"
    out: Dict[str, float] = {}
    if not p.exists():
        return out
    text = p.read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in re.finditer(r"(?m)^\t(\w+): \{", text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]
        sid = _norm(re.match(r"\t(\w+):", block).group(1))
        wm = re.search(r"weightkg:\s*([\d.]+)", block)
        if wm:
            out[sid] = float(wm.group(1))
    return out


SPECIES_WEIGHT: Dict[str, float] = _species_weights()


def _nfe_species() -> set:
    """Set of NOT-fully-evolved species ids (a non-empty pokedex `evos` list) — for Eviolite (v11 B3).
    Parsed once from pokedex.ts exactly like _species_weights so BOTH encoders share ONE source = parity by
    construction. (Showdown filters evos by obtainability; verified 0 gen-9 mismatches vs plain non-empty evos.)"""
    p = _DEX / "pokedex.ts"
    out: set = set()
    if not p.exists():
        return out
    text = p.read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in re.finditer(r"(?m)^\t(\w+): \{", text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]
        sid = _norm(re.match(r"\t(\w+):", block).group(1))
        m = re.search(r"evos:\s*\[([^\]]*)\]", block)
        if m and m.group(1).strip():
            out.add(sid)
    return out


NFE_SPECIES: set = _nfe_species()


def is_nfe(species: Optional[str]) -> bool:
    """True iff ``species`` is not-fully-evolved (Eviolite applies). Shared by both encoders (parity)."""
    return bool(species) and _norm(species) in NFE_SPECIES


def species_weight(species: Optional[str]) -> Optional[float]:
    """Weight in kg for a species (None if unknown). Used for the per-mon weight feature + weight-based BP."""
    if not species:
        return None
    sid = _norm(species)
    if sid in SPECIES_WEIGHT:
        return SPECIES_WEIGHT[sid]
    # mega/forme fallback: strip a trailing forme suffix to the base species
    for suf in ("megax", "megay", "mega", "gmax"):
        if sid.endswith(suf) and sid[:-len(suf)] in SPECIES_WEIGHT:
            return SPECIES_WEIGHT[sid[:-len(suf)]]
    return None


def _low_kick_bp(w: float) -> int:
    for thr, bp in ((200, 120), (100, 100), (50, 80), (25, 60), (10, 40)):
        if w >= thr:
            return bp
    return 20


def _heavy_slam_bp(att: float, tgt: float) -> int:
    r = att / max(0.1, tgt)
    if r >= 5:
        return 120
    if r >= 4:
        return 100
    if r >= 3:
        return 80
    if r >= 2:
        return 60
    return 40


def _electro_ball_bp(att: float, tgt: float) -> int:
    r = att / max(1.0, tgt)
    if r >= 4:
        return 150
    if r >= 3:
        return 120
    if r >= 2:
        return 80
    if r >= 1:
        return 60
    return 40


def variable_base_power(move_id: Optional[str], static_bp: float, *,
                        attacker_weight: Optional[float] = None, target_weight: Optional[float] = None,
                        attacker_hp_frac: Optional[float] = None,
                        attacker_speed: Optional[float] = None, target_speed: Optional[float] = None,
                        attacker_pos_boosts: int = 0, target_pos_boosts: int = 0,
                        fainted_allies: int = 0, times_hit: int = 0) -> float:
    """The REAL base power of a variable-BP move given context, else ``static_bp`` unchanged. Missing context
    for a variable move falls back to static_bp (graceful)."""
    mid = _norm(move_id)
    if mid in ("lowkick", "grassknot") and target_weight is not None:
        return _low_kick_bp(target_weight)
    if mid in ("heavyslam", "heatcrash") and attacker_weight and target_weight:
        return _heavy_slam_bp(attacker_weight, target_weight)
    if mid == "gyroball" and attacker_speed and target_speed:
        # Showdown: floor(25 * tgt_spe / usr_spe) + 1, clamped to [1,150] (moves.ts gyroball). The +1 was missing.
        return min(150.0, max(1.0, float(int(25 * target_speed / max(1.0, attacker_speed)) + 1)))
    if mid == "electroball" and attacker_speed and target_speed:
        return float(_electro_ball_bp(attacker_speed, target_speed))
    if mid in ("eruption", "waterspout", "dragonenergy") and attacker_hp_frac is not None:
        return max(1.0, float(int(150 * attacker_hp_frac)))
    if mid in ("storedpower", "powertrip"):
        return float(20 + 20 * max(0, attacker_pos_boosts))
    if mid == "punishment":
        return float(min(200, 60 + 20 * max(0, target_pos_boosts)))
    if mid == "lastrespects":
        return float(min(300, 50 + 50 * max(0, fainted_allies)))
    if mid == "ragefist":
        return float(min(350, 50 + 50 * max(0, times_hit)))
    return float(static_bp)
