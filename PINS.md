# Pinned versions (read before long self-play runs)

The two version-critical dependencies — **poke-env** and the **Pokémon Showdown**
server — are pinned to exact, tested versions so a long run can't be silently broken
by an upstream change mid-run (or have its results invalidated). Both own the battle
**protocol**, so a bump can change behaviour without any error.

> `pokemon-showdown/` is **gitignored**, so its commit SHA cannot live in the clone —
> **this file is the source of truth** for it.

> ⚠ **EXPECTED UPDATE (~2026-06-20, per user 2026-06-17):** a new Pokémon Showdown release with a
> **new VGC regulation** is due in a few days. When it lands: re-clone / `git pull` pokemon-showdown,
> bump `SHOWDOWN_SHA` in `setup.sh` (+ likely `poke-env`), regenerate `requirements.lock`, and
> **re-verify** (suite + live smoke). The new regulation may also change the battle **format string**
> (currently `gen9championsvgc2026regma`) and the legal **team pool** — and the BC model is trained on
> Reg M-A data, so its metagame relevance to a new reg is an open question (may want fresh Type-B data /
> retrain). Decide scope when it actually drops.

## The pins (as of 2026-06-17)

| What | Pinned to | Where it's enforced |
|---|---|---|
| **poke-env** | `hsahovic/poke-env@a6e4f67d204a390a3c8368c182197aff35816aca` (v0.15.0, from **git**, not PyPI) | `requirements.txt` (`poke-env @ git+…@<sha>`) |
| **Pokémon Showdown** | `smogon/pokemon-showdown@ecf39eef1e9cd2fd6ed2e9b9011b86610258d757` (`v0.11.10-1271`, master) | `setup.sh` → `SHOWDOWN_SHA` (checked out before `npm install`) |

### Supporting env (recorded; full freeze in `requirements.lock`)
- Python **3.12.10**, Node **v22.3.0** (venv-local, installed by `setup.sh`)
- torch **2.12.0+cu132**, torchvision **0.27.0+cu132** (installed separately by `setup.sh` via the PyTorch CUDA index — NOT from PyPI, so `requirements.lock` is a *record*, not a one-shot installer)
- numpy **2.4.6**, crawl4ai **0.8.9**, flask **3.1.3**, websockets **16.0**, orjson **3.11.9**

## Verify the live env matches the pins
```bash
.venv/Scripts/python.exe -m pip show poke-env | grep -i version          # expect 0.15.0
.venv/Scripts/python.exe -m pip freeze | grep -i poke                    # expect the git@<sha> line
git -C pokemon-showdown rev-parse HEAD                                    # expect ecf39eef1…
```

## Re-pinning (bumping a version)
1. Update the SHA/version in **`requirements.txt`** (poke-env) and/or **`setup.sh`** (`SHOWDOWN_SHA`), and this table.
2. Reinstall / re-checkout, regenerate the lock: `.venv/Scripts/python.exe -m pip freeze | grep -ivE "^-e | @ file://" > requirements.lock`.
3. **Re-verify before trusting it:** full suite (`pytest`) green AND a live self-play smoke
   (`python -m v_dance.selfplay.generation --live --generations 1 --games 30 -v`) — the protocol
   surface (illusion/mega/tera/force-switch) is where upstream drift bites.
