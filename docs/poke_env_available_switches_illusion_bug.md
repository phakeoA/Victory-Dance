# poke-env bug: `available_switches` drops a healthy own bench Pokémon after an Illusion (Zoroark) turn

**Status:** draft upstream report (poke-env). Characterised from the installed
poke-env source (git master, version string `0.15.0`) + the symptom seen in live
VGC doubles play. A minimal live reproduction recipe is below.

## Summary

In a Gen 9 VGC **doubles** battle, after one of the player's own Pokémon has used
**Illusion** (Zoroark / Zoroark-Hisui), a **healthy, benched** own Pokémon can be
missing from `DoubleBattle.available_switches` at a **forced replacement**
(`forceSwitch`). The Pokémon is not fainted and is a legal replacement, but
`available_switches[slot]` comes back without it (often empty), so a client that
relies on `available_switches` to choose a replacement has no legal option and is
forced to fall back to `/choose default`.

## Why it happens (root cause)

`DoubleBattle._parse_request` computes `available_switches` directly from each team
member's `active` / `fainted` flags
(`poke_env/battle/double_battle.py`, ~L264–272):

```python
for i in range(2):
    if not self.trapped[i]:
        for pkmn_json in side["pokemon"]:
            pokemon = self.team[pkmn_json["ident"]]
            if self.reviving:
                if pokemon.fainted:
                    self._available_switches[i].append(pokemon)
            elif not pokemon.active and not pokemon.fainted:
                self._available_switches[i].append(pokemon)
```

So a team member is offered as a switch **iff** `not pokemon.active and not
pokemon.fainted`. A *healthy* mon (not fainted) that is wrongly excluded must
therefore have `pokemon.active == True`.

That stale `active` flag comes from Illusion identity handling:

1. `self._team` is keyed by `"<role>: <name>"` where `<name>` is the nickname, or
   the **species** when the mon is unnicknamed (`abstract_battle.py`, ~L271–272).
2. While a Zoroark is disguised, the **`|switch|` / `|drag|` messages carry the
   DISGUISE's name** (the apparent teammate's species). `get_pokemon` tries to
   disambiguate the disguise from a same-named real teammate via the `active`
   slot + `ability == "illusion"` checks (`abstract_battle.py`, ~L236–269), but
   when the disguise's apparent species equals a **brought, unnicknamed teammate's
   species**, the disguise's switch-in can mark the **real teammate's object**
   `active = True` (its `_active` flag, set by `Pokemon.switch_in`).
3. `Pokemon.active` is only re-synced to the truth via `update_from_request`
   (`pokemon.py` L707, `self._active = request_pokemon["active"]`) — but that fix
   is applied *per resolved object*. When the disguise and the real teammate share
   a name-key, the request reconciliation in `_update_team_from_request`
   (`abstract_battle.py`, ~L1265–1306) does not reliably clear the benched
   teammate's `active` flag, so it persists into the `available_switches`
   computation above.

The net effect is the **own-side analogue of the well-known opponent-side
duplicate-species Illusion merge**: poke-env's species/name-keyed identity model
cannot cleanly separate a Zoroark disguised as a same-species teammate from that
teammate, and the teammate inherits the disguise's `active` state.

> Note: a transformed **Ditto (Imposter)** is sometimes named alongside this
> symptom, but Imposter copies the *opponent*, so it does not create a same-name
> own-side collision; the clearly-reproducible trigger is **Illusion**.

## Confirmed vs. inferred

- **Confirmed** (read directly from the installed source): the
  `available_switches` formula (`not active and not fainted`); `active` is set from
  the request and from `switch_in`/`switch_out`; `self._team` is name-keyed; the
  Illusion disambiguation and request-reconciliation code paths above exist.
- **Confirmed empirically** (this project): driving a real dual-Zoroark replay
  through `DoubleBattle.parse_message` (no `|request|` lines) produces **no** flag
  corruption — i.e. the desync requires the **`|request|` ⇄ switch-event
  reconciliation under Illusion**, not the public switch stream alone.
- **Inferred**: the precise object that ends up wrongly `active` (the real
  same-name teammate). A maintainer repro with `|request|` logging will confirm.

## Minimal reproduction recipe (live)

Format: `gen9vgc2026regulationg` / Reg M-A doubles, **open team sheet**, bring 4.

1. Bring a team containing **Zoroark-Hisui (Ability: Illusion)** and at least one
   other brought, **unnicknamed** Pokémon whose species Zoroark will mimic
   (Illusion copies the last non-fainted party slot — order the team so the
   disguise species is a brought bencher).
2. Play until that disguised-as-teammate Zoroark is on the field while the real
   same-species teammate sits healthy on the bench.
3. Get a **forced replacement** for the *other* active slot (KO it).
4. Inspect `battle.available_switches[slot]`: the healthy benched same-species
   teammate is missing (the slot may have **no** offered switch at all), even
   though `/switch <that mon>` is accepted by the server.

A convenient repro team ("Trickery"): Zoroark-Hisui @ Focus Sash + Sneasler,
Charizard-Mega-Y, Whimsicott, Basculegion, Ditto.

## Suggested upstream fix directions

- Key `self._team` by a **stable identity** (team-slot index / the request's
  `ident`) rather than a mutable display name, so a disguised Zoroark and a
  same-species teammate never share a dict entry.
- After `_update_team_from_request`, **re-assert** every team member's `active`
  flag strictly from `request["side"]["pokemon"][*]["active"]` (the request is the
  authoritative truth for the own side), clearing any flag the public switch
  stream set during an Illusion.
- Or expose the request's authoritative switchable set directly so clients are not
  forced to re-derive it from per-mon `active`/`fainted` flags.

## Our workaround (downstream, this repo)

Two layers, both keyed on the fact that a post-faint replacement is **never
trapped**, so every living **brought** bench mon is a legal replacement:

1. **Request-authoritative own bench** (`live_state_encoder.own_bench_mons` /
   `brought_team_mons`). Instead of poke-env's corruptible per-mon `active`/
   `fainted` flags, we read the live **`battle.last_request`** — Showdown's own
   side data, which (a) lists **only the brought team** (Showdown rebuilds
   `side.pokemon` to the picked mons at team preview, `sim/battle.ts` `'team'`
   action) and (b) gives each mon's true `active` flag + `condition`. We map those
   idents back to `battle.team` objects (poke-env keeps all 6 there, same keys) for
   order construction. This is immune to the Illusion flag/identity corruption, and
   because the request is brought-only it also fixes the un-brought-mon exclusion
   for free. Falls back to the flag heuristic only when no request exists (our
   offline replay-driven parity path).
2. **`build_replacement_mask` desync recovery** (forced replacement): when
   poke-env's `available_switches` offers nothing for a must-switch slot, offer the
   request-authoritative `own_bench_mons` **by position**; `_replacement_order`
   builds the `/switch` from the same list, so the server accepts the real mon and
   the model keeps driving instead of collapsing to `/choose default`.
3. **`build_legal_action_mask` normal-turn switch legality**: the voluntary-switch
   mask gates the same request-authoritative bench on `battle.trapped` (filled
   per-slot from `active[].trapped`, uncorruptible) instead of `available_switches`
   membership — so a corrupted drop can no longer hide a switch *option* on a
   non-forced turn either. (`maybe_trapped` still offers switches, matching
   poke-env; Revival Blessing is the one mechanic the living-bench codec can't
   express, so a reviving request offers no switch — its prior behaviour.)

**Coverage:** because the request carries every brought mon's true state, this
covers all three failure shapes (healthy wrongly-`active`, healthy wrongly-
`fainted`, brought-but-never-revealed), on BOTH forced replacements and normal-turn
switches. Verified across `data/scripts/tests/test_replacement.py`
(`test_recovers_mon_wrongly_flagged_active_by_illusion`,
`test_own_bench_uses_request_over_corrupted_fainted_flag`,
`test_request_aware_bench_excludes_unbrought_mon`,
`test_normal_switch_mask_recovers_from_corrupted_available_switches`,
`test_normal_switch_mask_respects_real_trapping`).
The upstream fix proper (key the team by a stable identity, or re-assert `active`
from the request) remains the right durable solution.
