// ═══════════════════════════════════════════════════════════
// tb_parser.js
// Client-side Showdown replay parser — offline fallback.
// Mirrors vod_parser.py; kept in sync with the Python version.
// Depends on: tb_constants.js
// ═══════════════════════════════════════════════════════════

function extractLog(html) {
  const m = html.match(/<script[^>]+class="battle-log-data"[^>]*>([\s\S]*?)<\/script>/);
  if (!m) throw new Error('No battle-log-data block found. Is this a valid Showdown replay?');
  return m[1].replace(/\\\//g, '/');
}

function extractReplayId(html) {
  const m = html.match(/name="replayid"\s+value="([^"]+)"/);
  return m ? m[1] : null;
}

function parseShowdownLog(rawLog, ourPlayer = 'p1') {
  const lines = rawLog.split('\n').map(l => l.trim()).filter(Boolean);

  const players = {}, ratings = {}, ratingDeltas = {};
  const rosters = { p1: [], p2: [] };
  const knownTeam = { p1: [], p2: [] };
  const teamSizes = {};
  const sideConditions = {
    p1: { tailwind: 0, screens: {} },
    p2: { tailwind: 0, screens: {} },
  };
  const fieldConditions = { weather: null, terrain: null, trick_room: 0 };
  const activeSlots = {};
  const seenMons = {};

  let winner = null, format = null;
  let currentTurn = 0;
  const turns = [];
  let turnActions = [], turnDamageEvents = [];
  let stateBefore = null;
  let lastMoveAction = null;
  let executionIndex = 0;

  // ── Internal helpers ────────────────────────────────────
  function slotKeyFromIdent(ident) {
    if (!ident) return '';
    const m = ident.match(/^(p[12][ab])/);
    return m ? m[1] : ident.split(':')[0].trim();
  }

  function speciesFromIdent(ident) {
    if (!ident) return null;
    return ident.includes(': ') ? ident.split(': ')[1].trim() : null;
  }

  function parseHp(s) {
    s = (s || '100/100').split(' ')[0];
    if (s.includes('/')) {
      const [a, b] = s.split('/');
      return [parseFloat(a), parseFloat(b)];
    }
    return [parseFloat(s) || 0, 100];
  }

  function snapshotState() {
    return {
      field: { ...fieldConditions },
      side_conditions: {
        p1: {
          tailwind_turns_remaining: sideConditions.p1.tailwind,
          screens: { ...sideConditions.p1.screens },
        },
        p2: {
          tailwind_turns_remaining: sideConditions.p2.tailwind,
          screens: { ...sideConditions.p2.screens },
        },
      },
      active_pokemon: Object.fromEntries(
        Object.entries(activeSlots)
          .filter(([, m]) => !m.is_fainted)
          .map(([k, m]) => [k, {
            ...m,
            boosts: { ...m.boosts },
            revealed_moves: [...(m.revealed_moves || [])],
          }])
      ),
      known_team: { p1: [...knownTeam.p1], p2: [...knownTeam.p2] },
    };
  }

  function flushTurn() {
    if (currentTurn === 0) return;
    turns.push({
      turn: currentTurn,
      state_before_actions: stateBefore || snapshotState(),
      actions: [...turnActions],
      damage_events: [...turnDamageEvents],
      state_after_actions: snapshotState(),
      predicted_action_by_bot: null,
    });
  }

  function setMegaStoneItem(slotKey, megaSpecies, preMegaSpecies) {
    if (activeSlots[slotKey]) activeSlots[slotKey].known_item = 'mega stone';
    const pid = slotKey.slice(0, 2);
    for (const sk of [`${pid}:${megaSpecies}`, `${pid}:${preMegaSpecies}`]) {
      if (seenMons[sk]) seenMons[sk].known_item = 'mega stone';
    }
  }

  /**
   * Bug 8 (mega ability split): a mega forme has exactly ONE ability,
   * fully determined by the species.  The instant a mega is observed:
   *   1. demote any previously revealed ability to pre_mega_ability
   *      (it can no longer be active), and
   *   2. set known_ability/mega_ability from the pokedex — or null if the
   *      dex isn't loaded; a later |-ability| line will fill it in.
   * Idempotent: safe to call from both |detailschange| and |-mega|.
   */
  function applyMegaAbility(mon, megaSpecies) {
    if (mon.pre_mega_ability == null && mon.known_ability && mon.known_ability !== mon.mega_ability)
      mon.pre_mega_ability = mon.known_ability;
    const megaAb = dexMegaAbility(megaSpecies);
    if (megaAb) {
      mon.mega_ability  = megaAb;
      mon.known_ability = megaAb;
    } else if (mon.mega_ability) {
      mon.known_ability = mon.mega_ability;   // already revealed earlier
    } else {
      mon.known_ability = null;               // stale pre-mega ability cleared
    }
  }

  // ── Main parse loop ──────────────────────────────────────
  for (const line of lines) {
    if (!line.startsWith('|')) continue;
    const parts = line.split('|');
    const cmd = parts[1];

    if (cmd === 'player') {
      const pid = parts[2];
      if (parts[3]) players[pid] = parts[3];
      if (parts[5]) { try { ratings[pid] = parseInt(parts[5]); } catch (e) { } }

    } else if (cmd === 'tier') {
      format = parts[2] || null;

    } else if (cmd === 'poke') {
      const pid = parts[2];
      const species = (parts[3] || '').split(',')[0].trim();
      rosters[pid].push(species);

    } else if (cmd === 'teamsize') {
      teamSizes[parts[2]] = parseInt(parts[3]);

    } else if (cmd === 'win') {
      winner = parts[2] || null;

    } else if (cmd === 'raw') {
      const raw = parts[2] || '';
      const dm = raw.match(/(\d+)\s*(?:&rarr;|→)\s*<strong>(\d+)</);
      if (dm) {
        for (const [pid, uname] of Object.entries(players)) {
          if (uname && raw.toLowerCase().includes(uname.toLowerCase()))
            ratingDeltas[pid] = parseInt(dm[2]) - parseInt(dm[1]);
        }
      }

    } else if (cmd === 'turn') {
      flushTurn();
      currentTurn = parseInt(parts[2]);
      turnActions = []; turnDamageEvents = [];
      lastMoveAction = null; executionIndex = 0;
      stateBefore = snapshotState();

    } else if (cmd === 'switch' || cmd === 'drag') {
      const ident   = parts[2], details = parts[3] || '', hpStr = parts[4] || '100/100';
      const slotKey = slotKeyFromIdent(ident);
      const nickname = ident.includes(': ') ? ident.split(': ')[1] : ident;
      const species = details.split(',')[0].trim();
      const pid = slotKey.slice(0, 2);
      const [hpC, hpM] = parseHp(hpStr);
      const seenKey = `${pid}:${species}`;

      let mon = seenMons[seenKey] || null;
      if (!mon) {
        // Bug 8 continuity: a mega'd mon switching back in shows its MEGA
        // forme name; seenMons is keyed by base species — match on current
        // species so we don't fork a duplicate and lose ability state.
        mon = Object.entries(seenMons)
          .filter(([k]) => k.startsWith(pid + ':'))
          .map(([, m]) => m)
          .find(m => m.species === species) || null;
      }
      if (mon) {
        mon.slot = slotKey[2];
        mon.hp_current = hpC;
        mon.is_fainted = false;
        activeSlots[slotKey] = mon;
      } else {
        mon = {
          species, base_species: species, nickname, player: pid, slot: slotKey[2],
          hp_current: hpC, hp_max: hpM,
          status: null, boosts: {}, is_mega: false, is_fainted: false,
          revealed_moves: [], known_item: null, known_tera_type: null,
          is_terastallized: false,
          known_ability: null, pre_mega_ability: null, mega_ability: null,
        };
        seenMons[seenKey] = mon;
        activeSlots[slotKey] = mon;
      }
      const knownSpecies = mon.base_species || species;
      if (!knownTeam[pid].includes(knownSpecies)) knownTeam[pid].push(knownSpecies);
      turnActions.push({ event: 'switch', slot: slotKey, species, player: pid });

    } else if (cmd === 'detailschange') {
      // Fires for megas AND non-mega forme changes (Palafin-Hero, …) —
      // only genuine megas get is_mega + the ability swap (Bug 8).
      const slotKey    = slotKeyFromIdent(parts[2]);
      const newSpecies = parts[3] ? parts[3].split(',')[0].trim() : null;
      const mon        = activeSlots[slotKey];
      const isMega     = !!newSpecies && isMegaSpecies(newSpecies);

      if (mon && newSpecies) {
        const preMegaSpecies = mon.species;
        mon.species = newSpecies;
        if (isMega) {
          mon.is_mega = true;
          applyMegaAbility(mon, newSpecies);
          setMegaStoneItem(slotKey, newSpecies, preMegaSpecies);
        }
      }
      if (isMega) {
        turnActions.push({
          event: 'mega_evolution', slot: slotKey, new_species: newSpecies,
          mega_stone: mon?.known_item || null,
          pre_mega_ability: mon?.pre_mega_ability || null,
          mega_ability: mon?.mega_ability || null,
        });
      } else {
        turnActions.push({ event: 'forme_change', slot: slotKey, new_species: newSpecies });
      }

    } else if (cmd === '-mega') {
      // |-mega|p1a: Floette|Floette|Floettite — confirms mega happened
      const slotKey = slotKeyFromIdent(parts[2]);
      const preMega = parts[3] || null;
      const mon     = activeSlots[slotKey];
      if (mon) {
        const megaSpecies = mon.species; // already mutated by detailschange
        setMegaStoneItem(slotKey, megaSpecies, preMega || megaSpecies);
        // Safety net for replays missing |detailschange| (Bug 8, idempotent)
        if (!mon.is_mega) {
          mon.is_mega = true;
          applyMegaAbility(mon, megaSpecies);
        }
        for (let i = turnActions.length - 1; i >= 0; i--) {
          if (turnActions[i].event === 'mega_evolution' && turnActions[i].slot === slotKey) {
            turnActions[i].mega_stone = 'mega stone';
            break;
          }
        }
      }

    } else if (cmd === 'move') {
      const userIdent  = parts[2] || '', moveName = parts[3] || '', targetIdent = parts[4] || null;
      const userSlot   = slotKeyFromIdent(userIdent);
      const isProtect  = [
        'protect', 'detect', 'wide guard', 'quick guard', 'baneful bunker',
        'spiky shield', 'silk trap', 'burning bulwark', 'max guard',
      ].includes(moveName.toLowerCase());

      if (activeSlots[userSlot] && moveName && !activeSlots[userSlot].revealed_moves.includes(moveName))
        activeSlots[userSlot].revealed_moves.push(moveName);

      const action = {
        event: 'move', execution_index: executionIndex++,
        user_slot: userSlot, user_species: speciesFromIdent(userIdent),
        move: moveName, target_slot: targetIdent ? slotKeyFromIdent(targetIdent) : null,
        target_species: targetIdent ? speciesFromIdent(targetIdent) : null,
        is_protect: isProtect,
      };
      lastMoveAction = action;
      turnActions.push(action);

    } else if (cmd === '-damage' || cmd === '-heal') {
      const slotKey = slotKeyFromIdent(parts[2]);
      const [hpC]   = parseHp(parts[3]);
      const prevHp  = activeSlots[slotKey]?.hp_current ?? null;
      if (activeSlots[slotKey]) activeSlots[slotKey].hp_current = hpC;
      const delta = (prevHp !== null) ? Math.round((hpC - prevHp) * 10) / 10 : null;
      const ev = {
        event: cmd.replace('-', ''), slot: slotKey,
        species: speciesFromIdent(parts[2]), hp_pct_after: hpC, hp_pct_delta: delta,
        source_slot:    (cmd === '-damage' && lastMoveAction) ? lastMoveAction.user_slot    : null,
        source_species: (cmd === '-damage' && lastMoveAction) ? lastMoveAction.user_species : null,
        source_move:    (cmd === '-damage' && lastMoveAction) ? lastMoveAction.move          : null,
      };
      turnActions.push(ev); turnDamageEvents.push(ev);

    } else if (cmd === 'faint') {
      const slotKey = slotKeyFromIdent(parts[2]);
      if (activeSlots[slotKey]) {
        activeSlots[slotKey].is_fainted = true;
        activeSlots[slotKey].hp_current = 0;
      }
      turnActions.push({ event: 'faint', slot: slotKey, species: speciesFromIdent(parts[2]) });

    } else if (cmd === '-boost' || cmd === '-unboost') {
      const slotKey = slotKeyFromIdent(parts[2]);
      const stat    = parts[3] || '';
      const amt     = (cmd === '-unboost' ? -1 : 1) * (parseInt(parts[4]) || 1);
      if (activeSlots[slotKey])
        activeSlots[slotKey].boosts[stat] = (activeSlots[slotKey].boosts[stat] || 0) + amt;
      turnActions.push({ event: 'stat_change', slot: slotKey, stat, stages: amt });

    } else if (cmd === '-status') {
      const slotKey = slotKeyFromIdent(parts[2]);
      if (activeSlots[slotKey]) activeSlots[slotKey].status = parts[3] || null;

    } else if (cmd === '-curestatus') {
      const slotKey = slotKeyFromIdent(parts[2]);
      if (activeSlots[slotKey]) activeSlots[slotKey].status = null;

    } else if (cmd === '-item') {
      const slotKey = slotKeyFromIdent(parts[2]);
      const item    = parts[3] || null;
      if (activeSlots[slotKey] && item) activeSlots[slotKey].known_item = item;
      turnActions.push({ event: 'item_revealed', slot: slotKey, species: speciesFromIdent(parts[2]), item });

    } else if (cmd === '-enditem') {
      const slotKey = slotKeyFromIdent(parts[2]);
      const item    = parts[3] || null;
      if (activeSlots[slotKey] && item) activeSlots[slotKey].known_item = item;
      turnActions.push({ event: 'item_consumed', slot: slotKey, species: speciesFromIdent(parts[2]), item });

    } else if (cmd === '-terastallize') {
      const slotKey  = slotKeyFromIdent(parts[2]);
      const teraType = parts[3] || null;
      if (activeSlots[slotKey]) {
        activeSlots[slotKey].known_tera_type = teraType;
        activeSlots[slotKey].is_terastallized = true;
      }
      turnActions.push({ event: 'terastallize', slot: slotKey, species: speciesFromIdent(parts[2]), tera_type: teraType });

    } else if (cmd === '-ability') {
      // |-ability|p1a: Aerodactyl|Unnerve
      const slotKey = slotKeyFromIdent(parts[2]);
      const ability = parts[3] || null;
      const mon     = activeSlots[slotKey];
      if (mon && ability) {
        // Bug 8: route the reveal into the correct ability context — a
        // mega'd mon's ability line IS its (fixed) mega ability; a base
        // forme's line is its chosen base ability.
        if (mon.is_mega) mon.mega_ability     = ability;
        else             mon.pre_mega_ability = ability;
        mon.known_ability = ability;
        // active_slots and seenMons share the same object, but keep the
        // explicit base-species-keyed write for refactor safety (Bug 7).
        const pid     = slotKey.slice(0, 2);
        const seenKey = `${pid}:${mon.base_species || mon.species}`;
        if (seenMons[seenKey]) seenMons[seenKey].known_ability = ability;
      }
      turnActions.push({
        event: 'ability_revealed', slot: slotKey,
        species: speciesFromIdent(parts[2]), ability,
        is_mega_ability: !!mon?.is_mega,
      });

    } else if (cmd === '-sidestart') {
      const pid = (parts[2] || '').split(':')[0].trim(), eff = parts[3] || '';
      if (eff.includes('Tailwind'))     sideConditions[pid].tailwind = 4;
      else if (eff.includes('Reflect')) sideConditions[pid].screens.reflect = 5;
      else if (eff.includes('Light Screen')) sideConditions[pid].screens.light_screen = 5;
      else if (eff.includes('Aurora Veil')) sideConditions[pid].screens.aurora_veil = 5;

    } else if (cmd === '-sideend') {
      const pid = (parts[2] || '').split(':')[0].trim(), eff = parts[3] || '';
      if (eff.includes('Tailwind'))          sideConditions[pid].tailwind = 0;
      else if (eff.includes('Reflect'))      delete sideConditions[pid].screens.reflect;
      else if (eff.includes('Light Screen')) delete sideConditions[pid].screens.light_screen;
      else if (eff.includes('Aurora Veil'))  delete sideConditions[pid].screens.aurora_veil;

    } else if (cmd === '-weather') {
      fieldConditions.weather = (parts[2] && parts[2] !== 'none') ? parts[2] : null;

    } else if (cmd === '-fieldstart') {
      const raw = parts[2] || '';
      if (raw.includes('Trick Room'))                  fieldConditions.trick_room = 5;
      else if (raw.toLowerCase().includes('terrain'))  fieldConditions.terrain = raw;

    } else if (cmd === '-fieldend') {
      const raw = parts[2] || '';
      if (raw.includes('Trick Room'))                  fieldConditions.trick_room = 0;
      else if (raw.toLowerCase().includes('terrain'))  fieldConditions.terrain = null;

    } else if (cmd === 'upkeep') {
      for (const sc of Object.values(sideConditions)) {
        if (sc.tailwind > 0) sc.tailwind--;
        for (const k of Object.keys(sc.screens)) {
          if (sc.screens[k] <= 1) delete sc.screens[k];
          else sc.screens[k]--;
        }
      }
      if (fieldConditions.trick_room > 0) fieldConditions.trick_room--;
    }
  }

  flushTurn();

  // Build the same revealed_info structure that vod_parser.py emits
  const revealed_info = {};
  for (const [seenKey, mon] of Object.entries(seenMons)) {
    const base = mon.base_species || mon.species;
    revealed_info[seenKey] = {
      revealed_moves:    [...(mon.revealed_moves || [])],
      known_item:        mon.is_mega ? 'mega stone' : (mon.known_item || null),
      known_tera_type:   mon.known_tera_type || null,
      is_terastallized:  mon.is_terastallized || false,
      // Bug 8: split ability contexts (see tb_parser applyMegaAbility)
      known_ability:     mon.known_ability || null,
      pre_mega_ability:  mon.pre_mega_ability || null,
      mega_ability:      mon.mega_ability || null,
      is_mega:           mon.is_mega || false,
      mega_species:      mon.is_mega ? mon.species : null,
      possible_abilities: dexAbilities(base),
      mega_formes:        dexMegaFormes(base),
    };
  }

  return {
    source_type: 'ranked_player_vod',
    replay_id: null,
    format,
    players: {
      our_side: ourPlayer,
      p1: { username: players.p1 || null, rating_before: ratings.p1 || null, rating_delta: ratingDeltas.p1 || null, roster: rosters.p1, team_size_chosen: teamSizes.p1 || null },
      p2: { username: players.p2 || null, rating_before: ratings.p2 || null, rating_delta: ratingDeltas.p2 || null, roster: rosters.p2, team_size_chosen: teamSizes.p2 || null },
    },
    winner,
    stats_quality: { our_side: 'distribution', opp_side: 'distribution' },
    known_team_overrides: {},
    revealed_info,
    turns,
  };
}
