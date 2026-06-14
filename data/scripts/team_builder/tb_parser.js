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
  // Choice-item constraint working state: mon → Set of distinct moves the
  // player selected during the mon's CURRENT stay on the field.  Two or more
  // distinct moves in one stint rule out the whole Choice item family
  // (Scarf/Band/Specs lock the holder into its first move until it switches
  // out) — mirrors replay_parser.py.
  const stintMoves = new Map();

  let winner = null, format = null;
  let currentTurn = 0;
  const turns = [];
  let turnActions = [], turnDamageEvents = [];
  let stateBefore = null;
  let lastMoveAction = null;
  let executionIndex = 0;
  // Edge-case flags surfaced for parity with replay_parser.py.
  let containsIllusion = false, containsTransform = false;

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

  /** Write a revealed ability into the holder's slot + seenMons state. */
  function recordAbility(slotKey, ability, species) {
    const mon = activeSlots[slotKey];
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
      species, ability,
      is_mega_ability: !!mon?.is_mega,
    });
  }

  /**
   * Bug 9: many abilities never get a standalone |-ability| line — the
   * reveal rides as a "[from] ability:" tag on some other line, e.g.
   *   |-weather|RainDance|[from] ability: Drizzle|[of] p2a: Pelipper
   *   |-heal|p2a: X|100/100|[from] ability: Water Absorb
   * The holder is the [of] mon when present, else the line's subject.
   * (Showdown points [of] at a different mon for Pickpocket/Magician —
   * accepted as a known limitation; mirrors replay_parser.py.)
   */
  function learnAbilityFromTags(parts) {
    let ability = null, ofIdent = null;
    for (const p of parts.slice(2)) {
      if (p.startsWith('[from] ability:')) ability = p.slice('[from] ability:'.length).trim();
      else if (p.startsWith('[of]'))       ofIdent = p.slice('[of]'.length).trim();
    }
    if (!ability) return;
    const holder  = ofIdent || parts[2] || '';
    const slotKey = slotKeyFromIdent(holder);
    if (!/^p[12][ab]$/.test(slotKey)) return;
    recordAbility(slotKey, ability, speciesFromIdent(holder));
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

  // ── Illusion (Zoroark) pre-scan ──────────────────────────
  // Map each disguised |switch| line index → the TRUE details string revealed
  // by a later |replace| at that slot, so the main pass builds the slot as the
  // real species (Zoroark-Hisui) from its first frame instead of crediting the
  // disguise. Maps to the MOST RECENT switch at the slot so a genuine copy of
  // the disguise species stays separate. Mirrors replay_parser._prescan_illusions.
  const illusionTruth = {};
  {
    const lastSwitch = {};
    lines.forEach((ln, idx) => {
      if (!ln.startsWith('|')) return;
      const p = ln.split('|'), c = p[1];
      if (c === 'switch' || c === 'drag') {
        lastSwitch[slotKeyFromIdent(p[2] || '')] = idx;
      } else if (c === 'replace') {
        const slot = slotKeyFromIdent(p[2] || ''), det = p[3] || '';
        if (lastSwitch[slot] !== undefined && det) illusionTruth[lastSwitch[slot]] = det;
      }
    });
  }

  // ── Main parse loop ──────────────────────────────────────
  for (let _i = 0; _i < lines.length; _i++) {
    const line = lines[_i];
    if (!line.startsWith('|')) continue;
    const parts = line.split('|');
    const cmd = parts[1];

    // Bug 9: scan every line for inline "[from] ability:" reveals.
    // |-ability| is excluded: it has its own handler, and a [from] tag
    // there (e.g. Trace) names a DIFFERENT ability than the active one.
    if (cmd !== '-ability') learnAbilityFromTags(parts);

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
      const ident   = parts[2];
      let details   = parts[3] || '';
      // Illusion: if the pre-scan flagged this switch as a disguised Zoroark,
      // swap in the TRUE details so the slot is tracked as the real species.
      if (illusionTruth[_i]) { details = illusionTruth[_i]; containsIllusion = true; }
      const hpStr   = parts[4] || '100/100';
      const slotKey = slotKeyFromIdent(ident);
      const outgoing = activeSlots[slotKey];   // transform reverts on switch-out
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
          can_have_choice_item: true,
          is_transformed: false, transformed_into: null, ever_transformed: false,
        };
        seenMons[seenKey] = mon;
        activeSlots[slotKey] = mon;
      }
      // A switch-in starts a fresh stay on the field — a Choice lock from a
      // previous stint no longer applies.
      stintMoves.set(mon, new Set());
      // Transform reverts the instant a mon leaves the field; the incoming mon
      // is its real self again (an Imposter Ditto re-fires |-transform| right
      // after, re-setting these).
      if (outgoing && outgoing !== mon) { outgoing.is_transformed = false; outgoing.transformed_into = null; }
      mon.is_transformed = false; mon.transformed_into = null;
      const knownSpecies = mon.base_species || species;
      if (!knownTeam[pid].includes(knownSpecies)) knownTeam[pid].push(knownSpecies);
      turnActions.push({ event: 'switch', slot: slotKey, species, player: pid });

    } else if (cmd === 'replace') {
      // |replace|p2a: Zoroark|Zoroark-Hisui, L50, M — Illusion unmasked.
      // The pre-scan normally relabeled the originating switch already; the
      // defensive relabel below covers a malformed log the pre-scan missed.
      const ident = parts[2] || '', details = parts[3] || '';
      const slotKey = slotKeyFromIdent(ident);
      const trueSpecies = details ? details.split(',')[0].trim() : null;
      containsIllusion = true;
      const mon = activeSlots[slotKey];
      if (mon && trueSpecies) {
        const currentBase = mon.base_species || mon.species;
        if (currentBase !== trueSpecies) {
          const pid = slotKey.slice(0, 2);
          delete seenMons[`${pid}:${currentBase}`];
          mon.species = trueSpecies;
          mon.base_species = trueSpecies;
          mon.nickname = ident.includes(': ') ? ident.split(': ')[1] : mon.nickname;
          seenMons[`${pid}:${trueSpecies}`] = mon;
          if (!knownTeam[pid].includes(trueSpecies)) knownTeam[pid].push(trueSpecies);
        }
      }
      turnActions.push({ event: 'illusion_revealed', slot: slotKey, species: trueSpecies });

    } else if (cmd === '-transform') {
      // |-transform|p1a: Ditto|p2b: Whimsicott|[from] ability: Imposter
      const ident = parts[2] || '', targetIdent = parts[3] || '';
      const slotKey = slotKeyFromIdent(ident);
      let intoSpecies = speciesFromIdent(targetIdent);
      if (!intoSpecies && targetIdent) intoSpecies = targetIdent.split(',')[0].trim() || null;
      containsTransform = true;
      const mon = activeSlots[slotKey];
      if (mon) { mon.is_transformed = true; mon.transformed_into = intoSpecies; mon.ever_transformed = true; }
      turnActions.push({ event: 'transform', slot: slotKey, species: speciesFromIdent(ident), into_species: intoSpecies });

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

      const mover = activeSlots[userSlot];
      // Transform (Ditto/Imposter): moves used while transformed are the COPIED
      // foe's moves — they reveal nothing about this mon's real set or item, so
      // they must not enter revealed_moves or the Choice-item constraint.
      if (mover && !mover.is_transformed && moveName && !mover.revealed_moves.includes(moveName))
        mover.revealed_moves.push(moveName);

      // Choice-item constraint (mirrors replay_parser.py): a [from]-tagged
      // move was CALLED by another effect (Sleep Talk, Dancer, Instruct,
      // locked-move continuations) — the player never selected it.  Struggle
      // is excluded too: a choice-locked mon Struggles once its locked move
      // runs out of PP.
      const wasCalled = parts.slice(4).some(p => (p || '').startsWith('[from]'));
      if (mover && !mover.is_transformed && moveName && !wasCalled && moveName.toLowerCase() !== 'struggle') {
        const stint = stintMoves.get(mover) || new Set();
        stint.add(moveName);
        stintMoves.set(mover, stint);
        if (stint.size >= 2) mover.can_have_choice_item = false;
      }

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
      if (activeSlots[slotKey]) {
        if (item) activeSlots[slotKey].known_item = item;
        // Item changed hands / was revealed (Trick, Frisk, …) — later moves
        // prove nothing about the item the mon BROUGHT; restart the stint.
        stintMoves.set(activeSlots[slotKey], new Set());
      }
      turnActions.push({ event: 'item_revealed', slot: slotKey, species: speciesFromIdent(parts[2]), item });

    } else if (cmd === '-enditem') {
      const slotKey = slotKeyFromIdent(parts[2]);
      const item    = parts[3] || null;
      if (activeSlots[slotKey]) {
        if (item) activeSlots[slotKey].known_item = item;
        // Item lost/consumed — same stint reset as '-item' above.
        stintMoves.set(activeSlots[slotKey], new Set());
      }
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
      recordAbility(slotKeyFromIdent(parts[2]), parts[3] || null, speciesFromIdent(parts[2]));

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
      // False = used 2+ different moves in one stay on the field → cannot
      // hold Choice Scarf/Band/Specs; the belief fill drops those items.
      can_have_choice_item: mon.can_have_choice_item !== false,
      // Transform/Imposter: latched True if this mon ever transformed; its
      // revealed_moves stay empty (copied moves filtered) so belief falls back
      // to the mon's own usage distribution.
      is_transformed:     mon.ever_transformed || false,
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
      // roster = full 6-mon teampreview pool; brought = the (≤4) mons that
      // actually entered the battle, in first switch-in order (leads first).
      p1: { username: players.p1 || null, rating_before: ratings.p1 || null, rating_delta: ratingDeltas.p1 || null, roster: rosters.p1, brought: [...knownTeam.p1], team_size_chosen: teamSizes.p1 || null },
      p2: { username: players.p2 || null, rating_before: ratings.p2 || null, rating_delta: ratingDeltas.p2 || null, roster: rosters.p2, brought: [...knownTeam.p2], team_size_chosen: teamSizes.p2 || null },
    },
    winner,
    stats_quality: { our_side: 'distribution', opp_side: 'distribution' },
    contains_illusion: containsIllusion,
    contains_transform: containsTransform,
    known_team_overrides: {},
    revealed_info,
    turns,
  };
}

// ═══════════════════════════════════════════════════════════
// Showdown team-paste parser (Type A "Import team")
// Mirrors vod_parser/team_sheet.py (parse_showdown_team) so the UI's import
// button and the headless bulk exporter produce identical inject data.
// ═══════════════════════════════════════════════════════════

const _TEAM_STAT_LABELS = { hp: 'hp', atk: 'atk', def: 'def', spa: 'spa', spd: 'spd', spe: 'spe' };
const _NATURE_SET = new Set(NATURES.map(n => n.toLowerCase()));

/** Parse a full team paste into an array of per-Pokémon objects. */
function parseShowdownTeam(text) {
  const mons = [];
  // Normalise CRLF/CR → LF first: a browser File.text() keeps Windows \r\n,
  // and the blank-line block separator below only matches \n…\n — without this
  // a CRLF paste collapses into a SINGLE mon.
  const normalized = (text || '').replace(/\r\n?/g, '\n');
  for (const block of normalized.trim().split(/\n[ \t]*\n/)) {
    const lines = block.split('\n').map(l => l.replace(/\s+$/, '')).filter(l => l.trim());
    if (!lines.length) continue;
    const mon = _parseTeamMon(lines);
    if (mon && mon.species) mons.push(mon);
  }
  return mons;
}

/** Split the first line → {species, nickname, item, gender}. */
function _parseTeamFirstLine(line) {
  let item = null, namepart = line.trim();
  const at = namepart.lastIndexOf(' @ ');
  if (at !== -1) { item = namepart.slice(at + 3).trim() || null; namepart = namepart.slice(0, at).trim(); }
  let gender = null;
  const gm = namepart.match(/\(([MFN])\)\s*$/);
  if (gm) { gender = gm[1]; namepart = namepart.slice(0, gm.index).trim(); }
  let species = namepart, nickname = null;
  const nm = namepart.match(/^(.*?)\s*\(([^)]+)\)\s*$/);
  if (nm) { nickname = nm[1].trim() || null; species = nm[2].trim(); }
  return { species, nickname, item, gender };
}

/** Parse "EVs: 2 HP / 32 Atk / 32 Spe" → {hp:2, atk:32, spe:32}. */
function _parseTeamStatLine(line) {
  const out = {};
  const body = line.includes(':') ? line.slice(line.indexOf(':') + 1) : line;
  for (const part of body.split('/')) {
    const m = part.match(/\s*(\d+)\s+([A-Za-z]+)/);
    if (!m) continue;
    const key = _TEAM_STAT_LABELS[m[2].toLowerCase()];
    if (key) out[key] = parseInt(m[1], 10);
  }
  return out;
}

function _parseTeamMon(lines) {
  const first = _parseTeamFirstLine(lines[0]);
  const mon = {
    species: first.species, nickname: first.nickname, gender: first.gender,
    item: first.item, ability: null, level: 50, nature: null,
    evs: {}, ivs: {}, moves: [], teraType: null,
  };
  for (const raw of lines.slice(1)) {
    const line = raw.trim(), low = line.toLowerCase();
    if (low.startsWith('ability:'))        mon.ability = line.slice(line.indexOf(':') + 1).trim() || null;
    else if (low.startsWith('level:'))     { const n = parseInt(line.slice(line.indexOf(':') + 1).trim(), 10); if (!isNaN(n)) mon.level = n; }
    else if (low.startsWith('tera type:')) mon.teraType = line.slice(line.indexOf(':') + 1).trim() || null;
    else if (low.startsWith('evs:'))       mon.evs = _parseTeamStatLine(line);
    else if (low.startsWith('ivs:'))       mon.ivs = _parseTeamStatLine(line);
    else if (line.startsWith('-'))         { const mv = line.slice(1).trim(); if (mv && mon.moves.length < 4) mon.moves.push(mv); }
    else if (low.endsWith('nature') && _NATURE_SET.has(line.split(/\s+/)[0].toLowerCase()))
      mon.nature = line.replace(/\s+Nature$/i, '').trim();
    // Shiny:/Happiness:/Gigantamax: etc. are ignored.
  }
  return mon;
}

/** Resolve a paste species to its replay-roster name (mega forme → base). */
function teamBaseSpecies(species) {
  if (isMegaSpecies(species)) {
    const e = dexEntry(species);
    if (e && e.baseSpecies) return e.baseSpecies;
    return species.replace(/-Mega(-[XY])?$/, '');
  }
  return species;
}
