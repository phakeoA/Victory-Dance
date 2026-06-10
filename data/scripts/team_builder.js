// ═══════════════════════════════════════════════════════════
// 0.  CONSTANTS & HELPERS
// ═══════════════════════════════════════════════════════════
const SERVER = 'http://localhost:5174';

const NATURES = ['Hardy','Lonely','Brave','Adamant','Naughty','Bold','Docile','Relaxed',
  'Impish','Lax','Timid','Hasty','Serious','Jolly','Naive','Modest','Mild','Quiet',
  'Bashful','Rash','Calm','Gentle','Sassy','Careful','Quirky'];
const STAT_NAMES = ['hp','atk','def','spa','spd','spe'];
const STAT_LABELS = { hp:'HP', atk:'Atk', def:'Def', spa:'SpA', spd:'SpD', spe:'Spe' };

// Mega stone lookup — mirrors vod_parser.py MEGA_STONE_MAP.
// Keys are the PRE-mega Showdown species name; values are the item string.
const MEGA_STONE_MAP = {
  Abomasnow:    'Abomasite',
  Absol:        'Absolite',
  Aerodactyl:   'Aerodactylite',
  Aggron:       'Aggronite',
  Alakazam:     'Alakazite',
  Altaria:      'Altarianite',
  Ampharos:     'Ampharosite',
  Audino:       'Audinite',
  Banette:      'Banettite',
  Beedrill:     'Beedrillite',
  Blastoise:    'Blastoisinite',
  Camerupt:     'Cameruptite',
  Chandelure:   'Chandelurite',
  Charizard:    'Charizardite Y',   // default; X/Y resolved by newSpecies below
  Chesnaught:   'Chesnaughtite',
  Chimecho:     'Chimechite',
  Clefable:     'Clefablite',
  Crabominable: 'Crabominite',
  Delphox:      'Delphoxite',
  Dragonite:    'Dragoninite',
  Drampa:       'Drampanite',
  Emboar:       'Emboarite',
  Excadrill:    'Excadrite',
  Feraligatr:   'Feraligite',
  Floette:      'Floettite',
  Froslass:     'Froslassite',
  Gallade:      'Galladite',
  Garchomp:     'Garchompite',
  Gardevoir:    'Gardevoirite',
  Gengar:       'Gengarite',
  Glalie:       'Glalitite',
  Glimmora:     'Glimmoranite',
  Golurk:       'Golurkite',
  Greninja:     'Greninjite',
  Gyarados:     'Gyaradosite',
  Hawlucha:     'Hawluchanite',
  Heracross:    'Heracronite',
  Houndoom:     'Houndoominite',
  Kangaskhan:   'Kangaskhanite',
  Lopunny:      'Lopunnite',
  Lucario:      'Lucarionite',
  Manectric:    'Manectite',
  Medicham:     'Medichamite',
  Meganium:     'Meganiumite',
  Meowstic:     'Meowsticite',
  Pidgeot:      'Pidgeotite',
  Pinsir:       'Pinsirite',
  Sableye:      'Sablenite',
  Scizor:       'Scizorite',
  Scovillain:   'Scovillainite',
  Sharpedo:     'Sharpedonite',
  Skarmory:     'Skarmorite',
  Slowbro:      'Slowbronite',
  Starmie:      'Starminite',
  Steelix:      'Steelixite',
  Tyranitar:    'Tyranitarite',
  Venusaur:     'Venusaurite',
  Victreebel:   'Victreebelite',
};

function resolveMegaStone(preMegaSpecies, newSpecies) {
  if (preMegaSpecies === 'Charizard') {
    return newSpecies && newSpecies.endsWith('-X') ? 'Charizardite X' : 'Charizardite Y';
  }
  return MEGA_STONE_MAP[preMegaSpecies] || null;
}

function hpColor(pct) {
  if (pct > 50) return 'hp-hi';
  if (pct > 20) return 'hp-mid';
  return 'hp-lo';
}
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ═══════════════════════════════════════════════════════════
// 1.  BACKEND COMMUNICATION
// ═══════════════════════════════════════════════════════════

/**
 * Check server health and update the status dot in the sidebar.
 */
async function checkServer() {
  const dot = document.getElementById('srv-dot');
  const lbl = document.getElementById('srv-status');
  try {
    const r = await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) {
      dot.style.background = 'var(--green)';
      lbl.textContent = 'Server: connected';
      return true;
    }
  } catch (_) {}
  dot.style.background = 'var(--red)';
  lbl.textContent = 'Server: offline';
  return false;
}

/**
 * POST a replay HTML file to /parse and return the structured battle object.
 * Falls back to client-side parsing if the server is unreachable.
 */
async function parseReplayViaServer(file) {
  const fd = new FormData();
  fd.append('replay_html', file);

  const r = await fetch(`${SERVER}/parse`, { method: 'POST', body: fd });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: r.statusText }));
    throw new Error(err.error || r.statusText);
  }
  return r.json();
}

/**
 * POST the approved battle entry to /export and trigger a JSONL download.
 */
async function exportViaServer(battle) {
  const body = {
    battle_id: battle.replay_id || ('battle-' + Date.now()),
    known_teams_entry: buildKnownTeamsEntry(battle),
    replay_html: battle._rawHtml || '',
  };

  const r = await fetch(`${SERVER}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: r.statusText }));
    throw new Error(err.error || r.statusText);
  }

  const blob = await r.blob();
  const count = r.headers.get('X-Transition-Count') || '?';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${body.battle_id}.jsonl`;
  a.click();
  showNotif(`Exported ${count} transitions`);
}

/**
 * Build the known_teams entry shape that server.py / replay_to_transitions expects.
 */
function buildKnownTeamsEntry(b) {
  const entry = { _meta: { yourSide: b.players.our_side, winner: b.winner, p1name: b.players.p1.username, p2name: b.players.p2.username } };
  for (const pid of ['p1', 'p2']) {
    entry[pid] = {};
    for (const species of (b.players[pid].roster || [])) {
      const key = `${pid}:${species}`;
      const inj = b.known_team_overrides?.[key] || {};
      entry[pid][species] = {
        nature: inj.nature || null,
        item: inj.item || null,
        ability: inj.ability || null,
        ev_spread: inj.ev_spread || null,
        moves: inj.moves || null,
      };
    }
  }
  return entry;
}

// ═══════════════════════════════════════════════════════════
// 2.  CLIENT-SIDE VOD PARSER  (fallback / offline)
//     Mirrors vod_parser.py — kept in sync with the Python version.
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
  const sideConditions = { p1: { tailwind: 0, screens: {} }, p2: { tailwind: 0, screens: {} } };
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
    if (s.includes('/')) { const [a, b] = s.split('/'); return [parseFloat(a), parseFloat(b)]; }
    return [parseFloat(s) || 0, 100];
  }
  function snapshotState() {
    return {
      field: { ...fieldConditions },
      side_conditions: {
        p1: { tailwind_turns_remaining: sideConditions.p1.tailwind, screens: { ...sideConditions.p1.screens } },
        p2: { tailwind_turns_remaining: sideConditions.p2.tailwind, screens: { ...sideConditions.p2.screens } },
      },
      active_pokemon: Object.fromEntries(
        Object.entries(activeSlots)
          .filter(([, m]) => !m.is_fainted)
          .map(([k, m]) => [k, { ...m, boosts: { ...m.boosts }, revealed_moves: [...(m.revealed_moves || [])] }])
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
  function setMegaStone(slotKey, megaSpecies, preMegaSpecies, stone) {
    if (activeSlots[slotKey]) activeSlots[slotKey].known_item = stone;
    const pid = slotKey.slice(0, 2);
    for (const sk of [`${pid}:${megaSpecies}`, `${pid}:${preMegaSpecies}`]) {
      if (seenMons[sk]) seenMons[sk].known_item = stone;
    }
  }

  for (const line of lines) {
    if (!line.startsWith('|')) continue;
    const parts = line.split('|');
    const cmd = parts[1];

    if (cmd === 'player') {
      const pid = parts[2];
      if (parts[3]) players[pid] = parts[3];
      if (parts[5]) { try { ratings[pid] = parseInt(parts[5]); } catch (e) {} }

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
      const ident = parts[2], details = parts[3] || '', hpStr = parts[4] || '100/100';
      const slotKey = slotKeyFromIdent(ident);
      const nickname = ident.includes(': ') ? ident.split(': ')[1] : ident;
      const species = details.split(',')[0].trim();
      const pid = slotKey.slice(0, 2);
      const [hpC, hpM] = parseHp(hpStr);
      const seenKey = `${pid}:${species}`;
      if (seenMons[seenKey]) {
        seenMons[seenKey].slot = slotKey[2];
        seenMons[seenKey].hp_current = hpC;
        seenMons[seenKey].is_fainted = false;
        activeSlots[slotKey] = seenMons[seenKey];
      } else {
        const mon = { species, nickname, player: pid, slot: slotKey[2], hp_current: hpC, hp_max: hpM, status: null, boosts: {}, is_mega: false, is_fainted: false, revealed_moves: [], known_item: null, known_tera_type: null, is_terastallized: false };
        seenMons[seenKey] = mon;
        activeSlots[slotKey] = mon;
      }
      if (!knownTeam[pid].includes(species)) knownTeam[pid].push(species);
      turnActions.push({ event: 'switch', slot: slotKey, species, player: pid });

    } else if (cmd === 'detailschange') {
      const slotKey = slotKeyFromIdent(parts[2]);
      const newSpecies = parts[3] ? parts[3].split(',')[0].trim() : null;
      if (activeSlots[slotKey] && newSpecies) {
        const preMegaSpecies = activeSlots[slotKey].species;
        activeSlots[slotKey].species = newSpecies;
        activeSlots[slotKey].is_mega = true;
        // Fallback only — |-mega| fires next and will set the explicit stone name
        if (!activeSlots[slotKey].known_item) {
          const inferred = resolveMegaStone(preMegaSpecies, newSpecies);
          if (inferred) setMegaStone(slotKey, newSpecies, preMegaSpecies, inferred);
        }
      }
      turnActions.push({ event: 'mega_evolution', slot: slotKey, new_species: newSpecies,
        mega_stone: activeSlots[slotKey]?.known_item || null });

    } else if (cmd === '-mega') {
      // |-mega|p1a: Floette|Floette|Floettite  — explicit stone name from the protocol
      const slotKey = slotKeyFromIdent(parts[2]);
      const preMega = parts[3] || null;
      const stone   = parts[4] || null;
      if (stone && activeSlots[slotKey]) {
        const megaSpecies = activeSlots[slotKey].species;  // already mutated by detailschange
        setMegaStone(slotKey, megaSpecies, preMega || megaSpecies, stone);
        // Patch the mega_stone on the most recent mega_evolution action
        for (let i = turnActions.length - 1; i >= 0; i--) {
          if (turnActions[i].event === 'mega_evolution' && turnActions[i].slot === slotKey) {
            turnActions[i].mega_stone = stone;
            break;
          }
        }
      }

    } else if (cmd === 'move') {
      const userIdent = parts[2] || '', moveName = parts[3] || '', targetIdent = parts[4] || null;
      const userSlot = slotKeyFromIdent(userIdent);
      const isProtect = ['protect','detect','wide guard','quick guard','baneful bunker',
        'spiky shield','silk trap','burning bulwark','max guard'].includes(moveName.toLowerCase());
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
      const [hpC] = parseHp(parts[3]);
      const prevHp = activeSlots[slotKey]?.hp_current ?? null;
      if (activeSlots[slotKey]) activeSlots[slotKey].hp_current = hpC;
      const delta = (prevHp !== null) ? Math.round((hpC - prevHp) * 10) / 10 : null;
      const ev = {
        event: cmd.replace('-', ''), slot: slotKey,
        species: speciesFromIdent(parts[2]), hp_pct_after: hpC, hp_pct_delta: delta,
        source_slot: (cmd === '-damage' && lastMoveAction) ? lastMoveAction.user_slot : null,
        source_species: (cmd === '-damage' && lastMoveAction) ? lastMoveAction.user_species : null,
        source_move: (cmd === '-damage' && lastMoveAction) ? lastMoveAction.move : null,
      };
      turnActions.push(ev); turnDamageEvents.push(ev);

    } else if (cmd === 'faint') {
      const slotKey = slotKeyFromIdent(parts[2]);
      if (activeSlots[slotKey]) { activeSlots[slotKey].is_fainted = true; activeSlots[slotKey].hp_current = 0; }
      turnActions.push({ event: 'faint', slot: slotKey, species: speciesFromIdent(parts[2]) });

    } else if (cmd === '-boost' || cmd === '-unboost') {
      const slotKey = slotKeyFromIdent(parts[2]);
      const stat = parts[3] || '', amt = (cmd === '-unboost' ? -1 : 1) * (parseInt(parts[4]) || 1);
      if (activeSlots[slotKey]) activeSlots[slotKey].boosts[stat] = (activeSlots[slotKey].boosts[stat] || 0) + amt;
      turnActions.push({ event: 'stat_change', slot: slotKey, stat, stages: amt });

    } else if (cmd === '-status') {
      const slotKey = slotKeyFromIdent(parts[2]);
      if (activeSlots[slotKey]) activeSlots[slotKey].status = parts[3] || null;

    } else if (cmd === '-curestatus') {
      const slotKey = slotKeyFromIdent(parts[2]);
      if (activeSlots[slotKey]) activeSlots[slotKey].status = null;

    } else if (cmd === '-item') {
      const slotKey = slotKeyFromIdent(parts[2]);
      const item = parts[3] || null;
      if (activeSlots[slotKey] && item) activeSlots[slotKey].known_item = item;
      turnActions.push({ event: 'item_revealed', slot: slotKey, species: speciesFromIdent(parts[2]), item });

    } else if (cmd === '-enditem') {
      const slotKey = slotKeyFromIdent(parts[2]);
      const item = parts[3] || null;
      if (activeSlots[slotKey] && item) activeSlots[slotKey].known_item = item;
      turnActions.push({ event: 'item_consumed', slot: slotKey, species: speciesFromIdent(parts[2]), item });

    } else if (cmd === '-terastallize') {
      const slotKey = slotKeyFromIdent(parts[2]);
      const teraType = parts[3] || null;
      if (activeSlots[slotKey]) { activeSlots[slotKey].known_tera_type = teraType; activeSlots[slotKey].is_terastallized = true; }
      turnActions.push({ event: 'terastallize', slot: slotKey, species: speciesFromIdent(parts[2]), tera_type: teraType });

    } else if (cmd === '-ability') {
      // |-ability|p1a: Aerodactyl|Unnerve
      const slotKey = slotKeyFromIdent(parts[2]);
      const ability = parts[3] || null;
      if (activeSlots[slotKey] && ability) {
        activeSlots[slotKey].known_ability = ability;
        const pid = slotKey.slice(0, 2);
        const seenKey = `${pid}:${activeSlots[slotKey].species}`;
        if (seenMons[seenKey]) seenMons[seenKey].known_ability = ability;
      }
      turnActions.push({ event: 'ability_revealed', slot: slotKey, species: speciesFromIdent(parts[2]), ability });

    } else if (cmd === '-sidestart') {
      const pid = (parts[2] || '').split(':')[0].trim(), eff = parts[3] || '';
      if (eff.includes('Tailwind')) sideConditions[pid].tailwind = 4;
      else if (eff.includes('Reflect')) sideConditions[pid].screens.reflect = 5;
      else if (eff.includes('Light Screen')) sideConditions[pid].screens.light_screen = 5;
      else if (eff.includes('Aurora Veil')) sideConditions[pid].screens.aurora_veil = 5;

    } else if (cmd === '-sideend') {
      const pid = (parts[2] || '').split(':')[0].trim(), eff = parts[3] || '';
      if (eff.includes('Tailwind')) sideConditions[pid].tailwind = 0;
      else if (eff.includes('Reflect')) delete sideConditions[pid].screens.reflect;
      else if (eff.includes('Light Screen')) delete sideConditions[pid].screens.light_screen;
      else if (eff.includes('Aurora Veil')) delete sideConditions[pid].screens.aurora_veil;

    } else if (cmd === '-weather') {
      fieldConditions.weather = (parts[2] && parts[2] !== 'none') ? parts[2] : null;

    } else if (cmd === '-fieldstart') {
      const raw = parts[2] || '';
      if (raw.includes('Trick Room')) fieldConditions.trick_room = 5;
      else if (raw.toLowerCase().includes('terrain')) fieldConditions.terrain = raw;

    } else if (cmd === '-fieldend') {
      const raw = parts[2] || '';
      if (raw.includes('Trick Room')) fieldConditions.trick_room = 0;
      else if (raw.toLowerCase().includes('terrain')) fieldConditions.terrain = null;

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
    revealed_info[seenKey] = {
      revealed_moves: [...(mon.revealed_moves || [])],
      known_item: mon.known_item || null,
      known_tera_type: mon.known_tera_type || null,
      is_terastallized: mon.is_terastallized || false,
      known_ability: mon.known_ability || null,
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

// ═══════════════════════════════════════════════════════════
// 3.  APP STATE
// ═══════════════════════════════════════════════════════════
let battles = [];
let activeBattle = null;
let activeTurnIdx = 0;
let activeTab = 'turns';

// ═══════════════════════════════════════════════════════════
// 4.  RENDER HELPERS
// ═══════════════════════════════════════════════════════════

function renderBattleList() {
  const el = document.getElementById('b-list');
  if (battles.length === 0) {
    el.innerHTML = '<div style="font-size:11px;color:var(--text3);padding:4px 9px;">No battles loaded yet.</div>';
    return;
  }
  el.innerHTML = battles.map((b, i) => {
    const p1 = b.players.p1.username || 'P1';
    const p2 = b.players.p2.username || 'P2';
    const srcColor = { ranked_player_vod:'var(--blue)', own_vod:'var(--green)', bot_vod:'var(--yellow)', self_play:'#ef4179' }[b.source_type] || 'var(--text3)';
    const active = b === activeBattle;
    return `<div class="b-item${active ? ' active' : ''}" onclick="selectBattle(${i})">
      <div style="flex:1;min-width:0">
        <div class="b-name">${p1} vs ${p2}</div>
        <div class="b-meta">${b.turns.length} turns · <span style="color:${srcColor}">${b.source_type}</span></div>
      </div>
      <button class="b-del" onclick="event.stopPropagation();deleteBattle(${i})" title="Remove">✕</button>
    </div>`;
  }).join('');
}

function renderMain() {
  const main = document.getElementById('main');
  if (!activeBattle) {
    main.innerHTML = `<div id="empty-state">
      <div class="big">🎮</div>
      <p>Load a Showdown replay HTML to get started, or create a new battle manually.<br><br>You can also drag &amp; drop a replay file here.</p>
    </div>`;
    return;
  }

  const b = activeBattle;
  const p1name = b.players.p1.username || 'Player 1';
  const p2name = b.players.p2.username || 'Player 2';
  const ourSide = b.players.our_side;

  main.innerHTML = `
  <div id="b-hdr">
    <div style="flex:1">
      <div class="lbl-sm">Battle</div>
      <div style="font-size:15px;font-weight:700;color:var(--text)">${p1name} <span style="color:var(--text3)">vs</span> ${p2name}</div>
      <div style="font-size:11px;color:var(--text3);margin-top:2px">${b.format || ''} · ${b.replay_id || 'no id'}</div>
    </div>
    <div>
      <div class="lbl-sm">Your side</div>
      <select class="fld" id="your-side" onchange="setOurSide(this.value)">
        <option value="p1"${ourSide === 'p1' ? ' selected' : ''}>p1 — ${p1name}</option>
        <option value="p2"${ourSide === 'p2' ? ' selected' : ''}>p2 — ${p2name}</option>
      </select>
    </div>
    <div>
      <div class="lbl-sm">Source type</div>
      <div id="src-type-row">
        ${['ranked_player_vod','own_vod','bot_vod','self_play'].map((t, i) => {
          const lbl = ['B: Ranked VOD','A: Own VOD','C: Bot vs Ranked','D: Self-Play'][i];
          const ac = ['b','a','c','d'][i];
          return `<button class="src-btn${b.source_type === t ? ' active-'+ac : ''}" onclick="setSourceType('${t}')">${lbl}</button>`;
        }).join('')}
      </div>
    </div>
    <button class="btn suc" onclick="doExport()">⬇ Export JSONL</button>
    <button class="btn" onclick="exportJSON()">⬇ Export JSON</button>
  </div>

  <div id="view-tabs"></div>
  <div id="tab-content"></div>`;

  renderTabBar();
  renderTabContent();
}

function renderTabBar() {
  const el = document.getElementById('view-tabs');
  if (!el) return;
  el.innerHTML = `
    <button class="vtab${activeTab === 'turns'  ? ' active' : ''}" onclick="setTab('turns')">📽 Turn Viewer</button>
    <button class="vtab${activeTab === 'inject' ? ' active' : ''}" onclick="setTab('inject')">✏️ Inject Stats</button>
    <button class="vtab${activeTab === 'json'   ? ' active' : ''}" onclick="setTab('json')">{ } Raw JSON</button>`;
}

function renderTabContent() {
  const el = document.getElementById('tab-content');
  if (!el) return;
  if (activeTab === 'turns') renderTurnViewer(el);
  else if (activeTab === 'inject') renderInjectPanel(el);
  else renderJsonTab(el);
}

// ── Turn Viewer ──────────────────────────────────────────
function renderTurnViewer(el) {
  const b = activeBattle;
  const nTurns = b.turns.length;
  if (nTurns === 0) { el.innerHTML = '<div style="padding:20px;color:var(--text3)">No turns recorded.</div>'; return; }
  const idx = Math.min(activeTurnIdx, nTurns - 1);
  const turn = b.turns[idx];
  const turnData = turn.state_after_actions || turn; // support both old and new format

  el.innerHTML = `
  <div id="turn-viewer">
    <div id="tv-bar">
      <span class="tv-label">Turn</span>
      <input type="range" id="turn-scrubber" min="0" max="${nTurns-1}" value="${idx}" oninput="setTurn(parseInt(this.value))">
      <span id="tv-info">${idx+1} / ${nTurns}</span>
      <div id="tv-btns">
        <button class="btn" onclick="setTurn(${Math.max(0,idx-1)})">◀</button>
        <button class="btn" onclick="setTurn(${Math.min(nTurns-1,idx+1)})">▶</button>
      </div>
    </div>
    <div id="field-bar">
      <span id="field-bar-label">Field:</span>
      ${renderFieldPills(turnData)}
    </div>
    <div id="turn-board">
      ${renderSideSlots(turn, 'p1', b)}
      ${renderSideSlots(turn, 'p2', b)}
    </div>
    <div id="action-log-wrap">
      <h4>Turn ${turn.turn} — Actions</h4>
      <div id="action-log">${renderActionLog(turn)}</div>
    </div>
  </div>`;
}

function renderFieldPills(turnData) {
  const pills = [];
  const f = turnData.field || {};
  const sc = turnData.side_conditions || {};
  if (f.weather) pills.push(`<span class="field-pill fp-weather">☁ ${f.weather}</span>`);
  if (f.terrain)  pills.push(`<span class="field-pill fp-terrain">🌿 ${f.terrain}</span>`);
  if (f.trick_room_turns_remaining > 0) pills.push(`<span class="field-pill fp-tr">⏲ Trick Room ×${f.trick_room_turns_remaining}</span>`);
  if (sc.p1?.tailwind_turns_remaining > 0) pills.push(`<span class="field-pill fp-tw">💨 P1 Tailwind ×${sc.p1.tailwind_turns_remaining}</span>`);
  if (sc.p2?.tailwind_turns_remaining > 0) pills.push(`<span class="field-pill fp-tw">💨 P2 Tailwind ×${sc.p2.tailwind_turns_remaining}</span>`);
  return pills.length ? pills.join('') : '<span style="font-size:11px;color:var(--text3)">None</span>';
}

function renderSideSlots(turn, pid, b) {
  const isP1 = pid === 'p1';
  const username = b.players[pid].username || pid.toUpperCase();
  const ourSide = b.players.our_side;
  const yourTag = pid === ourSide ? '<span class="your-side-tag">YOUR SIDE</span>' : '';
  const winTag = b.winner && b.players[pid].username === b.winner ? '<span class="winner-badge">🏆</span>' : '';

  const roster = b.players[pid].roster || [];
  const afterState = turn.state_after_actions || turn;
  const activeForPlayer = Object.entries(afterState.active_pokemon || {})
    .filter(([k]) => k.startsWith(pid))
    .map(([k, m]) => ({ ...m, slotKey: k }));

  // Helper: build data for one roster entry
  function monData(species) {
    const active = activeForPlayer.find(m => m.species === species || m.species.startsWith(species.split('-')[0]));
    const inj = b.known_team_overrides?.[`${pid}:${species}`];
    const hp = active ? active.hp_current : null;
    const fainted = active ? active.is_fainted : false;
    const isMega = active ? active.is_mega : false;
    const onField = !!active && !fainted;
    const boosts = active ? (active.boosts || {}) : {};
    return { active, inj, hp, fainted, isMega, onField, boosts, displaySpecies: active?.species || species, species };
  }

  // ── Active (on-field) slots: up to 2, shown large ─────────────────────
  // Drive from the actual activeForPlayer slots so order matches p1a/p1b
  const fieldSlots = activeForPlayer
    .filter(m => !m.is_fainted)
    .sort((a, b) => a.slotKey.localeCompare(b.slotKey));

  const fieldHtml = fieldSlots.length
    ? fieldSlots.map(m => {
        const inj = b.known_team_overrides?.[`${pid}:${m.species.replace(/-Mega$/,'')}`]
                 || b.known_team_overrides?.[`${pid}:${m.species}`];
        const boosts = m.boosts || {};
        const boostHtml = Object.entries(boosts).filter(([,v]) => v !== 0).map(([s,v]) =>
          `<span class="boost-pip ${v>0?'boost-pos':'boost-neg'}">${STAT_LABELS[s]||s} ${v>0?'+':''}${v}</span>`
        ).join('');
        const slotLetter = m.slotKey.slice(-1).toUpperCase(); // 'A' or 'B'
        const teraTag  = m.is_terastallized ? `<span class="active-badge tera-badge">TERA</span>` : '';
        const megaTag  = m.is_mega ? `<span class="active-badge mega-badge">MEGA</span>` : '';
        const slotTag  = `<span class="active-slot-letter">${slotLetter}</span>`;
        const itemLine = m.known_item ? `<div class="active-item">🎒 ${m.known_item}</div>` : '';
        const hp = m.hp_current ?? 100;
        return `<div class="active-slot-card">
          <div class="active-slot-top">${slotTag}${teraTag}${megaTag}</div>
          <div class="active-species">${m.species}${inj ? ' <span class="inj-known">✓</span>' : ''}</div>
          <div class="active-hp-bar"><div class="slot-hp-fill ${hpColor(hp)}" style="width:${hp}%"></div></div>
          <div class="active-hp-num">${hp.toFixed(0)}% HP</div>
          ${itemLine}
          ${boostHtml ? `<div class="slot-boosts">${boostHtml}</div>` : ''}
        </div>`;
      }).join('')
    : `<div class="active-empty">— no active Pokémon —</div>`;

  // ── Bench row: all 6 (or fewer) as compact chips ──────────────────────
  // A mon on field but fainted appears greyed out; not-yet-seen appears as species name only
  const benchHtml = roster.map(species => {
    const d = monData(species);
    let cls = 'bench-chip';
    if (d.fainted) cls += ' bench-fainted';
    else if (d.onField) cls += ' bench-onfield';

    const hp = d.hp !== null ? `<span class="bench-hp">${d.hp.toFixed(0)}%</span>` : '';
    const megaTag = d.isMega ? `<span class="bench-mega">M</span>` : '';
    const teraTag = d.active?.is_terastallized ? `<span class="bench-tera">T</span>` : '';
    return `<div class="${cls}" title="${d.displaySpecies}${d.hp !== null ? ' · ' + d.hp.toFixed(0) + '%' : ''}">
      <span class="bench-name">${d.displaySpecies}</span>${megaTag}${teraTag}${hp}
    </div>`;
  }).join('');

  return `<div class="tb-panel">
    <div class="tb-phdr ${isP1?'p1h':'p2h'}">${pid.toUpperCase()} · ${username} ${yourTag} ${winTag}</div>
    <div class="tb-field-row">${fieldSlots.length
      ? fieldHtml
      : '<div class="active-empty">— no active Pokémon recorded —</div>'}</div>
    ${roster.length ? `<div class="tb-bench-row">
      <span class="bench-label">Bench</span>
      <div class="bench-chips">${benchHtml}</div>
    </div>` : ''}
  </div>`;
}

function renderActionLog(turn) {
  if (!turn.actions?.length) return '<div style="font-size:11px;color:var(--text3);padding:4px 0">No actions recorded.</div>';
  return turn.actions.map(a => {
    let badge = '', text = '';
    if (a.event === 'move') {
      badge = `<span class="al-evt al-move">Move</span>`;
      text = `<b>${a.user_slot}</b> ${a.user_species||''} → <b>${a.move}</b>${a.is_protect?' 🛡':''}${a.target_slot ? ` on ${a.target_slot}` : ''}`;
    } else if (a.event === 'switch') {
      badge = `<span class="al-evt al-switch">Switch</span>`;
      text = `${a.player} sends out <b>${a.species}</b> to slot ${a.slot}`;
    } else if (a.event === 'damage') {
      badge = `<span class="al-evt al-damage">Dmg</span>`;
      text = `<b>${a.slot}</b> ${a.species||''} → ${a.hp_pct_after?.toFixed(0)}% HP`;
      if (a.source_move) text += ` (from ${a.source_move})`;
      if (a.hp_pct_delta !== null) text += ` <span class="al-delta">(${a.hp_pct_delta?.toFixed(0)}%)</span>`;
    } else if (a.event === 'heal') {
      badge = `<span class="al-evt al-damage" style="background:rgba(76,175,125,.15);color:var(--green);border-color:var(--green)">Heal</span>`;
      text = `<b>${a.slot}</b> → ${a.hp_pct_after?.toFixed(0)}% HP`;
      if (a.hp_pct_delta !== null) text += ` <span class="al-heal">(+${a.hp_pct_delta?.toFixed(0)}%)</span>`;
    } else if (a.event === 'faint') {
      badge = `<span class="al-evt al-faint">Faint</span>`;
      text = `<b>${a.slot}</b> ${a.species||''} fainted`;
    } else if (a.event === 'mega_evolution') {
      badge = `<span class="al-evt al-mega">Mega</span>`;
      text = `${a.slot} → <b>${a.new_species}</b>`;
    } else if (a.event === 'stat_change') {
      badge = `<span class="al-evt al-stat">Stat</span>`;
      text = `<b>${a.slot}</b> ${STAT_LABELS[a.stat]||a.stat} ${a.stages>0?'+':''}${a.stages}`;
    } else if (a.event === 'item_consumed' || a.event === 'item_revealed') {
      badge = `<span class="al-evt" style="background:rgba(245,200,66,.15);color:var(--yellow);border:1px solid var(--yellow)">Item</span>`;
      text = `<b>${a.slot}</b> ${a.event === 'item_consumed' ? 'consumed' : 'revealed'} <b>${a.item}</b>`;
    } else if (a.event === 'terastallize') {
      badge = `<span class="al-evt" style="background:rgba(239,65,121,.15);color:#ef4179;border:1px solid #ef4179">Tera</span>`;
      text = `<b>${a.slot}</b> ${a.species||''} terastallized → <b>${a.tera_type}</b>`;
    } else {
      badge = `<span class="al-evt" style="background:var(--surf2)">${a.event}</span>`;
      text = JSON.stringify(a).slice(0, 80);
    }
    return `<div class="al-row">${badge}<span class="al-text">${text}</span></div>`;
  }).join('');
}

// ── Inject Stats Panel ───────────────────────────────────
function renderInjectPanel(el) {
  const b = activeBattle;
  const allMons = [];
  for (const pid of ['p1','p2']) {
    for (const species of (b.players[pid].roster || [])) allMons.push({ pid, species });
  }

  const cards = allMons.map(({ pid, species }) => {
    const key = `${pid}:${species}`;
    const inj  = b.known_team_overrides?.[key] || {};
    const rev  = b.revealed_info?.[key] || {};   // what the parser saw in the log
    const isOurs = pid === b.players.our_side;

    // "Stats injected" badge — true if user has typed anything manually
    const hasManual = Object.values(inj).some(v => v !== null && v !== undefined && v !== '' &&
      !(Array.isArray(v) && v.every(x => !x)) &&
      !(typeof v === 'object' && !Array.isArray(v) && Object.values(v).every(x => !x)));

    // ── Confirmed-from-replay chips ─────────────────────────────────────
    // Item
    const revItem = rev.known_item || null;
    const itemChip = revItem
      ? `<span class="rev-chip">🎒 ${revItem}</span>`
      : '';

    // Tera type
    const revTera = rev.known_tera_type || null;
    const teraChip = revTera
      ? `<span class="rev-chip rev-tera">◈ Tera ${revTera}${rev.is_terastallized ? ' ✓' : ''}</span>`
      : '';

    // Revealed moves — up to 4; pre-fill editable inputs with what was seen
    const revMoves = rev.revealed_moves || [];

    // ── Editable fields ─────────────────────────────────────────────────
    const evHtml = STAT_NAMES.map(s => `
      <div class="ev-cell">
        <label>${STAT_LABELS[s]}</label>
        <input type="number" min="0" max="252" value="${inj.ev_spread?.[s] ?? ''}" placeholder="0"
          oninput="setInject('${key}','ev_${s}',this.value)" />
      </div>`).join('');

    // Move inputs: manual override wins; fall back to revealed move from log
    const moveHtml = [0,1,2,3].map(i => {
      const manualVal = inj.moves?.[i] ?? '';
      const revVal    = revMoves[i] ?? '';
      const val       = manualVal || revVal;
      const isConfirmed = !manualVal && !!revVal;
      return `<div class="inj-move-slot">
        <input class="inj-move-in${isConfirmed ? ' rev-prefilled' : ''}" type="text"
          placeholder="Move ${i+1}" value="${escHtml(val)}"
          oninput="setInject('${key}','move_${i}',this.value)" />
        ${isConfirmed ? '<span class="rev-move-badge" title="Confirmed in replay">VOD</span>' : ''}
      </div>`;
    }).join('');

    // Item input — pre-filled from replay if not manually set
    const itemVal      = inj.item ?? revItem ?? '';
    const itemPrefill  = !inj.item && !!revItem;

    // Ability input — pre-filled from |-ability| reveals in the log
    const revAbility  = rev.known_ability || null;
    const abilityVal  = inj.ability ?? revAbility ?? '';
    const abilityPrefill = !inj.ability && !!revAbility;

    // Ability chip (shown in the chip row alongside item/tera)
    const abilityChip = revAbility
      ? `<span class="rev-chip rev-ability">⚡ ${revAbility}</span>`
      : '';

    return `<div class="inj-card">
      <div class="inj-card-hdr">
        <span class="side-dot ${pid}-dot"></span>
        <span>${species}</span>
        <span style="font-size:9px;color:var(--text3)">${pid.toUpperCase()}${isOurs?' · YOUR SIDE':''}</span>
        <span class="${hasManual?'inj-known':'inj-unknown'}">${hasManual?'✓ Stats injected':'No stats yet'}</span>
      </div>

      ${(itemChip || teraChip || abilityChip) ? `<div class="rev-chips">${itemChip}${abilityChip}${teraChip}</div>` : ''}

      <div class="inj-body">
        <div class="inj-row">
          <label>Nature</label>
          <select onchange="setInject('${key}','nature',this.value)">
            <option value="">— unknown —</option>
            ${NATURES.map(n => `<option value="${n}"${inj.nature===n?' selected':''}>${n}</option>`).join('')}
          </select>
        </div>
        <div class="inj-row">
          <label>Item${itemPrefill ? ' <span class="rev-label-badge">VOD</span>' : ''}</label>
          <input type="text" placeholder="e.g. Choice Specs"
            value="${escHtml(itemVal)}"
            class="${itemPrefill ? 'rev-prefilled' : ''}"
            oninput="setInject('${key}','item',this.value)" />
        </div>
        <div class="inj-row">
          <label>Ability${abilityPrefill ? ' <span class="rev-label-badge">VOD</span>' : ''}</label>
          <input type="text" placeholder="e.g. Fairy Aura"
            value="${escHtml(abilityVal)}"
            class="${abilityPrefill ? 'rev-prefilled' : ''}"
            oninput="setInject('${key}','ability',this.value)" />
        </div>
        <div class="inj-section-label">EVs</div>
        <div class="ev-row">${evHtml}</div>
        <div class="inj-section-label">
          Moves
          ${revMoves.length ? `<span class="rev-label-badge">${revMoves.length} confirmed in VOD</span>` : ''}
        </div>
        <div class="inj-moves">${moveHtml}</div>
      </div>
    </div>`;
  }).join('');

  el.innerHTML = `<div id="inject-wrap">
    <h3>Manual Stat Injection
      <span style="color:var(--text3);font-size:10px;font-weight:400;text-transform:none;letter-spacing:0">
        Fill in known stats to upgrade Type B → Type A (your side) or Type C (bot side).
        <span class="rev-chip" style="margin-left:6px">VOD</span> = confirmed from replay log.
      </span>
    </h3>
    <div id="inject-grid">${cards}</div>
  </div>`;
}

// ── JSON Tab ─────────────────────────────────────────────
function renderJsonTab(el) {
  const json = buildExportJson();
  el.innerHTML = `<div style="position:relative">
    <button class="btn" style="position:absolute;top:8px;right:8px;z-index:2" onclick="copyJson()">📋 Copy</button>
    <pre id="json-pre" style="background:var(--bg2);border:1px solid var(--bdr);border-radius:var(--rad-lg);padding:14px;font-size:10px;line-height:1.6;overflow:auto;max-height:70vh;color:var(--text2)">${escHtml(JSON.stringify(json, null, 2))}</pre>
  </div>`;
}

// ═══════════════════════════════════════════════════════════
// 5.  ACTIONS
// ═══════════════════════════════════════════════════════════

function setTurn(idx) {
  activeTurnIdx = idx;
  renderTabContent();
  // Reset scroll so the board is always visible at the top after navigation
  const tv = document.getElementById('turn-viewer');
  if (tv) tv.scrollIntoView({ block: 'start', behavior: 'instant' });
  const main = document.getElementById('main');
  if (main) main.scrollTop = 0;
}
function setTab(tab)  { activeTab = tab; renderTabBar(); renderTabContent(); }

function setOurSide(val) {
  if (activeBattle) { activeBattle.players.our_side = val; renderMain(); }
}
function setSourceType(t) {
  if (!activeBattle) return;
  activeBattle.source_type = t;
  if (t === 'own_vod')   { activeBattle.stats_quality = { our_side:'exact', opp_side:'distribution' }; }
  else if (t === 'bot_vod')  { activeBattle.stats_quality = { our_side:'exact', opp_side:'distribution' }; }
  else if (t === 'self_play') { activeBattle.stats_quality = { our_side:'exact', opp_side:'exact' }; }
  else { activeBattle.stats_quality = { our_side:'distribution', opp_side:'distribution' }; }
  renderMain();
}
function setInject(key, field, value) {
  if (!activeBattle) return;
  if (!activeBattle.known_team_overrides) activeBattle.known_team_overrides = {};
  if (!activeBattle.known_team_overrides[key]) activeBattle.known_team_overrides[key] = {};
  const inj = activeBattle.known_team_overrides[key];
  if (field === 'nature') inj.nature = value || null;
  else if (field === 'item') inj.item = value || null;
  else if (field === 'ability') inj.ability = value || null;
  else if (field.startsWith('ev_')) { if (!inj.ev_spread) inj.ev_spread = {}; inj.ev_spread[field.slice(3)] = parseInt(value) || 0; }
  else if (field.startsWith('move_')) { if (!inj.moves) inj.moves = ['','','','']; inj.moves[parseInt(field.slice(5))] = value; }
  // Live badge refresh
  document.querySelectorAll('.inj-card-hdr').forEach(hdr => {
    const badge = hdr.querySelector('.inj-known, .inj-unknown');
    if (!badge) return;
    const dot = hdr.querySelector('.side-dot');
    const sp  = hdr.querySelector('span:nth-child(2)');
    if (!dot || !sp) return;
    const pid = dot.classList.contains('p1-dot') ? 'p1' : 'p2';
    const k2 = `${pid}:${sp.textContent.trim()}`;
    const has = Object.values(activeBattle.known_team_overrides[k2] || {}).some(v => v !== null && v !== undefined);
    badge.className = has ? 'inj-known' : 'inj-unknown';
    badge.textContent = has ? '✓ Stats injected' : 'No stats yet';
  });
}

function selectBattle(i) {
  activeBattle = battles[i];
  activeTurnIdx = 0;
  activeTab = 'turns';
  renderBattleList();
  renderMain();
}
function deleteBattle(i) {
  const wasActive = battles[i] === activeBattle;
  battles.splice(i, 1);
  if (wasActive) activeBattle = battles[0] || null;
  renderBattleList();
  renderMain();
}

function buildExportJson() {
  if (!activeBattle) return {};
  const b = activeBattle;
  const turns = b.turns.map(t => {
    const ap = {};
    const src = t.state_after_actions?.active_pokemon || t.active_pokemon || {};
    for (const [k, mon] of Object.entries(src)) {
      const injKey = `${mon.player}:${mon.species.replace(/-Mega$/,'')}`;
      const inj = b.known_team_overrides?.[injKey] || {};
      ap[k] = { ...mon, ev_spread: inj.ev_spread||null, nature: inj.nature||null, item: inj.item||null, ability: inj.ability||null, moves: inj.moves||null };
    }
    return { ...t, active_pokemon: ap };
  });
  return { ...b, turns };
}

function exportJSON() {
  const json = JSON.stringify(buildExportJson(), null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${activeBattle?.replay_id || 'battle'}.json`;
  a.click();
  showNotif('JSON exported!');
}

async function doExport() {
  if (!activeBattle) return;
  try {
    await exportViaServer(activeBattle);
  } catch (err) {
    showNotif('Server export failed — using JSON fallback');
    exportJSON();
  }
}

function copyJson() {
  navigator.clipboard.writeText(JSON.stringify(buildExportJson(), null, 2));
  showNotif('Copied to clipboard');
}

function showNotif(msg) {
  const n = document.getElementById('notif');
  n.textContent = msg; n.style.display = 'block';
  setTimeout(() => n.style.display = 'none', 2200);
}

// ═══════════════════════════════════════════════════════════
// 6.  FILE I/O
// ═══════════════════════════════════════════════════════════

async function loadReplayFile(file) {
  showNotif('Parsing replay…');
  let battle = null;

  // Always parse client-side for display — the server schema may differ from
  // what the render functions expect, and the client parser is fully validated.
  // The server is only needed at *export* time (doExport -> exportViaServer).
  const html = await file.text();
  try {
    const log = extractLog(html);
    const rid = extractReplayId(html);
    battle = parseShowdownLog(log, 'p1');
    battle.replay_id = rid;
    battle._rawHtml = html;   // kept so exportViaServer can POST it
    if (!battle.known_team_overrides) battle.known_team_overrides = {};
  } catch (err) {
    alert('Parse error: ' + err.message);
    return;
  }

  battle.source_file = file.name;
  battles.push(battle);
  activeBattle = battle;
  activeTurnIdx = 0;
  activeTab = 'turns';
  renderBattleList();
  renderMain();
  showNotif(`Loaded: ${file.name}`);
}

function loadJsonFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    try {
      const data = JSON.parse(e.target.result);
      const list = Array.isArray(data) ? data : [data];
      for (const b of list) {
        if (!b.known_team_overrides) b.known_team_overrides = {};
        battles.push(b);
      }
      activeBattle = battles[battles.length - 1];
      activeTurnIdx = 0; activeTab = 'turns';
      renderBattleList(); renderMain();
      showNotif(`Imported ${list.length} battle(s)`);
    } catch (err) { alert('JSON parse error: ' + err.message); }
  };
  reader.readAsText(file);
}

// ── Drag & drop ──
function setupDragDrop() {
  const main = document.getElementById('main');
  main.addEventListener('dragover', e => { e.preventDefault(); main.style.outline = '2px dashed var(--blue)'; });
  main.addEventListener('dragleave', () => { main.style.outline = ''; });
  main.addEventListener('drop', e => {
    e.preventDefault(); main.style.outline = '';
    for (const file of e.dataTransfer.files) {
      if (file.name.endsWith('.html')) loadReplayFile(file);
      else if (file.name.endsWith('.json')) loadJsonFile(file);
    }
  });
}

// ═══════════════════════════════════════════════════════════
// 7.  BOOT
// ═══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  renderBattleList();
  renderMain();
  setupDragDrop();

  // Wire header buttons
  const replayFileIn = document.getElementById('replay-file');
  const impFileIn    = document.getElementById('imp-file');

  document.querySelectorAll('#hdr button').forEach(btn => {
    const txt = btn.textContent.trim();
    if (txt.includes('Load replay') || txt.includes('📂')) {
      btn.addEventListener('click', () => replayFileIn.click());
    } else if (txt.includes('Import JSON')) {
      btn.addEventListener('click', () => impFileIn.click());
    } else if (txt.includes('Run Parser')) {
      btn.addEventListener('click', () => { if (activeBattle) { activeTurnIdx = 0; renderMain(); showNotif('Re-rendered'); } });
    } else if (txt.includes('Export')) {
      btn.addEventListener('click', doExport);
    } else if (txt.includes('New battle')) {
      btn.addEventListener('click', () => {
        const b = {
          source_type: 'own_vod', replay_id: 'new-battle-' + Date.now(),
          format: '[Gen 9 Champions] VGC 2026 Reg M-A',
          players: { our_side: 'p1', p1: { username:'Me', rating_before:null, rating_delta:null, roster:[], team_size_chosen:null }, p2: { username:'Opponent', rating_before:null, rating_delta:null, roster:[], team_size_chosen:null } },
          winner: null, stats_quality: { our_side:'exact', opp_side:'distribution' }, known_team_overrides:{}, turns:[]
        };
        battles.push(b); activeBattle = b; activeTurnIdx = 0; activeTab = 'inject';
        renderBattleList(); renderMain(); showNotif('New battle created');
      });
    }
  });

  replayFileIn.addEventListener('change', e => { if (e.target.files[0]) loadReplayFile(e.target.files[0]); e.target.value = ''; });
  impFileIn.addEventListener('change',    e => { if (e.target.files[0]) loadJsonFile(e.target.files[0]);   e.target.value = ''; });

  // Sidebar new battle button
  document.querySelector('#side .btn')?.addEventListener('click', () => {
    document.querySelectorAll('#hdr button')[0].click();
  });

  // Server status check
  checkServer();
  setInterval(checkServer, 15000);
});
