// ═══════════════════════════════════════════════════════════
// tb_actions.js
// User-triggered actions, file I/O, drag-and-drop, and boot.
// Depends on: tb_constants.js, tb_parser.js, tb_api.js, tb_render.js
// ═══════════════════════════════════════════════════════════

// ── Tab / turn navigation ────────────────────────────────
function setTurn(idx) {
  activeTurnIdx = idx;
  renderTabContent();
  const tv   = document.getElementById('turn-viewer');
  if (tv) tv.scrollIntoView({ block: 'start', behavior: 'instant' });
  const main = document.getElementById('main');
  if (main) main.scrollTop = 0;
}

function setTab(tab) {
  activeTab = tab;
  renderTabBar();
  renderTabContent();
}

// ── Battle-level setters ─────────────────────────────────
function setOurSide(val) {
  if (activeBattle) { activeBattle.players.our_side = val; renderMain(); }
}

function setSourceType(t) {
  if (!activeBattle) return;
  activeBattle.source_type = t;
  if      (t === 'own_vod')  activeBattle.stats_quality = { our_side: 'exact',        opp_side: 'distribution' };
  else if (t === 'bot_vod')  activeBattle.stats_quality = { our_side: 'exact',        opp_side: 'distribution' };
  else if (t === 'self_play') activeBattle.stats_quality = { our_side: 'exact',        opp_side: 'exact' };
  else                        activeBattle.stats_quality = { our_side: 'distribution', opp_side: 'distribution' };
  renderMain();
}

/**
 * Update a single injected field for one Pokémon and refresh the ✓/✗ badge
 * without re-rendering the whole panel.
 */
function setInject(key, field, value) {
  if (!activeBattle) return;
  if (!activeBattle.known_team_overrides)        activeBattle.known_team_overrides = {};
  if (!activeBattle.known_team_overrides[key])   activeBattle.known_team_overrides[key] = {};

  const inj = activeBattle.known_team_overrides[key];
  if      (field === 'nature')             inj.nature  = value || null;
  else if (field === 'item')               inj.item    = value || null;
  else if (field === 'ability')            inj.ability = value || null;
  else if (field.startsWith('ev_'))  { if (!inj.ev_spread) inj.ev_spread = {}; inj.ev_spread[field.slice(3)] = parseInt(value) || 0; }
  else if (field.startsWith('move_')) { if (!inj.moves) inj.moves = ['', '', '', '']; inj.moves[parseInt(field.slice(5))] = value; }

  // Live badge refresh — update only the affected card's badge
  document.querySelectorAll('.inj-card-hdr').forEach(hdr => {
    const badge = hdr.querySelector('.inj-known, .inj-unknown');
    if (!badge) return;
    const dot = hdr.querySelector('.side-dot');
    const sp  = hdr.querySelector('span:nth-child(2)');
    if (!dot || !sp) return;
    const pid = dot.classList.contains('p1-dot') ? 'p1' : 'p2';
    const k2  = `${pid}:${sp.textContent.trim()}`;
    const has = Object.values(activeBattle.known_team_overrides[k2] || {}).some(v => v !== null && v !== undefined);
    badge.className  = has ? 'inj-known' : 'inj-unknown';
    badge.textContent = has ? '✓ Stats injected' : 'No stats yet';
  });
}

// ── Battle list management ───────────────────────────────
function selectBattle(i) {
  activeBattle  = battles[i];
  activeTurnIdx = 0;
  activeTab     = 'turns';
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

// ── Export helpers ───────────────────────────────────────
function buildExportJson() {
  if (!activeBattle) return {};
  const b = activeBattle;
  const turns = b.turns.map(t => {
    const ap  = {};
    const src = t.state_after_actions?.active_pokemon || t.active_pokemon || {};
    for (const [k, mon] of Object.entries(src)) {
      // Bug 8: inj entries are keyed by BASE species ("Floette-Eternal"),
      // never the mega forme name — suffix-stripping "Floette-Mega" would
      // wrongly give "Floette".  base_species is frozen at first switch-in.
      const baseSpecies = mon.base_species || mon.species.replace(/-Mega(-[XY])?$/, '');
      const injKey = `${mon.player}:${baseSpecies}`;
      const inj    = b.known_team_overrides?.[injKey] || {};

      // A mega forme has exactly ONE ability, fixed by the forme — the
      // user-injected ability is always the base forme's and must never be
      // exported as the active ability of a mega'd mon.
      const isMega = !!mon.is_mega;
      const megaAb = isMega ? (mon.mega_ability || dexMegaAbility(mon.species) || null) : (mon.mega_ability || null);
      ap[k] = {
        ...mon,
        ev_spread: inj.ev_spread || null,
        nature:    inj.nature    || null,
        item:      inj.item      || null,
        ability:   isMega ? megaAb : (inj.ability || mon.known_ability || null),
        pre_mega_ability: mon.pre_mega_ability || inj.ability || null,
        mega_ability:     megaAb,
        moves:     inj.moves     || null,
      };
    }
    return { ...t, active_pokemon: ap };
  });
  return { ...b, turns };
}

/** Client-side JSON download (always available, no server needed). */
function exportJSON() {
  const json = JSON.stringify(buildExportJson(), null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `${activeBattle?.replay_id || 'battle'}.json`;
  a.click();
  showNotif('JSON exported!');
}

/**
 * Primary export: tries the server (/export → JSONL with transitions).
 * Falls back silently to the client-side JSON download if server is offline.
 */
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

// ── File I/O ─────────────────────────────────────────────

/**
 * Load a Showdown replay .html file.
 * Always parses client-side (tb_parser.js) for display.
 * Keeps _rawHtml on the battle object so doExport() can POST it to /export.
 */
async function loadReplayFile(file) {
  showNotif('Parsing replay…');
  const html = await file.text();
  let battle = null;
  try {
    const log = extractLog(html);
    const rid = extractReplayId(html);
    battle     = parseShowdownLog(log, 'p1');
    battle.replay_id  = rid;
    battle._rawHtml   = html;   // kept so exportViaServer can POST it
    if (!battle.known_team_overrides) battle.known_team_overrides = {};
  } catch (err) {
    alert('Parse error: ' + err.message);
    return;
  }
  battle.source_file = file.name;
  battles.push(battle);
  activeBattle  = battle;
  activeTurnIdx = 0;
  activeTab     = 'turns';
  renderBattleList();
  renderMain();
  showNotif(`Loaded: ${file.name}`);
}

/** Import a previously-exported .json battle file. */
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
      activeBattle  = battles[battles.length - 1];
      activeTurnIdx = 0;
      activeTab     = 'turns';
      renderBattleList();
      renderMain();
      showNotif(`Imported ${list.length} battle(s)`);
    } catch (err) { alert('JSON parse error: ' + err.message); }
  };
  reader.readAsText(file);
}

// ── Drag & drop ──────────────────────────────────────────
function setupDragDrop() {
  const main = document.getElementById('main');
  main.addEventListener('dragover', e => { e.preventDefault(); main.style.outline = '2px dashed var(--blue)'; });
  main.addEventListener('dragleave', () => { main.style.outline = ''; });
  main.addEventListener('drop', e => {
    e.preventDefault();
    main.style.outline = '';
    for (const file of e.dataTransfer.files) {
      if (file.name.endsWith('.html')) loadReplayFile(file);
      else if (file.name.endsWith('.json')) loadJsonFile(file);
    }
  });
}

// ── Boot ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderBattleList();
  renderMain();
  setupDragDrop();

  const replayFileIn = document.getElementById('replay-file');
  const impFileIn    = document.getElementById('imp-file');

  // Wire header buttons by their text content
  document.querySelectorAll('#hdr button').forEach(btn => {
    const txt = btn.textContent.trim();
    if (txt.includes('Load replay') || txt.includes('📂')) {
      btn.addEventListener('click', () => replayFileIn.click());
    } else if (txt.includes('Import JSON')) {
      btn.addEventListener('click', () => impFileIn.click());
    } else if (txt.includes('Run Parser')) {
      btn.addEventListener('click', () => {
        if (activeBattle) { activeTurnIdx = 0; renderMain(); showNotif('Re-rendered'); }
      });
    } else if (txt.includes('Export')) {
      btn.addEventListener('click', doExport);
    } else if (txt.includes('New battle')) {
      btn.addEventListener('click', () => {
        const b = {
          source_type: 'own_vod',
          replay_id:   'new-battle-' + Date.now(),
          format:      '[Gen 9 Champions] VGC 2026 Reg M-A',
          players: {
            our_side: 'p1',
            p1: { username: 'Me',       rating_before: null, rating_delta: null, roster: [], team_size_chosen: null },
            p2: { username: 'Opponent', rating_before: null, rating_delta: null, roster: [], team_size_chosen: null },
          },
          winner: null,
          stats_quality:       { our_side: 'exact', opp_side: 'distribution' },
          known_team_overrides: {},
          turns: [],
        };
        battles.push(b);
        activeBattle  = b;
        activeTurnIdx = 0;
        activeTab     = 'inject';
        renderBattleList();
        renderMain();
        showNotif('New battle created');
      });
    }
  });

  replayFileIn.addEventListener('change', e => { if (e.target.files[0]) loadReplayFile(e.target.files[0]); e.target.value = ''; });
  impFileIn.addEventListener('change',    e => { if (e.target.files[0]) loadJsonFile(e.target.files[0]);   e.target.value = ''; });

  // Sidebar "New battle" button delegates to the header button
  document.querySelector('#side .btn')?.addEventListener('click', () => {
    document.querySelectorAll('#hdr button')[0].click();
  });

  // Server health check on boot + every 15 s; pokedex loads alongside
  loadPokedex();
  checkServer();
  setInterval(checkServer, 15000);
});
