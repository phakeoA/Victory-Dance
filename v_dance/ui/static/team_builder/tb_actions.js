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
 *
 * `el` (optional) is the input/select that fired the event — used to strip
 * belief auto-fill styling live: any manual edit of an auto-filled field
 * revokes its "auto" status (the value stays, the AUTO badge goes).
 */
function setInject(key, field, value, el) {
  if (!activeBattle) return;
  if (!activeBattle.known_team_overrides)        activeBattle.known_team_overrides = {};
  if (!activeBattle.known_team_overrides[key])   activeBattle.known_team_overrides[key] = {};

  const inj = activeBattle.known_team_overrides[key];
  if      (field === 'nature')             inj.nature  = value || null;
  else if (field === 'item')               inj.item    = value || null;
  else if (field === 'ability')            inj.ability = value || null;
  else if (field.startsWith('ev_'))  { if (!inj.ev_spread) inj.ev_spread = {}; inj.ev_spread[field.slice(3)] = parseInt(value) || 0; }
  else if (field.startsWith('move_')) { if (!inj.moves) inj.moves = ['', '', '', '']; inj.moves[parseInt(field.slice(5))] = value; }

  // ── Belief auto-fill revocation — manual edit ends the auto status ──────
  const auto = activeBattle.belief_autofill?.[key];
  if (auto) {
    let cleared = false;
    if (field === 'nature' || field === 'item' || field === 'ability') {
      if (auto[field]) { auto[field] = false; cleared = true; }
    } else if (field.startsWith('ev_')) {
      // EVs are one logical unit (a spread): editing any cell makes the
      // whole spread manual.
      if (auto.ev_spread) {
        auto.ev_spread = false;
        cleared = true;
        const card = el?.closest('.inj-card');
        card?.querySelectorAll('.ev-row .belief-prefilled')
            .forEach(x => x.classList.remove('belief-prefilled'));
        card?.querySelector('.ev-auto-badge')?.remove();
        // Nature rides on the same Pikalytics spread — it stays auto (it has
        // its own flag and badge, revoked only when the nature itself is edited).
      }
    } else if (field.startsWith('move_')) {
      const idx = parseInt(field.slice(5));
      if (auto.moves && auto.moves[idx]) { auto.moves[idx] = false; cleared = true; }
    }
    if (cleared && el) {
      el.classList.remove('belief-prefilled');
      el.closest('.inj-row, .inj-move-slot')?.querySelector('.auto-badge')?.remove();
    }
  }

  // Live badge refresh — update only the affected card's badge
  document.querySelectorAll('.inj-card-hdr').forEach(hdr => {
    const badge = hdr.querySelector('.inj-known, .inj-unknown, .inj-auto');
    if (!badge) return;
    const dot = hdr.querySelector('.side-dot');
    const sp  = hdr.querySelector('span:nth-child(2)');
    if (!dot || !sp) return;
    const pid = dot.classList.contains('p1-dot') ? 'p1' : 'p2';
    const st  = injBadge(`${pid}:${sp.textContent.trim()}`);
    badge.className   = st.cls;
    badge.textContent = st.txt;
  });
}

/**
 * "✨ Use Belief Integration" button.
 * Fetches Pikalytics suggestions from /fill-beliefs for the current battle's
 * VOD type (B: both sides; A/C: opponent side; D: nothing) and drops them
 * into the inject panel.  Never overwrites a field the user already filled
 * in manually; every filled field is flagged auto (purple AUTO badge) until
 * the user edits it.
 */
async function useBeliefIntegration() {
  if (!activeBattle) return;
  showNotif('Fetching belief data…');

  let res;
  try {
    res = await fillBeliefsViaServer(activeBattle);
  } catch (err) {
    showNotif('Belief fill failed: ' + err.message);
    return;
  }

  const sugg = res.suggestions || {};
  if (!Object.keys(sugg).length) {
    showNotif(res.note || 'Nothing to auto-fill for this VOD type');
    return;
  }

  const b = activeBattle;
  if (!b.known_team_overrides) b.known_team_overrides = {};
  if (!b.belief_autofill)      b.belief_autofill = {};
  if (!b.belief_meta)          b.belief_meta = {};

  let nFields = 0, nMons = 0;
  for (const [key, s] of Object.entries(sugg)) {
    if (!b.known_team_overrides[key]) b.known_team_overrides[key] = {};
    if (!b.belief_autofill[key])      b.belief_autofill[key] = {};
    const inj  = b.known_team_overrides[key];
    const auto = b.belief_autofill[key];
    b.belief_meta[key] = s.meta || {};
    let touched = false;

    if (s.nature && !inj.nature)   { inj.nature  = s.nature;  auto.nature  = true; nFields++; touched = true; }
    if (s.item && !inj.item)       { inj.item    = s.item;    auto.item    = true; nFields++; touched = true; }
    if (s.ability && !inj.ability) { inj.ability = s.ability; auto.ability = true; nFields++; touched = true; }

    const hasManualEvs = inj.ev_spread && Object.values(inj.ev_spread).some(v => v);
    if (s.ev_spread && !hasManualEvs) {
      inj.ev_spread = { ...s.ev_spread };
      auto.ev_spread = true;
      nFields++; touched = true;
    }

    if (s.moves && s.moves.some(m => m)) {
      if (!inj.moves)  inj.moves  = ['', '', '', ''];
      if (!auto.moves) auto.moves = [false, false, false, false];
      s.moves.forEach((m, i) => {
        if (m && !inj.moves[i]) { inj.moves[i] = m; auto.moves[i] = true; nFields++; touched = true; }
      });
    }
    if (touched) nMons++;
  }

  activeTab = 'inject';
  renderMain();
  const sides = (res.filled_sides || []).join(' + ') || 'none';
  // skipped = bench mons not brought in the VOD + species without usage data
  const skip  = res.skipped?.length ? ` · ${res.skipped.length} skipped` : '';
  showNotif(nFields
    ? `✨ Auto-filled ${nFields} fields on ${nMons} Pokémon (${sides})${skip}`
    : 'Nothing new to fill — all fields already set');
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

/**
 * Import a Showdown team paste and fill the inject panel for YOUR side (Type A).
 * Auto-detects which side is yours by matching the team against each side's
 * teampreview roster, writes exact stats into known_team_overrides, and marks
 * the battle as an own-VOD.  Requires a replay to be loaded first (the inject
 * cards are keyed by that replay's roster).
 */
async function importTeamFile(file) {
  if (!activeBattle) { alert('Load a replay first, then import your team.'); return; }

  let text;
  try { text = await file.text(); }
  catch (err) { alert('Could not read team file: ' + err.message); return; }

  const mons = parseShowdownTeam(text);
  if (!mons.length) { alert('No Pokémon found in that team file.'); return; }

  const b = activeBattle;

  // ── Auto-detect your side by roster overlap ──────────────────────────────
  const wanted = new Set(mons.map(m => dexKey(teamBaseSpecies(m.species))));
  let side = b.players.our_side, best = -1;
  for (const pid of ['p1', 'p2']) {
    const overlap = (b.players[pid]?.roster || [])
      .filter(r => wanted.has(dexKey(r))).length;
    if (overlap > best) { best = overlap; side = pid; }
  }
  if (best > 0) b.players.our_side = side;

  // ── Fill inject cards for that side ───────────────────────────────────────
  if (!b.known_team_overrides) b.known_team_overrides = {};
  const roster = b.players[side]?.roster || [];
  let filled = 0;
  for (const mon of mons) {
    const base = teamBaseSpecies(mon.species);
    // Anchor to an actual roster entry when present so the key matches a card.
    const rosterSp = roster.find(r => dexKey(r) === dexKey(base)) || base;
    const ev = {};
    for (const s of STAT_NAMES) ev[s] = mon.evs[s] || 0;
    b.known_team_overrides[`${side}:${rosterSp}`] = {
      nature:    mon.nature  || null,
      item:      mon.item    || null,
      ability:   mon.ability || null,
      ev_spread: ev,
      moves:     [0, 1, 2, 3].map(i => mon.moves[i] || ''),
    };
    filled++;
  }

  // Importing your own team makes this a Type A own-VOD.
  activeTab = 'inject';
  setSourceType('own_vod');   // sets stats_quality + re-renders
  const matched = roster.filter(r => wanted.has(dexKey(r))).length;
  showNotif(`📋 Imported ${filled} Pokémon onto ${side.toUpperCase()} (your side) · `
    + `${matched}/${roster.length || '?'} matched this VOD's roster`);
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
  const teamFileIn   = document.getElementById('team-file');

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
  teamFileIn.addEventListener('change',   e => { if (e.target.files[0]) importTeamFile(e.target.files[0]); e.target.value = ''; });

  // "📋 Import team" — wired by id (text-match loop above skips it)
  document.getElementById('import-team-btn')?.addEventListener('click', () => teamFileIn.click());

  // Sidebar "New battle" button delegates to the header button
  document.querySelector('#side .btn')?.addEventListener('click', () => {
    document.querySelectorAll('#hdr button')[0].click();
  });

  // Server health check on boot + every 15 s; pokedex loads alongside
  loadPokedex();
  checkServer();
  setInterval(checkServer, 15000);
});
